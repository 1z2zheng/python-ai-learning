from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation_utils import contact_sheet, metrics as binary_metrics, overlay_comparison
from cropseg.data import SegmentationDataset, find_image
from cropseg.metrics import SegmentationMetrics
from cropseg.model import build_model, normalize_model_name
from cropseg.utils import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="在独立数据划分上评估语义分割模型")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="可选：覆盖checkpoint中记录的数据目录，用于在同一测试集上公平比较模型",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    data_root = args.data_root if args.data_root is not None else Path(config["data_root"])
    dataset = SegmentationDataset(
        data_root,
        args.split,
        config["image_size"],
        config["num_classes"],
        config.get("channel_mode", "RGB"),
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=config.get("batch_size", 4), shuffle=False)
    device = choose_device(config.get("device", "auto"))
    model_name = normalize_model_name(checkpoint.get("model_name", config.get("model", "unet")))
    model = build_model(
        model_name,
        checkpoint["in_channels"],
        config["num_classes"],
        config.get("base_channels", 32),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    aggregate = SegmentationMetrics(config["num_classes"])
    per_image = []
    previews: list[tuple[str, Image.Image]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"评估{args.split}"):
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                predictions = model(images).argmax(dim=1)
            aggregate.update(predictions, targets)
            for index, stem in enumerate(batch["stem"]):
                prediction = predictions[index].detach().cpu().numpy().astype(np.uint8)
                target = targets[index].detach().cpu().numpy().astype(np.uint8)
                score = binary_metrics(prediction, target)
                per_image.append({"image_id": stem, **score})

                image_path = find_image(dataset.images_dir, stem)
                mask_path = find_image(dataset.masks_dir, stem)
                with Image.open(image_path) as opened:
                    original = opened.convert("RGB")
                with Image.open(mask_path) as opened:
                    original_target = np.asarray(opened.convert("L"))
                original_prediction = np.asarray(
                    Image.fromarray(prediction).resize(original.size, Image.Resampling.NEAREST)
                )
                previews.append(
                    (
                        f"{stem} | wheat IoU={score['iou']:.3f}",
                        overlay_comparison(original, original_target, original_prediction),
                    )
                )

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "model_name": model_name,
        "parameter_count": checkpoint.get(
            "parameter_count", sum(parameter.numel() for parameter in model.parameters())
        ),
        "split": args.split,
        "sample_count": len(dataset),
        "aggregate": aggregate.compute(),
        "per_image": per_image,
        "preview_legend": {
            "green": "预测与真实Mask重合",
            "red": "漏检",
            "blue": "误检",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{args.split}_metrics.json"
    preview_path = args.output_dir / f"{args.split}_predictions.jpg"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_sheet(previews, preview_path)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(f"指标：{metrics_path}")
    print(f"预测总览：{preview_path}")


if __name__ == "__main__":
    main()
