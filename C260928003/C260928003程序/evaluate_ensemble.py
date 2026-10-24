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
    parser = argparse.ArgumentParser(description="评估多个U-Net检查点的logits平均集成")
    parser.add_argument("--checkpoints", required=True, type=Path, nargs="+")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False) for path in args.checkpoints
    ]
    config = checkpoints[0]["config"]
    architecture = (
        normalize_model_name(checkpoints[0].get("model_name", config.get("model", "unet"))),
        checkpoints[0]["in_channels"],
        config["num_classes"],
        config.get("base_channels", 32),
        config["image_size"],
        config.get("channel_mode", "RGB"),
    )
    for checkpoint in checkpoints[1:]:
        other = checkpoint["config"]
        other_architecture = (
            normalize_model_name(checkpoint.get("model_name", other.get("model", "unet"))),
            checkpoint["in_channels"],
            other["num_classes"],
            other.get("base_channels", 32),
            other["image_size"],
            other.get("channel_mode", "RGB"),
        )
        if other_architecture != architecture:
            raise ValueError("集成检查点的网络结构或输入配置不一致")

    dataset = SegmentationDataset(
        args.data_root,
        args.split,
        config["image_size"],
        config["num_classes"],
        config.get("channel_mode", "RGB"),
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=config.get("batch_size", 4), shuffle=False)
    device = choose_device(config.get("device", "auto"))
    models = []
    for checkpoint in checkpoints:
        model = build_model(
            architecture[0],
            checkpoint["in_channels"],
            config["num_classes"],
            config.get("base_channels", 32),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)

    aggregate = SegmentationMetrics(config["num_classes"])
    per_image = []
    previews: list[tuple[str, Image.Image]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"集成评估{args.split}"):
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                mean_logits = torch.stack([model(images) for model in models]).mean(dim=0)
                predictions = mean_logits.argmax(dim=1)
            aggregate.update(predictions, targets)
            for index, stem in enumerate(batch["stem"]):
                prediction = predictions[index].cpu().numpy().astype(np.uint8)
                target = targets[index].cpu().numpy().astype(np.uint8)
                score = binary_metrics(prediction, target)
                per_image.append({"image_id": stem, **score})
                with Image.open(find_image(dataset.images_dir, stem)) as opened:
                    original = opened.convert("RGB")
                with Image.open(find_image(dataset.masks_dir, stem)) as opened:
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
        "method": "mean_logits_ensemble",
        "checkpoints": [str(path.resolve()) for path in args.checkpoints],
        "checkpoint_epochs": [checkpoint["epoch"] for checkpoint in checkpoints],
        "split": args.split,
        "sample_count": len(dataset),
        "aggregate": aggregate.compute(),
        "per_image": per_image,
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
