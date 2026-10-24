from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cropseg.data import CHANNELS, IMAGE_SUFFIXES
from cropseg.model import build_model, normalize_model_name


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINTS = [
    PROJECT_DIR / "runs" / "ablation_multiscale_seed42" / "best.pt",
    PROJECT_DIR / "runs" / "ablation_multiscale_seed123" / "best.pt",
    PROJECT_DIR / "runs" / "ablation_multiscale_seed2026" / "best.pt",
]
WHEAT_COLOR = np.asarray([40, 220, 90], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="U-Net单模型或多模型集成预测麦穗像素级Mask")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--checkpoint", type=Path, help="兼容旧用法：使用单个模型")
    model_group.add_argument("--checkpoints", type=Path, nargs="+", help="多模型logits平均集成")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=Path, help="预测单张图片")
    input_group.add_argument("--input-dir", type=Path, help="批量预测文件夹中的图片")
    parser.add_argument("--output", type=Path, help="兼容旧用法：单张Mask输出路径")
    parser.add_argument("--output-dir", type=Path, help="输出目录")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if args.image is not None:
        if not args.image.is_file():
            raise FileNotFoundError(f"找不到输入图片：{args.image}")
        return [args.image]
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"找不到输入目录：{args.input_dir}")
    images = sorted(
        path for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"输入目录中没有可支持的图片：{args.input_dir}")
    return images


def load_models(paths: list[Path], device: torch.device) -> tuple[list[torch.nn.Module], dict]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少模型文件：{missing}")
    checkpoints = [torch.load(path, map_location=device, weights_only=False) for path in paths]
    config = checkpoints[0]["config"]
    signature = (
        normalize_model_name(checkpoints[0].get("model_name", config.get("model", "unet"))),
        checkpoints[0]["in_channels"], config["num_classes"],
        config.get("base_channels", 32), config["image_size"],
        config.get("channel_mode", "RGB").upper(),
    )
    models = []
    for checkpoint in checkpoints:
        other = checkpoint["config"]
        other_signature = (
            normalize_model_name(checkpoint.get("model_name", other.get("model", "unet"))),
            checkpoint["in_channels"], other["num_classes"],
            other.get("base_channels", 32), other["image_size"],
            other.get("channel_mode", "RGB").upper(),
        )
        if other_signature != signature:
            raise ValueError("多个检查点的模型结构或输入配置不一致")
        model = build_model(signature[0], signature[1], signature[2], signature[3]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)
    return models, config


def make_overlay(original: Image.Image, mask: np.ndarray, alpha: float) -> Image.Image:
    array = np.asarray(original, dtype=np.uint8).copy()
    foreground = mask == 1
    blended = (1.0 - alpha) * array[foreground].astype(np.float32) + alpha * WHEAT_COLOR
    array[foreground] = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def predict_image(
    image_path: Path, models: list[torch.nn.Module], config: dict, device: torch.device,
) -> tuple[Image.Image, np.ndarray]:
    original = Image.open(image_path).convert("RGB")
    resized = original.resize((config["image_size"], config["image_size"]), Image.Resampling.BILINEAR)
    mode = config.get("channel_mode", "RGB").upper()
    array = np.asarray(resized, dtype=np.float32)[:, :, CHANNELS[mode]] / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        mean_logits = torch.stack([model(tensor) for model in models]).mean(dim=0)
        resized_mask = mean_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    mask = np.asarray(Image.fromarray(resized_mask).resize(original.size, Image.Resampling.NEAREST))
    return original, mask


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha必须介于0和1之间")
    image_paths = resolve_inputs(args)
    checkpoint_paths = (
        [args.checkpoint] if args.checkpoint is not None
        else (args.checkpoints if args.checkpoints is not None else DEFAULT_CHECKPOINTS)
    )
    if args.output is not None and len(image_paths) != 1:
        raise ValueError("--output只能用于单张图片，批量预测请使用--output-dir")
    output_dir = args.output_dir or (args.output.parent if args.output is not None else PROJECT_DIR / "predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    models, config = load_models(checkpoint_paths, device)
    print(f"设备：{device}；模型数：{len(models)}；图片数：{len(image_paths)}")

    records = []
    for index, image_path in enumerate(image_paths, start=1):
        original, mask = predict_image(image_path, models, config, device)
        mask_path = args.output if args.output is not None else output_dir / f"{image_path.stem}_mask.png"
        overlay_path = output_dir / f"{image_path.stem}_overlay.png"
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
        make_overlay(original, mask, args.overlay_alpha).save(overlay_path)
        foreground_pixels = int((mask == 1).sum())
        total_pixels = int(mask.size)
        record = {
            "image": str(image_path.resolve()), "width": original.width, "height": original.height,
            "mask": str(mask_path.resolve()), "overlay": str(overlay_path.resolve()),
            "wheat_pixels": foreground_pixels, "total_pixels": total_pixels,
            "wheat_coverage_ratio": foreground_pixels / total_pixels,
        }
        records.append(record)
        print(f"[{index}/{len(image_paths)}] {image_path.name} | 麦穗覆盖率={record['wheat_coverage_ratio']:.2%}")

    summary = {
        "method": "single_model" if len(models) == 1 else "mean_logits_ensemble",
        "checkpoints": [str(path.resolve()) for path in checkpoint_paths],
        "device": str(device), "image_count": len(records), "records": records,
    }
    summary_path = output_dir / "prediction_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"预测完成：{output_dir}")
    print(f"统计：{summary_path}")


if __name__ == "__main__":
    main()
