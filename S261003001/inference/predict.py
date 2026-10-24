"""Run disease classification and lesion segmentation on one image."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy import ndimage
from torchvision import transforms
from torchvision.transforms import functional as TF

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from classification.train import build_model
from segmentation.train_deeplabv3plus import (
    DeepLabV3PlusMobileNetV3,
    MEAN,
    STD,
)


DEFAULT_CLASSIFIER = PROJECT / "runs" / "mobilenet_v3_large_plantdoc" / "best.pt"
DEFAULT_SEGMENTER = PROJECT / "runs" / "deeplabv3plus_plantseg" / "best.pt"
DEFAULT_OUTPUT = PROJECT / "runs" / "unified_inference"

CHINESE_NAMES = {
    "Apple_Apple_scab": "苹果黑星病",
    "Apple_Black_rot": "苹果黑腐病",
    "Apple_Cedar_apple_rust": "苹果雪松锈病",
    "Apple_healthy": "健康苹果",
    "Potato_Early_blight": "马铃薯早疫病",
    "Potato_Late_blight": "马铃薯晚疫病",
    "Potato_healthy": "健康马铃薯",
    "Tomato_Early_blight": "番茄早疫病",
    "Tomato_Late_blight": "番茄晚疫病",
    "Tomato_Leaf_Mold": "番茄叶霉病",
    "Tomato_healthy": "健康番茄",
}


def severity_level(lesion_ratio: float, healthy: bool, segmented: bool) -> str:
    """Reference level based on lesion pixels / all image pixels, not leaf area."""
    if healthy:
        return "不适用（分类为健康）"
    if not segmented:
        return "未评估"
    if lesion_ratio <= 0:
        return "未检出病斑"
    if lesion_ratio < 0.10:
        return "轻度（参考）"
    if lesion_ratio < 0.25:
        return "中度（参考）"
    return "重度（参考）"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input JPG/PNG image")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--segmenter", type=Path, default=DEFAULT_SEGMENTER)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.60,
        help="Predictions below this confidence are marked uncertain",
    )
    parser.add_argument(
        "--force-segmentation",
        action="store_true",
        help="Run segmentation even when the predicted class is healthy",
    )
    parser.add_argument(
        "--segmentation-threshold",
        type=float,
        default=None,
        help="Override the checkpoint threshold; a higher value produces smaller masks",
    )
    parser.add_argument(
        "--segmentation-size",
        type=int,
        default=None,
        help="Override segmentation inference size, for example 512",
    )
    parser.add_argument(
        "--min-component-percent",
        type=float,
        default=0.03,
        help="Remove connected components smaller than this percentage of the mask",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_classifier(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    model = build_model(
        checkpoint["model_name"], len(classes), pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    image_size = int(checkpoint.get("image_size", 224))
    transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return model, classes, transform, checkpoint


def classify(
    image: Image.Image,
    model: torch.nn.Module,
    classes: list[str],
    transform,
    device: torch.device,
    top_k: int,
) -> list[dict[str, object]]:
    with torch.inference_mode():
        logits = model(transform(image).unsqueeze(0).to(device))
        probabilities = logits.softmax(dim=1)[0]
    count = min(max(top_k, 1), len(classes))
    values, indices = probabilities.topk(count)
    return [
        {
            "rank": rank,
            "class": classes[index],
            "name_zh": CHINESE_NAMES.get(classes[index], classes[index]),
            "confidence": float(value),
        }
        for rank, (value, index) in enumerate(
            zip(values.cpu().tolist(), indices.cpu().tolist()), start=1
        )
    ]


def load_segmenter(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_name") != "deeplabv3plus_mobilenetv3_large":
        raise ValueError(f"Unsupported segmentation model: {checkpoint.get('model_name')}")
    model = DeepLabV3PlusMobileNetV3(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def segment(
    image: Image.Image,
    model: torch.nn.Module,
    checkpoint: dict,
    device: torch.device,
    threshold_override: float | None = None,
    image_size_override: int | None = None,
    min_component_percent: float = 0.0,
) -> tuple[Image.Image, Image.Image, float]:
    size = int(image_size_override or checkpoint["image_size"])
    resized = TF.resize(image, [size, size], antialias=True)
    tensor = TF.normalize(TF.to_tensor(resized), MEAN, STD).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type, enabled=device.type == "cuda"
    ):
        probability = torch.sigmoid(model(tensor))[0, 0].float().cpu().numpy()
    threshold = float(
        threshold_override
        if threshold_override is not None
        else checkpoint.get("threshold", 0.5)
    )
    binary_mask = probability >= threshold
    if min_component_percent > 0:
        labels, component_count = ndimage.label(
            binary_mask, structure=np.ones((3, 3), dtype=np.uint8)
        )
        if component_count:
            component_sizes = np.bincount(labels.reshape(-1))
            minimum_pixels = max(
                1, math.ceil(binary_mask.size * min_component_percent / 100.0)
            )
            keep = component_sizes >= minimum_pixels
            keep[0] = False
            binary_mask = keep[labels]
    small_mask = Image.fromarray(binary_mask.astype(np.uint8) * 255)
    mask = small_mask.resize(image.size, Image.Resampling.NEAREST)
    mask_array = np.asarray(mask, dtype=np.uint8) > 0
    lesion_ratio = float(mask_array.mean())

    source = np.asarray(image, dtype=np.float32).copy()
    red = np.zeros_like(source)
    red[..., 0] = 255
    source[mask_array] = source[mask_array] * 0.55 + red[mask_array] * 0.45
    overlay = Image.fromarray(np.clip(source, 0, 255).astype(np.uint8))
    return mask, overlay, lesion_ratio


def safe_stem(path: Path) -> str:
    value = "".join(character if character.isalnum() or character in "-_" else "_" for character in path.stem)
    return value[:100] or "prediction"


def main() -> None:
    args = parse_args()
    image_path = args.image.resolve()
    classifier_path = args.classifier.resolve()
    segmenter_path = args.segmenter.resolve()
    for path, label in (
        (image_path, "input image"),
        (classifier_path, "classification checkpoint"),
        (segmenter_path, "segmentation checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    output_dir = args.output.resolve() / safe_stem(image_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    started = time.perf_counter()
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    classifier, classes, classifier_transform, classifier_checkpoint = load_classifier(
        classifier_path, device
    )
    predictions = classify(
        image, classifier, classes, classifier_transform, device, args.top_k
    )
    primary = predictions[0]
    predicted_class = str(primary["class"])
    confidence = float(primary["confidence"])
    healthy = predicted_class.endswith("_healthy")
    uncertain = confidence < args.confidence_threshold
    should_segment = args.force_segmentation or not healthy

    mask_path = None
    overlay_path = None
    lesion_ratio = 0.0
    segmentation_threshold = None
    if should_segment:
        segmenter, segmentation_checkpoint = load_segmenter(segmenter_path, device)
        mask, overlay, lesion_ratio = segment(
            image,
            segmenter,
            segmentation_checkpoint,
            device,
            threshold_override=args.segmentation_threshold,
            image_size_override=args.segmentation_size,
            min_component_percent=args.min_component_percent,
        )
        segmentation_threshold = float(
            args.segmentation_threshold
            if args.segmentation_threshold is not None
            else segmentation_checkpoint.get("threshold", 0.5)
        )
        mask_path = output_dir / "lesion_mask.png"
        overlay_path = output_dir / "lesion_overlay.jpg"
        mask.save(mask_path)
        overlay.save(overlay_path, quality=95)

    original_path = output_dir / "input.jpg"
    image.save(original_path, quality=95)
    elapsed = time.perf_counter() - started
    result = {
        "input": str(image_path),
        "device": str(device),
        "classification": {
            "predicted_class": predicted_class,
            "predicted_name_zh": primary["name_zh"],
            "confidence": confidence,
            "confidence_threshold": args.confidence_threshold,
            "uncertain": uncertain,
            "healthy": healthy,
            "top_k": predictions,
            "model": classifier_checkpoint["model_name"],
        },
        "segmentation": {
            "executed": should_segment,
            "model": "deeplabv3plus_mobilenetv3_large" if should_segment else None,
            "threshold": segmentation_threshold,
            "lesion_ratio": lesion_ratio,
            "lesion_percent_of_image": lesion_ratio * 100.0,
            "severity_reference": severity_level(lesion_ratio, healthy, should_segment),
            "mask": str(mask_path) if mask_path else None,
            "overlay": str(overlay_path) if overlay_path else None,
        },
        "elapsed_seconds": elapsed,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    status = "置信度较低，请人工复核" if uncertain else "预测完成"
    print(f"状态: {status}")
    print(f"类别: {primary['name_zh']} ({predicted_class})")
    print(f"置信度: {confidence:.2%}")
    if should_segment:
        print(f"病斑占整幅图像比例: {lesion_ratio:.2%}")
        print(f"叠加图: {overlay_path}")
    else:
        print("分类结果为健康，已跳过病斑分割")
    print(f"结果: {result_path}")


if __name__ == "__main__":
    main()
