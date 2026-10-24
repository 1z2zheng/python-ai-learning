"""Predict the disease class of one leaf image."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from train import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    size = checkpoint.get("image_size", 224)
    model = build_model(checkpoint["model_name"], len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(int(size * 1.14)), transforms.CenterCrop(size), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(args.image).convert("RGB")
    with torch.inference_mode():
        probabilities = model(transform(image).unsqueeze(0).to(device)).softmax(1)[0]
    k = min(args.top_k, len(classes))
    values, indices = probabilities.topk(k)
    for rank, (value, index) in enumerate(zip(values.tolist(), indices.tolist()), 1):
        print(f"{rank}. {classes[index]}: {value:.2%}")


if __name__ == "__main__":
    main()

