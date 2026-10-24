from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from predict import load_models


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINTS = [
    PROJECT_DIR / "runs" / "ablation_multiscale_seed42" / "best.pt",
    PROJECT_DIR / "runs" / "ablation_multiscale_seed123" / "best.pt",
    PROJECT_DIR / "runs" / "ablation_multiscale_seed2026" / "best.pt",
]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(callable_, device: torch.device, warmup: int = 20, repeats: int = 100) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            callable_()
        synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            callable_()
        synchronize(device)
    mean_ms = (time.perf_counter() - start) * 1000.0 / repeats
    return {
        "warmup_iterations": warmup,
        "timed_iterations": repeats,
        "mean_latency_ms": mean_ms,
        "fps": 1000.0 / mean_ms,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, config = load_models(CHECKPOINTS, device)
    size = int(config["image_size"])
    in_channels = int(next(models[0].parameters()).shape[1])
    sample = torch.randn(1, in_channels, size, size, device=device)
    autocast_enabled = device.type == "cuda"

    def single_forward() -> torch.Tensor:
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            return models[1](sample).argmax(dim=1)

    def ensemble_forward() -> torch.Tensor:
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            logits = torch.stack([model(sample) for model in models]).mean(dim=0)
            return logits.argmax(dim=1)

    parameter_count = sum(parameter.numel() for parameter in models[0].parameters())
    checkpoint_sizes = [path.stat().st_size for path in CHECKPOINTS]
    result = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "input_shape": [1, in_channels, size, size],
        "precision": "automatic mixed precision" if autocast_enabled else "float32",
        "single_model": {
            "checkpoint": str(CHECKPOINTS[1].relative_to(PROJECT_DIR)),
            "parameters": parameter_count,
            "checkpoint_size_mb": checkpoint_sizes[1] / 1024**2,
            **benchmark(single_forward, device),
        },
        "three_model_ensemble": {
            "checkpoints": [str(path.relative_to(PROJECT_DIR)) for path in CHECKPOINTS],
            "total_parameters_loaded": parameter_count * len(models),
            "total_checkpoint_size_mb": sum(checkpoint_sizes) / 1024**2,
            **benchmark(ensemble_forward, device),
        },
        "scope": "forward pass plus argmax; excludes image decoding, resize, disk IO and visualization",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
