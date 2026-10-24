from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from cropseg.data import SegmentationDataset
from cropseg.losses import combined_loss
from cropseg.metrics import SegmentationMetrics
from cropseg.model import build_model, normalize_model_name
from cropseg.utils import choose_device, load_json, save_json, set_seed


def run_epoch(model, loader, device, config, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    metrics = SegmentationMetrics(config["num_classes"])
    loss_sum = 0.0
    amp_enabled = bool(config.get("amp", True) and device.type == "cuda")
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in tqdm(loader, leave=False, desc="训练" if training else "验证"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = combined_loss(
                    logits,
                    masks,
                    config["num_classes"],
                    config.get("cross_entropy_weight", 1.0),
                    config.get("dice_weight", 1.0),
                    config.get("ignore_background_in_dice", False),
                    config.get("foreground_ce_weight", 1.0),
                    config.get("tversky_weight", 0.0),
                    config.get("tversky_alpha", 0.3),
                    config.get("tversky_beta", 0.7),
                )
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            loss_sum += float(loss.detach()) * images.size(0)
            metrics.update(logits.argmax(dim=1), masks)
    result = metrics.compute()
    result["loss"] = loss_sum / max(len(loader.dataset), 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="训练语义分割模型")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    project_dir = Path(__file__).resolve().parent
    data_root = Path(config["data_root"])
    output_dir = Path(config["output_dir"])
    if not data_root.is_absolute():
        data_root = project_dir / data_root
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config["data_root"] = str(data_root.resolve())
    config["output_dir"] = str(output_dir.resolve())
    save_json(config, output_dir / "resolved_config.json")

    set_seed(int(config.get("seed", 42)))
    device = choose_device(config.get("device", "auto"))
    train_dataset = SegmentationDataset(
        data_root, "train", config["image_size"], config["num_classes"],
        config.get("channel_mode", "RGB"), augment=True,
        augmentation_config=config.get("augmentation", {}),
    )
    val_dataset = SegmentationDataset(
        data_root, "val", config["image_size"], config["num_classes"],
        config.get("channel_mode", "RGB"), augment=False,
    )
    loader_args = {
        "batch_size": config["batch_size"],
        "num_workers": config.get("num_workers", 0),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    model_name = normalize_model_name(config.get("model", "unet"))
    config["model"] = model_name
    save_json(config, output_dir / "resolved_config.json")
    model = build_model(
        model_name,
        train_dataset.in_channels,
        config["num_classes"],
        config.get("base_channels", 32),
    ).to(device)
    optimizer = AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config.get("weight_decay", 0.0)
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.get("amp", True) and device.type == "cuda"))
    history = []
    best_miou = -1.0
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"模型：{model_name}；参数量：{parameter_count:,}；设备：{device}；"
        f"训练样本：{len(train_dataset)}；验证样本：{len(val_dataset)}"
    )
    for epoch in range(1, int(config["epochs"]) + 1):
        train_result = run_epoch(model, train_loader, device, config, optimizer, scaler)
        val_result = run_epoch(model, val_loader, device, config)
        scheduler.step(val_result["miou"])
        record = {"epoch": epoch, "train": train_result, "val": val_result}
        history.append(record)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "in_channels": train_dataset.in_channels,
            "model_name": model_name,
            "parameter_count": parameter_count,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_result["miou"] > best_miou:
            best_miou = val_result["miou"]
            torch.save(checkpoint, output_dir / "best.pt")
        save_json(history, output_dir / "history.json")
        print(
            f"Epoch {epoch:03d} | train_loss={train_result['loss']:.4f} "
            f"| val_loss={val_result['loss']:.4f} | val_mIoU={val_result['miou']:.4f} "
            f"| val_F1={val_result['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()

