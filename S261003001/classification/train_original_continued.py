"""Conservatively continue the deployed classifier with validation-gated saving.

Starts from the actual original checkpoint. This is a warm-start experiment:
using a reviewed split now does not erase the original model's training history.
Test data is evaluated only after model selection, never to choose an epoch.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

try:
    from .train import build_model
    from .train_mixed import (MixedDataset, Record, evaluation_report, make_transforms,
                             plantseg_records, plantvillage_records, read_classes,
                             remove_cross_domain_leakage, run_epoch, seed_everything)
    from .train_refined import extra_records
except ImportError:
    from train import build_model
    from train_mixed import (MixedDataset, Record, evaluation_report, make_transforms,
                            plantseg_records, plantvillage_records, read_classes,
                            remove_cross_domain_leakage, run_epoch, seed_everything)
    from train_refined import extra_records

PROJECT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "runs/mobilenet_v3_large_plantdoc/best.pt")
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/mobilenet_v3_large_original_continued")
    parser.add_argument("--plantseg-splits", type=Path, default=PROJECT / "runs/plantseg_split_audit/splits_deduplicated.json")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--backbone-lr", type=float, default=2e-6)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--distillation-weight", type=float, default=0.7)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--check-only", action="store_true", help="Check data/model and a forward pass; do not train")
    args = parser.parse_args()
    if min(args.epochs, args.steps_per_epoch, args.batch_size, args.patience) < 1:
        parser.error("Epochs, steps, batch size and patience must be positive")
    if min(args.backbone_lr, args.head_lr) <= 0 or args.distillation_weight < 0:
        parser.error("Learning rates must be positive; distillation weight must be nonnegative")
    return args


def source_records(records):
    return [Record(r.path, r.label, r.class_name,
                   "appleleaf9" if r.path.name.startswith("appleleaf9_blackrot_") else "plantdoc")
            for r in records]


def sampler_for(records, samples, seed):
    mass = {"plantvillage": 0.30, "plantseg": 0.40, "plantdoc": 0.20, "appleleaf9": 0.10}
    counts = Counter((r.class_name, r.domain) for r in records)
    available = {}
    for r in records:
        available.setdefault(r.class_name, set()).add(r.domain)
    weights = [mass[r.domain] / sum(mass[d] for d in available[r.class_name])
               / counts[(r.class_name, r.domain)] for r in records]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), samples,
                                 replacement=True, generator=torch.Generator().manual_seed(seed))


def conservative_loss(logits, teacher_logits, labels, weight):
    ce = F.cross_entropy(logits, labels, label_smoothing=0.02)
    temperature = 2.0
    anchor = F.kl_div(F.log_softmax(logits / temperature, dim=1),
                      F.softmax(teacher_logits.detach() / temperature, dim=1),
                      reduction="batchmean") * temperature ** 2
    return ce + weight * anchor


def configure_student(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Adapt only the final feature blocks and classifier; retain early features.
    for module in (model.features[-4:], model.classifier):
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def freeze_batchnorm(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def validation_summary(model, loaders, device, classes):
    result = {}
    for domain in ("pv", "ps"):
        metrics, targets, predicted = run_epoch(model, loaders[domain + "_val"], nn.CrossEntropyLoss(), device)
        metrics["macro_f1_present"] = float(f1_score(targets, predicted, labels=sorted(set(targets)),
                                                    average="macro", zero_division=0))
        result[domain] = metrics
    return result


def eligible(candidate, baseline):
    return (candidate["ps"]["accuracy"] >= baseline["ps"]["accuracy"] - 1e-12
            and candidate["ps"]["macro_f1_present"] >= baseline["ps"]["macro_f1_present"] - 1e-12
            and candidate["pv"]["accuracy"] >= baseline["pv"]["accuracy"] - 0.005 - 1e-12)


def score(summary):
    return 0.6 * summary["ps"]["accuracy"] + 0.4 * summary["ps"]["macro_f1_present"]


def main():
    args = parse_args()
    seed_everything(args.seed)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Preserve the existing experiment; choose a new --output: {output}")
    checkpoint_path, split_path = args.checkpoint.resolve(), args.plantseg_splits.resolve()
    initial = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pv_root, ps_root = PROJECT / "dataset_classification", PROJECT / "dataset_segmentation_plantseg"
    classes = read_classes(pv_root)
    if initial["classes"] != classes or initial["model_name"] != "mobilenet_v3_large":
        raise ValueError("Expected the original MobileNetV3-Large with matching class order")
    pv = {s: plantvillage_records(pv_root, s, classes) for s in ("train", "val", "test")}
    ps = {s: plantseg_records(ps_root, s, classes, split_path) for s in ("train", "val", "test")}
    extra = source_records(extra_records(PROJECT / "dataset_natural_extra", classes))
    size = int(initial.get("image_size", 320))
    _, evaluate_transform = make_transforms(size)
    train_transform = transforms.Compose([
        transforms.Resize(int(size * 1.14)), transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02),
        transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = build_model("mobilenet_v3_large", len(classes), pretrained=False).to(device)
    student.load_state_dict(initial["model_state"])
    configure_student(student)
    if args.check_only:
        sample = MixedDataset([ps["train"][0]], train_transform)[0][0].unsqueeze(0).to(device)
        student.eval()
        with torch.no_grad():
            logits = student(sample)
        assert logits.shape == (1, len(classes)) and torch.isfinite(logits).all()
        print(json.dumps({"checkpoint": str(checkpoint_path), "device": str(device),
                          "plantseg": {s: len(v) for s, v in ps.items()},
                          "extra_by_source": dict(Counter(r.domain for r in extra)),
                          "trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
                          "forward_shape": list(logits.shape), "training_started": False}, indent=2))
        return

    pv_train, ps_train, dedup = remove_cross_domain_leakage(pv["train"], ps["train"],
                                                         pv["val"] + pv["test"], ps["val"] + ps["test"])
    records = pv_train + ps_train + extra
    train_loader = DataLoader(MixedDataset(records, train_transform), batch_size=args.batch_size,
                              sampler=sampler_for(records, args.steps_per_epoch * args.batch_size, args.seed),
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    loaders = {domain + "_" + split: DataLoader(MixedDataset(data[split], evaluate_transform),
               batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
               for domain, data in (("pv", pv), ("ps", ps)) for split in ("val", "test")}
    teacher = copy.deepcopy(student).eval()
    teacher.requires_grad_(False)
    baseline = validation_summary(teacher, loaders, device, classes)
    best_validation, best_score, best_epoch = baseline, score(baseline), 0
    output.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    provenance = {"warm_start_checkpoint": str(checkpoint_path), "reviewed_split_used_for_continuation": str(split_path),
                  "inherited_training_history": "Original deployed checkpoint was trained on the original split; this is not a clean-from-scratch experiment",
                  "extra_by_source": dict(Counter(r.domain for r in extra)), "deduplication": dedup,
                  "selection": "Validation only; PS accuracy and macro-F1 must both retain baseline; PV accuracy drop <=0.005"}
    saved = dict(initial)
    saved.update(continuation=provenance, continuation_epoch=0)
    torch.save(saved, output / "best.pt")
    (output / "config.json").write_text(json.dumps({"arguments": config, "provenance": provenance}, indent=2), encoding="utf-8")
    print(f"Original validation: PS_acc={baseline['ps']['accuracy']:.4f}, PS_F1={baseline['ps']['macro_f1_present']:.4f}, PV_acc={baseline['pv']['accuracy']:.4f}", flush=True)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in student.features.parameters() if p.requires_grad], "lr": args.backbone_lr},
        {"params": student.classifier.parameters(), "lr": args.head_lr}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, stale = [], 0
    for epoch in range(1, args.epochs + 1):
        student.train()
        freeze_batchnorm(student)
        losses = []
        for step, (images, labels, _, _) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                with torch.no_grad():
                    teacher_logits = teacher(images)
                loss = conservative_loss(student(images), teacher_logits, labels, args.distillation_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            if step % 50 == 0:
                print(f"Epoch {epoch}: {step}/{args.steps_per_epoch} batches, loss={np.mean(losses):.4f}", flush=True)
        scheduler.step()
        validation = validation_summary(student, loaders, device, classes)
        allowed = eligible(validation, baseline)
        improved = allowed and score(validation) > best_score + 1e-8
        if improved:
            best_validation, best_score, best_epoch, stale = validation, score(validation), epoch, 0
            saved = dict(initial)
            saved.update(model_state=student.state_dict(), continuation=provenance, continuation_epoch=epoch,
                         continuation_validation=validation)
            torch.save(saved, output / "best.pt")
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation,
                        "validation_gate_passed": allowed, "saved_as_best": improved})
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Epoch {epoch}: PS_acc={validation['ps']['accuracy']:.4f}, PS_F1={validation['ps']['macro_f1_present']:.4f}, PV_acc={validation['pv']['accuracy']:.4f}, saved={improved}", flush=True)
        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    chosen = torch.load(output / "best.pt", map_location=device, weights_only=False)
    student.load_state_dict(chosen["model_state"])
    # Only now touch test data: compare with the actual original, not a new baseline.
    tests = {}
    for name, model in (("original", teacher), ("selected", student)):
        tests[name] = {}
        for domain in ("pv", "ps"):
            metrics, targets, predicted = run_epoch(model, loaders[domain + "_test"], nn.CrossEntropyLoss(), device)
            report = evaluation_report(metrics, targets, predicted, classes)
            report["macro_f1_present"] = float(f1_score(targets, predicted, labels=sorted(set(targets)), average="macro", zero_division=0))
            tests[name][domain] = report
    result = {"classes": classes, "baseline_validation": baseline, "best_validation": best_validation,
              "best_epoch": best_epoch, "validation_improved": best_epoch > 0,
              "tests_after_selection": tests, "history": history, "provenance": provenance,
              "automatic_deployment": False}
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Best epoch: {best_epoch} (0 means original retained)")
    for name in ("original", "selected"):
        report = tests[name]["ps"]
        print(f"{name} PS test: accuracy={report['accuracy']:.4f}, macro_F1={report['macro_f1_present']:.4f}")
    print(f"Saved to {output}. App checkpoint unchanged.")


if __name__ == "__main__":
    main()
