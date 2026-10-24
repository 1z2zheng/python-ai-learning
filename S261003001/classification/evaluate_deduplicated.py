"""Re-evaluate classifiers after excluding cross-split near-duplicate images.

The exclusion rule uses only image hashes, never model predictions. Dataset
files and the original split manifest remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from classification.analyze_weak_class import predict  # noqa: E402
from classification.train_mixed import plantseg_records  # noqa: E402
from inference.predict import DEFAULT_CLASSIFIER  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "dataset_segmentation_plantseg")
    parser.add_argument("--candidates", type=Path, default=PROJECT / "runs/plantseg_split_audit/cross_split_candidates.csv")
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/plantseg_split_audit")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--baseline", type=Path, default=PROJECT / "runs/mobilenet_v3_large_mixed/best.pt")
    return parser.parse_args()


def exclusions(candidates: Path) -> dict[str, list[str]]:
    excluded: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    with candidates.open(newline="", encoding="utf-8-sig") as file:
        for pair in csv.DictReader(file):
            sides = [(pair["a_split"], Path(pair["a_path"]).stem), (pair["b_split"], Path(pair["b_path"]).stem)]
            if pair["a_class"] != pair["b_class"]:
                # The same photo has conflicting labels: do not guess either one.
                for split, stem in sides:
                    excluded[split].add(stem)
            elif any(split == "train" for split, _ in sides):
                for split, stem in sides:
                    if split != "train":
                        excluded[split].add(stem)
            elif {split for split, _ in sides} == {"val", "test"}:
                for split, stem in sides:
                    if split == "test":
                        excluded["test"].add(stem)
    return {name: sorted(stems) for name, stems in excluded.items()}


def evaluate(records, predictions, classes):
    true = [record.class_name for record in records]
    predicted = [row["class"] for row in predictions]
    present = sorted(set(true))
    details = classification_report(
        true, predicted, labels=present, output_dict=True, zero_division=0
    )
    return {
        "samples": len(records),
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(details["macro avg"]["f1-score"]),
        "apple_black_rot": details.get("Apple_Black_rot", {}),
        "support_by_class": dict(Counter(true)),
    }


def main():
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    excluded = exclusions(args.candidates.resolve())
    original = json.loads((args.data.resolve() / "splits.json").read_text(encoding="utf-8"))
    cleaned = {split: [stem for stem in stems if stem not in set(excluded.get(split, []))] for split, stems in original.items()}
    (output / "splits_deduplicated.json").write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "excluded_images.json").write_text(json.dumps(excluded, ensure_ascii=False, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {"exclusions": excluded, "models": {}}
    for label, path in (("current", args.checkpoint), ("baseline", args.baseline)):
        results["models"][label] = {}
        for split in ("val", "test"):
            full = plantseg_records(args.data.resolve(), split, classes)
            removed = set(excluded[split])
            keep = [record for record in full if record.path.stem not in removed]
            predictions = predict(path.resolve(), full, device, batch_size=16)
            filtered_predictions = [prediction for record, prediction in zip(full, predictions, strict=True) if record.path.stem not in removed]
            results["models"][label][split] = {
                "original": evaluate(full, predictions, classes),
                "deduplicated": evaluate(keep, filtered_predictions, classes),
            }
    (output / "deduplicated_evaluation.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {
        label: {split: {
            state: {key: metrics[key] for key in ("samples", "accuracy", "macro_f1")}
            for state, metrics in splits[split].items()
        } for split in ("val", "test")}
        for label, splits in results["models"].items()
    }
    print(json.dumps({"exclusions": excluded, "results": compact}, ensure_ascii=False, indent=2))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
