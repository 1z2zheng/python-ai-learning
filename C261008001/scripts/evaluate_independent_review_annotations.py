"""Evaluate real adjudicated labels only; empty templates yield pending status."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "review_analysis"


def main():
    annotation = pd.read_csv(OUT / "review_manual_annotation_template.csv").fillna("")
    labels = {"正向", "中性", "混合", "负向"}
    pairs = annotation[annotation["annotator_a_sentiment"].isin(labels) & annotation["annotator_b_sentiment"].isin(labels)]
    adjudicated = annotation[annotation["adjudicated_sentiment"].isin(labels)]
    result = {"status": "pending", "independent_double_labels": len(pairs),
              "adjudicated_labels": len(adjudicated), "cohen_kappa": None,
              "macro_f1": None, "note": "不得以脚本自动预测充当独立人工参考标签。"}
    if len(pairs):
        agreement = pairs["annotator_a_sentiment"].eq(pairs["annotator_b_sentiment"]).mean()
        chance = sum(pairs["annotator_a_sentiment"].eq(label).mean() * pairs["annotator_b_sentiment"].eq(label).mean() for label in labels)
        result["cohen_kappa"] = (agreement - chance) / (1 - chance) if chance < 1 else None
    if len(adjudicated):
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from esports_analysis.review_cleaning import classify_text_sentiment
        truth = adjudicated["adjudicated_sentiment"]
        predicted = adjudicated["content_clean"].map(lambda text: classify_text_sentiment(text)[0])
        metrics = []
        for label in sorted(labels):
            tp = int((truth.eq(label) & predicted.eq(label)).sum())
            fp = int((~truth.eq(label) & predicted.eq(label)).sum())
            fn = int((truth.eq(label) & ~predicted.eq(label)).sum())
            metrics.append({"label": label, "support": int(truth.eq(label).sum()),
                            "precision": tp / (tp + fp) if tp + fp else 0,
                            "recall": tp / (tp + fn) if tp + fn else 0,
                            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0})
        result["status"] = "partial_validation_requires_heldout_split"
        result["class_metrics"] = metrics
        result["macro_f1"] = sum(m["f1"] for m in metrics) / len(metrics)
        pd.crosstab(truth, predicted, rownames=["reference"], colnames=["prediction"]).to_csv(OUT / "manual_confusion_matrix.csv", encoding="utf-8-sig")
    (OUT / "independent_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
