from __future__ import annotations

import json
import os
import platform
import sqlite3
import sys
import ctypes
import winreg
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.comparison import build_parameter_comparison, merge_comparison_metrics
from esports_analysis.config import get_database_path
from esports_analysis.overview import load_overview_data
from esports_analysis.preference import DEFAULT_WEIGHTS, calculate_dimension_scores, rank_products
from esports_analysis.pricing import load_price_history
from esports_analysis.review_analysis import classify_sentiment, extract_aspect_mentions


OUTPUT_DIR = ROOT / "output" / "validation"
REVIEW_PATH = ROOT / "data" / "processed" / "jd_reviews_20261009.csv"
ASPECT_NAMES = ["手感", "性能响应", "连接稳定", "舒适重量", "做工质量", "外观灯效", "价格价值", "声音"]


# 历史参考标签，来源与独立性未核验。不得作为当前版本人工验证证据。
MANUAL_ASPECTS = {
    "5a433d0dce71b74cc6191872": ["手感", "声音"],
    "ad0eec89980ba41414fa712c": ["手感", "声音", "舒适重量", "外观灯效", "性能响应", "做工质量"],
    "89e9becf1e6ee5ad35dcffb5": ["手感", "声音", "性能响应", "外观灯效", "舒适重量"],
    "15b4230ee716361cecd4b89a": ["外观灯效", "做工质量", "舒适重量", "性能响应"],
    "097524ca35bc514c5aff22bf": ["外观灯效", "性能响应", "连接稳定", "舒适重量", "手感"],
    "546b4a442f0aa4261d2491ed": ["手感", "舒适重量", "性能响应", "做工质量", "价格价值"],
    "9c389d8ff50905908d106df1": ["手感", "舒适重量", "声音"],
    "a75d3956439d236c36fde349": ["手感", "舒适重量"],
    "8649cb50620675db15f28cc6": ["手感", "舒适重量", "性能响应"],
    "fe4ba76a29cc34daab94e8e6": ["外观灯效", "性能响应", "做工质量", "舒适重量"],
    "8c08b29582b9f62730f8a2b4": ["舒适重量", "做工质量", "声音", "外观灯效"],
    "71b7df2e9c35dd296bd5f31b": ["做工质量", "性能响应", "连接稳定", "外观灯效", "舒适重量"],
    "87cffc2d28e40b0932b6d6e2": ["外观灯效", "连接稳定", "性能响应", "舒适重量", "做工质量"],
    "93c92f0b6d541cedcac11dce": ["外观灯效", "手感", "舒适重量", "性能响应"],
    "b3ca67db78a61a9c90e64f06": ["做工质量", "手感", "舒适重量", "性能响应"],
    "1a5f414caa249aeff577911a": ["价格价值", "手感", "舒适重量", "声音", "性能响应", "连接稳定", "外观灯效"],
    "704bae75d730873596faeca5": ["手感", "舒适重量", "连接稳定", "性能响应", "外观灯效", "价格价值"],
    "0f97f1521b641aab50e38a81": ["外观灯效", "舒适重量", "手感", "连接稳定", "性能响应", "声音"],
    "f7573b950c9fadc31a425703": ["声音", "性能响应", "手感", "舒适重量", "连接稳定", "做工质量"],
    "039d0fcde368b376f1ff5b77": ["手感", "外观灯效", "舒适重量"],
    "9c0f6cee95d6b9e4ad78d33c": ["手感", "价格价值"],
    "a289655613411de5d23a8a9c": ["手感", "声音", "性能响应", "做工质量", "外观灯效", "连接稳定", "舒适重量", "价格价值"],
    "aced951b6debbd743adaee51": ["做工质量", "舒适重量", "价格价值"],
    "d1ccfa2f09999ebda7f0f7bb": ["外观灯效", "手感", "价格价值", "做工质量"],
    "32d97a44a8682c71aab0e35a": ["做工质量", "舒适重量", "声音", "手感"],
    "a8593603c95999dc72dd5639": ["外观灯效", "手感"],
    "b99f5b27cfabc2ab085c4ced": ["声音", "做工质量", "手感"],
    "7b3eacafb5f693d0a3b6c2d4": ["手感", "舒适重量"],
    "2da05b173b3250b42bce8530": ["外观灯效", "舒适重量", "做工质量", "手感", "性能响应"],
    "4fd8c74162f6572163e379c9": ["手感", "舒适重量", "性能响应", "做工质量", "外观灯效"],
    "bbe3f823a918328cf1a90005": ["外观灯效"],
    "f369c10f53fb21cd6c1ae0b1": ["手感", "舒适重量", "做工质量"],
    "4bf7042939beff8a509f7a11": ["做工质量", "价格价值"],
    "4a535d4540d1040fb9da8edc": ["手感", "做工质量", "外观灯效"],
    "3ef3cc76bdd18659e32b5f21": ["外观灯效", "手感", "性能响应", "价格价值"],
    "4d5bef0f3ef5008cdfa87052": ["外观灯效", "性能响应", "手感", "舒适重量"],
    "bca3962dd284878b82227149": ["做工质量", "手感", "外观灯效", "性能响应", "价格价值"],
    "8ffbb1cd5aea163de51b5da6": ["手感", "性能响应", "声音", "外观灯效", "做工质量"],
    "e07bd0deafd3ab316eab282a": ["做工质量", "手感", "声音", "外观灯效", "价格价值", "舒适重量"],
    "1ba3e97bf5af6384149c73f1": ["外观灯效", "做工质量", "声音"],
    "5246bba262b804ffc4e49cda": ["手感", "声音", "外观灯效", "做工质量"],
    "fde130d5d87512a9a46d3d07": ["外观灯效", "连接稳定"],
    "a0109bfd756b11bafc31ebc4": ["声音", "手感", "舒适重量", "连接稳定", "外观灯效"],
    "be638733898d2b8720a402be": ["做工质量", "声音", "手感"],
    "3bdec93097ec9bbabf83a020": ["做工质量", "舒适重量", "外观灯效", "性能响应"],
    "dca914c6f20b3b2dffe76ddd": ["手感", "声音", "外观灯效", "做工质量", "价格价值"],
    "5f1993dd819aac145fec4024": ["外观灯效", "舒适重量", "性能响应", "手感"],
    "5ddb8856ef84e095ee2feb64": ["手感", "声音", "舒适重量", "外观灯效", "连接稳定", "性能响应", "价格价值"],
}


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def hardware_environment() -> dict[str, object]:
    processor = platform.processor() or "未由运行时返回型号"
    memory_gb = None
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            processor = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except OSError:
        pass

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        memory_gb = round(status.total_physical / (1024 ** 3), 1)
    return {
        "operating_system": platform.platform(),
        "processor": processor,
        "logical_cpu_count": os.cpu_count(),
        "memory_gb": memory_gb,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "sqlite_version": sqlite3.sqlite_version,
        "database_path": str(get_database_path()),
    }


def build_performance_evidence(database_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest = load_overview_data(database_path)
    history = load_price_history(database_path)
    scored = calculate_dimension_scores(latest)
    category = "mouse" if (scored["category"] == "mouse").any() else scored["category"].iloc[0]
    category_scored = scored[scored["category"] == category].copy()
    merged = merge_comparison_metrics(scored, history)
    comparison_ids = merged[merged["category"] == category].head(5)["product_id"].tolist()

    def score_and_rank():
        current_scored = calculate_dimension_scores(latest)
        return rank_products(current_scored[current_scored["category"] == category], DEFAULT_WEIGHTS)

    operations = {
        "数据概览查询": lambda: load_overview_data(database_path),
        "偏好评分与排名": score_and_rank,
        "竞品对比生成": lambda: build_parameter_comparison(merged, comparison_ids),
    }

    for operation in operations.values():
        for _ in range(3):
            operation()

    records: list[dict[str, object]] = []

    def timed_call(name: str, operation) -> dict[str, object]:
        started = perf_counter()
        result = operation()
        elapsed = (perf_counter() - started) * 1000
        return {"operation": name, "latency_ms": elapsed, "result_rows": len(result)}

    for concurrency in (1, 5, 10):
        for run in range(1, 31):
            for name, operation in operations.items():
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [executor.submit(timed_call, name, operation) for _ in range(concurrency)]
                    for user_index, future in enumerate(futures, start=1):
                        record = future.result()
                        record.update({"concurrency": concurrency, "run": run, "virtual_user": user_index})
                        records.append(record)

    raw = pd.DataFrame(records)
    summary_rows = []
    for (concurrency, operation), group in raw.groupby(["concurrency", "operation"], sort=True):
        values = group["latency_ms"].tolist()
        summary_rows.append({
            "concurrency": int(concurrency),
            "operation": operation,
            "sample_count": len(values),
            "mean_ms": mean(values),
            "median_ms": median(values),
            "p95_ms": percentile(values, 95),
            "max_ms": max(values),
        })
    return raw, pd.DataFrame(summary_rows)


def build_review_evidence() -> tuple[pd.DataFrame, dict[str, object]]:
    reviews = pd.read_csv(REVIEW_PATH)
    rows = []
    tp = fp = fn = 0
    sentiment_correct = 0
    for _, review in reviews.iterrows():
        review_id = str(review["source_review_id"])
        if review_id not in MANUAL_ASPECTS:
            raise KeyError(f"缺少复核标签: {review_id}")
        gold = set(MANUAL_ASPECTS[review_id])
        predicted = set(extract_aspect_mentions(str(review["content"])))
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        gold_sentiment = "正向"
        predicted_sentiment = classify_sentiment(str(review["content"]), float(review["rating"]))
        sentiment_correct += int(gold_sentiment == predicted_sentiment)
        rows.append({
            "source_review_id": review_id,
            "rating": review["rating"],
            "content": review["content"],
            "reference_sentiment": gold_sentiment,
            "predicted_sentiment": predicted_sentiment,
            "reference_aspects": "、".join(name for name in ASPECT_NAMES if name in gold),
            "predicted_aspects": "、".join(name for name in ASPECT_NAMES if name in predicted),
        })
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "review_count": len(reviews),
        "aspect_true_positive": tp,
        "aspect_false_positive": fp,
        "aspect_false_negative": fn,
        "aspect_micro_precision": precision,
        "aspect_micro_recall": recall,
        "aspect_micro_f1": f1,
        "sentiment_accuracy": sentiment_correct / len(reviews),
        "sentiment_weighted_precision": sentiment_correct / len(reviews),
        "sentiment_weighted_recall": sentiment_correct / len(reviews),
        "sentiment_weighted_f1": sentiment_correct / len(reviews),
        "sentiment_class_note": "48条参考标签均为正向，情感指标仅反映单一类别，不代表对中性或负向评论的识别能力。",
        "annotation_note": "单一复核标注版本，尚未进行双人独立标注与一致性检验。",
    }
    return pd.DataFrame(rows), metrics


def build_preference_sensitivity(database_path: Path, samples: int = 1000) -> dict[str, object]:
    latest = load_overview_data(database_path)
    demo_mask = latest["product_name"].fillna("").str.contains("演示") | latest["shop_name"].fillna("").str.contains("功能测试数据")
    research = latest[~demo_mask & latest["current_price"].notna()].copy()
    scored = calculate_dimension_scores(research)
    rng = np.random.default_rng(261008001)
    results: dict[str, object] = {"random_weight_samples": samples, "seed": 261008001, "categories": {}}
    for category in ("mouse", "keyboard"):
        data = scored[scored["category"] == category].copy()
        if len(data) < 3:
            continue
        baseline = rank_products(data, DEFAULT_WEIGHTS).set_index("product_id")
        baseline_order = baseline["rank"].copy()
        baseline_top3 = set(baseline.nsmallest(3, "rank").index)
        baseline_top1 = int(baseline["rank"].idxmin())
        correlations = []
        top3_overlaps = []
        top1_matches = 0
        for weights_array in rng.dirichlet(np.ones(len(DEFAULT_WEIGHTS)), size=samples):
            weights = dict(zip(DEFAULT_WEIGHTS, weights_array * 100))
            ranked = rank_products(data, weights).set_index("product_id")
            candidate_order = ranked["rank"].reindex(baseline_order.index)
            correlations.append(float(baseline_order.corr(candidate_order, method="pearson")))
            candidate_top3 = set(ranked.nsmallest(3, "rank").index)
            top3_overlaps.append(len(baseline_top3 & candidate_top3) / 3)
            top1_matches += int(int(ranked["rank"].idxmin()) == baseline_top1)
        results["categories"][category] = {
            "product_count": len(data),
            "mean_rank_correlation": mean(correlations),
            "median_rank_correlation": median(correlations),
            "p05_rank_correlation": percentile(correlations, 5),
            "mean_top3_overlap": mean(top3_overlaps),
            "top1_retention_rate": top1_matches / samples,
        }
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    database_path = get_database_path()
    raw_performance, performance_summary = build_performance_evidence(database_path)
    review_annotations, review_metrics = build_review_evidence()
    preference_sensitivity = build_preference_sensitivity(database_path)
    evidence = {
        "review_validation_status": "historical_reference_not_independently_verified",
        "review_metrics_usage": "以下旧参考标签指标不得用作text-rule-v2.0准确率；新版本以independent_validation.json为准",
        "test_environment": hardware_environment(),
        "performance_summary": performance_summary.to_dict(orient="records"),
        "review_metrics": review_metrics,
        "preference_sensitivity": preference_sensitivity,
    }
    raw_performance.to_csv(OUTPUT_DIR / "performance_runs_20261013.csv", index=False, encoding="utf-8-sig")
    performance_summary.to_csv(OUTPUT_DIR / "performance_summary_20261013.csv", index=False, encoding="utf-8-sig")
    review_annotations.to_csv(OUTPUT_DIR / "review_reference_annotations_20261013.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "validation_evidence_20261013.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
