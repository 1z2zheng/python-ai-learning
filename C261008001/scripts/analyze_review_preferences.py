from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.config import get_database_path
from esports_analysis.db import initialize_database
from esports_analysis.review_identity import load_resolved_reviews
from esports_analysis.review_cleaning import (
    DIMENSION_KEYWORDS,
    classify_dimension_sentiment,
    extract_dimensions,
    normalize_review_text,
    quality_reasons,
    review_scope,
)


def configure_plotting() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160


def load_reviews(database: Path) -> pd.DataFrame:
    query = """
        SELECT r.id AS review_row_id, p.id AS product_id, p.platform_product_id,
               p.category, p.brand, p.product_name, r.source_review_id, r.rating,
               r.content, r.reviewed_at, r.variant_text, r.source_product_id,
               r.sample_group, r.source_url, r.collected_at
        FROM reviews r
        JOIN products p ON p.id = r.product_id
        WHERE r.is_valid = 1
        ORDER BY r.id
    """
    with sqlite3.connect(database) as connection:
        return load_resolved_reviews(connection)


def prepare_reviews(raw: pd.DataFrame) -> pd.DataFrame:
    reviews = raw.copy()
    reviews["content_clean"] = reviews["content"].map(normalize_review_text)
    reviews["content_key"] = reviews["content_clean"].str.lower()
    reviews["duplicate_content"] = (
        reviews.duplicated(["platform_product_id", "content_key"], keep="first")
        | reviews.duplicated(["platform_product_id", "source_review_id"], keep="first")
    )
    reviews["sku_scope"] = reviews.apply(
        lambda row: review_scope(row["platform_product_id"], row["source_product_id"]), axis=1
    )
    reviews["dimensions"] = reviews["content_clean"].map(extract_dimensions)
    reviews["quality_reasons"] = reviews.apply(
        lambda row: quality_reasons(row["content_clean"], bool(row["duplicate_content"])), axis=1
    )
    reviews["preference_eligible"] = reviews["quality_reasons"].map(len).eq(0)
    reviews["product_level_eligible_strict"] = (
        reviews["preference_eligible"] & reviews["sku_scope"].eq("目标SKU")
    )
    reviews["product_level_eligible_extended"] = (
        reviews["preference_eligible"] & ~reviews["sku_scope"].eq("其他SKU待核验")
    )
    reviews["quality_status"] = reviews["quality_reasons"].map(
        lambda values: "可用于偏好分析" if not values else "排除"
    )
    reviews["quality_reasons"] = reviews["quality_reasons"].map(lambda values: "；".join(values))
    reviews["dimensions"] = reviews["dimensions"].map(lambda values: "；".join(values))
    reviews["sampling_method"] = np.where(reviews["sample_group"].notna(), "评分组定额首页", "历史展示样本")
    reviews["sample_group"] = reviews["sample_group"].fillna("legacy")
    return reviews


def build_mentions(reviews: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for review in reviews.loc[reviews["preference_eligible"]].itertuples(index=False):
        dimensions = [value for value in str(review.dimensions).split("；") if value]
        rating = float(review.rating) if pd.notna(review.rating) else None
        for dimension in dimensions:
            label, score = classify_dimension_sentiment(review.content_clean, dimension, rating)
            rows.append({
                "review_row_id": review.review_row_id,
                "product_id": review.product_id,
                "platform_product_id": review.platform_product_id,
                "category": review.category,
                "brand": review.brand,
                "product_name": review.product_name,
                "source_review_id": review.source_review_id,
                "source_product_id": review.source_product_id,
                "sku_scope": review.sku_scope,
                "sample_group": review.sample_group,
                "sampling_method": review.sampling_method,
                "collection_date": str(review.collected_at)[:10],
                "rating": rating,
                "dimension": dimension,
                "sentiment": label,
                "sentiment_score": score,
                "content_clean": review.content_clean,
            })
    return pd.DataFrame(rows)


def summarize_dimensions(mentions: pd.DataFrame, eligible_reviews: int) -> pd.DataFrame:
    dimensions = list(DIMENSION_KEYWORDS)
    rows = []
    for dimension in dimensions:
        part = mentions[mentions["dimension"] == dimension]
        positive = int((part["sentiment"] == "正向").sum())
        neutral = int((part["sentiment"] == "中性").sum())
        negative = int((part["sentiment"] == "负向").sum())
        total = len(part)
        rows.append({
            "dimension": dimension,
            "mention_count": total,
            "unique_review_count": int(part["review_row_id"].nunique()),
            "mention_rate_pct": round(part["review_row_id"].nunique() / eligible_reviews * 100, 2) if eligible_reviews else 0,
            "positive_count": positive,
            "neutral_count": neutral,
            "mixed_count": int((part["sentiment"] == "混合").sum()),
            "negative_count": negative,
            "net_sentiment": round((positive - negative) / total, 4) if total else None,
        })
    return pd.DataFrame(rows)


def summarize_categories(mentions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (category, dimension), part in mentions.groupby(["category", "dimension"], sort=True):
        rows.append({
            "category": category,
            "dimension": dimension,
            "mention_count": len(part),
            "unique_review_count": int(part["review_row_id"].nunique()),
            "positive_count": int((part["sentiment"] == "正向").sum()),
            "neutral_count": int((part["sentiment"] == "中性").sum()),
            "mixed_count": int((part["sentiment"] == "混合").sum()),
            "negative_count": int((part["sentiment"] == "负向").sum()),
            "net_sentiment": round(float(part["sentiment_score"].mean()), 4),
        })
    return pd.DataFrame(rows)


def summarize_products(mentions: pd.DataFrame) -> pd.DataFrame:
    strict = mentions[mentions["sku_scope"] == "目标SKU"].copy()
    rows = []
    for (product_id, dimension), part in strict.groupby(["product_id", "dimension"], sort=True):
        base = part.iloc[0]
        rows.append({
            "product_id": product_id,
            "platform_product_id": base["platform_product_id"],
            "category": base["category"],
            "brand": base["brand"],
            "product_name": base["product_name"],
            "dimension": dimension,
            "mention_count": len(part),
            "positive_count": int((part["sentiment"] == "正向").sum()),
            "neutral_count": int((part["sentiment"] == "中性").sum()),
            "mixed_count": int((part["sentiment"] == "混合").sum()),
            "negative_count": int((part["sentiment"] == "负向").sum()),
            "net_sentiment": round(float(part["sentiment_score"].mean()), 4),
        })
    return pd.DataFrame(rows)


def save_figures(summary: pd.DataFrame, category_summary: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    plot = summary.set_index("dimension")[["positive_count", "neutral_count", "mixed_count", "negative_count"]]
    plot.columns = ["正向", "中性", "混合", "负向"]
    ax = plot.plot(kind="bar", stacked=True, figsize=(9.2, 5.2), color=["#4C956C", "#B8B8B8", "#D9A441", "#D1495B"])
    ax.set_title("六维偏好评价的情感结构")
    ax.set_xlabel("偏好维度")
    ax.set_ylabel("方面级提及次数")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(figure_dir / "review_dimension_sentiment.png", bbox_inches="tight", facecolor="white")
    plt.close()

    dimensions = list(DIMENSION_KEYWORDS)
    categories = [value for value in ("keyboard", "mouse") if value in set(category_summary["category"])]
    matrix = np.array([
        [
            category_summary.loc[
                (category_summary["category"] == category) & (category_summary["dimension"] == dimension),
                "net_sentiment",
            ].iloc[0]
            if not category_summary.loc[
                (category_summary["category"] == category) & (category_summary["dimension"] == dimension)
            ].empty else np.nan
            for dimension in dimensions
        ]
        for category in categories
    ])
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(dimensions)), dimensions)
    ax.set_yticks(range(len(categories)), ["键盘" if value == "keyboard" else "鼠标" for value in categories])
    ax.set_title("样本六维净情感值（品类抽样不一致，禁止解释为质量差异）")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, "无样本" if np.isnan(value) else f"{value:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="净情感值")
    plt.tight_layout()
    plt.savefig(figure_dir / "review_category_sentiment_heatmap.png", bbox_inches="tight", facecolor="white")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="清洗京东评论并完成六维偏好分析")
    parser.add_argument("--database", type=Path, default=get_database_path())
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "review_analysis")
    args = parser.parse_args()

    initialize_database(args.database)
    raw = load_reviews(args.database)
    reviews = prepare_reviews(raw)
    mentions = build_mentions(reviews)
    eligible_count = int(reviews["preference_eligible"].sum())
    dimension_summary = summarize_dimensions(mentions, eligible_count)
    category_summary = summarize_categories(mentions)
    product_summary = summarize_products(mentions)
    stratified = mentions.groupby(
        ["category", "platform_product_id", "sampling_method", "collection_date", "sample_group", "sku_scope", "dimension"],
        dropna=False,
    ).agg(
        mention_count=("sentiment_score", "size"),
        net_sentiment=("sentiment_score", "mean"),
        positive_count=("sentiment", lambda s: int(s.eq("正向").sum())),
        negative_count=("sentiment", lambda s: int(s.eq("负向").sum())),
        mixed_count=("sentiment", lambda s: int(s.eq("混合").sum())),
        neutral_count=("sentiment", lambda s: int(s.eq("中性").sum())),
    ).reset_index()

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reviews_path = args.processed_dir / "jd_reviews_cleaned_20261014.csv"
    mentions_path = args.processed_dir / "jd_review_dimension_mentions_20261014.csv"
    dimension_path = args.output_dir / "review_dimension_summary.csv"
    category_path = args.output_dir / "review_category_dimension_summary.csv"
    product_path = args.output_dir / "review_product_dimension_summary_strict.csv"
    reviews.to_csv(reviews_path, index=False, encoding="utf-8-sig")
    mentions.to_csv(mentions_path, index=False, encoding="utf-8-sig")
    dimension_summary.to_csv(dimension_path, index=False, encoding="utf-8-sig")
    category_summary.to_csv(category_path, index=False, encoding="utf-8-sig")
    product_summary.to_csv(product_path, index=False, encoding="utf-8-sig")
    stratified_path = args.output_dir / "review_stratified_summary.csv"
    stratified.to_csv(stratified_path, index=False, encoding="utf-8-sig")
    annotation_path = args.output_dir / "review_manual_annotation_template.csv"
    if not annotation_path.exists():
        annotation = reviews[["review_row_id", "source_review_id", "platform_product_id", "sku_scope", "sample_group", "rating", "content_clean"]].copy()
        for column in ("annotator_a_dimensions", "annotator_a_sentiment", "annotator_b_dimensions", "annotator_b_sentiment", "adjudicated_dimensions", "adjudicated_sentiment", "reviewer_note"):
            annotation[column] = ""
        annotation.to_csv(annotation_path, index=False, encoding="utf-8-sig")
    save_figures(dimension_summary, category_summary, args.output_dir / "figures")

    scope_counts = reviews["sku_scope"].value_counts().to_dict()
    exclusion_counts = (
        reviews.loc[~reviews["preference_eligible"], "quality_reasons"]
        .str.split("；").explode().value_counts().to_dict()
    )
    top_dimensions = dimension_summary.sort_values(["mention_count", "net_sentiment"], ascending=[False, False])
    result = {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "raw_review_count": int(len(reviews)),
        "distinct_content_count": int(reviews["content_key"].nunique()),
        "preference_eligible_count": eligible_count,
        "excluded_count": int((~reviews["preference_eligible"]).sum()),
        "scope_counts": {str(key): int(value) for key, value in scope_counts.items()},
        "exclusion_reason_counts": {str(key): int(value) for key, value in exclusion_counts.items()},
        "dimension_mention_count": int(len(mentions)),
        "strict_eligible_review_count": int(reviews["product_level_eligible_strict"].sum()),
        "strict_eligible_product_count": int(reviews.loc[reviews["product_level_eligible_strict"], "product_id"].nunique()),
        "analysis_version": "text-rule-v2.0",
        "manual_validation_status": "pending_independent_annotation",
        "sampling_note": "按评分组定额采集，混合汇总仅描述样本；禁止推断总体好评率或品类质量差异。",
        "top_dimensions": top_dimensions.to_dict("records"),
        "method": {
            "unit": "一条评论可提及多个偏好维度",
            "net_sentiment": "正向提及数减负向提及数，再除以该维度全部提及数，取值范围为-1至1",
            "strict_product_scope": "仅source_product_id与目标SKU一致的评论用于严格商品级汇总",
            "category_scope": "其他SKU未核验，仅作为采集入口下的探索文本，不推断目标型号质量",
        },
        "outputs": {
            "cleaned_reviews": str(reviews_path),
            "dimension_mentions": str(mentions_path),
            "dimension_summary": str(dimension_path),
            "category_summary": str(category_path),
            "product_summary_strict": str(product_path),
            "stratified_summary": str(stratified_path),
            "annotation_template": str(annotation_path),
        },
    }
    result_path = args.output_dir / "review_analysis_summary.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# 评论数据清洗与六维偏好分析",
        "",
        f"生成时间：{result['generated_at']}",
        "",
        "## 样本处理",
        "",
        f"数据库包含{len(reviews)}条脱敏评论，其中{eligible_count}条通过偏好分析质量规则，{int((~reviews['preference_eligible']).sum())}条因占位、重复、过短、仅涉及服务或缺乏产品属性信息而排除。清洗过程只生成派生文件，不修改原始评论表。",
        "",
        f"目标SKU原始评论{scope_counts.get('目标SKU', 0)}条，其中{result['strict_eligible_review_count']}条通过清洗。其他SKU待核验{scope_counts.get('其他SKU待核验', 0)}条，历史样本未记录来源SKU评论{scope_counts.get('历史样本未记录来源SKU', 0)}条。严格商品级汇总只使用目标SKU评论。",
        "",
        "## 分析口径",
        "",
        "六维偏好包括手感体验、响应性能、舒适度、耐用性、外观设计和性价比。一条评论可以同时贡献多个维度提及。情感由文本判断，评分只保留为来源属性。否定表达采用最长词语匹配和局部否定检查，混合体验单独记录。中性与混合记为0，净情感值等于正向提及数减负向提及数后除以全部提及数。",
        "",
        "## 核心结果",
        "",
    ]
    for row in top_dimensions.itertuples(index=False):
        report_lines.append(
            f"- {row.dimension}：{row.mention_count}次提及，覆盖{row.unique_review_count}条评论，提及率{row.mention_rate_pct:.2f}%，净情感值{row.net_sentiment if pd.notna(row.net_sentiment) else '无样本'}。"
        )
    report_lines.extend([
        "",
        "## 使用限制",
        "",
        "评分组定额首页抽样与历史展示抽样不一致。混合汇总只描述样本，禁止将键盘鼠标差异解释为总体质量差异。其他SKU待核验，历史样本缺少来源SKU。规则法仍缺少独立人工标注验证，不能报告模型准确率。已生成双人标注模板，等待真实标注后计算一致性与分类指标。",
    ])
    (args.output_dir / "review_analysis_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
