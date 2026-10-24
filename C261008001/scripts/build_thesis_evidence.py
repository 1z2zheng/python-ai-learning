from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\C261008001")
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from esports_analysis.overview import load_overview_data
from esports_analysis.preference import DEFAULT_WEIGHTS, calculate_dimension_scores, rank_products
from esports_analysis.pricing import (
    build_cross_platform_comparison,
    build_product_history_summary,
    load_price_history,
)
from esports_analysis.review_analysis import build_product_feedback_summary, load_review_research_data


DB = ROOT / "data" / "esports_analysis.db"
OUT = ROOT / "output" / "thesis"
FIG = OUT / "figures"


def configure_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160


def save_figure(name: str) -> Path:
    path = FIG / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def short_product_label(brand: str, name: str) -> str:
    rules = [
        ("COOL68", "达尔优 COOL68磁轴键盘"),
        ("月刃2", "ROG 月刃2 ACE"),
        ("炼狱蝰蛇V3", "雷蛇 炼狱蝰蛇V3极速版"),
        ("龙鳞ACE", "ROG 龙鳞ACE"),
        ("V98Pro V4", "VGN V98Pro V4"),
        ("专业迷你版", "雷蛇 猎魂光蛛V3专业迷你版"),
        ("GPW二代", "罗技 GPW 2代"),
        ("EK75", "达尔优 EK75"),
        ("专业竞技版", "雷蛇 猎魂光蛛V3专业竞技版"),
        ("猎魂光蛛V3专业版", "雷蛇 猎魂光蛛V3专业版"),
    ]
    for needle, label in rules:
        if needle in name:
            return label
    return f"{brand} {name[:14]}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    configure_chinese_font()

    history = load_price_history(DB)
    historical = history[history["price_source"] == "manmanbuy"].copy()
    historical_products = build_product_history_summary(historical)
    pairs = pd.read_csv(ROOT / "data" / "manmanbuy" / "cross_platform_pairs.csv", dtype=str)
    cross = build_cross_platform_comparison(historical, pairs)
    verified_cross = cross[cross["comparison_eligible"]].copy()

    overview = load_overview_data(DB)
    historical_ids = set(historical["product_id"].astype(int))
    research_overview = overview[overview["product_id"].isin(historical_ids)].copy()
    scored = calculate_dimension_scores(research_overview)
    ranked_parts = []
    for _, part in scored.groupby("category"):
        ranked_parts.append(rank_products(part, DEFAULT_WEIGHTS))
    ranked = pd.concat(ranked_parts, ignore_index=True) if ranked_parts else scored

    reviews, heat = load_review_research_data(DB)
    feedback = build_product_feedback_summary(reviews, heat)
    feedback = feedback[feedback["product_id"].isin(historical_ids)].copy()

    with sqlite3.connect(DB) as connection:
        spec_coverage = pd.read_sql_query(
            """
            SELECT p.category,
                   COUNT(DISTINCT p.id) AS products,
                   SUM(CASE WHEN s.dpi_max IS NOT NULL THEN 1 ELSE 0 END) AS dpi_known,
                   SUM(CASE WHEN s.weight_g IS NOT NULL THEN 1 ELSE 0 END) AS weight_known,
                   SUM(CASE WHEN s.polling_rate_hz IS NOT NULL THEN 1 ELSE 0 END) AS polling_known,
                   SUM(CASE WHEN s.switch_type IS NOT NULL THEN 1 ELSE 0 END) AS switch_known,
                   SUM(CASE WHEN s.layout_size IS NOT NULL THEN 1 ELSE 0 END) AS layout_known
            FROM products p
            LEFT JOIN product_specs s ON s.product_id = p.id
            WHERE p.id IN (
                SELECT DISTINCT product_id FROM price_snapshots WHERE price_source = 'manmanbuy'
            )
            GROUP BY p.category
            ORDER BY p.category
            """,
            connection,
        )

    brand_category = (
        research_overview.groupby(["brand", "category"])["product_id"]
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    )

    payload = {
        "review_analysis_version": "text-rule-v2.0",
        "interpretation_note": "评论为非概率样本；配对为候选关系；规格评分为规则代理指标。旧版分类准确率不得沿用。",
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "historical_price_sample": {
            "products": int(historical["product_id"].nunique()),
            "snapshots": int(len(historical)),
            "jd_products": int(historical.loc[historical["platform"] == "jd", "product_id"].nunique()),
            "jd_snapshots": int((historical["platform"] == "jd").sum()),
            "tmall_products": int(historical.loc[historical["platform"] == "tmall", "product_id"].nunique()),
            "tmall_snapshots": int((historical["platform"] == "tmall").sum()),
            "start": str(historical["collected_date"].min().date()),
            "end": str(historical["collected_date"].max().date()),
            "median_snapshots_per_product": float(historical_products["snapshot_count"].median()),
            "median_latest_price": float(historical_products["latest_price"].median()),
            "median_lowest_price": float(historical_products["lowest_price"].median()),
            "median_volatility_rate": float(historical_products["volatility_rate"].median()),
            "median_potential_saving": float(historical_products["potential_saving"].median()),
        },
        "cross_platform_sample": {
            "matched_models": int(len(cross)),
            "verified_comparable_models": int(cross["comparison_eligible"].sum()),
            "matched_snapshots": int(cross["jd_snapshot_count"].sum() + cross["tmall_snapshot_count"].sum()),
            "jd_cheaper": int((cross["cheaper_platform"] == "京东").sum()),
            "tmall_cheaper": int((cross["cheaper_platform"] == "天猫").sum()),
            "same_price": int((cross["cheaper_platform"] == "同价").sum()),
            "median_absolute_latest_gap": float(verified_cross["latest_gap"].abs().median()) if len(verified_cross) else None,
            "median_jd_lowest": float(verified_cross["jd_lowest_price"].median()) if len(verified_cross) else None,
            "median_tmall_lowest": float(verified_cross["tmall_lowest_price"].median()) if len(verified_cross) else None,
        },
        "review_sample": {
            "products_with_reviews": int(reviews["product_id"].nunique()),
            "representative_reviews": int(len(reviews)),
            "products_with_heat_proxy": int(len(heat)),
            "review_count_lower_bound_total": int(pd.to_numeric(heat["review_count"], errors="coerce").fillna(0).sum()),
            "positive_rate": float((reviews["sentiment"] == "正向").mean() * 100) if len(reviews) else None,
            "scope": "清洗后文本情感样本占比，不代表总体好评率",
        },
        "spec_coverage": spec_coverage.to_dict("records"),
        "brand_category_counts": brand_category.reset_index().to_dict("records"),
        "cross_platform_rows": cross.to_dict("records"),
        "top_ranked_products": ranked.sort_values("match_score", ascending=False)[
            ["category", "brand", "product_name", "current_price", "match_score"]
        ].head(10).to_dict("records"),
    }

    review_evidence = ROOT / "output" / "review_analysis" / "review_analysis_summary.json"
    if review_evidence.exists():
        payload["review_cleaning_evidence"] = json.loads(review_evidence.read_text(encoding="utf-8"))

    evidence_path = OUT / "thesis_evidence.json"
    evidence_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    ax = brand_category.plot(kind="bar", stacked=True, figsize=(8.6, 4.8), color=["#355C7D", "#C06C84"])
    ax.set_title("历史价格研究样本的品牌与品类分布")
    ax.set_xlabel("品牌")
    ax.set_ylabel("商品数量")
    ax.legend(["键盘", "鼠标"], frameon=False)
    ax.grid(axis="y", alpha=0.2)
    save_figure("figure_4_1_brand_category_sample.png")

    daily = (
        historical.assign(date=historical["collected_at"].dt.tz_convert("Asia/Shanghai").dt.date)
        .groupby(["date", "platform"], as_index=False)["effective_price"]
        .median()
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for platform, part in daily.groupby("platform"):
        label = "京东" if platform == "jd" else "天猫"
        ax.plot(pd.to_datetime(part["date"]), part["effective_price"], label=label, linewidth=1.8)
    ax.set_title("京东与天猫历史价格样本的每日中位数")
    ax.set_xlabel("日期")
    ax.set_ylabel("有效价格 元")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save_figure("figure_5_1_daily_price_median.png")

    cross_plot = cross.sort_values("latest_gap").copy()
    y = np.arange(len(cross_plot))
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.barh(y - 0.18, cross_plot["jd_latest_price"], height=0.34, label="京东", color="#355C7D")
    ax.barh(y + 0.18, cross_plot["tmall_latest_price"], height=0.34, label="天猫", color="#C06C84")
    ax.set_yticks(y, cross_plot["model"])
    ax.set_xlabel("最新观测价格 元")
    ax.set_title("候选型号跨平台报价（配置及优惠条件待核验）")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    save_figure("figure_5_2_cross_platform_latest_price.png")

    top = ranked.sort_values("match_score", ascending=False).head(10).sort_values("match_score")
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    labels = [short_product_label(str(row.brand), str(row.product_name)) for row in top.itertuples()]
    ax.barh(labels, top["match_score"], color="#4C78A8")
    ax.set_xlim(0, 100)
    ax.set_xlabel("默认权重综合匹配度")
    ax.set_title("历史价格样本的综合匹配度前十名")
    ax.grid(axis="x", alpha=0.2)
    save_figure("figure_5_3_preference_ranking.png")

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.axis("off")
    boxes = [
        (0.04, 0.68, 0.18, 0.17, "数据来源\n慢慢买公开查询\n厂商官网\n合规商品页面"),
        (0.29, 0.68, 0.18, 0.17, "采集与导入\n目标清单\n字段校验\n增量写入"),
        (0.54, 0.68, 0.18, 0.17, "SQLite存储\n商品\n规格\n价格\n评论"),
        (0.79, 0.68, 0.17, 0.17, "质量控制\n缺失标记\n去重\n来源追溯"),
        (0.17, 0.23, 0.25, 0.18, "分析服务\n六维偏好评分\n价格带与波动\n跨平台同型号对比"),
        (0.58, 0.23, 0.25, 0.18, "Streamlit交互层\n概览与筛选\n趋势与定位\n竞品并排比较"),
    ]
    for x, y0, w, h, label in boxes:
        rect = plt.Rectangle((x, y0), w, h, facecolor="#EAF0F6", edgecolor="#355C7D", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y0 + h / 2, label, ha="center", va="center", fontsize=10)
    arrows = [((0.22, 0.765), (0.29, 0.765)), ((0.47, 0.765), (0.54, 0.765)), ((0.72, 0.765), (0.79, 0.765)), ((0.64, 0.68), (0.35, 0.41)), ((0.42, 0.32), (0.58, 0.32))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#355C7D", "lw": 1.4})
    ax.set_title("系统数据流与分析架构", fontsize=14, pad=12)
    save_figure("figure_3_1_system_architecture.png")

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print(f"evidence={evidence_path}")
    print(f"figures={FIG}")


if __name__ == "__main__":
    main()
