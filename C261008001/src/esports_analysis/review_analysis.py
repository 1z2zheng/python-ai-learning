from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .db import connect, initialize_database
from .review_cleaning import classify_text_sentiment, normalize_review_text, quality_reasons
from .review_identity import load_resolved_reviews


POSITIVE_WORDS = (
    "满意", "不错", "很好", "舒服", "顺畅", "稳定", "灵敏", "精致", "耐用",
    "扎实", "清脆", "轻便", "好看", "超赞", "性价比", "推荐", "喜欢",
)
NEGATIVE_WORDS = (
    "失望", "不满", "卡顿", "断连", "延迟", "异响", "松动", "累手", "太重",
    "偏重", "太贵", "价格高", "小贵", "吵", "缺点", "压坏", "破损",
)
ASPECTS = {
    "手感": ("手感", "触感", "回弹", "键程", "按键", "微动", "轴体"),
    "性能响应": ("灵敏", "响应", "延迟", "回报率", "DPI", "精准", "顺畅", "不卡"),
    "连接稳定": ("连接", "无线", "蓝牙", "断连", "稳定", "接收器"),
    "舒适重量": ("舒服", "舒适", "累手", "重量", "轻便", "轻量", "握持", "偏重"),
    "做工质量": ("做工", "质量", "扎实", "耐用", "包装", "松动", "破损"),
    "外观灯效": ("颜值", "外观", "好看", "灯光", "RGB", "颜色"),
    "价格价值": ("价格", "性价比", "划算", "便宜", "贵", "小贵"),
    "声音": ("静音", "声音", "清脆", "吵", "杂音", "异响"),
}


def classify_sentiment(content: str, rating: float | None) -> str:
    return classify_text_sentiment(normalize_review_text(content))[0]


def extract_aspect_mentions(content: str) -> list[str]:
    return [name for name, words in ASPECTS.items() if any(word in content for word in words)]


def extract_opinion_snippets(content: str, polarity: str) -> list[str]:
    words = POSITIVE_WORDS if polarity == "正向" else NEGATIVE_WORDS
    parts = [part.strip() for part in re.split(r"[。！？!?；;，,\n]", content) if part.strip()]
    def contains_opinion(part: str, word: str) -> bool:
        position = part.find(word)
        while position >= 0:
            prefix = part[max(0, position - 6):position]
            suffix = part[position + len(word):position + len(word) + 6]
            if polarity == "正向" or not any(
                negation in prefix for negation in ("不", "没", "没有", "不会", "无", "未", "几乎无")
            ) and not any(negation in suffix for negation in ("无感", "没有", "不明显", "很低", "极低")):
                return True
            position = part.find(word, position + len(word))
        return False

    return [part[:36] for part in parts if any(contains_opinion(part, word) for word in words)][:3]


def load_review_research_data(database_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    initialize_database(database_path)
    review_query = """
        SELECT p.id AS product_id, p.platform_product_id, p.category, p.brand,
               p.product_name, r.source_review_id, r.rating, r.content,
               r.reviewed_at, r.variant_text, r.source_product_id, r.sample_group,
               r.collected_at
        FROM reviews r
        JOIN products p ON p.id = r.product_id
        WHERE r.is_valid = 1
        ORDER BY r.id
    """
    heat_query = """
        WITH latest AS (
            SELECT product_id, MAX(id) AS latest_id
            FROM sales_snapshots
            WHERE metric_name = 'review_count_lower_bound'
            GROUP BY product_id
        )
        SELECT p.id AS product_id, p.platform_product_id, p.category, p.brand,
               p.product_name, s.review_count, s.raw_metric_text,
               s.source_payload_json, s.collected_at
        FROM latest l
        JOIN sales_snapshots s ON s.id = l.latest_id
        JOIN products p ON p.id = s.product_id
    """
    with connect(database_path) as connection:
        reviews = load_resolved_reviews(connection)
        heat = pd.read_sql_query(heat_query, connection)
    if not reviews.empty:
        reviews["content"] = reviews["content"].map(normalize_review_text)
        duplicates = (reviews.duplicated(["platform_product_id", "content"], keep="first")
                      | reviews.duplicated(["platform_product_id", "source_review_id"], keep="first"))
        keep = [not quality_reasons(text, bool(duplicate))
                for text, duplicate in zip(reviews["content"], duplicates)]
        reviews = reviews.loc[keep].copy()
        reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
        reviews["sentiment"] = reviews.apply(
            lambda row: classify_sentiment(row["content"], row["rating"]), axis=1
        )
        reviews["aspects"] = reviews["content"].map(extract_aspect_mentions)
        reviews["pros"] = reviews["content"].map(lambda value: extract_opinion_snippets(value, "正向"))
        reviews["cons"] = reviews["content"].map(lambda value: extract_opinion_snippets(value, "负向"))
    return reviews, heat


def build_product_feedback_summary(reviews: pd.DataFrame, heat: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "product_id", "category", "brand", "product_name", "review_count_lower_bound",
        "review_count_text", "sample_review_count", "average_rating", "positive_rate",
        "top_aspects", "top_positive_points", "top_negative_points", "platform_tags",
    ]
    if reviews.empty and heat.empty:
        return pd.DataFrame(columns=columns)
    product_ids = sorted(set(reviews.get("product_id", [])) | set(heat.get("product_id", [])))
    rows = []
    for product_id in product_ids:
        product_reviews = reviews[reviews["product_id"] == product_id] if not reviews.empty else reviews
        product_heat = heat[heat["product_id"] == product_id] if not heat.empty else heat
        base = product_reviews.iloc[0] if not product_reviews.empty else product_heat.iloc[0]
        aspect_counts: dict[str, int] = {}
        pros: list[str] = []
        cons: list[str] = []
        for value in product_reviews.get("aspects", []):
            for aspect in value:
                aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
        for value in product_reviews.get("pros", []):
            pros.extend(value)
        for value in product_reviews.get("cons", []):
            cons.extend(value)
        tags = []
        if not product_heat.empty:
            try:
                payload = json.loads(product_heat.iloc[0]["source_payload_json"] or "{}")
                tags = [
                    f"{item.get('name')}（{item.get('countText') or item.get('count') or ''}）"
                    for item in (payload.get("tags") or [])[:10]
                    if item.get("name")
                ]
            except (json.JSONDecodeError, TypeError):
                tags = []
        positive_rate = None
        if not product_reviews.empty:
            positive_rate = (product_reviews["sentiment"] == "正向").mean() * 100
        rows.append({
            "product_id": product_id,
            "category": base["category"],
            "brand": base["brand"],
            "product_name": base["product_name"],
            "review_count_lower_bound": int(product_heat.iloc[0]["review_count"]) if not product_heat.empty and pd.notna(product_heat.iloc[0]["review_count"]) else None,
            "review_count_text": product_heat.iloc[0]["raw_metric_text"] if not product_heat.empty else None,
            "sample_review_count": len(product_reviews),
            "average_rating": round(float(product_reviews["rating"].mean()), 2) if not product_reviews.empty else None,
            "positive_rate": round(float(positive_rate), 2) if positive_rate is not None else None,
            "top_aspects": "、".join(name for name, _ in sorted(aspect_counts.items(), key=lambda item: (-item[1], item[0]))[:3]),
            "top_positive_points": "；".join(dict.fromkeys(pros))[:160],
            "top_negative_points": "；".join(dict.fromkeys(cons))[:160],
            "platform_tags": "、".join(tags),
        })
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["category", "review_count_lower_bound"], ascending=[True, False], na_position="last"
    ).reset_index(drop=True)
