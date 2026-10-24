from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .db import connect
from .overview import load_overview_data
from .pricing import load_price_history, build_product_history_summary
from .review_analysis import load_review_research_data
from .review_cleaning import DIMENSION_KEYWORDS, classify_dimension_sentiment
from .review_identity import load_verified_product_catalog


def load_evidence_data(database_path: str | Path) -> dict[str, pd.DataFrame]:
    products = load_overview_data(database_path)
    history = load_price_history(database_path)
    history = history[history["price_source"].eq("manmanbuy")].copy()
    reviews, heat = load_review_research_data(database_path)
    if reviews.empty:
        strict = reviews.copy()
    else:
        match = reviews["source_product_id"].astype("string").eq(
            reviews["platform_product_id"].astype("string")
        ).fillna(False)
        strict = reviews[match].copy()
    verified = load_verified_product_catalog(database_path)
    ids = set(history["product_id"]) | set(strict["product_id"]) | set(verified["product_id"])
    products = products[products["product_id"].isin(ids)].copy()
    counts = strict.groupby("product_id")["source_review_id"].nunique() if not strict.empty else pd.Series(dtype=int)
    products["strict_review_count"] = products["product_id"].map(counts).fillna(0).astype(int)
    products["identity_verified"] = products["product_id"].isin(set(verified["product_id"]))
    products = products.sort_values(["strict_review_count", "identity_verified", "brand", "product_name"],
                                    ascending=[False, False, True, True]).reset_index(drop=True)
    return {"products": products, "history": history, "strict_reviews": strict,
            "heat": heat, "verified": verified}


def product_evidence(bundle: dict[str, pd.DataFrame], product_id: int, database_path) -> dict:
    matches = bundle["products"][bundle["products"]["product_id"].eq(product_id)]
    if matches.empty:
        raise ValueError("商品不在当前研究证据范围内")
    product = matches.iloc[0].to_dict()
    history = bundle["history"][bundle["history"]["product_id"].eq(product_id)].copy()
    reviews = bundle["strict_reviews"][bundle["strict_reviews"]["product_id"].eq(product_id)].copy()
    verification = bundle["verified"][bundle["verified"]["product_id"].eq(product_id)].copy()
    price_summary = build_product_history_summary(history)
    with connect(database_path) as db:
        raw = db.execute("SELECT raw_specs_json,source_text,updated_at FROM product_specs WHERE product_id=?", (product_id,)).fetchone()
    raw_attributes = []
    source_text = ""
    updated_at = ""
    if raw:
        source_text = raw["source_text"] or ""
        updated_at = raw["updated_at"] or ""
        try:
            values = json.loads(raw["raw_specs_json"] or "[]")
            if isinstance(values, list):
                raw_attributes = [{"参数": v.get("labelName", ""), "来源原值": v.get("labelValue", "")}
                                  for v in values if isinstance(v, dict) and v.get("labelName")]
        except (ValueError, TypeError):
            pass
    dimensions = []
    excerpts = []
    for dimension in DIMENSION_KEYWORDS:
        count = {"正向": 0, "负向": 0, "中性": 0, "混合": 0}
        for review in reviews.to_dict("records"):
            # Keep the existing six-dimension extraction rule authoritative.
            from .review_cleaning import extract_dimensions
            if dimension not in extract_dimensions(review["content"]):
                continue
            sentiment, _ = classify_dimension_sentiment(review["content"], dimension, None)
            count[sentiment] += 1
            excerpts.append({"维度": dimension, "规则情感": sentiment, "正文": review["content"],
                             "评价时间": review["reviewed_at"], "来源SKU": review["source_product_id"],
                             "原采集入口SKU": review["requested_sku"], "核验方式": review["identity_resolution"]})
        total = sum(count.values())
        dimensions.append({"维度": dimension, "提及数": total, **count,
                           "净情感值": (count["正向"] - count["负向"]) / total if total else None})
    return {"product": product, "history": history, "reviews": reviews,
            "verification": verification, "price_summary": price_summary,
            "dimensions": pd.DataFrame(dimensions), "excerpts": pd.DataFrame(excerpts),
            "raw_attributes": pd.DataFrame(raw_attributes), "spec_source": source_text,
            "spec_updated_at": updated_at}
