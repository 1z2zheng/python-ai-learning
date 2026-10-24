from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .db import connect, initialize_database
from .preference import DIMENSIONS


PARAMETER_OPTIONS = {
    "mouse": {
        "dpi_max": "最大 DPI",
        "weight_g": "重量 g",
        "polling_rate_hz": "回报率 Hz",
    },
    "keyboard": {
        "actuation_mm": "最小触发行程 mm",
        "polling_rate_hz": "回报率 Hz",
        "weight_g": "重量 g",
    },
}

PRICE_BANDS = {
    "mouse": (
        [0, 300, 600, 1000, float("inf")],
        ["入门 0-299", "主流 300-599", "中高端 600-999", "旗舰 1000以上"],
    ),
    "keyboard": (
        [0, 300, 500, 800, float("inf")],
        ["入门 0-299", "主流 300-499", "中高端 500-799", "旗舰 800以上"],
    ),
}


def load_price_history(database_path: str | Path) -> pd.DataFrame:
    initialize_database(database_path)
    query = """
        SELECT
            p.id AS product_id,
            p.platform_product_id,
            COALESCE(NULLIF(p.source_site, ''), p.platform) AS platform,
            p.category,
            p.brand,
            p.product_name,
            ps.list_price,
            ps.sale_price,
            ps.coupon_price,
            COALESCE(ps.coupon_price, ps.sale_price, ps.list_price) AS effective_price,
            ps.promotion_text,
            ps.price_source,
            ps.collected_at,
            ps.collected_date
        FROM price_snapshots ps
        JOIN products p ON p.id = ps.product_id
        WHERE p.is_active = 1
        ORDER BY ps.collected_at, p.brand, p.product_name
    """
    with connect(database_path) as connection:
        cursor = connection.execute(query)
        columns = [item[0] for item in cursor.description]
        rows = cursor.fetchall()
    frame = pd.DataFrame([tuple(row) for row in rows], columns=columns)
    if not frame.empty:
        frame["effective_price"] = pd.to_numeric(frame["effective_price"], errors="coerce")
        frame["collected_at"] = pd.to_datetime(frame["collected_at"], errors="coerce", utc=True)
        frame["collected_date"] = pd.to_datetime(frame["collected_date"], errors="coerce")
    return frame


CROSS_PLATFORM_COLUMNS = [
    "pair_key",
    "category",
    "brand",
    "model",
    "jd_sku",
    "tmall_item_id",
    "jd_snapshot_count",
    "tmall_snapshot_count",
    "overlap_start",
    "overlap_end",
    "jd_latest_price",
    "tmall_latest_price",
    "latest_gap",
    "cheaper_platform",
    "jd_lowest_price",
    "tmall_lowest_price",
    "verification_status",
    "comparison_eligible",
    "comparison_date",
    "risk_note",
]


def build_cross_platform_comparison(
    history: pd.DataFrame, pairs: pd.DataFrame, verification_records: dict | None = None
) -> pd.DataFrame:
    """Compare matched JD and Tmall models using their real historical prices."""
    if history.empty or pairs.empty:
        return pd.DataFrame(columns=CROSS_PLATFORM_COLUMNS)

    required_history = {
        "platform_product_id", "platform", "effective_price", "collected_at", "collected_date"
    }
    required_pairs = {"pair_key", "category", "brand", "model", "jd_sku", "tmall_item_id"}
    if not required_history.issubset(history.columns) or not required_pairs.issubset(pairs.columns):
        return pd.DataFrame(columns=CROSS_PLATFORM_COLUMNS)

    valid = history.dropna(subset=["effective_price", "collected_at"]).copy()
    valid["platform_product_id"] = valid["platform_product_id"].astype(str)
    audit_path = Path(__file__).resolve().parents[2] / "data" / "manmanbuy" / "cross_platform_verification.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    if verification_records is not None:
        audit = verification_records
    rows: list[dict[str, object]] = []
    for pair in pairs.to_dict("records"):
        jd = valid[
            (valid["platform"] == "jd")
            & (valid["platform_product_id"] == str(pair["jd_sku"]))
        ].sort_values("collected_at")
        tmall = valid[
            (valid["platform"] == "tmall")
            & (valid["platform_product_id"] == str(pair["tmall_item_id"]))
        ].sort_values("collected_at")
        if jd.empty or tmall.empty:
            continue
        jd_latest = float(jd.iloc[-1]["effective_price"])
        tmall_latest = float(tmall.iloc[-1]["effective_price"])
        gap = tmall_latest - jd_latest
        same_day = jd.iloc[-1]["collected_date"] == tmall.iloc[-1]["collected_date"]
        record = audit.get(str(pair["pair_key"]), {})
        verified = record.get("status") == "verified" and all(record.get(name) for name in (
            "configuration_evidence", "offer_evidence", "verified_by", "verified_at",
            "jd_variant", "tmall_variant", "shop_evidence"
        ))
        quote_date = str(jd.iloc[-1]["collected_date"].date()) if same_day and pd.notna(jd.iloc[-1]["collected_date"]) else None
        bound = (str(record.get("jd_sku")) == str(pair["jd_sku"])
                 and str(record.get("tmall_item_id")) == str(pair["tmall_item_id"])
                 and record.get("comparison_date") == quote_date
                 and record.get("jd_latest_price") == jd_latest
                 and record.get("tmall_latest_price") == tmall_latest)
        risks = []
        if not verified:
            risks.append("配置与优惠条件缺少完整核验记录")
        if verified and not bound:
            risks.append("SKU、报价日期或价格已变化，需重新核验")
        if record.get("status") == "different":
            risks.append("已记录配置不同，不参与价差排名")
        if not same_day:
            risks.append("末次观测日期不同")
        if jd_latest <= 0 or tmall_latest <= 0:
            risks.append("存在非正报价")
        if min(jd_latest, tmall_latest) > 0 and max(jd_latest, tmall_latest) / min(jd_latest, tmall_latest) >= 2:
            risks.append("价格比超过2倍，需检查默认SKU")
        eligible = verified and not risks
        rows.append({
            "pair_key": pair["pair_key"],
            "category": pair["category"],
            "brand": pair["brand"],
            "model": pair["model"],
            "jd_sku": str(pair["jd_sku"]),
            "tmall_item_id": str(pair["tmall_item_id"]),
            "jd_snapshot_count": len(jd),
            "tmall_snapshot_count": len(tmall),
            "overlap_start": max(jd["collected_date"].min(), tmall["collected_date"].min()),
            "overlap_end": min(jd["collected_date"].max(), tmall["collected_date"].max()),
            "jd_latest_price": jd_latest,
            "tmall_latest_price": tmall_latest,
            "latest_gap": round(gap, 2),
            "cheaper_platform": ("京东" if gap > 0 else ("天猫" if gap < 0 else "同价")) if eligible else "待核验",
            "verification_status": "different" if record.get("status") == "different" else ("verified" if verified and bound else "pending"),
            "comparison_eligible": eligible,
            "comparison_date": jd.iloc[-1]["collected_date"] if same_day else pd.NaT,
            "risk_note": "；".join(risks),
            "jd_lowest_price": float(jd["effective_price"].min()),
            "tmall_lowest_price": float(tmall["effective_price"].min()),
        })
    result = pd.DataFrame(rows, columns=CROSS_PLATFORM_COLUMNS)
    if result.empty:
        return result
    numeric = [
        "jd_latest_price", "tmall_latest_price", "latest_gap",
        "jd_lowest_price", "tmall_lowest_price",
    ]
    result[numeric] = result[numeric].round(2)
    return result.sort_values(["category", "brand", "model"]).reset_index(drop=True)


def assign_price_bands(data: pd.DataFrame, category: str) -> pd.DataFrame:
    if category not in PRICE_BANDS:
        raise ValueError(f"不支持的产品类别: {category}")
    result = data.copy()
    bins, labels = PRICE_BANDS[category]
    result["price_band"] = pd.cut(
        result["current_price"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    return result


def build_price_band_summary(data: pd.DataFrame, category: str) -> pd.DataFrame:
    banded = assign_price_bands(data, category)
    _, labels = PRICE_BANDS[category]
    summary = (
        banded.groupby("price_band", observed=False)
        .agg(商品数=("product_id", "nunique"), 平均价格=("current_price", "mean"))
        .reindex(labels)
        .reset_index()
        .rename(columns={"price_band": "价格带"})
    )
    total = int(summary["商品数"].sum())
    summary["商品占比"] = summary["商品数"] / total * 100 if total else 0.0
    return summary


def build_positioning_data(scored_data: pd.DataFrame) -> pd.DataFrame:
    result = scored_data.copy()
    performance_dimensions = [name for name in DIMENSIONS if name != "value_score"]
    result["performance_score"] = result[performance_dimensions].mean(axis=1).round(2)
    if result.empty:
        result["positioning_group"] = pd.Series(dtype=str)
        return result
    price_middle = float(result["current_price"].median())
    performance_middle = float(result["performance_score"].median())
    conditions = [
        (result["performance_score"] >= performance_middle) & (result["current_price"] < price_middle),
        (result["performance_score"] >= performance_middle) & (result["current_price"] >= price_middle),
        (result["performance_score"] < performance_middle) & (result["current_price"] < price_middle),
    ]
    choices = ["高性能低价格", "高性能高价格", "低性能低价格"]
    result["positioning_group"] = np.select(conditions, choices, default="低性能高价格")
    result["price_median"] = price_middle
    result["performance_median"] = performance_middle
    return result


def summarize_price_history(history: pd.DataFrame) -> dict[str, object]:
    if history.empty:
        return {
            "snapshot_count": 0,
            "product_count": 0,
            "average_price": None,
            "lowest_price": None,
            "date_start": None,
            "date_end": None,
        }
    return {
        "snapshot_count": len(history),
        "product_count": int(history["product_id"].nunique()),
        "average_price": float(history["effective_price"].mean()),
        "lowest_price": float(history["effective_price"].min()),
        "date_start": history["collected_date"].min(),
        "date_end": history["collected_date"].max(),
    }


PRODUCT_HISTORY_COLUMNS = [
    "product_id",
    "platform",
    "category",
    "brand",
    "product_name",
    "snapshot_count",
    "date_start",
    "date_end",
    "average_price",
    "lowest_price",
    "highest_price",
    "latest_price",
    "price_range",
    "volatility_rate",
    "lowest_gap_rate",
    "potential_saving",
]


def build_product_history_summary(history: pd.DataFrame) -> pd.DataFrame:
    """Build one research-ready price profile per product."""
    if history.empty:
        return pd.DataFrame(columns=PRODUCT_HISTORY_COLUMNS)
    valid = history.dropna(subset=["effective_price", "collected_at"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=PRODUCT_HISTORY_COLUMNS)
    valid = valid.sort_values(["product_id", "collected_at"])
    keys = ["product_id", "platform", "category", "brand", "product_name"]
    summary = (
        valid.groupby(keys, dropna=False)
        .agg(
            snapshot_count=("effective_price", "size"),
            date_start=("collected_date", "min"),
            date_end=("collected_date", "max"),
            average_price=("effective_price", "mean"),
            lowest_price=("effective_price", "min"),
            highest_price=("effective_price", "max"),
            price_std=("effective_price", lambda values: values.std(ddof=0)),
        )
        .reset_index()
    )
    latest = (
        valid.groupby("product_id", as_index=False).tail(1)[["product_id", "effective_price"]]
        .rename(columns={"effective_price": "latest_price"})
    )
    summary = summary.merge(latest, on="product_id", how="left")
    summary["price_range"] = summary["highest_price"] - summary["lowest_price"]
    summary["volatility_rate"] = np.where(
        summary["average_price"] > 0,
        summary["price_std"] / summary["average_price"] * 100,
        np.nan,
    )
    summary["lowest_gap_rate"] = np.where(
        summary["lowest_price"] > 0,
        (summary["latest_price"] - summary["lowest_price"]) / summary["lowest_price"] * 100,
        np.nan,
    )
    summary["potential_saving"] = (
        summary["latest_price"] - summary["lowest_price"]
    ).clip(lower=0)
    numeric_columns = [
        "average_price",
        "lowest_price",
        "highest_price",
        "latest_price",
        "price_range",
        "volatility_rate",
        "lowest_gap_rate",
        "potential_saving",
    ]
    summary[numeric_columns] = summary[numeric_columns].round(2)
    return summary[PRODUCT_HISTORY_COLUMNS].sort_values(
        ["category", "brand", "product_name"]
    ).reset_index(drop=True)


def build_brand_history_summary(product_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate product profiles without letting long series dominate a brand."""
    columns = [
        "brand",
        "product_count",
        "snapshot_count",
        "average_latest_price",
        "average_lowest_price",
        "average_volatility_rate",
        "average_lowest_gap_rate",
    ]
    if product_summary.empty:
        return pd.DataFrame(columns=columns)
    result = (
        product_summary.groupby("brand", dropna=False)
        .agg(
            product_count=("product_id", "nunique"),
            snapshot_count=("snapshot_count", "sum"),
            average_latest_price=("latest_price", "mean"),
            average_lowest_price=("lowest_price", "mean"),
            average_volatility_rate=("volatility_rate", "mean"),
            average_lowest_gap_rate=("lowest_gap_rate", "mean"),
        )
        .reset_index()
    )
    metric_columns = columns[3:]
    result[metric_columns] = result[metric_columns].round(2)
    return result[columns].sort_values(
        ["average_volatility_rate", "brand"], ascending=[True, True]
    ).reset_index(drop=True)

