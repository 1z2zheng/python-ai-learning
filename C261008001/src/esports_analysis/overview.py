from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .db import connect, initialize_database


CATEGORY_LABELS = {"mouse": "鼠标", "keyboard": "键盘"}
PLATFORM_LABELS = {
    "jd": "京东",
    "tmall": "天猫",
    "manual": "人工导入",
    "third_party": "第三方数据",
    "logitech_official": "罗技官网",
    "razer_official": "雷蛇官网",
    "rog_official": "ROG 官网",
    "dareu_official": "达尔优官网",
    "vgn_official": "VGN 官网",
}
CONNECTION_LABELS = {
    "wired": "有线",
    "wireless_2_4g": "2.4G 无线",
    "bluetooth": "蓝牙",
    "multi_mode": "多模",
    "unknown": "未提供",
}


def load_overview_data(database_path: str | Path) -> pd.DataFrame:
    """Load one row per active product using its newest price snapshot."""
    initialize_database(database_path)
    query = """
        WITH ranked_prices AS (
            SELECT
                ps.*,
                ROW_NUMBER() OVER (
                    PARTITION BY ps.product_id
                    ORDER BY ps.collected_at DESC, ps.id DESC
                ) AS price_rank
            FROM price_snapshots ps
        )
        SELECT
            p.id AS product_id,
            COALESCE(NULLIF(p.source_site, ''), p.platform) AS platform,
            p.source_site,
            p.platform_product_id,
            p.category,
            p.brand,
            p.model,
            p.product_name,
            p.shop_name,
            p.product_url,
            p.is_active,
            s.connection_type,
            s.weight_g,
            s.polling_rate_hz,
            s.rgb,
            s.warranty_months,
            s.dpi_max,
            s.sensor_model,
            s.button_count,
            s.handedness,
            s.switch_type,
            s.switch_name,
            s.layout_size,
            s.key_count,
            s.actuation_mm,
            s.hot_swappable,
            rp.list_price,
            rp.sale_price,
            rp.coupon_price,
            COALESCE(rp.coupon_price, rp.sale_price, rp.list_price) AS current_price,
            rp.currency,
            COALESCE(rp.collected_at, p.last_seen_at) AS collected_at
        FROM products p
        LEFT JOIN product_specs s ON s.product_id = p.id
        LEFT JOIN ranked_prices rp
            ON rp.product_id = p.id AND rp.price_rank = 1
        WHERE p.is_active = 1
        ORDER BY p.category, p.brand, p.product_name
    """
    with connect(database_path) as connection:
        cursor = connection.execute(query)
        columns = [item[0] for item in cursor.description]
        rows = cursor.fetchall()
    frame = pd.DataFrame([tuple(row) for row in rows], columns=columns)
    if not frame.empty:
        frame["current_price"] = pd.to_numeric(frame["current_price"], errors="coerce")
        frame["collected_at"] = pd.to_datetime(frame["collected_at"], errors="coerce", utc=True)
    return frame


def filter_overview_data(
    data: pd.DataFrame,
    *,
    category: str,
    platforms: Iterable[str] | None = None,
    brands: Iterable[str] | None = None,
    price_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    filtered = data[data["category"] == category].copy()
    platform_values = list(platforms or [])
    brand_values = list(brands or [])
    if platform_values:
        filtered = filtered[filtered["platform"].isin(platform_values)]
    if brand_values:
        filtered = filtered[filtered["brand"].isin(brand_values)]
    if price_range is not None:
        minimum, maximum = price_range
        filtered = filtered[filtered["current_price"].between(minimum, maximum, inclusive="both")]
    return filtered.reset_index(drop=True)


def calculate_summary(data: pd.DataFrame) -> dict[str, object]:
    if data.empty:
        return {
            "product_count": 0,
            "brand_count": 0,
            "platform_count": 0,
            "average_price": None,
            "completeness_rate": 0.0,
            "latest_update": None,
        }
    critical_fields = ["brand", "product_name", "category", "current_price"]
    completeness = data[critical_fields].notna().mean().mean() * 100
    latest_update = data["collected_at"].max()
    average_price = data["current_price"].mean()
    return {
        "product_count": int(data["product_id"].nunique()),
        "brand_count": int(data["brand"].nunique()),
        "platform_count": int(data["platform"].nunique()),
        "average_price": float(average_price) if pd.notna(average_price) else None,
        "completeness_rate": float(completeness),
        "latest_update": latest_update,
    }


def build_quality_report(data: pd.DataFrame, category: str) -> pd.DataFrame:
    common_fields = [
        ("brand", "品牌"),
        ("product_name", "商品名称"),
        ("current_price", "有效价格"),
        ("connection_type", "连接方式"),
        ("weight_g", "重量"),
        ("polling_rate_hz", "回报率"),
    ]
    category_fields = (
        [("dpi_max", "最大 DPI"), ("sensor_model", "传感器型号")]
        if category == "mouse"
        else [("switch_type", "轴体类型"), ("layout_size", "键盘配列")]
    )
    total = len(data)
    records = []
    for field, label in common_fields + category_fields:
        valid = int(data[field].notna().sum()) if total else 0
        rate = valid / total * 100 if total else 0.0
        records.append(
            {
                "字段": label,
                "有效记录": valid,
                "缺失记录": total - valid,
                "完整率": rate,
            }
        )
    return pd.DataFrame(records)

