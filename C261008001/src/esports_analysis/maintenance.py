from __future__ import annotations

from pathlib import Path

import pandas as pd

from .db import connect, initialize_database
from .overview import build_quality_report, load_overview_data
from .product_evidence import load_evidence_data


TASK_COLUMNS = ["商品ID", "平台", "SKU", "类别", "品牌", "商品", "优先级", "任务类型",
                "缺口", "严格评论数", "报价日期数", "商品链接"]
CORE_SPECS = {
    "mouse": {"connection_type": "连接方式", "weight_g": "重量", "polling_rate_hz": "回报率",
              "dpi_max": "DPI", "sensor_model": "传感器"},
    "keyboard": {"connection_type": "连接方式", "polling_rate_hz": "回报率",
                 "switch_type": "轴体类型", "switch_name": "轴体名称", "layout_size": "配列"},
}


def build_research_collection_tasks(database_path: str | Path, review_target: int = 20) -> pd.DataFrame:
    """Read-only evidence gaps; targets are planning conventions, not significance tests."""
    if not isinstance(review_target, int) or isinstance(review_target, bool) or review_target < 1:
        raise ValueError("评论补采目标必须为正整数")
    bundle = load_evidence_data(database_path)
    products = bundle["products"]
    products = products[~products["product_name"].fillna("").str.contains("演示", regex=False)
                        & ~products["shop_name"].fillna("").str.contains("功能测试数据", regex=False)
                        & products["platform"].isin(["jd", "tmall"])].copy()
    history = bundle["history"].copy()
    history["quote_day"] = pd.to_datetime(history["collected_at"], errors="coerce", utc=True).dt.date
    date_counts = history.groupby("product_id")["quote_day"].nunique()
    rows = []
    for product in products.to_dict("records"):
        count = int(product["strict_review_count"])
        days = int(date_counts.get(product["product_id"], 0))
        base = {"商品ID": product["product_id"], "平台": product["platform"],
                "SKU": str(product["platform_product_id"]),
                "类别": {"mouse": "鼠标", "keyboard": "键盘"}.get(product["category"], product["category"]),
                "品牌": product["brand"], "商品": product["product_name"],
                "严格评论数": count, "报价日期数": days, "商品链接": product["product_url"]}
        def add(priority, kind, gap):
            rows.append({**base, "优先级": priority, "任务类型": kind, "缺口": gap})
        if not product["identity_verified"]:
            add("高", "型号核验", "未建立商品页核验记录；核对SKU、型号、颜色和配置，跨平台同型号需单独核验")
        missing = []
        for field, label in CORE_SPECS.get(product["category"], {}).items():
            value = product.get(field)
            if pd.isna(value) or (isinstance(value, str) and value.strip().lower() in {"", "unknown", "未知"}):
                missing.append(label)
        if missing:
            add("中", "商品规格", "待核实：" + "、".join(missing) + "；保留来源原值，不推断精确参数")
        if count < review_target:
            add("高" if count == 0 else "中", "严格评论", f"当前{count}条，补采目标{review_target}条；需来源SKU一致，去重后计数，兼顾正中负评价")
        if days < 3:
            add("高" if days == 0 else "中", "历史报价", f"仅{days}个有效报价日期；补充到至少3个日期仅为维护目标，不等同促销期覆盖达标")
    result = pd.DataFrame(rows, columns=TASK_COLUMNS)
    if not result.empty:
        result["_priority"] = result["优先级"].map({"高": 0, "中": 1})
        result = result.sort_values(["_priority", "严格评论数", "商品ID", "任务类型"]).drop(columns="_priority")
    return result.reset_index(drop=True)


def load_recent_collection_jobs(database_path: str | Path, limit: int = 20) -> pd.DataFrame:
    initialize_database(database_path)
    query = """
        SELECT
            id AS 批次,
            source AS 来源,
            status AS 状态,
            started_at AS 开始时间,
            finished_at AS 完成时间,
            fetched_count AS 读取数,
            inserted_count AS 新增数,
            updated_count AS 更新数,
            failed_count AS 失败数,
            error_summary AS 错误摘要
        FROM collection_jobs
        ORDER BY started_at DESC
        LIMIT ?
    """
    with connect(database_path) as connection:
        cursor = connection.execute(query, (limit,))
        columns = [item[0] for item in cursor.description]
        rows = cursor.fetchall()
    return pd.DataFrame([tuple(row) for row in rows], columns=columns)


def build_database_quality_summary(database_path: str | Path) -> pd.DataFrame:
    data = load_overview_data(database_path)
    parts = []
    for category, label in (("mouse", "鼠标"), ("keyboard", "键盘")):
        category_data = data[data["category"] == category]
        quality = build_quality_report(category_data, category)
        quality.insert(0, "类别", label)
        parts.append(quality)
    return pd.concat(parts, ignore_index=True)

