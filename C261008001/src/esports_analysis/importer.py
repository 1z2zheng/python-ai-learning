from __future__ import annotations

import csv
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .db import connect, initialize_database


REQUIRED_COLUMNS = {
    "platform",
    "platform_product_id",
    "category",
    "brand",
    "product_name",
    "collected_at",
}

PLATFORM_ALIASES = {
    "jd": "jd",
    "京东": "jd",
    "tmall": "tmall",
    "天猫": "tmall",
    "manual": "manual",
    "人工": "manual",
    "third_party": "third_party",
    "logitech_official": "third_party",
    "razer_official": "third_party",
    "rog_official": "third_party",
    "dareu_official": "third_party",
    "vgn_official": "third_party",
}

CATEGORY_ALIASES = {
    "mouse": "mouse",
    "鼠标": "mouse",
    "keyboard": "keyboard",
    "键盘": "keyboard",
}

BRAND_ALIASES = {
    "logitech": "罗技",
    "logitech g": "罗技",
    "罗技g": "罗技",
    "罗技": "罗技",
    "razer": "雷蛇",
    "雷蛇": "雷蛇",
    "rog": "ROG",
    "vgn": "VGN",
    "dareu": "达尔优",
    "达尔优": "达尔优",
}

CONNECTION_ALIASES = {
    "wired": "wired",
    "有线": "wired",
    "wireless_2_4g": "wireless_2_4g",
    "2.4g": "wireless_2_4g",
    "bluetooth": "bluetooth",
    "蓝牙": "bluetooth",
    "multi_mode": "multi_mode",
    "三模": "multi_mode",
    "双模": "multi_mode",
    "unknown": "unknown",
    "未知": "unknown",
}


class RowValidationError(ValueError):
    pass


@dataclass
class ImportResult:
    batch_id: str
    fetched_count: int = 0
    inserted_count: int = 0
    updated_count: int = 0
    failed_count: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.failed_count == 0:
            return "success"
        if self.inserted_count + self.updated_count > 0:
            return "partial"
        return "failed"


@dataclass
class ValidationResult:
    row_count: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    errors: list[str] = field(default_factory=list)
    normalized_rows: list[dict[str, Any]] = field(default_factory=list)


def validate_products_csv(csv_path: str | Path, preview_limit: int = 20) -> ValidationResult:
    """Validate a product CSV without changing the database."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")
    result = ValidationResult()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - headers
        if missing:
            raise ValueError(f"CSV 缺少列: {', '.join(sorted(missing))}")
        for line_number, raw_row in enumerate(reader, start=2):
            result.row_count += 1
            try:
                normalized = normalize_row(raw_row)
                result.valid_count += 1
                if len(result.normalized_rows) < preview_limit:
                    result.normalized_rows.append(normalized)
            except RowValidationError as exc:
                result.invalid_count += 1
                result.errors.append(f"第 {line_number} 行: {exc}")
    return result


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"-", "n/a", "null", "none", "未知", "未提供"}:
        return None
    return value


def _required(row: dict[str, str], name: str) -> str:
    value = _clean(row.get(name))
    if value is None:
        raise RowValidationError(f"缺少必填字段 {name}")
    return value


def _number(value: Any, name: str, integer: bool = False) -> int | float | None:
    cleaned = _clean(value)
    if cleaned is None:
        return None
    try:
        number = float(cleaned.replace(",", ""))
    except ValueError as exc:
        raise RowValidationError(f"{name} 不是有效数字: {cleaned}") from exc
    if number < 0:
        raise RowValidationError(f"{name} 不能为负数")
    return int(number) if integer else number


def _boolean(value: Any, name: str, default: bool | None = None) -> int | None:
    cleaned = _clean(value)
    if cleaned is None:
        return None if default is None else int(default)
    lowered = cleaned.lower()
    if lowered in {"1", "true", "yes", "y", "是"}:
        return 1
    if lowered in {"0", "false", "no", "n", "否"}:
        return 0
    raise RowValidationError(f"{name} 不是有效布尔值: {cleaned}")


def _timestamp(value: Any) -> tuple[str, str]:
    cleaned = _clean(value)
    if cleaned is None:
        raise RowValidationError("缺少必填字段 collected_at")
    normalized = cleaned.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise RowValidationError(f"collected_at 不是 ISO 8601 时间: {cleaned}") from exc
    if parsed.tzinfo is None:
        raise RowValidationError("collected_at 必须包含时区")
    return parsed.isoformat(), parsed.date().isoformat()


def normalize_row(row: dict[str, str]) -> dict[str, Any]:
    platform_raw = _required(row, "platform")
    platform = PLATFORM_ALIASES.get(platform_raw.lower())
    if platform is None:
        raise RowValidationError(f"不支持的平台: {platform_raw}")

    category_raw = _required(row, "category")
    category = CATEGORY_ALIASES.get(category_raw.lower())
    if category is None:
        raise RowValidationError(f"不支持的产品类别: {category_raw}")

    brand_raw = _required(row, "brand")
    brand = BRAND_ALIASES.get(brand_raw.lower(), brand_raw)

    sale_price = _number(row.get("sale_price"), "sale_price")
    if sale_price is not None and sale_price <= 0:
        raise RowValidationError("sale_price 必须大于零")

    collected_at, collected_date = _timestamp(row.get("collected_at"))
    connection_raw = _clean(row.get("connection_type"))
    connection_type = None
    if connection_raw is not None:
        connection_type = CONNECTION_ALIASES.get(connection_raw.lower())
        if connection_type is None:
            raise RowValidationError(f"不支持的连接方式: {connection_raw}")

    source_site = _clean(row.get("source_site"))
    if source_site is None and platform_raw.lower().endswith("_official"):
        source_site = platform_raw.lower()
    platform_product_id = _required(row, "platform_product_id")
    if platform == "third_party" and source_site:
        platform_product_id = f"{source_site}:{platform_product_id}"

    normalized = {
        "platform": platform,
        "source_site": source_site,
        "platform_product_id": platform_product_id,
        "category": category,
        "brand": brand,
        "model": _clean(row.get("model")),
        "product_name": _required(row, "product_name"),
        "shop_name": _clean(row.get("shop_name")),
        "product_url": _clean(row.get("product_url")),
        "image_url": _clean(row.get("image_url")),
        "is_active": _boolean(row.get("is_active"), "is_active", True),
        "connection_type": connection_type,
        "weight_g": _number(row.get("weight_g"), "weight_g"),
        "polling_rate_hz": _number(row.get("polling_rate_hz"), "polling_rate_hz", True),
        "rgb": _boolean(row.get("rgb"), "rgb"),
        "warranty_months": _number(row.get("warranty_months"), "warranty_months", True),
        "dpi_max": _number(row.get("dpi_max"), "dpi_max", True),
        "sensor_model": _clean(row.get("sensor_model")),
        "button_count": _number(row.get("button_count"), "button_count", True),
        "handedness": _clean(row.get("handedness")),
        "switch_type": _clean(row.get("switch_type")),
        "switch_name": _clean(row.get("switch_name")),
        "layout_size": _clean(row.get("layout_size")),
        "key_count": _number(row.get("key_count"), "key_count", True),
        "actuation_mm": _number(row.get("actuation_mm"), "actuation_mm"),
        "hot_swappable": _boolean(row.get("hot_swappable"), "hot_swappable"),
        "source_text": _clean(row.get("source_text")),
        "list_price": _number(row.get("list_price"), "list_price"),
        "sale_price": sale_price,
        "coupon_price": _number(row.get("coupon_price"), "coupon_price"),
        "currency": _clean(row.get("currency")) or "CNY",
        "promotion_text": _clean(row.get("promotion_text")),
        "price_source": _clean(row.get("price_source")) or platform,
        "collected_at": collected_at,
        "collected_date": collected_date,
    }

    if category == "mouse" and normalized["switch_type"]:
        raise RowValidationError("鼠标记录不应填写 switch_type")
    if category == "keyboard" and normalized["dpi_max"] is not None:
        raise RowValidationError("键盘记录不应填写 dpi_max")
    return normalized


def _upsert_row(connection: sqlite3.Connection, row: dict[str, Any], batch_id: str) -> str:
    existing = connection.execute(
        "SELECT id FROM products WHERE platform = ? AND platform_product_id = ?",
        (row["platform"], row["platform_product_id"]),
    ).fetchone()
    now = datetime.now(timezone.utc).isoformat()
    action = "updated" if existing else "inserted"

    connection.execute(
        """
        INSERT INTO products (
            platform, source_site, platform_product_id, category, brand, model, product_name,
            shop_name, product_url, image_url, is_active, first_seen_at,
            last_seen_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(platform, platform_product_id) DO UPDATE SET
            source_site=excluded.source_site, category=excluded.category, brand=excluded.brand, model=excluded.model,
            product_name=excluded.product_name, shop_name=excluded.shop_name,
            product_url=excluded.product_url, image_url=excluded.image_url,
            is_active=excluded.is_active, last_seen_at=excluded.last_seen_at,
            updated_at=excluded.updated_at
        """,
        (
            row["platform"], row["source_site"], row["platform_product_id"], row["category"],
            row["brand"], row["model"], row["product_name"], row["shop_name"],
            row["product_url"], row["image_url"], row["is_active"],
            row["collected_at"], row["collected_at"], now, now,
        ),
    )
    product_id = connection.execute(
        "SELECT id FROM products WHERE platform = ? AND platform_product_id = ?",
        (row["platform"], row["platform_product_id"]),
    ).fetchone()[0]

    spec_columns = [
        "connection_type", "weight_g", "polling_rate_hz", "rgb",
        "warranty_months", "dpi_max", "sensor_model", "button_count",
        "handedness", "switch_type", "switch_name", "layout_size", "key_count",
        "actuation_mm", "hot_swappable", "source_text",
    ]
    connection.execute(
        f"""
        INSERT INTO product_specs (product_id, {', '.join(spec_columns)}, updated_at)
        VALUES ({', '.join(['?'] * (len(spec_columns) + 2))})
        ON CONFLICT(product_id) DO UPDATE SET
        {', '.join(f'{name}=COALESCE(excluded.{name}, product_specs.{name})' for name in spec_columns)},
        updated_at=excluded.updated_at
        """,
        (product_id, *(row[name] for name in spec_columns), now),
    )

    if row["sale_price"] is not None:
        connection.execute(
            """
            INSERT INTO price_snapshots (
                product_id, list_price, sale_price, coupon_price, currency,
                promotion_text, price_source, collected_at, collected_date, source_batch_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(product_id, collected_date) DO UPDATE SET
                list_price=excluded.list_price, sale_price=excluded.sale_price,
                coupon_price=excluded.coupon_price, currency=excluded.currency,
                promotion_text=excluded.promotion_text, price_source=excluded.price_source,
                collected_at=excluded.collected_at,
                source_batch_id=excluded.source_batch_id
            """,
            (
                product_id, row["list_price"], row["sale_price"], row["coupon_price"],
                row["currency"], row["promotion_text"], row["price_source"], row["collected_at"],
                row["collected_date"], batch_id,
            ),
        )
    return action


def import_products_csv(csv_path: str | Path, database_path: str | Path) -> ImportResult:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    initialize_database(database_path)
    batch_id = f"csv_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    result = ImportResult(batch_id=batch_id)
    started_at = datetime.now(timezone.utc).isoformat()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - headers
        if missing:
            raise ValueError(f"CSV 缺少列: {', '.join(sorted(missing))}")

        with connect(database_path) as connection:
            connection.execute(
                "INSERT INTO collection_jobs (id, source, status, started_at) VALUES (?, ?, 'running', ?)",
                (batch_id, "csv", started_at),
            )
            for line_number, raw_row in enumerate(reader, start=2):
                result.fetched_count += 1
                connection.execute("SAVEPOINT import_row")
                try:
                    row = normalize_row(raw_row)
                    action = _upsert_row(connection, row, batch_id)
                    if action == "inserted":
                        result.inserted_count += 1
                    else:
                        result.updated_count += 1
                    connection.execute("RELEASE SAVEPOINT import_row")
                except (RowValidationError, sqlite3.DatabaseError) as exc:
                    connection.execute("ROLLBACK TO SAVEPOINT import_row")
                    connection.execute("RELEASE SAVEPOINT import_row")
                    result.failed_count += 1
                    result.errors.append(f"第 {line_number} 行: {exc}")

            finished_at = datetime.now(timezone.utc).isoformat()
            connection.execute(
                """
                UPDATE collection_jobs SET status=?, finished_at=?, fetched_count=?,
                    inserted_count=?, updated_count=?, failed_count=?, error_summary=?
                WHERE id=?
                """,
                (
                    result.status, finished_at, result.fetched_count,
                    result.inserted_count, result.updated_count, result.failed_count,
                    "\n".join(result.errors[:20]) or None, batch_id,
                ),
            )
    return result

