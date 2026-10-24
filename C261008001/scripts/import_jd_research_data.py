from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.config import get_database_path
from esports_analysis.db import connect, initialize_database


SPEC_FIELDS = [
    "connection_type", "weight_g", "polling_rate_hz", "rgb", "dpi_max",
    "sensor_model", "button_count", "handedness", "switch_type", "switch_name",
    "layout_size", "key_count", "actuation_mm", "hot_swappable",
]


def number(value: Any, integer: bool = False) -> int | float | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = float(str(value).replace(",", "").strip())
    return int(parsed) if integer else parsed


def boolean(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return 1 if str(value).strip().lower() in {"1", "true", "yes", "是"} else 0


def parse_count(text: Any, fallback: Any = None) -> int | None:
    raw = str(text or "").replace(",", "").strip()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(万|千)?", raw)
    if match:
        multiplier = {"万": 10000, "千": 1000}.get(match.group(2), 1)
        return int(float(match.group(1)) * multiplier)
    fallback_match = re.search(r"\d+", str(fallback or ""))
    return int(fallback_match.group()) if fallback_match else None


def infer_page_specs(product: dict[str, Any]) -> dict[str, Any]:
    title = str(product.get("title") or "")
    attrs = {
        str(item.get("labelName") or "").strip(): str(item.get("labelValue") or "").strip()
        for item in product.get("attributes") or []
    }
    text = " ".join([title, *[f"{key}:{value}" for key, value in attrs.items()]])
    connection = None
    connection_text = attrs.get("连接方式", "") + title
    if "三模" in connection_text or all(name in connection_text for name in ("有线", "无线", "蓝牙")):
        connection = "multi_mode"
    elif "无线" in connection_text:
        connection = "wireless_2_4g"
    elif "有线" in connection_text or "有线供电" in attrs.get("供电方式", ""):
        connection = "wired"

    weight_match = re.search(r"(?<!\d)(\d{2,4}(?:\.\d+)?)\s*g(?:超轻|轻量|\b)", title, re.I)
    polling_matches = re.findall(r"(?<!\d)(\d+(?:\.\d+)?)\s*([kK]?)\s*Hz", text)
    polling = max(
        (int(float(value) * (1000 if unit else 1)) for value, unit in polling_matches),
        default=None,
    )
    dpi_matches = re.findall(r"(?<!\d)(\d+(?:\.\d+)?)\s*([kK]?)\s*(?:DPI|传感器)", text, re.I)
    dpi = max(
        (int(float(value) * (1000 if unit else 1)) for value, unit in dpi_matches),
        default=None,
    )
    key_match = re.search(r"(?<!\d)(\d{2,3})\s*(?:键|配列)", title)
    key_count = int(key_match.group(1)) if key_match else None
    layout = None
    if "TKL" in title.upper() or key_count == 87:
        layout = "TKL"
    elif key_count:
        if key_count <= 61:
            layout = "60%"
        elif key_count <= 68:
            layout = "65%"
        elif key_count <= 82:
            layout = "75%"
        elif key_count >= 96 and key_count < 104:
            layout = "98%"
        elif key_count >= 104:
            layout = "100%"
    switch_type = None
    if "磁轴" in title:
        switch_type = "magnetic"
    elif "光轴" in title:
        switch_type = "optical"
    elif "机械" in title or "机械键盘" in attrs.get("类型", ""):
        switch_type = "mechanical"

    return {
        "connection_type": connection,
        "weight_g": float(weight_match.group(1)) if weight_match else None,
        "polling_rate_hz": polling,
        "rgb": 1 if "RGB" in text.upper() else (0 if "无光" in text else None),
        "dpi_max": dpi,
        "sensor_model": None,
        "button_count": None,
        "handedness": "right" if "右手" in text else ("ambidextrous" if "对称" in text else None),
        "switch_type": switch_type,
        "switch_name": None,
        "layout_size": layout,
        "key_count": key_count,
        "actuation_mm": None,
        "hot_swappable": 1 if "全键热插拔" in text else (0 if "不支持热插拔" in text else None),
    }


def load_curated(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    integer_fields = {"polling_rate_hz", "dpi_max", "button_count", "key_count"}
    boolean_fields = {"rgb", "hot_swappable"}
    float_fields = {"weight_g", "actuation_mm"}
    for row in rows:
        normalized: dict[str, Any] = {"source_url": row.get("source_url"), "source_note": row.get("source_note")}
        for field in SPEC_FIELDS:
            value = row.get(field)
            if field in integer_fields:
                normalized[field] = number(value, True)
            elif field in float_fields:
                normalized[field] = number(value)
            elif field in boolean_fields:
                normalized[field] = boolean(value)
            else:
                normalized[field] = value.strip() if value and value.strip() else None
        result[row["sku"]] = normalized
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="导入京东公开评价、热度代理指标和规格")
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/jd_research_20261009.json")
    parser.add_argument("--curated", type=Path, default=ROOT / "data/spec_sources/esports_specs_curated.csv")
    parser.add_argument("--database", type=Path, default=get_database_path())
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    products = payload.get("products") or []
    curated = load_curated(args.curated)
    china = ZoneInfo("Asia/Shanghai")
    collected_at = datetime.now(china).isoformat(timespec="seconds")
    review_rows: list[dict[str, Any]] = []
    tag_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    spec_rows: list[dict[str, Any]] = []

    initialize_database(args.database)
    with connect(args.database) as connection:
        for product in products:
            sku = str(product["sku"])
            db_product = connection.execute(
                "SELECT id, category, brand, product_name FROM products WHERE platform='jd' AND platform_product_id=?",
                (sku,),
            ).fetchone()
            if not db_product:
                continue
            product_id = int(db_product["id"])
            page_specs = infer_page_specs(product)
            verified = curated.get(sku, {})
            specs = {
                field: verified.get(field) if verified.get(field) is not None else page_specs.get(field)
                for field in SPEC_FIELDS
            }
            raw_specs_json = json.dumps(product.get("attributes") or [], ensure_ascii=False)
            source_text = json.dumps({
                "jd_page": product.get("product_url"),
                "manufacturer_source": verified.get("source_url"),
                "note": verified.get("source_note"),
            }, ensure_ascii=False)
            assignments = ", ".join(f"{field}=COALESCE(?, {field})" for field in SPEC_FIELDS)
            connection.execute(
                f"""UPDATE product_specs SET {assignments}, raw_specs_json=?, source_text=?, updated_at=?
                    WHERE product_id=?""",
                (*[specs[field] for field in SPEC_FIELDS], raw_specs_json, source_text, collected_at, product_id),
            )
            spec_rows.append({
                "sku": sku, "category": db_product["category"], "brand": db_product["brand"],
                "product_name": db_product["product_name"], **specs,
                "source_url": verified.get("source_url"), "source_note": verified.get("source_note"),
            })

            summary = product.get("review_summary")
            if summary:
                review_count = parse_count(summary.get("all_count_text"), summary.get("all_count"))
                raw_text = summary.get("all_count_text") or summary.get("all_count")
                source_payload = json.dumps(summary, ensure_ascii=False)
                connection.execute(
                    """DELETE FROM sales_snapshots
                       WHERE product_id=? AND metric_name='review_count_lower_bound'
                         AND substr(collected_at, 1, 10)=?""",
                    (product_id, collected_at[:10]),
                )
                connection.execute(
                    """INSERT INTO sales_snapshots (
                        product_id, sales_value, metric_type, metric_name, review_count, rating,
                        raw_metric_text, source_url, source_payload_json, collected_at
                    ) VALUES (?, ?, 'proxy', 'review_count_lower_bound', ?, NULL, ?, ?, ?, ?)""",
                    (product_id, review_count, review_count, raw_text, product.get("product_url"), source_payload, collected_at),
                )
                summary_rows.append({
                    "sku": sku, "category": db_product["category"], "brand": db_product["brand"],
                    "product_name": db_product["product_name"], "review_count_lower_bound": review_count,
                    "review_count_text": raw_text, "good_rate_text": summary.get("good_rate_text"),
                    "good_count": summary.get("good_count"), "normal_count": summary.get("normal_count"),
                    "bad_count": summary.get("bad_count"),
                    "tags_json": json.dumps(summary.get("tags") or [], ensure_ascii=False),
                    "collected_at": collected_at, "source_url": product.get("product_url"),
                })
                for tag in summary.get("tags") or []:
                    tag_name = str(tag.get("name") or "").strip()
                    if not tag_name:
                        continue
                    tag_count = parse_count(tag.get("countText"), tag.get("count"))
                    connection.execute(
                        """INSERT INTO review_tags (
                            product_id, tag_name, tag_count, raw_count_text,
                            source_url, collected_at, collected_date
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(product_id, tag_name, collected_date) DO UPDATE SET
                            tag_count=excluded.tag_count,
                            raw_count_text=excluded.raw_count_text,
                            source_url=excluded.source_url,
                            collected_at=excluded.collected_at""",
                        (
                            product_id, tag_name, tag_count, tag.get("countText"),
                            product.get("product_url"), collected_at, collected_at[:10],
                        ),
                    )
                    tag_rows.append({
                        "sku": sku,
                        "category": db_product["category"],
                        "brand": db_product["brand"],
                        "product_name": db_product["product_name"],
                        "tag_name": tag_name,
                        "tag_count": tag_count,
                        "raw_count_text": tag.get("countText"),
                        "collected_at": collected_at,
                        "source_url": product.get("product_url"),
                    })

            for review in product.get("comments") or []:
                variant = json.dumps(review.get("variant") or [], ensure_ascii=False)
                connection.execute(
                    """INSERT INTO reviews (
                        product_id, source_review_id, rating, content, reviewed_at, variant_text,
                        source_url, collected_at, is_valid
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(product_id, source_review_id) DO UPDATE SET
                        rating=excluded.rating, content=excluded.content, reviewed_at=excluded.reviewed_at,
                        variant_text=excluded.variant_text, source_url=excluded.source_url,
                        collected_at=excluded.collected_at, is_valid=1""",
                    (product_id, review["source_review_id"], review.get("rating"), review["content"],
                     review.get("reviewed_at"), variant, product.get("product_url"), collected_at),
                )
                review_rows.append({
                    "sku": sku, "category": db_product["category"], "brand": db_product["brand"],
                    "product_name": db_product["product_name"], **review,
                    "variant": variant, "collected_at": collected_at,
                    "source_url": product.get("product_url"),
                })

        # Four throttled products still receive verified manufacturer specifications.
        for sku, verified in curated.items():
            if any(row["sku"] == sku for row in spec_rows):
                continue
            db_product = connection.execute(
                "SELECT id, category, brand, product_name FROM products WHERE platform='jd' AND platform_product_id=?",
                (sku,),
            ).fetchone()
            if not db_product:
                continue
            assignments = ", ".join(f"{field}=COALESCE(?, {field})" for field in SPEC_FIELDS)
            source_text = json.dumps({
                "manufacturer_source": verified.get("source_url"), "note": verified.get("source_note")
            }, ensure_ascii=False)
            connection.execute(
                f"UPDATE product_specs SET {assignments}, source_text=?, updated_at=? WHERE product_id=?",
                (*[verified.get(field) for field in SPEC_FIELDS], source_text, collected_at, db_product["id"]),
            )
            spec_rows.append({
                "sku": sku, "category": db_product["category"], "brand": db_product["brand"],
                "product_name": db_product["product_name"],
                **{field: verified.get(field) for field in SPEC_FIELDS},
                "source_url": verified.get("source_url"), "source_note": verified.get("source_note"),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "reviews": args.output_dir / "jd_reviews_20261009.csv",
        "summary": args.output_dir / "jd_review_summary_20261009.csv",
        "specs": args.output_dir / "jd_product_specs_20261009.csv",
    }
    for key, rows in (("reviews", review_rows), ("summary", summary_rows), ("specs", spec_rows)):
        if not rows:
            continue
        with outputs[key].open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({
        "products_with_page_data": len(products),
        "spec_products": len(spec_rows),
        "review_summary_products": len(summary_rows),
        "reviews": len(review_rows),
        "review_tags": len(tag_rows),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
