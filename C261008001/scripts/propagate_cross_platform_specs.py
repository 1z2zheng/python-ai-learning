from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.config import get_database_path
from esports_analysis.db import connect, initialize_database


COPY_FIELDS = [
    "connection_type",
    "weight_g",
    "polling_rate_hz",
    "rgb",
    "dpi_max",
    "sensor_model",
    "button_count",
    "handedness",
    "switch_type",
    "layout_size",
    "key_count",
    "actuation_mm",
    "hot_swappable",
]


def main() -> int:
    pairs_path = ROOT / "data" / "manmanbuy" / "cross_platform_pairs.csv"
    database_path = get_database_path()
    initialize_database(database_path)
    with pairs_path.open("r", encoding="utf-8-sig", newline="") as handle:
        pairs = list(csv.DictReader(handle))

    updated = 0
    skipped = []
    collected_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    with connect(database_path) as connection:
        for pair in pairs:
            jd = connection.execute(
                """SELECT p.id, s.* FROM products p
                   JOIN product_specs s ON s.product_id=p.id
                   WHERE p.platform='jd' AND p.platform_product_id=?""",
                (pair["jd_sku"],),
            ).fetchone()
            tmall = connection.execute(
                "SELECT id FROM products WHERE platform='tmall' AND platform_product_id=?",
                (pair["tmall_item_id"],),
            ).fetchone()
            if not jd or not tmall:
                skipped.append(pair["pair_key"])
                continue

            source_text = json.dumps(
                {
                    "method": "exact_model_cross_platform_match",
                    "pair_key": pair["pair_key"],
                    "model": pair["model"],
                    "jd_sku": pair["jd_sku"],
                    "jd_url": pair["jd_url"],
                    "tmall_url": pair["tmall_url"],
                    "note": "型号无关技术参数继承自同型号京东记录；未继承轴体名称等可选变体字段。",
                },
                ensure_ascii=False,
            )
            assignments = ", ".join(f"{field}=COALESCE(?, {field})" for field in COPY_FIELDS)
            connection.execute(
                f"""UPDATE product_specs SET {assignments}, source_text=?, updated_at=?
                    WHERE product_id=?""",
                (*[jd[field] for field in COPY_FIELDS], source_text, collected_at, tmall["id"]),
            )
            updated += 1

    print(json.dumps({
        "pair_count": len(pairs),
        "updated": updated,
        "skipped": skipped,
        "copied_fields": COPY_FIELDS,
        "source": str(pairs_path),
    }, ensure_ascii=False, indent=2))
    return 0 if not skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
