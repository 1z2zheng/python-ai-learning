from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.config import get_database_path
from esports_analysis.db import connect, initialize_database


def main() -> int:
    parser = argparse.ArgumentParser(description="仅导入京东分层评论，不改写商品规格或价格")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=get_database_path())
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    collected_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    inserted = 0
    updated = 0
    skipped = 0
    products_matched: set[str] = set()

    initialize_database(args.database)
    with connect(args.database) as connection:
        for product in payload.get("products") or []:
            sku = str(product.get("sku") or "").strip()
            db_product = connection.execute(
                "SELECT id FROM products WHERE platform='jd' AND platform_product_id=?",
                (sku,),
            ).fetchone()
            if not db_product:
                skipped += len(product.get("comments") or [])
                continue
            products_matched.add(sku)
            product_id = int(db_product["id"])
            source_url = product.get("product_url")
            for review in product.get("comments") or []:
                review_id = str(review.get("source_review_id") or "").strip()
                content = str(review.get("content") or "").strip()
                if not review_id or not content:
                    skipped += 1
                    continue
                existed = connection.execute(
                    "SELECT 1 FROM reviews WHERE product_id=? AND source_review_id=?",
                    (product_id, review_id),
                ).fetchone()
                variant = json.dumps(review.get("variant") or [], ensure_ascii=False)
                connection.execute(
                    """INSERT INTO reviews (
                        product_id, source_review_id, rating, content, reviewed_at, variant_text,
                        source_product_id, sample_group, source_url, collected_at, is_valid
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(product_id, source_review_id) DO UPDATE SET
                        rating=excluded.rating,
                        content=excluded.content,
                        reviewed_at=excluded.reviewed_at,
                        variant_text=excluded.variant_text,
                        source_product_id=excluded.source_product_id,
                        sample_group=excluded.sample_group,
                        source_url=excluded.source_url,
                        collected_at=excluded.collected_at,
                        is_valid=1""",
                    (
                        product_id,
                        review_id,
                        review.get("rating"),
                        content,
                        review.get("reviewed_at"),
                        variant,
                        str(review.get("source_product_id") or sku),
                        review.get("sample_group"),
                        source_url,
                        collected_at,
                    ),
                )
                if existed:
                    updated += 1
                else:
                    inserted += 1

        totals = connection.execute(
            """SELECT COUNT(*) AS review_count,
                      COUNT(DISTINCT product_id) AS product_count
               FROM reviews WHERE is_valid=1"""
        ).fetchone()
        distribution = {
            str(row["rating"]): int(row["count"])
            for row in connection.execute(
                """SELECT CAST(rating AS INTEGER) AS rating, COUNT(*) AS count
                   FROM reviews WHERE is_valid=1 GROUP BY CAST(rating AS INTEGER)
                   ORDER BY rating"""
            ).fetchall()
        }

    print(json.dumps({
        "input": str(args.input),
        "matched_products": len(products_matched),
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
        "database_review_count": int(totals["review_count"]),
        "database_product_count": int(totals["product_count"]),
        "rating_distribution": distribution,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
