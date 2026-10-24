"""Build a pending expansion manifest without modifying research records."""
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from esports_analysis.config import get_database_path
from esports_analysis.db import connect


def prepare(database_path):
    with connect(database_path) as db:
        reviews = db.execute("""SELECT r.source_product_id sku,p.category,p.brand,
            COUNT(*) raw_review_count,GROUP_CONCAT(DISTINCT r.variant_text) variant_evidence
            FROM reviews r JOIN products p ON p.id=r.product_id
            WHERE r.source_product_id IS NOT NULL AND TRIM(r.source_product_id) <> ''
            AND NOT EXISTS(SELECT 1 FROM products q WHERE q.platform_product_id=r.source_product_id)
            GROUP BY r.source_product_id,p.category,p.brand
            ORDER BY raw_review_count DESC,r.source_product_id""").fetchall()
        catalog = db.execute("""SELECT id product_id,brand,category,product_name,product_url,source_site
            FROM products WHERE is_active=1 AND platform='third_party' AND source_site LIKE '%official'
            AND product_name NOT LIKE '%演示%'
            ORDER BY brand,category,product_name""").fetchall()
    return {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "pending_source_verification",
        "research_records_added": 0,
        "counting_rule": "不同颜色、轴体和套装先视为SKU变体；仅型号归一与来源核验通过后统计新增型号。官方目录不是平台价格样本。",
        "review_source_candidates": [{**dict(row), "product_url": f"https://item.jd.com/{row['sku']}.html",
                                      "verification_status": "pending", "independent_model_status": "unresolved"}
                                     for row in reviews if str(row["sku"]).isdigit()],
        "official_catalog_candidates": [dict(row) for row in catalog],
    }


if __name__ == "__main__":
    result = prepare(get_database_path())
    target = ROOT / "data" / "raw" / "sample_expansion_pending_20261015.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"review_source_candidates": len(result["review_source_candidates"]),
                      "official_catalog_candidates": len(result["official_catalog_candidates"]),
                      "research_records_added": 0}, ensure_ascii=False))
