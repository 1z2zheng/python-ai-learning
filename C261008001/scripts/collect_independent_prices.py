"""Bounded acquisition: two new model targets, stop on any service failure."""
import csv
import json
import re
import sys
import time
from datetime import datetime, time as daytime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "historical-price-spider"))
from main import fetch_product, parse_history
from esports_analysis.config import get_database_path
from esports_analysis.importer import import_products_csv

TARGETS = [
    {"sku": "7571300", "brand": "罗技", "model": "logitech_g304_lightspeed",
     "tokens": ["g304", "g304黑色"], "forbidden": []},
    {"sku": "10098016952114", "brand": "雷蛇", "model": "razer_basilisk_v3_wired",
     "tokens": ["巴塞利斯蛇v3", "有线"], "forbidden": ["v3x", "35k", "专业版"]},
]


def main():
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    rows, reports = [], []
    for index, target in enumerate(TARGETS):
        if index:
            time.sleep(8)
        url = f"https://item.jd.com/{target['sku']}.html"
        try:
            product = fetch_product(url, timeout=20)
        except Exception as exc:
            reports.append({"sku": target["sku"], "status": "stopped_service_error", "error_type": type(exc).__name__, "error": str(exc)[:300]})
            break
        raw_path = ROOT / f"data/raw/manmanbuy_independent_{target['sku']}_{stamp}.json"
        raw_path.write_text(json.dumps({"requested_url": url, "fetched_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                                        "response": product}, ensure_ascii=False, indent=2), encoding="utf-8")
        title = str(product.get("title") or "")
        compact = re.sub(r"\s+", "", title).lower()
        if not all(token in compact for token in target["tokens"]) or any(token in compact for token in target["forbidden"]):
            reports.append({"sku": target["sku"], "status": "rejected_title_mismatch", "title": title})
            continue
        try:
            history = parse_history(str(product.get("jiagequshi") or ""))
        except ValueError:
            reports.append({"sku": target["sku"], "status": "no_parseable_history"})
            break
        accepted = 0
        for point in history.to_dict("records"):
            date, price = point["date_日期"], float(point["price_价格"])
            if price <= 0 or date > datetime.now(ZoneInfo("Asia/Shanghai")).date():
                continue
            rows.append({"platform": "jd", "platform_product_id": target["sku"], "category": "mouse",
                         "brand": target["brand"], "model": target["model"], "product_name": title,
                         "shop_name": "未独立核验店铺", "product_url": url, "sale_price": price,
                         "collected_at": datetime.combine(date, daytime(12), tzinfo=ZoneInfo("Asia/Shanghai")).isoformat(),
                         "price_source": "manmanbuy", "promotion_text": "慢慢买历史报价；日期精度为日，12时仅为存储约定；不保证当前成交及优惠资格"})
            accepted += 1
        reports.append({"sku": target["sku"], "model": target["model"], "status": "accepted_history",
                        "title": title, "price_points": accepted, "raw_source": str(raw_path)})
    if rows:
        output = ROOT / f"data/processed/independent_prices_{stamp}.csv"
        with output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result = import_products_csv(output, get_database_path())
        if result.failed_count:
            raise RuntimeError(f"导入失败：{result.errors}")
    report = {"targets": reports, "accepted_price_rows": len(rows)}
    (ROOT / f"logs/independent_prices_{stamp}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
