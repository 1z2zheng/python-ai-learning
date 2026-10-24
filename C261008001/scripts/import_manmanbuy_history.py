from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, time as clock_time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SPIDER = ROOT / "historical-price-spider"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SPIDER))

from esports_analysis.config import get_database_path
from esports_analysis.importer import import_products_csv
from main import fetch_product, fetch_product_official_web, parse_history, parse_official_history


FIELDS = [
    "platform", "platform_product_id", "category", "brand", "product_name",
    "shop_name", "product_url", "is_active", "sale_price", "currency",
    "collected_at", "price_source", "promotion_text",
]


def enabled(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "是"}


def fetch_target_history(
    target: dict[str, str],
    provider: str,
    timeout: float,
    cdp_url: str,
    node_executable: str | None,
):
    if provider == "official-web":
        product = fetch_product_official_web(
            target["product_url"],
            timeout=timeout,
            cdp_url=cdp_url,
            node_executable=node_executable,
        )
        return product, parse_official_history(str(product.get("datePrice") or "")), product.get("spName")
    product = fetch_product(target["product_url"], timeout=timeout)
    return product, parse_history(str(product.get("jiagequshi") or "")), product.get("title")


def collect(
    targets_path: Path,
    output_path: Path,
    log_path: Path,
    delay: float,
    timeout: float,
    provider: str,
    cdp_url: str,
    node_executable: str | None,
    retries: int,
) -> dict:
    with targets_path.open("r", encoding="utf-8-sig", newline="") as handle:
        targets = [row for row in csv.DictReader(handle) if enabled(row.get("enabled", ""))]

    rows: list[dict[str, object]] = []
    products: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    china = ZoneInfo("Asia/Shanghai")

    for index, target in enumerate(targets, start=1):
        product = history = title = None
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                product, history, title = fetch_target_history(
                    target, provider, timeout, cdp_url, node_executable
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    wait_seconds = max(delay, 1.0) * attempt
                    print(
                        f"[{index}/{len(targets)}] {target['sku']}："
                        f"第 {attempt} 次失败，{wait_seconds:.1f} 秒后重试"
                    )
                    time.sleep(wait_seconds)
        if product is None or history is None:
            error_text = str(last_error or "未知错误")
            errors.append({"sku": target["sku"], "url": target["product_url"], "error": error_text})
            print(f"[{index}/{len(targets)}] {target['sku']}：失败：{error_text}")
        else:
            for record in history.to_dict("records"):
                collected_at = datetime.combine(record["date_日期"], clock_time(12, 0), tzinfo=china)
                rows.append({
                    "platform": target["platform"],
                    "platform_product_id": target["sku"],
                    "category": target["category"],
                    "brand": target["brand"],
                    "product_name": str(title or target["product_name"]),
                    "shop_name": target["shop_name"],
                    "product_url": target["product_url"],
                    "is_active": "true",
                    "sale_price": record["price_价格"],
                    "currency": "CNY",
                    "collected_at": collected_at.isoformat(),
                    "price_source": "manmanbuy",
                    "promotion_text": record.get("promotion_text") or "慢慢买第三方历史价格",
                })
            products.append({
                "sku": target["sku"],
                "title": title,
                "history_points": len(history),
                "date_start": str(history.iloc[0]["date_日期"]),
                "date_end": str(history.iloc[-1]["date_日期"]),
                "lowest_price": product.get("lowerPrice"),
            })
            print(f"[{index}/{len(targets)}] {target['sku']}：{len(history)} 条")
        if index < len(targets):
            time.sleep(max(delay, 0.5))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "collected_at": datetime.now(china).isoformat(timespec="seconds"),
        "source": f"manmanbuy:{provider}",
        "target_count": len(targets),
        "success_count": len(products),
        "history_row_count": len(rows),
        "products": products,
        "errors": errors,
        "output": str(output_path),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="采集慢慢买电竞键鼠历史价格并导入项目")
    parser.add_argument("--targets", type=Path, default=ROOT / "data/manmanbuy/esports_targets.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/manmanbuy_esports_history.csv")
    parser.add_argument("--log", type=Path, default=ROOT / "logs/manmanbuy_esports_history.json")
    parser.add_argument("--database", type=Path, default=get_database_path())
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=75.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--provider", choices=["official-web", "legacy"], default="official-web",
    )
    parser.add_argument("--cdp-url", default="http://127.0.0.1:9222")
    parser.add_argument("--node", dest="node_executable")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    if args.retries < 1:
        parser.error("--retries 必须大于等于 1")

    report = collect(
        args.targets,
        args.output,
        args.log,
        args.delay,
        args.timeout,
        args.provider,
        args.cdp_url,
        args.node_executable,
        args.retries,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["history_row_count"]:
        return 1
    if not args.collect_only:
        result = import_products_csv(args.output, args.database)
        print(
            f"导入状态={result.status}，读取={result.fetched_count}，"
            f"新增={result.inserted_count}，更新={result.updated_count}，失败={result.failed_count}"
        )
        return 0 if result.status != "failed" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
