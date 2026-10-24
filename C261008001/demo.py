"""Prepare an isolated demo database and launch the existing Streamlit app."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.db import get_database_summary, initialize_database
from esports_analysis.importer import import_products_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="启动内置合成数据演示，不使用生产数据库")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    database = ROOT / "data" / "demo.db"
    initialize_database(database)
    if get_database_summary(database)["product_count"] == 0:
        result = import_products_csv(ROOT / "data" / "seed" / "demo_80_products.csv", database)
        if result.failed_count:
            raise RuntimeError("演示数据导入失败：" + "; ".join(result.errors))
    summary = get_database_summary(database)
    if summary["product_count"] != 80 or summary["price_snapshot_count"] != 240:
        raise RuntimeError(f"演示数据数量异常：{summary}")
    print("演示数据：80 个合成商品、240 条价格快照；非真实市场数据。", flush=True)
    if args.prepare_only:
        return 0
    environment = os.environ.copy()
    environment["ESPORTS_DB_PATH"] = str(database)
    environment["PORTFOLIO_DEMO"] = "1"
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
         "--server.port", str(args.port), "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT, env=environment,
    )


if __name__ == "__main__":
    raise SystemExit(main())
