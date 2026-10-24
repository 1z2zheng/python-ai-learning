from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.config import get_database_path
from esports_analysis.db import get_database_summary


if __name__ == "__main__":
    database_path = get_database_path()
    summary = get_database_summary(database_path)
    print(f"数据库: {database_path}")
    print(f"商品数: {summary['product_count']}")
    print(f"品牌数: {summary['brand_count']}")
    print(f"参数记录: {summary['spec_count']}")
    print(f"价格快照: {summary['price_snapshot_count']}")

