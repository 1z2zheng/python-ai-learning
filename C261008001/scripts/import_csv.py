import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.config import get_database_path
from esports_analysis.importer import import_products_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="导入电竞外设商品 CSV")
    parser.add_argument("csv_path", type=Path, help="CSV 文件路径")
    parser.add_argument("--database", type=Path, default=get_database_path())
    args = parser.parse_args()

    try:
        result = import_products_csv(args.csv_path, args.database)
    except (FileNotFoundError, ValueError) as exc:
        print(f"导入失败: {exc}", file=sys.stderr)
        return 1

    print(f"批次: {result.batch_id}")
    print(f"状态: {result.status}")
    print(f"读取: {result.fetched_count}")
    print(f"新增: {result.inserted_count}")
    print(f"更新: {result.updated_count}")
    print(f"失败: {result.failed_count}")
    for error in result.errors:
        print(error, file=sys.stderr)
    return 0 if result.status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

