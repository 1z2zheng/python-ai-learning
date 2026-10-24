from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.config import get_database_path
from esports_analysis.db import initialize_database


if __name__ == "__main__":
    database_path = get_database_path()
    initialize_database(database_path)
    print(f"数据库初始化完成: {database_path}")

