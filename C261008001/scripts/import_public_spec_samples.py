"""Store indexed primary-source specs separately from live/price research."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from esports_analysis.config import get_database_path
from esports_analysis.db import connect


def import_samples(database_path, source_path):
    data = json.loads(Path(source_path).read_text(encoding="utf-8"))
    samples = data["samples"]
    for row in samples:
        if str(row["attributes"].get("商品编号")) != row["source_sku"]:
            raise ValueError("来源商品编号不一致")
        if row.get("price") is not None or row.get("reviews"):
            raise ValueError("索引规格样本不能包含未经核验的价格或评论")
    with connect(database_path) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS public_spec_research_samples (
            model_key TEXT PRIMARY KEY,model_name TEXT NOT NULL,brand TEXT NOT NULL,
            category TEXT NOT NULL,source_sku TEXT NOT NULL,source_url TEXT NOT NULL,
            attributes_json TEXT NOT NULL,retrieved_date TEXT NOT NULL,
            source_method TEXT NOT NULL,freshness_note TEXT NOT NULL)""")
        inserted = 0
        for row in samples:
            result = db.execute("""INSERT INTO public_spec_research_samples VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(model_key) DO NOTHING""", (
                row["model_key"], row["model_name"], row["brand"], row["category"], row["source_sku"],
                row["source_url"], json.dumps(row["attributes"], ensure_ascii=False), data["retrieved_date"],
                data["method"], data["freshness"]))
            inserted += result.rowcount
    return inserted


if __name__ == "__main__":
    print("新增独立型号的历史索引规格样本：", import_samples(get_database_path(),
          ROOT / "data/raw/public_spec_samples_20261015.json"))
