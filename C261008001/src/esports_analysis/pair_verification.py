"""Append-only, quote-bound cross-platform verification records."""
import json
from datetime import datetime, timezone

from .db import connect


def _ensure_table(db):
    db.execute("""CREATE TABLE IF NOT EXISTS cross_platform_verification_events (
        id INTEGER PRIMARY KEY, pair_key TEXT NOT NULL, payload_json TEXT NOT NULL,
        saved_at TEXT NOT NULL)""")


def load_pair_verifications(database_path):
    with connect(database_path) as db:
        _ensure_table(db)
        rows = db.execute("SELECT pair_key,payload_json FROM cross_platform_verification_events ORDER BY id").fetchall()
    return {row["pair_key"]: json.loads(row["payload_json"]) for row in rows}


def save_pair_verification(database_path, pair, record):
    payload = dict(record)
    if payload.get("status") not in {"pending", "different", "verified"}:
        raise ValueError("不支持的核验状态")
    required = ["verified_by", "note"]
    if payload["status"] == "verified":
        required += ["jd_variant", "tmall_variant", "configuration_evidence", "offer_evidence", "shop_evidence"]
    if any(not str(payload.get(name, "")).strip() for name in required):
        raise ValueError("请填写核验人和核验说明；已核验还须提供两侧配置、配置依据、店铺及优惠依据")
    if payload["status"] == "verified" and not pair.get("comparison_date"):
        raise ValueError("两侧报价日期不同，不能保存为已核验可比")
    for field in ("jd_sku", "tmall_item_id", "jd_latest_price", "tmall_latest_price", "comparison_date"):
        payload[field] = pair.get(field)
    payload["verified_at"] = datetime.now(timezone.utc).isoformat()
    with connect(database_path) as db:
        _ensure_table(db)
        db.execute("INSERT INTO cross_platform_verification_events(pair_key,payload_json,saved_at) VALUES(?,?,?)",
                   (str(pair["pair_key"]), json.dumps(payload, ensure_ascii=False, allow_nan=False), payload["verified_at"]))
