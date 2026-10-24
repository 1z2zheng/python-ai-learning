from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .db import connect, initialize_database
from .review_cleaning import DIMENSION_KEYWORDS, normalize_review_text

SENTIMENTS = ("正向", "中性", "混合", "负向")
DECISIONS = ("已确认", "已修改", "排除", "需再核对")
DIMENSION_NAMES = tuple(DIMENSION_KEYWORDS)


def content_hash(content: str) -> str:
    return hashlib.sha256(normalize_review_text(content).encode("utf-8")).hexdigest()


def load_annotation_data(path: str | Path, database_path: str | Path) -> pd.DataFrame:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    initialize_database(database_path)
    with connect(database_path) as db:
        from .review_identity import load_resolved_reviews
        source_rows=load_resolved_reviews(db)
        known={(str(r['入口SKU']),r['原评论哈希ID']) for r in rows}
        for source in source_rows.to_dict('records'):
            identity=(str(source['requested_sku']),source['source_review_id'])
            if identity in known:
                continue
            rows.append({
                '复核状态':'待人工复核','评论编号':int(source['review_row_id']),
                '评论正文':normalize_review_text(source['content']), 'AI整体情感':'未标注',
                'AI涉及维度':'', **{'AI'+d:'未涉及' for d in DIMENSION_NAMES},
                'AI判断依据及疑点':'新采集评论，尚未生成辅助标签',
                '人工复核整体情感':'','人工复核维度及情感':'','复核人':'','复核日期':'','人工复核备注':'',
                '入口SKU':identity[0], 'SKU归属':'来源SKU与入口一致' if source['source_product_id']==identity[0] else '来源归属待核验',
                '标签来源':'未标注','原评论哈希ID':identity[1],
            })
            known.add(identity)
        frame=pd.DataFrame(rows)
        resolved={(str(r['requested_sku']),r['source_review_id']):r for r in source_rows.to_dict('records')}
        latest = db.execute("""
            SELECT e.* FROM review_annotation_events e
            JOIN (SELECT target_sku, source_review_id, MAX(id) latest_id
                  FROM review_annotation_events GROUP BY target_sku, source_review_id) l
            ON e.id=l.latest_id
        """).fetchall()
    indexed = {(r["target_sku"], r["source_review_id"]): dict(r) for r in latest}
    for index, row in frame.iterrows():
        frame.at[index, "入口SKU"] = str(row["入口SKU"])
        identity=resolved.get((str(row['入口SKU']),row['原评论哈希ID']))
        if identity:
            frame.at[index,'解析后SKU']=identity['platform_product_id']
            frame.at[index,'解析后商品']=identity['product_name']
            frame.at[index,'来源核验']=identity['identity_resolution']
        saved = indexed.get((str(row["入口SKU"]), row["原评论哈希ID"]))
        if not saved:
            continue
        if saved["content_sha256"] != content_hash(row["评论正文"]):
            frame.at[index, "复核状态"] = "文本已更新需复核"
            continue
        frame.at[index, "复核状态"] = saved["decision"]
        frame.at[index, "人工复核整体情感"] = saved["sentiment"]
        frame.at[index, "人工复核维度及情感"] = saved["aspects_json"]
        frame.at[index, "复核人"] = saved["reviewer"]
        frame.at[index, "复核日期"] = saved["reviewed_at"]
        frame.at[index, "人工复核备注"] = saved["note"]
    return frame


def save_annotation(database_path: str | Path, row: dict, reviewer: str,
                    sentiment: str, aspects: dict[str, str], decision: str, note: str = "") -> int:
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("请填写复核人昵称或编号")
    if sentiment not in SENTIMENTS or decision not in DECISIONS:
        raise ValueError("复核标签或状态不合法")
    if set(aspects) - set(DIMENSION_NAMES) or set(aspects.values()) - set(SENTIMENTS):
        raise ValueError("维度或维度情感不合法")
    if decision in ("排除", "需再核对") and not note.strip():
        raise ValueError("排除或待核对评论需要填写原因")
    if decision == "已确认":
        expected = {d: row.get("AI"+d) for d in DIMENSION_NAMES if row.get("AI"+d) in SENTIMENTS}
        if sentiment != row.get("AI整体情感") or aspects != expected:
            raise ValueError("标签已改变，请选择已修改")
    initialize_database(database_path)
    with connect(database_path) as db:
        source = db.execute("""
            SELECT r.content FROM reviews r JOIN products p ON p.id=r.product_id
            WHERE p.platform='jd' AND p.platform_product_id=? AND r.source_review_id=?
        """, (str(row["入口SKU"]), row["原评论哈希ID"])).fetchone()
        if source is None or content_hash(source["content"]) != content_hash(row["评论正文"]):
            raise ValueError("原始评论已改变或无法匹配，请先刷新辅助标签后复核")
        cursor = db.execute("""INSERT INTO review_annotation_events (
            target_sku, source_review_id, content_sha256, reviewer, sentiment,
            aspects_json, decision, note, reviewed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            str(row["入口SKU"]), row["原评论哈希ID"], content_hash(row["评论正文"]),
            reviewer, sentiment, json.dumps(aspects, ensure_ascii=False), decision,
            note.strip(), datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        ))
        return int(cursor.lastrowid)


def annotation_history(database_path: str | Path, sku: str, review_id: str) -> pd.DataFrame:
    initialize_database(database_path)
    with connect(database_path) as db:
        return pd.read_sql_query("""
            SELECT decision AS 状态, sentiment AS 整体情感, aspects_json AS 六维标签,
                   reviewer AS 复核人, note AS 备注, reviewed_at AS 复核时间
            FROM review_annotation_events WHERE target_sku=? AND source_review_id=?
            ORDER BY id DESC
        """, db, params=(sku, review_id))
