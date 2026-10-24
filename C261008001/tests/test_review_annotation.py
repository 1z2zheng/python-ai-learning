import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from esports_analysis.db import connect, initialize_database
from esports_analysis.review_annotation import (
    DIMENSION_NAMES, save_annotation, load_annotation_data, annotation_history,
)


class ReviewAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "test.db"
        initialize_database(self.db)
        with connect(self.db) as connection:
            connection.execute("""INSERT INTO products (
                platform, platform_product_id, category, brand, product_name,
                first_seen_at, last_seen_at, created_at, updated_at
            ) VALUES ('jd', '1001', 'mouse', '测试', '测试鼠标', '2026', '2026', '2026', '2026')""")
            connection.execute("""INSERT INTO reviews
                (product_id, source_review_id, content, collected_at)
                VALUES (1, 'hash1', '手感不错，但是价格太贵', '2026')""")
        self.row = {"入口SKU": "1001", "原评论哈希ID": "hash1", "评论正文": "手感不错，但是价格太贵",
                    "AI整体情感": "混合", "复核状态": "待人工复核", "评论编号": 1}
        for dimension in DIMENSION_NAMES:
            self.row["AI" + dimension] = "未涉及"
        self.row["AI手感体验"] = "正向"
        self.row["AI性价比"] = "负向"
        self.ai_path = Path(self.temp.name) / "ai.json"
        self.ai_path.write_text(json.dumps([self.row], ensure_ascii=False), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_save_and_reload_preserves_raw_review_and_history(self):
        save_annotation(self.db, self.row, "A", "混合", {"手感体验": "正向", "性价比": "负向"}, "已确认")
        save_annotation(self.db, self.row, "A", "负向", {"性价比": "负向"}, "已修改", "修正整体判断")
        frame = load_annotation_data(self.ai_path, self.db)
        self.assertEqual(frame.iloc[0]["复核状态"], "已修改")
        self.assertEqual(len(annotation_history(self.db, "1001", "hash1")), 2)
        with connect(self.db) as db:
            self.assertEqual(db.execute("SELECT content FROM reviews").fetchone()[0], self.row["评论正文"])

    def test_validation_rejects_missing_identity_or_false_confirmation(self):
        with self.assertRaises(ValueError):
            save_annotation(self.db, self.row, "", "混合", {}, "已修改")
        with self.assertRaises(ValueError):
            save_annotation(self.db, self.row, "A", "正向", {}, "已确认")
        with self.assertRaises(ValueError):
            save_annotation(self.db, self.row, "A", "混合", {}, "排除")

    def test_changed_text_cannot_reuse_old_confirmation(self):
        save_annotation(self.db, self.row, "A", "混合", {"手感体验": "正向", "性价比": "负向"}, "已确认")
        self.row["评论正文"] = "文字已更新"
        self.ai_path.write_text(json.dumps([self.row], ensure_ascii=False), encoding="utf-8")
        self.assertEqual(load_annotation_data(self.ai_path, self.db).iloc[0]["复核状态"], "文本已更新需复核")
        with self.assertRaises(ValueError):
            save_annotation(self.db, self.row, "A", "混合", {}, "已修改")


if __name__ == "__main__":
    unittest.main()
