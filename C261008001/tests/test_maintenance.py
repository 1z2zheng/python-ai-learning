import csv
import sys
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.importer import import_products_csv, validate_products_csv
from esports_analysis.maintenance import build_database_quality_summary, load_recent_collection_jobs, build_research_collection_tasks


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "maintenance.db"
        self.sample_path = ROOT / "data" / "seed" / "products_sample.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sample_file_validates_without_writing_database(self):
        result = validate_products_csv(self.sample_path)
        self.assertEqual(result.row_count, 15)
        self.assertEqual(result.valid_count, 15)
        self.assertEqual(result.invalid_count, 0)
        self.assertFalse(self.database_path.exists())

    def test_invalid_file_reports_row_error(self):
        invalid_path = Path(self.temp_dir.name) / "invalid.csv"
        with self.sample_path.open("r", encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
            fieldnames = list(rows[0])
        rows[0]["sale_price"] = "bad"
        with invalid_path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(rows[0])
        result = validate_products_csv(invalid_path)
        self.assertEqual(result.invalid_count, 1)
        self.assertIn("sale_price", result.errors[0])

    def test_jobs_and_quality_are_available_after_import(self):
        import_products_csv(self.sample_path, self.database_path)
        jobs = load_recent_collection_jobs(self.database_path)
        quality = build_database_quality_summary(self.database_path)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs.iloc[0]["状态"], "success")
        self.assertIn("鼠标", quality["类别"].tolist())
        self.assertIn("键盘", quality["类别"].tolist())

    def test_empty_research_queue_has_stable_columns(self):
        tasks = build_research_collection_tasks(self.database_path)
        self.assertTrue(tasks.empty)
        self.assertIn("SKU", tasks.columns)

    def test_queue_counts_dates_excludes_demo_and_respects_review_target(self):
        product = dict(product_id=1, platform="jd", platform_product_id="123", category="mouse",
                       brand="品牌", product_name="鼠标", shop_name="官方", product_url="https://item.jd.com/123.html",
                       strict_review_count=20, identity_verified=True, connection_type="wired", weight_g=0,
                       polling_rate_hz=1000, dpi_max=26000, sensor_model=" ")
        demo = dict(product, product_id=2, product_name="演示鼠标", strict_review_count=0)
        history = pd.DataFrame({"product_id": [1, 1, 1, 1],
                                "collected_at": ["2026-10-01", "2026-10-01", "2026-10-02", "bad"]})
        with patch("esports_analysis.maintenance.load_evidence_data",
                   return_value={"products": pd.DataFrame([product, demo]), "history": history}):
            tasks = build_research_collection_tasks(self.database_path)
            self.assertEqual(set(tasks["商品ID"]), {1})
            self.assertEqual(set(tasks["任务类型"]), {"商品规格", "历史报价"})
            self.assertEqual(tasks["报价日期数"].tolist(), [2, 2])
            specs = tasks[tasks["任务类型"].eq("商品规格")].iloc[0]["缺口"]
            self.assertIn("传感器", specs)
            self.assertNotIn("重量", specs)
            expanded = build_research_collection_tasks(self.database_path, 21)
            self.assertIn("严格评论", expanded["任务类型"].tolist())

    def test_invalid_target_rejected(self):
        for target in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                build_research_collection_tasks(self.database_path, target)


if __name__ == "__main__":
    unittest.main()

