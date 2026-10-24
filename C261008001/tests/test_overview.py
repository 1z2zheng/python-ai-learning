import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.importer import import_products_csv
from esports_analysis.overview import (
    build_quality_report,
    calculate_summary,
    filter_overview_data,
    load_overview_data,
)


class OverviewTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "overview.db"
        self.sample_path = ROOT / "data" / "seed" / "products_sample.csv"
        import_products_csv(self.sample_path, self.database_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loads_latest_product_rows(self):
        data = load_overview_data(self.database_path)
        self.assertEqual(len(data), 5)
        self.assertEqual(data["product_id"].nunique(), 5)
        self.assertTrue(data["current_price"].notna().all())

    def test_filters_by_category_brand_and_price(self):
        data = load_overview_data(self.database_path)
        filtered = filter_overview_data(
            data,
            category="mouse",
            platforms=["manual"],
            brands=["罗技"],
            price_range=(700, 800),
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["brand"], "罗技")
        self.assertEqual(filtered.iloc[0]["current_price"], 749)

    def test_summary_uses_filtered_data(self):
        data = load_overview_data(self.database_path)
        filtered = filter_overview_data(data, category="keyboard", platforms=["manual"])
        summary = calculate_summary(filtered)
        self.assertEqual(summary["product_count"], 3)
        self.assertEqual(summary["brand_count"], 3)
        self.assertAlmostEqual(summary["average_price"], (1199 + 329 + 469) / 3)

    def test_quality_report_uses_category_specific_fields(self):
        data = load_overview_data(self.database_path)
        mouse_data = filter_overview_data(data, category="mouse", platforms=["manual"])
        report = build_quality_report(mouse_data, "mouse")
        self.assertIn("最大 DPI", report["字段"].tolist())
        self.assertNotIn("轴体类型", report["字段"].tolist())


if __name__ == "__main__":
    unittest.main()

