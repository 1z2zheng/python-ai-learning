import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.importer import import_products_csv
from esports_analysis.overview import load_overview_data
from esports_analysis.preference import calculate_dimension_scores
from esports_analysis.pricing import (
    assign_price_bands,
    build_brand_history_summary,
    build_cross_platform_comparison,
    build_positioning_data,
    build_price_band_summary,
    build_product_history_summary,
    load_price_history,
    summarize_price_history,
)


class PricingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "pricing.db"
        import_products_csv(ROOT / "data" / "seed" / "products_sample.csv", self.database_path)
        self.latest = calculate_dimension_scores(load_overview_data(self.database_path))
        self.history = load_price_history(self.database_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_history_contains_three_snapshots_per_product(self):
        self.assertEqual(len(self.history), 15)
        counts = self.history.groupby("product_id").size()
        self.assertTrue((counts == 3).all())

    def test_assigns_all_products_to_price_bands(self):
        mouse = self.latest[self.latest["category"] == "mouse"]
        banded = assign_price_bands(mouse, "mouse")
        self.assertTrue(banded["price_band"].notna().all())

    def test_price_band_summary_preserves_product_total(self):
        keyboard = self.latest[self.latest["category"] == "keyboard"]
        summary = build_price_band_summary(keyboard, "keyboard")
        self.assertEqual(int(summary["商品数"].sum()), 3)
        self.assertAlmostEqual(float(summary["商品占比"].sum()), 100.0)

    def test_positioning_has_performance_and_group(self):
        keyboard = self.latest[self.latest["category"] == "keyboard"]
        positioning = build_positioning_data(keyboard)
        self.assertTrue(positioning["performance_score"].between(0, 100).all())
        self.assertTrue(positioning["positioning_group"].notna().all())

    def test_history_summary_reports_date_range(self):
        summary = summarize_price_history(self.history)
        self.assertEqual(summary["snapshot_count"], 15)
        self.assertEqual(summary["product_count"], 5)
        self.assertLess(summary["date_start"], summary["date_end"])

    def test_product_history_summary_has_price_research_metrics(self):
        summary = build_product_history_summary(self.history)
        self.assertEqual(len(summary), 5)
        self.assertTrue((summary["snapshot_count"] == 3).all())
        self.assertTrue((summary["lowest_price"] <= summary["latest_price"]).all())
        self.assertTrue((summary["latest_price"] <= summary["highest_price"]).all())
        self.assertTrue((summary["volatility_rate"] >= 0).all())
        self.assertTrue((summary["potential_saving"] >= 0).all())

    def test_brand_history_summary_preserves_products_and_snapshots(self):
        products = build_product_history_summary(self.history)
        brands = build_brand_history_summary(products)
        self.assertEqual(int(brands["product_count"].sum()), 5)
        self.assertEqual(int(brands["snapshot_count"].sum()), 15)
        self.assertTrue((brands["average_volatility_rate"] >= 0).all())

    def test_cross_platform_comparison_matches_platform_product_ids(self):
        history = pd.DataFrame([
            {"platform_product_id": "jd-1", "platform": "jd", "effective_price": 320,
             "collected_at": pd.Timestamp("2026-01-01", tz="UTC"),
             "collected_date": pd.Timestamp("2026-01-01")},
            {"platform_product_id": "jd-1", "platform": "jd", "effective_price": 300,
             "collected_at": pd.Timestamp("2026-01-02", tz="UTC"),
             "collected_date": pd.Timestamp("2026-01-02")},
            {"platform_product_id": "tm-1", "platform": "tmall", "effective_price": 280,
             "collected_at": pd.Timestamp("2026-01-02", tz="UTC"),
             "collected_date": pd.Timestamp("2026-01-02")},
        ])
        pairs = pd.DataFrame([{
            "pair_key": "sample", "category": "mouse", "brand": "示例",
            "model": "M1", "jd_sku": "jd-1", "tmall_item_id": "tm-1",
        }])
        result = build_cross_platform_comparison(history, pairs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["cheaper_platform"], "待核验")
        self.assertFalse(result.iloc[0]["comparison_eligible"])
        self.assertEqual(result.iloc[0]["latest_gap"], -20)
        self.assertEqual(result.iloc[0]["jd_snapshot_count"], 2)


if __name__ == "__main__":
    unittest.main()

