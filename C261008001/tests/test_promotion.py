import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.importer import import_products_csv
from esports_analysis.pricing import load_price_history
from esports_analysis.promotion import (
    build_brand_promotion_summary,
    build_daily_promotion_summary,
    build_product_promotion_metrics,
    detect_raise_then_discount,
    filter_history_by_date,
    summarize_promotion_metrics,
)


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "promotion.db"
        import_products_csv(ROOT / "data" / "seed" / "products_sample.csv", self.database_path)
        self.history = load_price_history(self.database_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_date_filter_is_inclusive(self):
        filtered = filter_history_by_date(self.history, "2026-09-08", "2026-09-22")
        self.assertEqual(len(filtered), 10)

    def test_product_metrics_use_latest_and_historical_values(self):
        metrics = build_product_promotion_metrics(self.history)
        self.assertEqual(len(metrics), 5)
        logitech = metrics[metrics["product_name"].str.contains("罗技")].iloc[0]
        self.assertEqual(logitech["current_price"], 749)
        self.assertEqual(logitech["historical_minimum"], 749)
        self.assertTrue(logitech["is_historical_low"])

    def test_brand_and_daily_summaries_preserve_counts(self):
        metrics = build_product_promotion_metrics(self.history)
        brands = build_brand_promotion_summary(metrics)
        daily = build_daily_promotion_summary(self.history)
        self.assertEqual(int(brands["商品数"].sum()), 5)
        self.assertEqual(int(daily["价格记录"].sum()), 15)

    def test_summary_reports_historical_low_count(self):
        metrics = build_product_promotion_metrics(self.history)
        summary = summarize_promotion_metrics(metrics)
        self.assertEqual(summary["product_count"], 5)
        self.assertEqual(summary["historical_low_count"], 5)

    def test_detects_raise_then_discount_pattern(self):
        data = pd.DataFrame(
            {
                "product_id": [1, 1, 1],
                "product_name": ["测试商品"] * 3,
                "brand": ["测试品牌"] * 3,
                "effective_price": [100.0, 120.0, 105.0],
                "collected_at": pd.to_datetime(
                    ["2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z", "2026-09-03T00:00:00Z"],
                    utc=True,
                ),
                "collected_date": pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03"]),
            }
        )
        alerts = detect_raise_then_discount(data)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts.iloc[0]["上涨幅度"], 20.0)
        self.assertEqual(alerts.iloc[0]["回落幅度"], 12.5)


if __name__ == "__main__":
    unittest.main()

