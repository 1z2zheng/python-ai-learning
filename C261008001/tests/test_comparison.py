import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.comparison import (
    build_comparison_summaries,
    build_parameter_comparison,
    merge_comparison_metrics,
    validate_product_selection,
)
from esports_analysis.importer import import_products_csv
from esports_analysis.overview import load_overview_data
from esports_analysis.preference import calculate_dimension_scores
from esports_analysis.pricing import load_price_history


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "comparison.db"
        import_products_csv(ROOT / "data" / "seed" / "products_sample.csv", self.database_path)
        latest = calculate_dimension_scores(load_overview_data(self.database_path))
        self.data = merge_comparison_metrics(latest, load_price_history(self.database_path))
        self.mouse_ids = self.data[self.data["category"] == "mouse"]["product_id"].tolist()
        self.keyboard_ids = self.data[self.data["category"] == "keyboard"]["product_id"].tolist()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_requires_two_to_five_products(self):
        with self.assertRaisesRegex(ValueError, "2至5"):
            validate_product_selection(self.data, self.mouse_ids[:1])

    def test_rejects_mixed_categories(self):
        with self.assertRaisesRegex(ValueError, "同一类别"):
            validate_product_selection(self.data, [self.mouse_ids[0], self.keyboard_ids[0]])

    def test_parameter_table_has_one_column_per_product(self):
        table = build_parameter_comparison(self.data, self.keyboard_ids)
        self.assertEqual(len(table.columns), 1 + len(self.keyboard_ids))
        self.assertIn("轴体类型", table["对比项"].tolist())
        self.assertNotIn("最大 DPI", table["对比项"].tolist())

    def test_mouse_table_uses_mouse_parameters(self):
        table = build_parameter_comparison(self.data, self.mouse_ids)
        self.assertIn("最大 DPI", table["对比项"].tolist())
        self.assertNotIn("轴体类型", table["对比项"].tolist())

    def test_summaries_include_two_strengths_and_weaknesses(self):
        summaries = build_comparison_summaries(self.data, self.keyboard_ids)
        self.assertEqual(len(summaries), 3)
        for summary in summaries.values():
            self.assertEqual(len(summary["strengths"]), 2)
            self.assertEqual(len(summary["weaknesses"]), 2)
            self.assertIsNotNone(summary["difference_from_historical_minimum"])

    def test_same_title_different_skus_keep_separate_columns(self):
        data=self.data.copy()
        mask=data['product_id'].isin(self.mouse_ids)
        data.loc[mask,'brand']='同品牌'
        data.loc[mask,'product_name']='同名商品不同版本'
        table=build_parameter_comparison(data,self.mouse_ids)
        self.assertEqual(len(table.columns),len(self.mouse_ids)+1)
        self.assertTrue(all('SKU ' in c for c in table.columns[1:]))


if __name__ == "__main__":
    unittest.main()

