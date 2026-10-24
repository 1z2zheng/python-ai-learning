import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.importer import import_products_csv
from esports_analysis.overview import load_overview_data
from esports_analysis.preference import (
    DEFAULT_WEIGHTS,
    DIMENSIONS,
    calculate_dimension_scores,
    normalize_weights,
    rank_products,
)


class PreferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "preference.db"
        import_products_csv(ROOT / "data" / "seed" / "products_sample.csv", self.database_path)
        self.scored = calculate_dimension_scores(load_overview_data(self.database_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scores_cover_six_dimensions_and_stay_in_range(self):
        self.assertEqual(len(self.scored), 5)
        for dimension in DIMENSIONS:
            self.assertIn(dimension, self.scored.columns)
            self.assertTrue(self.scored[dimension].between(0, 100).all())

    def test_weights_are_normalized(self):
        normalized = normalize_weights(DEFAULT_WEIGHTS)
        self.assertAlmostEqual(sum(normalized.values()), 1.0)

    def test_all_zero_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "至少设置一个"):
            normalize_weights({dimension: 0 for dimension in DIMENSIONS})

    def test_rank_is_sorted_and_has_contributions(self):
        mouse_data = self.scored[self.scored["category"] == "mouse"]
        ranked = rank_products(mouse_data, DEFAULT_WEIGHTS)
        self.assertEqual(ranked["rank"].tolist(), [1, 2])
        self.assertGreaterEqual(ranked.iloc[0]["match_score"], ranked.iloc[1]["match_score"])
        contribution_columns = [f"contribution_{dimension}" for dimension in DIMENSIONS]
        self.assertTrue(set(contribution_columns).issubset(ranked.columns))
        self.assertAlmostEqual(
            ranked.iloc[0][contribution_columns].sum(),
            ranked.iloc[0]["match_score"],
            places=1,
        )

    def test_keyboard_and_mouse_use_different_rules(self):
        mouse = self.scored[self.scored["category"] == "mouse"].iloc[0]
        keyboard = self.scored[self.scored["category"] == "keyboard"].iloc[0]
        self.assertNotEqual(mouse["score_evidence"], keyboard["score_evidence"])

    def test_price_dimension_does_not_repeat_performance(self):
        source = load_overview_data(self.database_path)
        before = calculate_dimension_scores(source)
        source["dpi_max"] = 35000
        source["polling_rate_hz"] = 8000
        after = calculate_dimension_scores(source)
        self.assertTrue(before["value_score"].equals(after["value_score"]))

    def test_missing_scenario_does_not_leak_to_next_call(self):
        source = load_overview_data(self.database_path)
        baseline = calculate_dimension_scores(source)
        calculate_dimension_scores(source, missing_score=25)
        repeated = calculate_dimension_scores(source)
        self.assertTrue(baseline[list(DIMENSIONS)].equals(repeated[list(DIMENSIONS)]))


if __name__ == "__main__":
    unittest.main()

