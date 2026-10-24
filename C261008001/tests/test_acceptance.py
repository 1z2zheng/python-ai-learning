import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.acceptance import generate_acceptance_markdown, run_acceptance_checks
from esports_analysis.importer import import_products_csv


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "acceptance.db"
        import_products_csv(ROOT / "data" / "seed" / "products_sample.csv", self.database_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_acceptance_runs_all_check_categories(self):
        checks, timings = run_acceptance_checks(self.database_path)
        categories = {item.category for item in checks}
        self.assertIn("功能", categories)
        self.assertIn("性能", categories)
        self.assertIn("数据质量", categories)
        self.assertIn("论文研究数据", categories)
        self.assertIn("偏好排名", timings)

    def test_sample_database_fails_scale_but_passes_core_functions(self):
        checks, _ = run_acceptance_checks(self.database_path)
        by_id = {item.check_id: item for item in checks}
        self.assertFalse(by_id["DATA-001"].passed)
        self.assertTrue(by_id["FUNC-001"].passed)
        self.assertTrue(by_id["FUNC-005"].passed)

    def test_markdown_separates_system_and_research_readiness(self):
        checks, timings = run_acceptance_checks(self.database_path)
        report = generate_acceptance_markdown(checks, timings)
        self.assertIn("系统功能和性能验收", report)
        self.assertIn("论文研究数据就绪度", report)
        self.assertIn("演示数据", report)


if __name__ == "__main__":
    unittest.main()

