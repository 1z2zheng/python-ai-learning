import unittest
from pathlib import Path
import os
import sys
import tempfile
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from esports_analysis.db import initialize_database
from esports_analysis.importer import import_products_csv


class StreamlitPreferencePageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        database = Path(self.temp.name) / "demo.db"
        initialize_database(database)
        import_products_csv(ROOT / "data/seed/demo_80_products.csv", database)
        environment = patch.dict(os.environ, {"ESPORTS_DB_PATH": str(database)})
        environment.start()
        self.addCleanup(environment.stop)

    def test_review_preference_section_loads_for_both_categories(self):
        app = AppTest.from_file(
            str(ROOT / "pages" / "2_偏好分析.py"), default_timeout=40
        ).run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("六维评论偏好", [item.value for item in app.subheader])

        app.radio[0].set_value("keyboard").run()
        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.dataframe), 1)
        ranking = app.dataframe[0].value
        self.assertGreater(len(ranking), 0)
        self.assertTrue(ranking["商品名称"].str.contains("演示键盘").all())
        self.assertTrue(ranking["综合匹配度"].between(0, 100).all())
        self.assertTrue(any("六维评论偏好数据尚未生成" in item.value for item in app.info))


if __name__ == "__main__":
    unittest.main()
