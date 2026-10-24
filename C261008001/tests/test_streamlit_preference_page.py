import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class StreamlitPreferencePageTests(unittest.TestCase):
    def test_review_preference_section_loads_for_both_categories(self):
        app = AppTest.from_file(
            str(ROOT / "pages" / "2_偏好分析.py"), default_timeout=40
        ).run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("六维评论偏好", [item.value for item in app.subheader])

        app.radio[0].set_value("keyboard").run()
        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.dataframe), 4)


if __name__ == "__main__":
    unittest.main()
