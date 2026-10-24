import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "historical-price-spider"))

from main import parse_official_history


class ManmanbuyParserTests(unittest.TestCase):
    def test_keeps_promotion_for_lowest_daily_price(self):
        raw = (
            '[1760000000000,399,"到手价399"],'
            '[1760003600000,329,"满300减30"],'
            '[1760086400000,349,""]'
        )
        frame = parse_official_history(raw)

        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[0]["price_价格"], 329)
        self.assertEqual(frame.iloc[0]["promotion_text"], "满300减30")
        self.assertEqual(frame.iloc[1]["promotion_text"], "")


if __name__ == "__main__":
    unittest.main()
