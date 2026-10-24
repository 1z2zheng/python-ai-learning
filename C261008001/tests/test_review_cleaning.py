import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.review_cleaning import (
    classify_dimension_sentiment,
    extract_dimensions,
    normalize_review_text,
    quality_reasons,
    review_scope,
)


class ReviewCleaningTests(unittest.TestCase):
    def test_normalize_review_text(self):
        self.assertEqual(normalize_review_text("  手感\n 很好  "), "手感 很好")

    def test_review_scope(self):
        self.assertEqual(review_scope("1001", "1001"), "目标SKU")
        self.assertEqual(review_scope("1001", "1002"), "其他SKU待核验")
        self.assertEqual(review_scope("1001", None), "历史样本未记录来源SKU")

    def test_extract_six_dimensions(self):
        dimensions = extract_dimensions("鼠标握持舒服，响应快，价格也划算")
        self.assertIn("舒适度", dimensions)
        self.assertIn("响应性能", dimensions)
        self.assertIn("性价比", dimensions)

    def test_placeholder_is_excluded(self):
        reasons = quality_reasons("此用户未及时填写评价内容")
        self.assertIn("平台占位评价", reasons)

    def test_negative_dimension_sentiment(self):
        label, score = classify_dimension_sentiment("无线连接经常断连，延迟明显", "响应性能", 2)
        self.assertEqual((label, score), ("负向", -1))

    def test_negation_objective_and_mixed_text(self):
        cases = [("不舒服", "舒适度", "负向"),
                 ("没有延迟", "响应性能", "正向"),
                 ("没有异响", "耐用性", "正向"),
                 ("重量60克", "舒适度", "中性"),
                 ("手感舒服，但是按键松动", "手感体验", "混合")]
        for text, dimension, expected in cases:
            for rating in (1, 3, 5):
                self.assertEqual(classify_dimension_sentiment(text, dimension, rating)[0], expected)
        self.assertEqual(extract_dimensions("客服回复快，值得信赖"), [])


if __name__ == "__main__":
    unittest.main()
