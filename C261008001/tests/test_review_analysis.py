import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.review_analysis import (
    build_product_feedback_summary,
    classify_sentiment,
    extract_aspect_mentions,
    extract_opinion_snippets,
)


class ReviewAnalysisTests(unittest.TestCase):
    def test_text_sentiment_is_independent_of_rating(self):
        self.assertEqual(classify_sentiment("价格太贵", 5), "负向")
        self.assertEqual(classify_sentiment("外观还可以", 1), "中性")

    def test_extracts_aspects_and_negative_snippet(self):
        text = "连接稳定，手感不错，就是价格有点小贵"
        aspects = extract_aspect_mentions(text)
        self.assertIn("连接稳定", aspects)
        self.assertIn("手感", aspects)
        self.assertIn("价格价值", aspects)
        self.assertIn("就是价格有点小贵", extract_opinion_snippets(text, "负向"))
        self.assertEqual(extract_opinion_snippets("连接没有延迟，用久了也不累手", "负向"), [])
        self.assertEqual(extract_opinion_snippets("延迟几乎无感", "负向"), [])

    def test_feedback_summary_keeps_heat_proxy_and_tags(self):
        reviews = pd.DataFrame([{
            "product_id": 1, "category": "mouse", "brand": "示例",
            "product_name": "M1", "rating": 5.0, "content": "轻便且连接稳定",
            "sentiment": "正向", "aspects": ["舒适重量", "连接稳定"],
            "pros": ["轻便且连接稳定"], "cons": [],
        }])
        heat = pd.DataFrame([{
            "product_id": 1, "category": "mouse", "brand": "示例",
            "product_name": "M1", "review_count": 1000, "raw_metric_text": "1000+",
            "source_payload_json": '{"tags":[{"name":"轻便","countText":"20人说"}]}',
        }])
        result = build_product_feedback_summary(reviews, heat)
        self.assertEqual(result.iloc[0]["review_count_lower_bound"], 1000)
        self.assertEqual(result.iloc[0]["positive_rate"], 100.0)
        self.assertIn("轻便（20人说）", result.iloc[0]["platform_tags"])


if __name__ == "__main__":
    unittest.main()
