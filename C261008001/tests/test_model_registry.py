import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from esports_analysis.model_registry import normalize_model_records


def product(pid, platform, sku, title="商品", brand="品牌", category="mouse"):
    return dict(product_id=pid, platform=platform, platform_product_id=sku, product_name=title,
                brand=brand, category=category, strict_review_count=0, product_url="https://example.com")


class ModelRegistryTests(unittest.TestCase):
    def test_candidate_pair_not_marked_verified(self):
        pairs = [dict(pair_key="model", model="M", jd_sku="1", tmall_item_id="2")]
        records = normalize_model_records(pd.DataFrame([product(1,"jd","1"),product(2,"tmall","2")]), pairs)
        self.assertEqual(records["型号组编号"].nunique(), 1)
        self.assertEqual(records.iloc[1]["归属状态"], "跨平台配置待核验")

    def test_similar_titles_and_unknown_versions_are_not_fuzzy_merged(self):
        records = normalize_model_records(pd.DataFrame([product(1,"jd","1","V3白色"),
            product(2,"jd","2","V3专业版"), product(3,"tmall","1","V3白色")]), [])
        self.assertEqual(records["型号组编号"].nunique(), 3)

    def test_known_conflicting_variants_split(self):
        for left,right in (("100066734479","827716604537"),("100036347936","673074579811")):
            records = normalize_model_records(pd.DataFrame([product(1,"jd",left),product(2,"tmall",right)]),
                [dict(pair_key="old_pair",model="same",jd_sku=left,tmall_item_id=right)])
            self.assertEqual(records["型号组编号"].nunique(), 2)
            self.assertTrue(records["版本风险"].ne("").all())

    def test_wired_and_wireless_versions_stay_separate(self):
        records = normalize_model_records(pd.DataFrame([product(1,"jd","100041149481"),
            product(2,"jd","100041149519")]),
            [dict(pair_key="logitech_g502x_wired",model="G502X有线",jd_sku="100041149481",tmall_item_id="2")])
        self.assertEqual(records["型号组编号"].nunique(),2)

    def test_multiple_color_skus_mapped_to_same_model_do_not_inflate_groups(self):
        pairs = [dict(pair_key="m",model="M",jd_sku="1",tmall_item_id="2"),
                 dict(pair_key="m",model="M",jd_sku="3",tmall_item_id="4")]
        records = normalize_model_records(pd.DataFrame([product(1,"jd","1","M白"),product(2,"jd","3","M黑")]),pairs)
        self.assertEqual(records["型号组编号"].nunique(),1)

    def test_conflicting_assignments_raise(self):
        pairs = [dict(pair_key="m",model="M",jd_sku="1",tmall_item_id="2"),
                 dict(pair_key="n",model="N",jd_sku="1",tmall_item_id="3")]
        with self.assertRaises(ValueError):
            normalize_model_records(pd.DataFrame(),pairs)
