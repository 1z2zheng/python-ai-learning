import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from esports_analysis.pair_verification import load_pair_verifications, save_pair_verification
from esports_analysis.pricing import build_cross_platform_comparison
from esports_analysis.db import connect


class PairVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "pairs.db"
        self.pair = dict(pair_key="test", category="mouse", brand="test", model="M",
                         jd_sku="1", tmall_item_id="2", jd_latest_price=100.0,
                         tmall_latest_price=110.0, comparison_date="2026-10-01")
        self.record = dict(status="verified", verified_by="reviewer", note="核对原文",
                           jd_variant="白色无线单品", tmall_variant="白色无线单品",
                           configuration_evidence="商品页所选规格一致", shop_evidence="两侧店铺名称",
                           offer_evidence="对应历史日期无会员门槛报价，含运费")
        self.history = pd.DataFrame([
            dict(platform_product_id="1", platform="jd", effective_price=100.0,
                 collected_at=pd.Timestamp("2026-10-01", tz="UTC"), collected_date=pd.Timestamp("2026-10-01")),
            dict(platform_product_id="2", platform="tmall", effective_price=110.0,
                 collected_at=pd.Timestamp("2026-10-01", tz="UTC"), collected_date=pd.Timestamp("2026-10-01")),
        ])

    def tearDown(self):
        self.temp.cleanup()

    def comparison(self, history=None):
        return build_cross_platform_comparison(self.history if history is None else history,
                    pd.DataFrame([self.pair]), load_pair_verifications(self.db)).iloc[0]

    def test_verified_quotes_become_ineligible_when_price_or_sku_changes(self):
        save_pair_verification(self.db, self.pair, self.record)
        self.assertTrue(self.comparison()["comparison_eligible"])
        changed = self.history.copy()
        changed.loc[1, "effective_price"] = 105.0
        self.assertFalse(self.comparison(changed)["comparison_eligible"])
        self.pair["jd_sku"] = "3"
        changed.loc[0, "platform_product_id"] = "3"
        self.assertFalse(self.comparison(changed)["comparison_eligible"])

    def test_different_configuration_appends_history_and_blocks_comparison(self):
        save_pair_verification(self.db, self.pair, self.record)
        save_pair_verification(self.db, self.pair, dict(status="different", verified_by="reviewer", note="轴体不同"))
        self.assertFalse(self.comparison()["comparison_eligible"])
        self.assertEqual(self.comparison()["verification_status"], "different")
        with connect(self.db) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM cross_platform_verification_events").fetchone()[0], 2)

    def test_missing_evidence_and_different_dates_rejected(self):
        with self.assertRaises(ValueError):
            save_pair_verification(self.db, self.pair, dict(self.record, shop_evidence=""))
        with self.assertRaises(ValueError):
            save_pair_verification(self.db, dict(self.pair, comparison_date=None), self.record)
        self.assertEqual(load_pair_verifications(self.db), {})

    def test_date_change_and_nonpositive_quote_block_comparison(self):
        save_pair_verification(self.db, self.pair, self.record)
        changed = self.history.copy()
        changed["collected_date"] = pd.Timestamp("2026-10-02")
        self.assertFalse(self.comparison(changed)["comparison_eligible"])
        changed.loc[0, "effective_price"] = 0
        self.assertFalse(self.comparison(changed)["comparison_eligible"])
