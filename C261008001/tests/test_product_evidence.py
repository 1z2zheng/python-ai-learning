import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from esports_analysis.db import initialize_database, connect
from esports_analysis.product_evidence import load_evidence_data, product_evidence


class ProductEvidenceTests(unittest.TestCase):
    def test_price_evidence_and_strict_reviews_use_matching_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            database=Path(tmp)/'test.db'
            initialize_database(database)
            with connect(database) as db:
                db.execute("""INSERT INTO products (platform,platform_product_id,category,brand,product_name,
                    product_url,first_seen_at,last_seen_at,created_at,updated_at)
                    VALUES ('jd','1001','mouse','测试','商品A','https://item.jd.com/1001.html','now','now','now','now')""")
                db.execute("""INSERT INTO price_snapshots (product_id,sale_price,collected_at,collected_date,source_batch_id,price_source)
                    VALUES (1,299,'2026-10-14T12:00:00+08:00','2026-10-14','batch','manmanbuy')""")
                for rid, source, text in [('a','1001','手感舒服'),('b','1002','价格太贵'),('c',None,'响应很快')]:
                    db.execute("INSERT INTO reviews (product_id,source_review_id,content,source_product_id,collected_at) VALUES (1,?,?,?,'now')",(rid,text,source))
            bundle=load_evidence_data(database)
            self.assertEqual(bundle['products'].iloc[0]['strict_review_count'],1)
            evidence=product_evidence(bundle,1,database)
            self.assertEqual(len(evidence['reviews']),1)
            self.assertEqual(evidence['reviews'].iloc[0]['content'],'手感舒服')
            self.assertEqual(evidence['price_summary'].iloc[0]['lowest_price'],299)
            self.assertEqual(evidence['dimensions']['提及数'].sum(),2)
            with self.assertRaises(ValueError):
                product_evidence(bundle,999,database)

    def test_empty_database_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            database=Path(tmp)/'test.db'
            bundle=load_evidence_data(database)
            self.assertTrue(bundle['products'].empty)


if __name__=='__main__':
    unittest.main()
