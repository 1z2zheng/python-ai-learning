import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from esports_analysis.db import initialize_database,connect
from esports_analysis.review_identity import load_resolved_reviews


class ReviewIdentityTests(unittest.TestCase):
    def test_only_verified_source_sku_resolves_and_original_stays_intact(self):
        with tempfile.TemporaryDirectory() as temp:
            database=Path(temp)/'test.db'
            initialize_database(database)
            with connect(database) as db:
                for sku in ('1001','1002'):
                    db.execute('''INSERT INTO products (platform,platform_product_id,category,brand,
                        product_name,first_seen_at,last_seen_at,created_at,updated_at)
                        VALUES ('jd',?,'mouse','测试',?,'now','now','now','now')''',(sku,sku))
                db.execute("INSERT INTO reviews (product_id,source_review_id,content,source_product_id,collected_at) VALUES (1,'hash','手感不错','1002','now')")
                unresolved=load_resolved_reviews(db)
                self.assertEqual(unresolved.iloc[0]['product_id'],1)
                db.execute("INSERT INTO product_identity_verifications VALUES (2,'1002','model2','title','url','[]','now','verified')")
                resolved=load_resolved_reviews(db)
                self.assertEqual(resolved.iloc[0]['product_id'],2)
                self.assertEqual(resolved.iloc[0]['requested_product_id'],1)
                self.assertEqual(resolved.iloc[0]['platform_product_id'],'1002')
                self.assertEqual(db.execute('SELECT product_id FROM reviews').fetchone()[0],1)
                db.execute("UPDATE product_identity_verifications SET status='pending'")
                self.assertEqual(load_resolved_reviews(db).iloc[0]['product_id'],1)


if __name__=='__main__':
    unittest.main()
