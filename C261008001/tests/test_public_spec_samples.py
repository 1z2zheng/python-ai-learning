import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_specs_import", ROOT / "scripts/import_public_spec_samples.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from esports_analysis.db import connect, initialize_database


class PublicSpecSamplesTests(unittest.TestCase):
    def test_import_is_idempotent_and_does_not_inflate_price_or_comment_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            initialize_database(db_path)
            source = ROOT / "data/raw/public_spec_samples_20261015.json"
            self.assertEqual(module.import_samples(db_path, source), 2)
            self.assertEqual(module.import_samples(db_path, source), 0)
            with connect(db_path) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM price_snapshots").fetchone()[0], 0)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0], 0)
                row = db.execute("SELECT attributes_json FROM public_spec_research_samples WHERE source_sku='10098016952114'").fetchone()
                self.assertIn("≥100g", row[0])
