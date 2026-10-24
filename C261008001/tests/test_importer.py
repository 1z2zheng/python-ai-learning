import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.db import connect, initialize_database
from esports_analysis.importer import import_products_csv, normalize_row, RowValidationError


class ImporterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "test.db"
        initialize_database(self.database_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_csv(self, rows):
        path = Path(self.temp_dir.name) / "products.csv"
        fieldnames = [
            "platform", "platform_product_id", "category", "brand",
            "product_name", "sale_price", "collected_at", "connection_type",
            "weight_g", "dpi_max", "switch_type",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def valid_row(self):
        return {
            "platform": "京东",
            "platform_product_id": "1001",
            "category": "鼠标",
            "brand": "Logitech G",
            "product_name": "测试鼠标",
            "sale_price": "399",
            "collected_at": "2026-10-08T10:00:00+08:00",
            "connection_type": "2.4G",
            "weight_g": "60",
            "dpi_max": "26000",
            "switch_type": "",
        }

    def test_normalize_aliases(self):
        row = normalize_row(self.valid_row())
        self.assertEqual(row["platform"], "jd")
        self.assertEqual(row["category"], "mouse")
        self.assertEqual(row["brand"], "罗技")
        self.assertEqual(row["connection_type"], "wireless_2_4g")

    def test_import_and_repeat_updates_without_duplicates(self):
        csv_path = self.write_csv([self.valid_row()])
        first = import_products_csv(csv_path, self.database_path)
        second = import_products_csv(csv_path, self.database_path)

        self.assertEqual(first.inserted_count, 1)
        self.assertEqual(second.updated_count, 1)
        with connect(self.database_path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM products").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM price_snapshots").fetchone()[0], 1)

    def test_price_only_refresh_preserves_existing_specs(self):
        original = self.valid_row()
        import_products_csv(self.write_csv([original]), self.database_path)
        refresh = self.valid_row()
        refresh["collected_at"] = "2026-10-09T10:00:00+08:00"
        refresh["sale_price"] = "379"
        refresh["connection_type"] = ""
        refresh["weight_g"] = ""
        refresh["dpi_max"] = ""
        import_products_csv(self.write_csv([refresh]), self.database_path)

        with connect(self.database_path) as connection:
            specs = connection.execute(
                "SELECT connection_type, weight_g, dpi_max FROM product_specs"
            ).fetchone()
            self.assertEqual(specs["connection_type"], "wireless_2_4g")
            self.assertEqual(specs["weight_g"], 60)
            self.assertEqual(specs["dpi_max"], 26000)

    def test_invalid_row_is_reported_and_not_inserted(self):
        row = self.valid_row()
        row["sale_price"] = "not-a-price"
        result = import_products_csv(self.write_csv([row]), self.database_path)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failed_count, 1)
        with connect(self.database_path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM products").fetchone()[0], 0)

    def test_catalog_row_without_price_imports_without_snapshot(self):
        row = self.valid_row()
        row["platform"] = "vgn_official"
        row["sale_price"] = ""
        result = import_products_csv(self.write_csv([row]), self.database_path)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.inserted_count, 1)
        with connect(self.database_path) as connection:
            product = connection.execute(
                "SELECT platform, source_site, platform_product_id FROM products"
            ).fetchone()
            self.assertEqual(product["platform"], "third_party")
            self.assertEqual(product["source_site"], "vgn_official")
            self.assertEqual(product["platform_product_id"], "vgn_official:1001")
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM price_snapshots").fetchone()[0], 0)

    def test_keyboard_cannot_have_dpi(self):
        row = self.valid_row()
        row["category"] = "keyboard"
        with self.assertRaises(RowValidationError):
            normalize_row(row)


if __name__ == "__main__":
    unittest.main()

