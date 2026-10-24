from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import get_schema_path


@contextmanager
def connect(database_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open a transaction and always close the SQLite handle on exit."""
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(database_path: str | Path) -> None:
    schema = get_schema_path().read_text(encoding="utf-8")
    with connect(database_path) as connection:
        connection.executescript(schema)
        product_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(products)").fetchall()
        }
        if "source_site" not in product_columns:
            connection.execute("ALTER TABLE products ADD COLUMN source_site TEXT")
        price_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(price_snapshots)").fetchall()
        }
        if "price_source" not in price_columns:
            connection.execute(
                "ALTER TABLE price_snapshots ADD COLUMN price_source TEXT NOT NULL DEFAULT 'unspecified'"
            )
        sales_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(sales_snapshots)").fetchall()
        }
        for name in ("raw_metric_text", "source_url", "source_payload_json"):
            if name not in sales_columns:
                connection.execute(f"ALTER TABLE sales_snapshots ADD COLUMN {name} TEXT")
        review_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(reviews)").fetchall()
        }
        for name in ("variant_text", "source_url", "source_product_id", "sample_group"):
            if name not in review_columns:
                connection.execute(f"ALTER TABLE reviews ADD COLUMN {name} TEXT")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_reviews_rating_group ON reviews (rating, sample_group)"
        )


def get_database_summary(database_path: str | Path) -> dict[str, int]:
    initialize_database(database_path)
    queries = {
        "product_count": "SELECT COUNT(*) FROM products",
        "brand_count": "SELECT COUNT(DISTINCT brand) FROM products",
        "price_snapshot_count": "SELECT COUNT(*) FROM price_snapshots",
        "spec_count": "SELECT COUNT(*) FROM product_specs",
    }
    with connect(database_path) as connection:
        return {
            key: int(connection.execute(sql).fetchone()[0])
            for key, sql in queries.items()
        }

