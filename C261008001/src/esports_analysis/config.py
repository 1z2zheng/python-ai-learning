from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_database_path() -> Path:
    """Return the configured database path and ensure its parent exists."""
    override = os.getenv("ESPORTS_DB_PATH")
    path = Path(override) if override else PROJECT_ROOT / "data" / "esports_analysis.db"
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_schema_path() -> Path:
    return PROJECT_ROOT / "src" / "esports_analysis" / "schema.sql"

