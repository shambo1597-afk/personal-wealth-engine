"""
Shared pytest fixtures for the Personal Wealth Engine test suite.

`isolated_db` gives each test its own throwaway SQLite database file so that
FIFO lot / trade-ledger tests never touch the real `wealth_ledger.db` used by
the running Streamlit app.
"""
import sqlite3
from pathlib import Path
from typing import Generator

import pytest

import database


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[database.__class__, None, None]:  # type: ignore[name-defined]
    """Points `database.DB_FILE` at a temp file, initializes the schema, and yields the module."""
    test_db_path = tmp_path / "test_wealth_ledger.db"
    monkeypatch.setattr(database, "DB_FILE", str(test_db_path))
    database.init_db()
    yield database


@pytest.fixture
def db_connection(isolated_db) -> Generator[sqlite3.Connection, None, None]:
    """A live connection bound to the isolated test database."""
    conn = isolated_db.get_db_connection()
    try:
        yield conn
    finally:
        conn.close()
