"""Read-only access to the local Chinook sample database."""

import asyncio
import os
import sqlite3
from contextlib import closing
from pathlib import Path


CHINOOK_DB_PATH = Path(os.getenv("CHINOOK_DB_PATH", "chinook.db")).expanduser()
if not CHINOOK_DB_PATH.is_absolute():
    CHINOOK_DB_PATH = Path(__file__).resolve().parent / CHINOOK_DB_PATH


def _query(sql: str, parameters: dict) -> list[dict]:
    if not CHINOOK_DB_PATH.is_file():
        raise FileNotFoundError(
            f"Chinook database not found: {CHINOOK_DB_PATH}. "
            "Set CHINOOK_DB_PATH to an existing Chinook SQLite database."
        )
    uri = CHINOOK_DB_PATH.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(sql, parameters).fetchall()]


async def query(sql: str, parameters: dict | None = None) -> list[dict]:
    """Run a query off the event loop and return rows as dictionaries."""
    return await asyncio.to_thread(_query, sql, parameters or {})
