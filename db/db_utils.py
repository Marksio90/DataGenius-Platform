# db/db_utils.py
"""
Lightweight DB utilities for TMIV – Advanced ML Platform.

Features
--------
- URL parsing for SQLite/DuckDB (no heavy deps).
- `get_connection()` that reads `config.settings.Settings.database_url` by default.
- `init_db()` to apply `db/schema.sql` and optional incremental migrations from `db/migrations/`.
- Safe helpers: `execute`, `executemany`, `fetch_one`, `fetch_all`.
- Context manager `db_transaction()` for explicit transactions.

Supported URLs
--------------
- None / ""                  -> SQLite file `./tmiv.db`
- sqlite:///:memory:         -> SQLite in-memory
- sqlite:///path/to.db       -> SQLite on disk
- duckdb:///path/to.duckdb   -> DuckDB on disk  (requires `duckdb` package)

Notes
-----
- Utilities keep dependencies minimal (stdlib + optional duckdb).
- For higher-level DB operations see `core/services/db_service.py`.
"""

from __future__ import annotations

import io
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

# ---- Settings (soft import) ----
try:  # pragma: no cover
    from config.settings import get_settings
except Exception:  # pragma: no cover
    def get_settings():
        class _S:
            database_url: str | None = None
        return _S()


# =========================
# URL parsing
# =========================

@dataclass(frozen=True)
class DBConfig:
    kind: str  # "sqlite" | "duckdb"
    path: str  # db path or ":memory:"


def parse_db_url(url: Optional[str]) -> DBConfig:
    if not url:
        return DBConfig("sqlite", str(Path("tmiv.db").resolve()))
    u = url.strip()
    if u.startswith("sqlite:///"):
        return DBConfig("sqlite", u.replace("sqlite:///", "", 1))
    if u == "sqlite:///:memory:":
        return DBConfig("sqlite", ":memory:")
    if u.startswith("duckdb:///"):
        return DBConfig("duckdb", u.replace("duckdb:///", "", 1))
    # fallback: treat as sqlite path
    return DBConfig("sqlite", u)


# =========================
# Connections
# =========================

def get_connection(database_url: Optional[str] = None):
    """
    Return a new DB connection based on URL. Caller is responsible for closing.
    """
    if database_url is None:
        try:
            database_url = get_settings().database_url
        except Exception:
            database_url = None
    cfg = parse_db_url(database_url)

    if cfg.kind == "duckdb":
        try:  # pragma: no cover
            import duckdb  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("DuckDB URL provided but 'duckdb' package is not installed.") from e
        # Ensure directory exists
        if cfg.path not in {":memory:"}:
            Path(cfg.path).parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(cfg.path)

    # sqlite
    if cfg.path not in {":memory:"}:
        Path(cfg.path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cfg.path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
    except Exception:
        pass
    return conn


@contextmanager
def db_transaction(database_url: Optional[str] = None):
    """
    Context manager for an explicit transaction:

        with db_transaction() as conn:
            conn.execute(...)
            ...

    Commits on success, rolls back on exception.
    """
    conn = get_connection(database_url)
    try:
        yield conn
        try:
            conn.commit()
        except Exception:
            pass
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


# =========================
# Initialization / schema / migrations
# =========================

def init_db(
    database_url: Optional[str] = None,
    *,
    schema_path: str | Path = "db/schema.sql",
    run_migrations: bool = True,
    migrations_dir: str | Path = "db/migrations",
) -> None:
    """
    Apply base schema and (optionally) run migrations from `db/migrations/*.sql`.

    - Creates DB parent directory when needed.
    - Migrations are applied once and tracked in table `_migrations(name, applied_at)`.
    """
    schema_path = Path(schema_path)
    migrations_dir = Path(migrations_dir)

    with db_transaction(database_url) as conn:
        # Base schema
        if schema_path.exists():
            _apply_sql_file(conn, schema_path)

        # Migrations
        if run_migrations and migrations_dir.exists():
            _ensure_migrations_table(conn)
            applied = _get_applied_migrations(conn)
            for f in sorted(migrations_dir.glob("*.sql")):
                name = f.name
                if name in applied:
                    continue
                _apply_sql_file(conn, f)
                _mark_migration_applied(conn, name)


def _apply_sql_file(conn, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    _exec_script(conn, sql)


def _exec_script(conn, sql: str) -> None:
    """
    Execute a potentially multi-statement SQL script (SQLite/DuckDB compatible).
    """
    # DuckDB has `execute` that accepts full script; SQLite `executescript`.
    driver = conn.__class__.__module__.split(".", 1)[0]
    if driver == "sqlite3":
        conn.executescript(sql)
    else:  # duckdb or others
        # naive splitter – DuckDB can execute whole script as well
        conn.execute(sql)


def _ensure_migrations_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _migrations(
            name TEXT PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _get_applied_migrations(conn) -> set[str]:
    try:
        cur = conn.execute("SELECT name FROM _migrations")
        return {r[0] for r in cur.fetchall()}
    except Exception:
        return set()


def _mark_migration_applied(conn, name: str) -> None:
    conn.execute("INSERT OR REPLACE INTO _migrations(name) VALUES (?)", (name,))


# =========================
# Simple helpers
# =========================

def execute(query: str, params: Sequence[Any] | None = None, *, database_url: Optional[str] = None) -> None:
    with db_transaction(database_url) as conn:
        conn.execute(query, params or [])


def executemany(query: str, rows: Iterable[Sequence[Any]], *, database_url: Optional[str] = None) -> None:
    with db_transaction(database_url) as conn:
        conn.executemany(query, list(rows))


def fetch_one(query: str, params: Sequence[Any] | None = None, *, database_url: Optional[str] = None) -> dict | None:
    with get_connection(database_url) as conn:
        cur = conn.execute(query, params or [])
        row = cur.fetchone()
        if row is None:
            return None
        if isinstance(row, sqlite3.Row):
            return {k: row[k] for k in row.keys()}
        return dict(row)


def fetch_all(query: str, params: Sequence[Any] | None = None, *, database_url: Optional[str] = None) -> list[dict]:
    with get_connection(database_url) as conn:
        cur = conn.execute(query, params or [])
        rows = cur.fetchall()
        out: list[dict] = []
        for r in rows:
            if isinstance(r, sqlite3.Row):
                out.append({k: r[k] for k in r.keys()})
            else:
                out.append(dict(r))
        return out


__all__ = [
    "DBConfig",
    "parse_db_url",
    "get_connection",
    "db_transaction",
    "init_db",
    "execute",
    "executemany",
    "fetch_one",
    "fetch_all",
]
