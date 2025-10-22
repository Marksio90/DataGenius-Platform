# core/services/db_service.py
"""
DBService – lekka, opcjonalna warstwa bazy danych dla TMIV – Advanced ML Platform.

Założenia
---------
- Zero twardych zależności: domyślnie SQLite (stdlib `sqlite3`); opcjonalnie DuckDB.
- Samodzielna inicjalizacja schematu (CREATE TABLE IF NOT EXISTS).
- Proste API pod zapisy runów, metryk, FI i leaderboardu.

Obsługiwane URL-e (config.settings.Settings.database_url)
--------------------------------------------------------
- brak / None               -> SQLite w pliku ./tmiv.db
- sqlite:///:memory:        -> SQLite in-memory
- sqlite:///path/to.db      -> SQLite na dysku
- duckdb:///path/to.duckdb  -> DuckDB na dysku (wymaga `duckdb`)

Tabela (uproszczona)
--------------------
datasets(name, fingerprint, rows, cols, created_at)
runs(run_id, dataset_name, fingerprint, problem_type, created_at)
model_metrics(run_id, model_name, metric_name, metric_value)
feature_importance(run_id, feature, importance, rank)
leaderboard(run_id, model_name, rank, primary_metric, payload_json)

Zobacz też: interfaces.IDBService
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

try:  # only for typing / convenience in save_feature_importance / save_leaderboard
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# --------------------------
# Helpers: DB URL parsing
# --------------------------

@dataclass(frozen=True)
class _DBConfig:
    kind: str  # "sqlite" | "duckdb"
    path: str  # db path or ":memory:"


def _parse_db_url(url: Optional[str]) -> _DBConfig:
    if not url:
        return _DBConfig("sqlite", str(Path("tmiv.db").resolve()))
    u = url.strip()
    if u.startswith("sqlite:///"):
        return _DBConfig("sqlite", u.replace("sqlite:///", "", 1))
    if u == "sqlite:///:memory:":
        return _DBConfig("sqlite", ":memory:")
    if u.startswith("duckdb:///"):
        return _DBConfig("duckdb", u.replace("duckdb:///", "", 1))
    # Fallback to sqlite path
    return _DBConfig("sqlite", u)


# --------------------------
# DBService
# --------------------------

class DBService:
    def __init__(self, database_url: Optional[str] = None) -> None:
        # Lazy-read from settings if not provided
        if database_url is None:
            try:
                from config.settings import get_settings
                database_url = get_settings().database_url
            except Exception:
                database_url = None

        self.cfg = _parse_db_url(database_url)
        self._lock = threading.Lock()
        # Ensure directory exists for file-based DBs
        if self.cfg.path not in {":memory:"}:
            Path(self.cfg.path).parent.mkdir(parents=True, exist_ok=True)
        # Initialize schema
        self._init_schema()

    # ---------- Public API (interfaces.IDBService) ----------

    def save_dataset_meta(self, *, name: str, fingerprint: str, rows: int, cols: int) -> bool:
        q = """
        INSERT INTO datasets(name, fingerprint, rows, cols)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(name, fingerprint) DO NOTHING;
        """
        self._execute(q, (name, fingerprint, int(rows), int(cols)))
        return True

    def save_metrics(self, *, run_id: str, model_name: str, metrics: Mapping[str, float]) -> bool:
        self._ensure_run(run_id)
        rows = [(run_id, model_name, str(k), float(v)) for k, v in (metrics or {}).items()]
        if not rows:
            return True
        self._executemany(
            "INSERT INTO model_metrics(run_id, model_name, metric_name, metric_value) VALUES (?, ?, ?, ?)",
            rows,
        )
        return True

    def save_feature_importance(self, *, run_id: str, df) -> bool:
        self._ensure_run(run_id)
        if df is None:
            return True
        rows = []
        if pd is not None and hasattr(df, "iterrows"):
            # Expect columns: feature, importance (and optionally rank)
            for i, row in df.reset_index(drop=True).iterrows():  # type: ignore[attr-defined]
                feat = str(row.get("feature", row.get("Feature", f"f_{i}")))
                imp = float(row.get("importance", row.get("importance_mean", 0.0)) or 0.0)
                rk = int(row.get("rank", i + 1))
                rows.append((run_id, feat, imp, rk))
        else:
            # accept iterable of (feature, importance)
            for i, (feat, imp) in enumerate(df):
                rows.append((run_id, str(feat), float(imp), int(i + 1)))

        if rows:
            self._executemany(
                "INSERT INTO feature_importance(run_id, feature, importance, rank) VALUES (?, ?, ?, ?)",
                rows,
            )
        return True

    def save_leaderboard(self, *, run_id: str, df) -> bool:
        self._ensure_run(run_id)
        if df is None:
            return True
        rows = []
        if pd is not None and hasattr(df, "to_dict"):
            # keep core fields and stash full row as JSON payload
            for _, r in df.iterrows():  # type: ignore[attr-defined]
                model = str(r.get("model", "model"))
                rank = int(r.get("rank", 0) or 0)
                primary = r.get("primary_metric", None)
                try:
                    primary_f = float(primary) if primary is not None else None
                except Exception:
                    primary_f = None
                payload = json.dumps({k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}, default=str)  # type: ignore[attr-defined]
                rows.append((run_id, model, rank, primary_f, payload))
        else:
            return True

        if rows:
            self._executemany(
                "INSERT INTO leaderboard(run_id, model_name, rank, primary_metric, payload_json) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
        return True

    def get_run(self, run_id: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {"run_id": run_id}
        out["metrics"] = self._fetch_all(
            "SELECT model_name, metric_name, metric_value FROM model_metrics WHERE run_id = ?",
            (run_id,),
        )
        out["feature_importance"] = self._fetch_all(
            "SELECT feature, importance, rank FROM feature_importance WHERE run_id = ? ORDER BY rank ASC",
            (run_id,),
        )
        out["leaderboard"] = self._fetch_all(
            "SELECT model_name, rank, primary_metric, payload_json FROM leaderboard WHERE run_id = ? ORDER BY rank ASC",
            (run_id,),
        )
        run_row = self._fetch_one("SELECT run_id, dataset_name, fingerprint, problem_type, created_at FROM runs WHERE run_id = ?", (run_id,))
        out["meta"] = run_row or {}
        return out

    # ---------- Convenience ----------

    def ensure_run_meta(
        self,
        *,
        run_id: str,
        dataset_name: Optional[str] = None,
        fingerprint: Optional[str] = None,
        problem_type: Optional[str] = None,
    ) -> None:
        """Idempotent upsert metadanych runu (wygodne z warstwy ML/eksportów)."""
        with self._lock, self._connect() as conn:
            if self.cfg.kind == "duckdb":
                # duckdb lacks "INSERT OR IGNORE" → emulate
                conn.execute(
                    "DELETE FROM runs WHERE run_id = ?",
                    [run_id],
                )
                conn.execute(
                    "INSERT INTO runs(run_id, dataset_name, fingerprint, problem_type) VALUES (?, ?, ?, ?)",
                    [run_id, dataset_name, fingerprint, problem_type],
                )
            else:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO runs(run_id, dataset_name, fingerprint, problem_type)
                    VALUES (?, ?, ?, ?)
                    """,
                    (run_id, dataset_name, fingerprint, problem_type),
                )

    # --------------------------
    # Internal: schema & connect
    # --------------------------

    def _connect(self):
        if self.cfg.kind == "duckdb":
            try:
                import duckdb  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError("DuckDB URL provided but 'duckdb' package is not installed.") from e
            return duckdb.connect(self.cfg.path)
        # sqlite
        conn = sqlite3.connect(self.cfg.path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
        except Exception:
            pass
        return conn

    def _init_schema(self) -> None:
        ddl_common = [
            """
            CREATE TABLE IF NOT EXISTS datasets(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                rows INTEGER,
                cols INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, fingerprint)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS runs(
                run_id TEXT PRIMARY KEY,
                dataset_name TEXT,
                fingerprint TEXT,
                problem_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS model_metrics(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS feature_importance(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                feature TEXT NOT NULL,
                importance REAL,
                rank INTEGER,
                FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS leaderboard(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                rank INTEGER,
                primary_metric REAL,
                payload_json TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );
            """,
        ]
        with self._lock, self._connect() as conn:
            for q in ddl_common:
                conn.execute(q)

    # --------------------------
    # Internal helpers (SQL)
    # --------------------------

    def _execute(self, query: str, params: tuple | list | None = None) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(query, params or [])
            try:
                conn.commit()
            except Exception:
                pass

    def _executemany(self, query: str, rows: list[tuple]) -> None:
        if not rows:
            return
        with self._lock, self._connect() as conn:
            conn.executemany(query, rows)
            try:
                conn.commit()
            except Exception:
                pass

    def _fetch_all(self, query: str, params: tuple | list | None = None) -> list[dict]:
        with self._lock, self._connect() as conn:
            cur = conn.execute(query, params or [])
            cols = [c[0] for c in cur.description] if getattr(cur, "description", None) else []
            out = [dict(zip(cols, row)) for row in cur.fetchall()]
            return out

    def _fetch_one(self, query: str, params: tuple | list | None = None) -> dict | None:
        with self._lock, self._connect() as conn:
            cur = conn.execute(query, params or [])
            row = cur.fetchone()
            if not row:
                return None
            cols = [c[0] for c in cur.description] if getattr(cur, "description", None) else []
            return dict(zip(cols, row))

    def _ensure_run(self, run_id: str) -> None:
        # Ensure entry exists
        if not self._fetch_one("SELECT run_id FROM runs WHERE run_id = ?", (run_id,)):
            self._execute("INSERT INTO runs(run_id) VALUES (?)", (run_id,))


__all__ = ["DBService"]
