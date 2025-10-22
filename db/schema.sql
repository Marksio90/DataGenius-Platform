-- db/schema.sql
-- TMIV – Advanced ML Platform
-- Minimal, portable schema (SQLite / DuckDB compatible).
-- Tables:
--   datasets, runs, model_metrics, feature_importance, leaderboard
-- Plus:
--   _migrations (used by db_utils.init_db) and a few helpful indexes.

-- SQLite pragmas are safe no-ops in DuckDB
PRAGMA foreign_keys = ON;

-- =========================
-- Datasets
-- =========================
CREATE TABLE IF NOT EXISTS datasets(
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT        NOT NULL,
    fingerprint  TEXT        NOT NULL,
    rows         INTEGER,
    cols         INTEGER,
    created_at   TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, fingerprint)
);

-- =========================
-- Runs (one training/evaluation session)
-- =========================
CREATE TABLE IF NOT EXISTS runs(
    run_id        TEXT PRIMARY KEY,
    dataset_name  TEXT,
    fingerprint   TEXT,
    problem_type  TEXT,           -- classification | regression | timeseries
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =========================
-- Metrics per model
-- =========================
CREATE TABLE IF NOT EXISTS model_metrics(
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT       NOT NULL,
    model_name   TEXT       NOT NULL,
    metric_name  TEXT       NOT NULL,
    metric_value REAL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_model_metrics_run ON model_metrics(run_id);
CREATE INDEX IF NOT EXISTS ix_model_metrics_model ON model_metrics(model_name);
CREATE INDEX IF NOT EXISTS ix_model_metrics_metric ON model_metrics(metric_name);

-- =========================
-- Feature importance (normalized if possible)
-- =========================
CREATE TABLE IF NOT EXISTS feature_importance(
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT     NOT NULL,
    feature     TEXT     NOT NULL,
    importance  REAL,
    rank        INTEGER,
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_fi_run ON feature_importance(run_id);
CREATE INDEX IF NOT EXISTS ix_fi_rank ON feature_importance(run_id, rank);

-- =========================
-- Leaderboard (one row per model for a given run)
-- =========================
CREATE TABLE IF NOT EXISTS leaderboard(
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT     NOT NULL,
    model_name      TEXT     NOT NULL,
    rank            INTEGER,
    primary_metric  REAL,
    payload_json    TEXT,      -- full row snapshot as JSON (metrics, params, etc.)
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_lb_run ON leaderboard(run_id);
CREATE INDEX IF NOT EXISTS ix_lb_rank ON leaderboard(run_id, rank);

-- =========================
-- Migrations bookkeeping (used by db_utils.init_db)
-- =========================
CREATE TABLE IF NOT EXISTS _migrations(
    name       TEXT PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =========================
-- Helpful views (optional)
-- =========================

-- Best model per run (by lowest rank)
CREATE VIEW IF NOT EXISTS v_best_model_per_run AS
SELECT
    l.run_id,
    l.model_name,
    l.rank,
    l.primary_metric
FROM leaderboard l
WHERE l.rank = (
    SELECT MIN(l2.rank) FROM leaderboard l2 WHERE l2.run_id = l.run_id
);

-- Aggregated metrics per run/model (pivot-like, limited)
-- Note: For full pivoting, do it at the app layer.
CREATE VIEW IF NOT EXISTS v_metrics_flat AS
SELECT
    m.run_id,
    m.model_name,
    m.metric_name,
    m.metric_value
FROM model_metrics m;
