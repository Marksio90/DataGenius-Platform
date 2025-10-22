# backend/export_everything.py
"""
Export all run artifacts (models, metrics, plots, configs) into a clean ZIP.

What it does
------------
- Creates a reproducible export tree under ./exports/<run_id>/ ...
- Saves:
  - manifest.json (who/what/when)
  - metrics.json (+ optional leaderboard.csv)
  - feature_importance.csv (if provided)
  - models/*.joblib (or .pkl as fallback)
  - plots/* (copied from given absolute/relative paths)
  - params/configs/context as JSON
  - README.txt with quick info
- Zips the folder to ./exports/<run_id>.zip and returns absolute path.

Public API
----------
export_everything(
    run_id: str,
    *,
    problem_type: str,
    metrics: dict,
    dataset_name: str | None = None,
    dataset_fingerprint: str | None = None,
    plan: dict | None = None,
    cv_metrics: list[dict] | None = None,
    leaderboard: "pd.DataFrame | None" = None,
    feature_importance: "pd.DataFrame | None" = None,
    models: dict[str, object] | None = None,
    plots: dict[str, str] | None = None,  # {name: filepath}
    configs: dict | None = None,
    params: dict | None = None,
    notes: str | None = None,
    exports_dir: str | None = None,
) -> str

Notes
-----
- Safe to call without optional arguments; it will just export what is provided.
- No Streamlit/UI side effects. Return value is the ZIP absolute path.

Example
-------
    zip_path = export_everything(
        "run_2025-10-22_12-00-00",
        problem_type="classification",
        metrics={"roc_auc": 0.89, "f1": 0.79},
        dataset_name="avocado",
        dataset_fingerprint=fingerprint_df(df),
        feature_importance=fi_df,
        leaderboard=lb_df,
        models={"xgb": xgb_model, "lr": lr_model},
        plots={"roc": "/abs/path/roc.png"},
        plan=training_plan_dict,
        configs={"cv": 5, "strategy": "balanced"},
        params={"seed": 42},
        notes="Baseline experiment.",
    )
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

# Optional heavy deps only used if present
try:  # typing only
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

from .cache_manager import cached_path  # not required, but handy for future use


# =========================
# Helpers
# =========================


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sanitize_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
    return safe.strip("._") or "artifact"


def _exports_root(default: str | None) -> Path:
    root = Path(default or os.getenv("TMIV_EXPORTS_DIR", "exports")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _save_dataframe_csv(df, path: Path) -> bool:
    if df is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(df, "to_csv"):
            df.to_csv(path, index=False)
            return True
    except Exception:
        return False
    return False


def _copy_if_exists(src: str | os.PathLike[str], dst: Path) -> bool:
    try:
        src_path = Path(src)
        if src_path.exists() and src_path.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)
            return True
    except Exception:
        return False
    return False


def _save_model(obj: Any, path: Path) -> str | None:
    """
    Try to persist a model using joblib (preferred) and pickle as fallback.
    Returns the actual saved file path (str) or None on failure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Prefer joblib when available
    if joblib is not None:
        try:
            out = path.with_suffix(".joblib")
            joblib.dump(obj, out)
            return str(out)
        except Exception:
            pass
    # Fallback: pickle
    import pickle

    try:
        out = path.with_suffix(".pkl")
        with out.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        return str(out)
    except Exception:
        return None


@dataclass
class Manifest:
    run_id: str
    created_at: str
    problem_type: str
    dataset_name: str | None = None
    dataset_fingerprint: str | None = None
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=lambda: platform.platform())
    packages: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)  # logical_name -> relative path

    def to_dict(self) -> dict:
        return asdict(self)


def _collect_pkg_versions(pkgs: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in pkgs:
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", None)
            if ver:
                out[name] = str(ver)
        except Exception:
            continue
    return out


# =========================
# Main API
# =========================


def export_everything(
    run_id: str,
    *,
    problem_type: str,
    metrics: Mapping[str, Any],
    dataset_name: str | None = None,
    dataset_fingerprint: str | None = None,
    plan: Mapping[str, Any] | None = None,
    cv_metrics: list[Mapping[str, Any]] | None = None,
    leaderboard=None,  # pd.DataFrame | None, typed loosely to avoid hard dependency
    feature_importance=None,  # pd.DataFrame | None
    models: Mapping[str, Any] | None = None,
    plots: Mapping[str, str] | None = None,  # {name: filepath}
    configs: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
    notes: str | None = None,
    exports_dir: str | None = None,
) -> str:
    """
    Build export tree and zip it. Returns absolute path to the created ZIP file.
    """
    # --- Prepare paths ---
    safe_run = _sanitize_name(run_id)
    root = _exports_root(exports_dir)
    run_dir = root / safe_run
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Subdirs
    d_models = run_dir / "models"
    d_plots = run_dir / "plots"
    d_tables = run_dir / "tables"
    d_meta = run_dir / "meta"
    for d in (d_models, d_plots, d_tables, d_meta):
        d.mkdir(parents=True, exist_ok=True)

    # --- Manifest ---
    pkgs = _collect_pkg_versions(
        [
            "pandas",
            "numpy",
            "scikit_learn",
            "xgboost",
            "lightgbm",
            "catboost",
            "shap",
            "lime",
        ]
    )
    manifest = Manifest(
        run_id=safe_run,
        created_at=_now_iso(),
        problem_type=str(problem_type),
        dataset_name=dataset_name,
        dataset_fingerprint=dataset_fingerprint,
        packages=pkgs,
    )

    # --- Metrics & tables ---
    _write_json(run_dir / "metrics.json", dict(metrics))
    manifest.files["metrics_json"] = "metrics.json"

    if cv_metrics:
        _write_json(d_tables / "cv_metrics.json", list(cv_metrics))
        manifest.files["cv_metrics_json"] = f"tables/{(d_tables / 'cv_metrics.json').name}"

    if leaderboard is not None:
        _save_dataframe_csv(leaderboard, d_tables / "leaderboard.csv")
        manifest.files["leaderboard_csv"] = f"tables/leaderboard.csv"

    if feature_importance is not None:
        _save_dataframe_csv(feature_importance, d_tables / "feature_importance.csv")
        manifest.files["feature_importance_csv"] = f"tables/feature_importance.csv"

    # --- Configs, params, plan, notes ---
    if plan is not None:
        _write_json(d_meta / "training_plan.json", dict(plan))
        manifest.files["training_plan"] = f"meta/training_plan.json"
    if configs is not None:
        _write_json(d_meta / "configs.json", dict(configs))
        manifest.files["configs"] = f"meta/configs.json"
    if params is not None:
        _write_json(d_meta / "params.json", dict(params))
        manifest.files["params"] = f"meta/params.json"
    if notes:
        (run_dir / "README.txt").write_text(str(notes).strip() + "\n", encoding="utf-8")
        manifest.files["readme_txt"] = "README.txt"

    # --- Models ---
    if models:
        for name, model in models.items():
            base = _sanitize_name(name)
            saved = _save_model(model, d_models / base)
            if saved:
                manifest.files[f"model:{base}"] = f"models/{Path(saved).name}"

    # --- Plots ---
    if plots:
        for name, src in plots.items():
            base = _sanitize_name(name)
            ext = Path(src).suffix or ".png"
            dst = d_plots / f"{base}{ext}"
            if _copy_if_exists(src, dst):
                manifest.files[f"plot:{base}"] = f"plots/{dst.name}"

    # --- Manifest last ---
    _write_json(run_dir / "manifest.json", manifest.to_dict())

    # --- Zip everything ---
    zip_path = root / f"{safe_run}.zip"
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root, safe_run)

    return str(zip_path.resolve())


# =========================
# Minimal CLI (manual test)
# =========================

if __name__ == "__main__":  # pragma: no cover
    # Quick manual test without pandas dependency
    zip_file = export_everything(
        "run_demo",
        problem_type="classification",
        metrics={"roc_auc": 0.88, "f1": 0.77},
        dataset_name="demo",
        models={"dummy": {"coef": [1, 2, 3]}},  # picklable
        plots={},
        configs={"cv": 5},
        params={"seed": 42},
        notes="This is a demonstration export.",
    )
    print("Exported ZIP:", zip_file)
