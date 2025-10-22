from __future__ import annotations
# -*- coding: utf-8 -*-

# backend/model_export_utils.py

from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import os
import json
import csv
import shutil
import numpy as np

# optional
try:
    import joblib  # type: ignore
except Exception:
    joblib = None  # pragma: no cover


# =========================== HELPERS ===========================

def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _n_features_from_sample(X_sample) -> int:
    """Zwraca liczbę cech (obsługuje NumPy i pandas)."""
    try:
        import pandas as pd  # type: ignore
        if isinstance(X_sample, pd.DataFrame):
            return int(X_sample.shape[1])
    except Exception:
        pass
    return int(np.asarray(X_sample).shape[1])


# =========================== ONNX / PMML ===========================

def export_onnx(model, X_sample, path: str) -> Optional[str]:
    """
    Eksport modelu do ONNX przez skl2onnx (jeśli dostępny).
    Dla XGBoost fallback do .json (save_model) jeśli ONNX się nie powiedzie.
    Zwraca ścieżkę do .onnx albo None (gdy nie udało się wyeksportować).
    """
    try:
        from skl2onnx import convert_sklearn  # type: ignore
        from skl2onnx.common.data_types import FloatTensorType  # type: ignore

        n_features = _n_features_from_sample(X_sample)
        initial_type = [('input', FloatTensorType([None, n_features]))]
        onx = convert_sklearn(model, initial_types=initial_type)

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            f.write(onx.SerializeToString())
        return str(out_path)

    except Exception:
        # Fallback: XGBoost native (JSON)
        try:
            import xgboost as xgb  # type: ignore  # noqa: F401
            if hasattr(model, "save_model"):
                json_path = path.replace(".onnx", ".json")
                Path(json_path).parent.mkdir(parents=True, exist_ok=True)
                model.save_model(json_path)
                # ONNX nie powstał, ale mamy JSON — zwracamy None by nie udawać ONNX-a
                return None
        except Exception:
            pass

    return None


def export_pmml(model, X_sample, path: str) -> Optional[str]:
    """
    Eksport do PMML:
      - preferowany: nyoka.skl_to_pmml,
      - alternatywa: sklearn2pmml (wymaga PMMLPipeline — tutaj bez automatycznej budowy).
    Zwraca ścieżkę do .pmml lub None.
    """
    # Nyoka
    try:
        from nyoka import skl_to_pmml  # type: ignore
        feature_names = [f"f{i}" for i in range(_n_features_from_sample(X_sample))]
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        skl_to_pmml(model, feature_names, 'target', str(out_path))
        return str(out_path)
    except Exception:
        pass

    # sklearn2pmml (bez PMMLPipeline nie idziemy dalej)
    try:
        from sklearn2pmml import sklearn2pmml  # type: ignore  # noqa: F401
        return None
    except Exception:
        return None


# =========================== METRYKI / META ===========================

def export_metrics_csv(metrics: Dict[str, Any], outdir: Path) -> Path:
    """
    Zapisuje metryki do CSV (kolumny: metric,value).
    """
    outdir = _ensure_dir(outdir)
    p = outdir / "metrics.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in (metrics or {}).items():
            w.writerow([k, v])
    return p


def export_plan_json(plan: Dict[str, Any], outdir: Path) -> Path:
    """
    Zapisuje plan (np. data-prep/modeling plan) do JSON: plan.json
    """
    outdir = _ensure_dir(outdir)
    p = outdir / "plan.json"
    p.write_text(json.dumps(plan or {}, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def export_recs_json(recs: Dict[str, Any], outdir: Path) -> Path:
    """
    Zapisuje rekomendacje do JSON: recs.json
    """
    outdir = _ensure_dir(outdir)
    p = outdir / "recs.json"
    p.write_text(json.dumps(recs or {}, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


# =========================== WYKRESY ===========================

def _safe_mpl():
    """Ładuje Matplotlib w trybie headless (Agg)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        return None


def save_figure_png(fig, outdir: Path, name: str = "plot.png", dpi: int = 120) -> Path:
    """
    Zapisuje figurę Matplotlib do PNG z bezpiecznym zamykaniem figury.
    """
    outdir = _ensure_dir(outdir)
    p = outdir / name
    try:
        plt = _safe_mpl()
        if hasattr(fig, "savefig"):
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
        # w innym wypadku — ignorujemy, ale zwracamy ścieżkę dla spójności
    except Exception:
        pass
    finally:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            if hasattr(fig, "number"):
                plt.close(fig)
        except Exception:
            pass
    return p


# =========================== LEGACY SHIMS (zgodność wstecz) ===========================
# Starsze wersje `backend/exporters.py` importowały funkcje `save_*`. Dodajemy aliasy.

def save_plan_json(plan: Union[Dict[str, Any], List[Dict[str, Any]]], out_dir: Union[str, Path], filename: str = "plan.json") -> str:
    """
    Zgodność wsteczna: zapisuje plan do JSON.
    Ignoruje nazwę pliku jeśli to 'plan.json' (zapisuje tam, gdzie oczekuje exporters.py).
    """
    out_path = export_plan_json(plan if isinstance(plan, dict) else {"plan": plan}, Path(out_dir))
    # zachowujemy nazwę — jeśli ktoś podał inną nazwę, przesuń plik
    if filename != "plan.json":
        dst = Path(out_dir) / filename
        shutil.copyfile(out_path, dst)
        return str(dst)
    return str(out_path)


def save_metrics_csv(metrics: Union[List[Dict[str, Any]], Dict[str, Any]], out_dir: Union[str, Path], filename: str = "metrics.csv") -> str:
    """
    Zgodność wsteczna: zapis metryk do CSV. Obsługuje listę rekordów lub dict {metric: value}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Jeśli to lista rekordów (np. [{'metric': 'f1', 'value': 0.9}, ...]) — spróbuj dopasować kolumny.
    if isinstance(metrics, list):
        # Spróbuj nagłówków 'metric','value'; jeśli nie ma — użyj wszystkich kluczy.
        headers = set()
        for r in metrics:
            headers.update(r.keys())
        headers = list(headers) if headers else ["metric", "value"]
        p = out_dir / filename
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for r in metrics:
                w.writerow([r.get(h, "") for h in headers])
        return str(p)
    # dict {metric: value}
    p = export_metrics_csv(metrics, out_dir)
    if filename != "metrics.csv":
        dst = out_dir / filename
        shutil.copyfile(p, dst)
        return str(dst)
    return str(p)


def save_recs_json(recs: Dict[str, Any], out_dir: Union[str, Path], filename: str = "recs.json") -> str:
    """
    Zgodność wsteczna: zapis rekomendacji do JSON.
    """
    p = export_recs_json(recs, Path(out_dir))
    if filename != "recs.json":
        dst = Path(out_dir) / filename
        shutil.copyfile(p, dst)
        return str(dst)
    return str(p)


def save_model_joblib(model: Any, out_dir: Union[str, Path], filename: str = "model.joblib") -> str:
    """
    Zapis modelu przez joblib, jeśli dostępne; w przeciwnym razie JSON z metadanymi.
    """
    out_dir = _ensure_dir(out_dir)
    path = out_dir / filename
    if joblib is not None:
        try:
            joblib.dump(model, str(path))
            return str(path)
        except Exception:
            pass  # fallback do JSON poniżej

    meta_path = out_dir / "model_meta.json"
    payload = {
        "note": "joblib not available or dump failed; storing meta only",
        "repr": str(type(model)),
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(meta_path)


def write_html_report(html_or_path: str, out_dir: Union[str, Path], filename: str = "report.html") -> str:
    """
    Przyjmuje już wyrenderowany HTML albo ścieżkę do niego; kopiuje/zapisuje do artifacts.
    """
    out_dir = _ensure_dir(out_dir)
    dst = out_dir / filename
    src_path = Path(html_or_path)
    if src_path.exists():
        shutil.copyfile(src_path, dst)
        return str(dst)
    dst.write_text(html_or_path, encoding="utf-8")
    return str(dst)


def write_md_readme(md_text: str, out_dir: Union[str, Path], filename: str = "README_RESULTS.md") -> str:
    out_dir = _ensure_dir(out_dir)
    p = out_dir / filename
    p.write_text(md_text, encoding="utf-8")
    return str(p)