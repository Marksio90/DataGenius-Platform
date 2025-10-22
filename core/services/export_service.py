# core/services/export_service.py
"""
ExportService – wygodne eksporty artefaktów dla TMIV – Advanced ML Platform.

Zakres:
- Zapisy pomocnicze: JSON/CSV/TXT/Model (joblib/pickle) do katalogu cache/artifacts.
- Raport PDF (delegacja do backend/export_explain_pdf.py, z bezpiecznym fallbackiem).
- Pełny ZIP artefaktów (delegacja do backend/export_everything.py, z fallbackiem).

Cechy:
- Zwracane są **absolutne** ścieżki plików.
- Brak twardych zależności – każda integracja ma soft-fallback.
"""

from __future__ import annotations

import json
import os
import pickle
import zipfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # pandas tylko gdy potrzebne do CSV
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# --- cache / ścieżki ---
try:  # prefer official cache manager
    from backend.cache_manager import cached_path
except Exception:  # pragma: no cover
    # minimal fallback
    def cached_path(subdir: str, name: str) -> Path:
        p = Path("cache").joinpath("artifacts", subdir)
        p.mkdir(parents=True, exist_ok=True)
        return p.joinpath(name).resolve()

# --- PDF (opcjonalny moduł; zapewniamy fallback) ---
try:  # pragma: no cover
    from backend.export_explain_pdf import build_pdf_report as _build_pdf_report
except Exception:  # pragma: no cover
    _build_pdf_report = None  # type: ignore

# --- ZIP (opcjonalny moduł; zapewniamy fallback) ---
try:  # pragma: no cover
    from backend.export_everything import export_everything as _export_everything
except Exception:  # pragma: no cover
    _export_everything = None  # type: ignore

# --- model serializer (prefer joblib) ---
try:  # pragma: no cover
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


class ExportService:
    # =========================
    # Low-level writers
    # =========================
    def write_json(self, obj: Any, name: str, *, subdir: str = "exports/tmp", indent: int = 2) -> str:
        """
        Zapisz JSON do cache/artifacts/<subdir>/<name>.json (gdy brak suffixa, zostanie dodany).
        """
        filename = name if name.lower().endswith(".json") else f"{name}.json"
        path = cached_path(subdir, filename)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent, default=_json_default)
        return str(path)

    def write_text(self, text: str, name: str, *, subdir: str = "exports/tmp") -> str:
        """
        Zapisz TXT/MD. Suffix wyniknie z `name`.
        """
        path = cached_path(subdir, name)
        with path.open("w", encoding="utf-8") as f:
            f.write(text or "")
        return str(path)

    def save_dataframe_csv(self, df, name: str, *, subdir: str = "exports/tmp") -> str:
        """
        Zapisz DataFrame jako CSV. Jeśli `pandas` nie jest dostępne, spróbuj obsłużyć list[dict].
        """
        path = cached_path(subdir, name if name.lower().endswith(".csv") else f"{name}.csv")
        if pd is not None and hasattr(getattr(df, "__class__", None), "__name__") and df.__class__.__name__ in {"DataFrame"}:
            df.to_csv(path, index=False)  # type: ignore[attr-defined]
            return str(path)
        # Fallback: list[dict] -> CSV
        rows = df or []
        if isinstance(rows, (list, tuple)) and rows and isinstance(rows[0], dict):
            # prosty CSV – kolumny z unii kluczy
            cols = sorted({k for r in rows for k in r.keys()})
            with path.open("w", encoding="utf-8") as f:
                f.write(",".join(map(str, cols)) + "\n")
                for r in rows:
                    line = ",".join(_csv_escape(r.get(c, "")) for c in cols)
                    f.write(line + "\n")
            return str(path)
        # Ostatecznie – zapis JSON
        jpath = Path(str(path).replace(".csv", ".json"))
        with jpath.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2, default=_json_default)
        return str(jpath)

    def save_model(self, model: Any, name: str, *, subdir: str = "models") -> str:
        """
        Zapisz model binarnie (prefer joblib, fallback pickle).
        """
        path = cached_path(subdir, name if name.lower().endswith(".joblib") else f"{name}.joblib")
        try:
            if joblib is not None:
                joblib.dump(model, path)  # type: ignore[attr-defined]
            else:
                # pickle fallback
                with path.open("wb") as f:
                    pickle.dump(model, f)
            return str(path)
        except Exception:
            # awaryjny pickle z innym rozszerzeniem
            p2 = path.with_suffix(".pkl")
            with p2.open("wb") as f:
                pickle.dump(model, f)
            return str(p2)

    # =========================
    # PDF
    # =========================
    def build_pdf_report(
        self,
        output_path: str,
        *,
        metrics: Mapping[str, Any] | None = None,
        feature_importance: Any | None = None,
        charts: Mapping[str, str] | Sequence[Tuple[str, str]] | None = None,
        title: str = "Explainability Report",
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        problem_type: Optional[str] = None,
        created_by: Optional[str] = None,
        run_id: Optional[str] = None,
        notes: Sequence[str] | str | None = None,
        page_size: str = "A4",
    ) -> str:
        """
        Zwraca **absolutną** ścieżkę do PDF (albo TXT fallback).
        """
        out = Path(output_path)
        if not out.is_absolute():
            out = cached_path("reports", Path(output_path).name)

        if _build_pdf_report is not None:
            return _build_pdf_report(
                str(out),
                metrics=metrics,
                feature_importance=feature_importance,
                charts=charts,
                title=title,
                model_name=model_name,
                dataset_name=dataset_name,
                problem_type=problem_type,
                created_by=created_by,
                run_id=run_id,
                notes=notes,
                page_size=page_size,
            )

        # Fallback: krótki TXT jako artefakt „raportu”
        txt = out.with_suffix(".txt")
        payload = {
            "title": title,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "problem_type": problem_type,
            "run_id": run_id,
            "metrics": metrics or {},
            "notes": notes or [],
            "charts": dict(charts or {}),
        }
        with txt.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
        return str(txt)

    # =========================
    # ZIP – pełny eksport
    # =========================
    def export_everything(
        self,
        run_id: str,
        *,
        problem_type: str,
        metrics: Mapping[str, Any],
        dataset_name: Optional[str] = None,
        dataset_fingerprint: Optional[str] = None,
        plan: Mapping[str, Any] | None = None,
        cv_metrics: Sequence[Mapping[str, Any]] | None = None,
        leaderboard: Any | None = None,           # pd.DataFrame | None
        feature_importance: Any | None = None,    # pd.DataFrame | None
        models: Mapping[str, Any] | None = None,  # map: name -> path|file|object
        plots: Mapping[str, str] | None = None,
        configs: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        notes: str | None = None,
        exports_dir: str | None = None,
    ) -> str:
        """
        Spakuj wszystkie artefakty do ZIP. Zwraca absolutną ścieżkę ZIP.
        Preferuje delegację do backend/export_everything.
        """
        if _export_everything is not None:
            return _export_everything(
                run_id,
                problem_type=problem_type,
                metrics=metrics,
                dataset_name=dataset_name,
                dataset_fingerprint=dataset_fingerprint,
                plan=plan,
                cv_metrics=cv_metrics,
                leaderboard=leaderboard,
                feature_importance=feature_importance,
                models=models,
                plots=plots,
                configs=configs,
                params=params,
                notes=notes,
                exports_dir=exports_dir,
            )

        # --- Fallback: lokalne budowanie ZIP ---
        base_dir = Path(exports_dir) if exports_dir else cached_path("exports", f"{run_id}.zip").parent
        base_dir.mkdir(parents=True, exist_ok=True)
        zip_path = base_dir.joinpath(f"{run_id}.zip").resolve()

        # Tymczasowe JSON-y
        def _dump_json(name: str, obj: Any) -> Path:
            p = cached_path("tmp_export", f"{run_id}_{name}.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
            return p

        manifest = {
            "run_id": run_id,
            "problem_type": problem_type,
            "dataset_name": dataset_name,
            "dataset_fingerprint": dataset_fingerprint,
            "notes": notes,
        }
        tmp_files: list[Path] = [
            _dump_json("manifest", manifest),
            _dump_json("metrics", metrics or {}),
            _dump_json("plan", plan or {}),
            _dump_json("cv_metrics", list(cv_metrics or [])),
            _dump_json("configs", configs or {}),
            _dump_json("params", params or {}),
        ]

        # Leaderboard / FI
        if pd is not None and hasattr(leaderboard, "to_csv") and getattr(leaderboard, "empty", False) is False:
            p = cached_path("tmp_export", f"{run_id}_leaderboard.csv")
            leaderboard.to_csv(p, index=False)  # type: ignore[attr-defined]
            tmp_files.append(p)
        if pd is not None and hasattr(feature_importance, "to_csv") and getattr(feature_importance, "empty", False) is False:
            p = cached_path("tmp_export", f"{run_id}_feature_importance.csv")
            feature_importance.to_csv(p, index=False)  # type: ignore[attr-defined]
            tmp_files.append(p)

        # Wykresy
        plot_files: list[Path] = []
        for _, p in (plots or {}).items():
            pp = Path(str(p))
            if pp.exists():
                plot_files.append(pp)

        # Modele – jeśli trzymamy ścieżki, dołącz; jeśli obiekty -> zapisz tymczasowo
        model_files: list[Tuple[str, Path]] = []
        for name, obj in (models or {}).items():
            if _is_pathlike(obj):
                mp = Path(obj)
                if mp.exists() and mp.is_file():
                    model_files.append((name, mp))
            else:
                # spróbuj zapisać obiekt
                mp = cached_path("tmp_export/models", f"{run_id}_{name}.joblib")
                try:
                    if joblib is not None:
                        joblib.dump(obj, mp)  # type: ignore[attr-defined]
                    else:
                        with mp.open("wb") as f:
                            pickle.dump(obj, f)
                    model_files.append((name, mp))
                except Exception:
                    # pomiń niemożliwe do serializacji
                    continue

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            # JSON-y
            for f in tmp_files:
                z.write(f, arcname=f.name)
            # Wykresy
            for p in plot_files:
                z.write(p, arcname=f"plots/{p.name}")
            # Modele
            for name, mp in model_files:
                z.write(mp, arcname=f"models/{mp.name}")

        return str(zip_path)

    # =========================
    # Uproszczone „bundle” ścieżek
    # =========================
    def bundle_paths(
        self,
        paths: Mapping[str, str] | Sequence[str],
        *,
        run_id: Optional[str] = None,
        exports_dir: Optional[str] = None,
        bundle_name: Optional[str] = None,
    ) -> str:
        """
        Zbierz dowolne ścieżki w pojedynczy ZIP.
        """
        name = bundle_name or (run_id or "bundle")
        base_dir = Path(exports_dir) if exports_dir else cached_path("exports", f"{name}.zip").parent
        base_dir.mkdir(parents=True, exist_ok=True)
        zip_path = base_dir.joinpath(f"{name}.zip").resolve()

        files: list[Tuple[str, Path]] = []
        if isinstance(paths, Mapping):
            for arc, p in paths.items():
                pp = Path(p)
                if pp.exists():
                    files.append((arc, pp))
        else:
            for p in paths:
                pp = Path(p)
                if pp.exists():
                    files.append((pp.name, pp))

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for arc, fp in files:
                z.write(fp, arcname=arc)

        return str(zip_path)


# =========================
# Helpers
# =========================

def _is_pathlike(x: Any) -> bool:
    try:
        return isinstance(x, (str, os.PathLike)) or hasattr(x, "__fspath__")
    except Exception:
        return False


def _csv_escape(x: Any) -> str:
    s = "" if x is None else str(x)
    if any(ch in s for ch in [",", '"', "\n"]):
        s = '"' + s.replace('"', '""') + '"'
    return s


def _json_default(obj: Any) -> Any:
    try:
        if isinstance(obj, (Path,)):
            return str(obj)
        if isinstance(obj, (np.generic,)):
            return obj.item()
        if pd is not None:
            if isinstance(obj, pd.DataFrame):
                return {"columns": obj.columns.tolist(), "shape": obj.shape}
            if isinstance(obj, pd.Series):
                return obj.to_dict()
        return str(obj)
    except Exception:
        return "<unserializable>"


__all__ = ["ExportService"]
