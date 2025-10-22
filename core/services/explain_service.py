# core/services/explain_service.py
"""
ExplainService – wizualizacje, raport PDF i pełny eksport artefaktów
dla TMIV – Advanced ML Platform.

Implementuje `IExplainService` (zob. core/interfaces.py).

Zakres:
- Wykresy (delegacja do backend/plots.py – tylko matplotlib).
- Raport PDF (delegacja do backend/export_explain_pdf.py, z bezpiecznym fallbackiem).
- Eksport ZIP (delegacja do backend/export_everything.py, z fallbackiem zipfile).

Uwaga:
- Brak twardych zależności na Streamlit/UI.
- Wszystkie funkcje zwracają **absolutne** ścieżki do plików.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --- wykresy (wymagane) ---
from backend import plots as P  # matplotlib-only helpers

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


class ExplainService:
    # =========================
    # WYKRESY
    # =========================
    def plot_classification_curves(
        self,
        y_true: np.ndarray | Sequence[int],
        y_proba: np.ndarray,
        *,
        model_name: str = "model",
    ) -> dict[str, str]:
        return P.plot_classification_curves(y_true, y_proba, model_name=model_name)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray | Sequence[int],
        y_pred: np.ndarray | Sequence[int],
        *,
        class_names: Optional[Sequence[str]] = None,
        model_name: str = "model",
        normalize: str | None = "true",
    ) -> str:
        return P.plot_confusion_matrix(
            y_true, y_pred, class_names=class_names, model_name=model_name, normalize=normalize
        )

    def plot_calibration_curve(
        self,
        y_true: np.ndarray | Sequence[int],
        y_proba: np.ndarray,
        *,
        model_name: str = "model",
        n_bins: int = 10,
    ) -> str:
        return P.plot_calibration_curve(y_true, y_proba, model_name=model_name, n_bins=int(n_bins))

    def plot_regression_diagnostics(
        self,
        y_true: np.ndarray | Sequence[float],
        y_pred: np.ndarray | Sequence[float],
        *,
        model_name: str = "model",
    ) -> dict[str, str]:
        return P.plot_regression_diagnostics(y_true, y_pred, model_name=model_name)

    def plot_feature_importance(
        self,
        fi: pd.DataFrame | Sequence[Mapping[str, Any]] | Sequence[Tuple[str, float]],
        *,
        top_k: int = 20,
        name: str = "fi",
    ) -> str:
        return P.plot_feature_importance(fi, top_k=int(top_k), name=name)

    def plot_radar_leaderboard(
        self,
        leaderboard: pd.DataFrame,
        *,
        metrics: Optional[Sequence[str]] = None,
        name: str = "radar",
    ) -> str:
        return P.plot_radar_leaderboard(leaderboard, metrics=metrics, name=name)

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
        Zwraca **absolutną** ścieżkę do PDF.
        """
        out = Path(output_path)
        if not out.is_absolute():
            out = cached_path("reports", Path(output_path).name)

        if _build_pdf_report is not None:
            # delegacja do pełnej implementacji (ReportLab)
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

        # --- Fallback: prosty PDF (ReportLab jeśli dostępny), inaczej TXT->PDF namiastka ---
        try:  # pragma: no cover
            from reportlab.lib.pagesizes import A4, letter
            from reportlab.lib.units import cm
            from reportlab.pdfgen import canvas

            pagesize = A4 if str(page_size).upper() == "A4" else letter
            out.parent.mkdir(parents=True, exist_ok=True)
            c = canvas.Canvas(str(out), pagesize=pagesize)
            width, height = pagesize
            y = height - 2 * cm

            def write_line(txt: str, dy: float = 14):
                nonlocal y
                c.drawString(2 * cm, y, txt[:120])
                y -= dy
                if y < 2 * cm:
                    c.showPage()
                    y = height - 2 * cm

            write_line(title or "Explainability Report")
            for meta in [
                f"Model: {model_name or '-'}",
                f"Dataset: {dataset_name or '-'}",
                f"Problem: {problem_type or '-'}",
                f"Run ID: {run_id or '-'}",
                f"Author: {created_by or '-'}",
            ]:
                write_line(meta)

            write_line(" ")
            write_line("== Metrics ==")
            for k, v in (metrics or {}).items():
                write_line(f"- {k}: {v}")

            if feature_importance is not None:
                write_line(" ")
                write_line("== Feature Importance (head) ==")
                try:
                    df = feature_importance if isinstance(feature_importance, pd.DataFrame) else None
                    if df is not None and not df.empty:
                        for _, row in df.head(20).iterrows():
                            write_line(f"- {row.get('feature')}: {row.get('importance')}")
                except Exception:
                    write_line("(unable to render FI table)")

            if charts:
                write_line(" ")
                write_line("== Charts ==")
                items = charts.items() if isinstance(charts, Mapping) else list(charts)
                for name, path in items:
                    p = Path(path)
                    write_line(f"- {name}: {p.name}")

            if notes:
                write_line(" ")
                write_line("== Notes ==")
                if isinstance(notes, (list, tuple)):
                    for n in notes:
                        write_line(f"- {n}")
                else:
                    write_line(str(notes))

            c.showPage()
            c.save()
            return str(out)
        except Exception:
            # Najprostszy fallback: plik .txt + rename na .pdf (wymienialny artefakt)
            out.parent.mkdir(parents=True, exist_ok=True)
            txt = out.with_suffix(".txt")
            with txt.open("w", encoding="utf-8") as f:
                f.write(f"{title}\n\n")
                f.write(json.dumps({"metrics": metrics, "notes": notes}, ensure_ascii=False, indent=2))
            # Zwróć ścieżkę .txt (UI może obsłużyć jako artefakt)
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
        models: Mapping[str, Any] | None = None,
        plots: Mapping[str, str] | None = None,
        configs: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        notes: str | None = None,
        exports_dir: str | None = None,
    ) -> str:
        """
        Spakuj wszystkie artefakty do ZIP. Zwraca absolutną ścieżkę ZIP.
        """
        # Prefer delegację do backend/export_everything.py
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

        # --- Fallback: własny ZIP (bez serializacji modeli binarnych) ---
        base_dir = Path(exports_dir) if exports_dir else cached_path("exports", f"{run_id}.zip").parent
        base_dir.mkdir(parents=True, exist_ok=True)
        zip_path = base_dir.joinpath(f"{run_id}.zip").resolve()

        # Przygotuj tymczasowe pliki JSON
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
        f_manifest = _dump_json("manifest", manifest)
        f_metrics = _dump_json("metrics", metrics or {})
        f_plan = _dump_json("plan", plan or {})
        f_cv = _dump_json("cv_metrics", list(cv_metrics or []))
        f_configs = _dump_json("configs", configs or {})
        f_params = _dump_json("params", params or {})

        # Zebrane wykresy
        plot_files: list[Path] = []
        for _, p in (plots or {}).items():
            pp = Path(p)
            if pp.exists():
                plot_files.append(pp)

        # Leaderboard / FI (CSV jeśli DataFrame)
        tmp_files: list[Path] = [f_manifest, f_metrics, f_plan, f_cv, f_configs, f_params]
        if isinstance(leaderboard, pd.DataFrame) and not leaderboard.empty:
            p = cached_path("tmp_export", f"{run_id}_leaderboard.csv")
            leaderboard.to_csv(p, index=False)
            tmp_files.append(p)
        if isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty:
            p = cached_path("tmp_export", f"{run_id}_feature_importance.csv")
            feature_importance.to_csv(p, index=False)
            tmp_files.append(p)

        # Budowa ZIP
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            # JSON-y
            for f in tmp_files:
                z.write(f, arcname=f.name)
            # Wykresy
            for p in plot_files:
                z.write(p, arcname=f"plots/{p.name}")
            # Modele (jeśli są ścieżki do plików)
            for name, model_obj in (models or {}).items():
                if isinstance(model_obj, (str, os.PathLike)):
                    mp = Path(model_obj)
                    if mp.exists() and mp.is_file():
                        z.write(mp, arcname=f"models/{mp.name}")

        return str(zip_path)


# =========================
# Helpers
# =========================

def _json_default(obj: Any) -> Any:
    try:
        if isinstance(obj, (Path,)):
            return str(obj)
        if isinstance(obj, (np.generic,)):
            return obj.item()
        if isinstance(obj, (pd.DataFrame,)):
            return {"columns": obj.columns.tolist(), "shape": obj.shape}
        if isinstance(obj, (pd.Series,)):
            return obj.to_dict()
        return str(obj)
    except Exception:
        return "<unserializable>"


__all__ = ["ExplainService"]
