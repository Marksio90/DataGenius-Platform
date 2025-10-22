from __future__ import annotations
# -*- coding: utf-8 -*-
"""
Compatibility shim for older imports: `from backend import exporters`.

Zapewnia spójne API eksportu bez potrzeby dodatkowych modułów.
- build_report(results, out_dir="artifacts/last_run") -> str (ścieżka do report.html)
- export_bundle(results, out_dir="artifacts/last_run", zip_name="artifacts.zip") -> Dict[str, str] (mapa ścieżek)

Dodatkowo re-eksportuje helpery (ONNX/PMML/plan/recs/fig) dla zgodności wstecznej.
"""

from backend.safe_utils import truthy_df_safe

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import csv
import zipfile
import datetime as _dt

# --- Opcjonalne importy z model_export_utils (z fallbackami) ---
try:
    from .model_export_utils import (
        export_onnx,
        export_pmml,
        save_plan_json,       # alias do export_plan_json (zgodność legacy)
        export_recs_json,
        save_figure_png,
    )
except Exception:
    # Minimalne stuby, żeby nie wywracać się na imporcie
    def export_onnx(*a, **k): return None
    def export_pmml(*a, **k): return None
    def save_plan_json(plan, out_dir, filename: str = "plan.json"):
        p = Path(out_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"plan": plan}, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(p)
    def export_recs_json(recs, out_dir):
        p = Path(out_dir) / "recs.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(recs or {}, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(p)
    def save_figure_png(fig, out_dir: Path, name: str = "plot.png", dpi: int = 120):
        p = Path(out_dir) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(str(p), dpi=dpi)
        except Exception:
            pass
        return p

# Te funkcje mogły nie być zdefiniowane w Twojej wersji model_export_utils — zapewniamy fallback:
try:
    from .model_export_utils import save_model_joblib, write_html_report, write_md_readme
except Exception:
    def save_model_joblib(model: Any, out_dir: Union[str, Path], filename: str = "model.joblib") -> str:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        meta = out / "model_meta.json"
        meta.write_text(json.dumps({"repr": str(type(model))}, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(meta)
    def write_html_report(html_or_path: str, out_dir: Union[str, Path], filename: str = "report.html") -> str:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        dst = out / filename
        # jeśli podano ścieżkę pliku, skopiuj; inaczej potraktuj jako treść
        src = Path(html_or_path)
        if src.exists():
            dst.write_bytes(src.read_bytes())
        else:
            dst.write_text(html_or_path, encoding="utf-8")
        return str(dst)
    def write_md_readme(md_text: str, out_dir: Union[str, Path], filename: str = "README_RESULTS.md") -> str:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        p = out / filename
        p.write_text(md_text, encoding="utf-8")
        return str(p)

# Przyda się do opcjonalnego CSV z metrykami
def _save_metrics_csv_any(metrics: Union[Dict[str, Any], List[Dict[str, Any]]], out_dir: Union[str, Path], filename: str = "metrics.csv") -> str:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    p = out / filename
    if isinstance(metrics, dict):
        rows = [{"metric": k, "value": v} for k, v in metrics.items()]
        headers = ["metric", "value"]
    else:
        # lista rekordów — spłaszcz klucze
        headers = set()
        for r in metrics:
            headers.update(r.keys())
        headers = list(headers) or ["metric", "value"]
        rows = metrics
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in headers})
    return str(p)

# === build_report ============================================================

def build_report(results: Dict[str, Any], out_dir: Union[str, Path] = "artifacts/last_run") -> str:
    """
    Buduje raport HTML:
      1) próbuje backend.reporting.generate_training_report (jeśli istnieje),
      2) w razie braku — generuje prosty HTML z meta i leaderboardem.
    Zwraca ścieżkę do report.html
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) spróbuj oficjalnego generatora
    try:
        from .reporting import generate_training_report  # type: ignore
        return str(generate_training_report(results, out_dir=str(out_dir)))
    except Exception:
        pass

    # 2) fallback — prosty HTML
    lb: List[Dict[str, Any]] = results.get("leaderboard") or []
    meta: Dict[str, Any] = results.get("meta") or {}
    now = _dt.datetime.now().isoformat()

    # kolumny leaderboardu (zachowaj sensowną kolejność)
    preferred = ["model_name", "metric_name", "metric_value", "model_key"]
    keys = set()
    for row in lb:
        keys.update(row.keys())
    ordered = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]

    html_lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>TMIV Report</title>",
        "<style>body{font-family:Inter,Arial,sans-serif;padding:24px;}table{border-collapse:collapse;}th,td{border:1px solid #ddd;padding:8px;}th{background:#fafafa;}</style>",
        "</head><body>",
        f"<h1>TMIV — Training Report</h1><p><em>Generated: {now}</em></p>",
    ]
    if truthy_df_safe(meta):
        html_lines.append("<h2>Meta</h2><pre>")
        html_lines.append(json.dumps(meta, ensure_ascii=False, indent=2))
        html_lines.append("</pre>")
    if truthy_df_safe(lb):
        html_lines.append("<h2>Leaderboard</h2>")
        html_lines.append("<table><thead><tr><th>rank</th>" + "".join(f"<th>{k}</th>" for k in ordered) + "</tr></thead><tbody>")
        for i, row in enumerate(lb, 1):
            html_lines.append("<tr><td>{}</td>{}</tr>".format(i, "".join(f"<td>{row.get(k,'')}</td>" for k in ordered)))
        html_lines.append("</tbody></table>")
    html_lines.append("</body></html>")
    html = "\n".join(html_lines)

    return write_html_report(html, out_dir, filename="report.html")

# === export_bundle ===========================================================

def export_bundle(
    results: Dict[str, Any],
    out_dir: Union[str, Path] = "artifacts/last_run",
    zip_name: str = "artifacts.zip",
    include_readme: bool = True,
    extra_files: Optional[Dict[str, Union[str, Path]]] = None,
) -> Dict[str, str]:
    """
    Pakuje artefakty jednego runu:
      - plan.json (jeśli `results['plan']` lub `results['config']`),
      - recs.json (jeśli `results['recs']` lub `results['recommendations']`),
      - metrics.csv (jeśli dostępne metryki lub z pierwszego wpisu leaderboardu),
      - report.html (używa build_report),
      - README_RESULTS.md (opcjonalnie, proste podsumowanie),
      - + pliki z `extra_files` (mapa: nazwa->ścieżka).

    Zwraca mapę ścieżek; tworzy ZIP z całego katalogu out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, str] = {}

    # plan.json
    plan = results.get("plan") or results.get("config")
    if plan is not None:
        written["plan_json"] = save_plan_json(plan, out_dir, filename="plan.json")

    # recs.json
    recs = results.get("recs") or results.get("recommendations")
    if recs is not None:
        written["recs_json"] = export_recs_json(recs, out_dir)

    # metrics.csv – spróbuj z results['metrics'] albo z pierwszego leadera
    metrics = results.get("metrics")
    if metrics is None:
        best = (results.get("leaderboard") or [{}])[0]
        # zbuduj minimalny dict jeżeli coś mamy
        if truthy_df_safe(best):
            metrics = {
                str(best.get("metric_name", "metric")): best.get("metric_value", ""),
                "model_name": best.get("model_name", ""),
                "model_key": best.get("model_key", ""),
            }
    if truthy_df_safe(metrics):
        written["metrics_csv"] = _save_metrics_csv_any(metrics, out_dir, filename="metrics.csv")

    # report.html
    try:
        written["report_html"] = build_report(results, out_dir=str(out_dir))
    except Exception as e:
        # nie blokuj całego bundla
        written["report_error"] = str(e)

    # README_RESULTS.md (opcjonalnie)
    if truthy_df_safe(include_readme):
        lb: List[Dict[str, Any]] = results.get("leaderboard") or []
        meta: Dict[str, Any] = results.get("meta") or {}
        lines: List[str] = [
            "# TMIV — Training Results",
            f"_Generated: {_dt.datetime.now().isoformat()}_  ",
            "",
        ]
        if truthy_df_safe(meta):
            lines += ["## Meta", "```json", json.dumps(meta, ensure_ascii=False, indent=2), "```", ""]
        if truthy_df_safe(lb):
            pref = ["model_name", "metric_name", "metric_value", "model_key"]
            keys = set()
            for r in lb: keys.update(r.keys())
            ordered = [k for k in pref if k in keys] + [k for k in keys if k not in pref]
            lines.append("## Leaderboard")
            lines.append("| rank | " + " | ".join(ordered) + " |")
            lines.append("|" + "|".join(["---"]*(len(ordered)+1)) + "|")
            for i, r in enumerate(lb, 1):
                lines.append("| " + " | ".join([str(i)] + [str(r.get(k, "")) for k in ordered]) + " |")
            lines.append("")
        written["readme_md"] = write_md_readme("\n".join(lines), out_dir, filename="README_RESULTS.md")

    # extra pliki do dołączenia (np. wygenerowane figi)
    if truthy_df_safe(extra_files):
        for name, src in (extra_files or {}).items():
            src_p = Path(src)
            if src_p.exists():
                dst = out_dir / name
                dst.write_bytes(src_p.read_bytes())
                written[f"extra::{name}"] = str(dst)

    # ZIP z całego out_dir
    zip_path = out_dir / zip_name
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            if p.is_file() and p.name != zip_name:
                zf.write(p, arcname=p.relative_to(out_dir))
    written["zip"] = str(zip_path)

    return written


# ===== Re-exports (API compat) =====
__all__ = [
    # helpers
    "export_onnx", "export_pmml", "save_plan_json", "export_recs_json", "save_figure_png",
    # high-level
    "build_report", "export_bundle",
]