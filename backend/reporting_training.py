from __future__ import annotations

from backend.safe_utils import truthy_df_safe

# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional
from datetime import datetime
import os, json

def _get(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _safe_number(x, ndigits: int = 6) -> str:
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)

_LOWER_IS_BETTER = {"rmse", "mae", "mape", "medae", "rmsle", "logloss"}

def _choose_primary(df_cols: List[str], requested: Optional[str], problem_hint: Optional[str]) -> Optional[str]:
    """Wybiera kolumnę do sortowania leaderboardu."""
    if requested in df_cols:
        return requested
    # sensowne fallbacki
    cls_pref = ["f1", "f1_weighted", "accuracy", "auc", "roc_auc_ovr", "average_precision"]
    reg_pref = ["rmse", "mae", "r2", "mape"]
    pref = cls_pref if (problem_hint or "").lower().startswith("class") else reg_pref
    for m in pref:
        if m in df_cols:
            return m
    return None

def _topn_leaderboard(results: Dict[str, Any], n: int = 10) -> List[Dict[str, Any]]:
    lb = results.get("leaderboard") or results.get("models") or results.get("candidates") or results.get("model_results")
    if lb is None:
        return []
    try:
        import pandas as pd
    except Exception:
        # brak pandas – jeśli to lista słowników, zwróć top n
        return lb[:n] if isinstance(lb, list) else []

    try:
        if isinstance(lb, pd.DataFrame):
            df = lb.copy()
        else:
            df = pd.DataFrame(lb)
        if df is None or df.empty:
            return []
        # wybór metryki sortowania
        training_plan = results.get('training_plan', {}) or {}
        primary_req = training_plan.get('metrics_primary') or results.get('primary_metric')
        problem_hint = results.get('problem_type') or training_plan.get('problem_type') or ""
        primary = _choose_primary(list(df.columns), primary_req, problem_hint)
        if primary:
            asc = primary in _LOWER_IS_BETTER
            df = df.sort_values(primary, ascending=asc, kind="mergesort")  # stabilne
        # kolumny do wyświetlenia
        pref = ['name','model','algorithm','metric','rmse','mae','r2','mape','accuracy','f1','f1_weighted','auc','roc_auc_ovr','average_precision']
        cols = [c for c in pref if c in df.columns]
        if not cols:
            cols = df.columns.tolist()
        return df[cols].head(max(0, int(n))).to_dict(orient='records')
    except Exception:
        # fallback: przytnij listę
        return lb[:n] if isinstance(lb, list) else []

def _extract_context(results: Dict[str, Any]) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    tp = results.get('problem_type') or _get(results.get('training_plan', {}) or {}, 'problem_type', default=None)
    ctx['problem_type'] = tp or 'unknown'
    ctx['target'] = results.get('target') or results.get('target_column') or _get(results, 'y_name', default='(unknown)')
    da = results.get('data_analysis') or {}
    ctx['n_rows'] = da.get('n_rows') or _get(results, 'n_rows', default='?')
    ctx['n_cols'] = da.get('n_cols') or _get(results, 'n_cols', default='?')
    tpobj = results.get('training_plan') or {}
    ctx['cv'] = tpobj.get('cv') or {}
    ctx['primary_metric'] = tpobj.get('metrics_primary') or results.get('primary_metric')
    ctx['extra_metrics'] = tpobj.get('metrics_extra') or results.get('metrics', {})
    ctx['best_model_name'] = _get(results, 'best_model_name', 'best_model_class', 'best_model', default='(unknown)')
    m = results.get('metrics') or results.get('best_metrics') or {}
    ctx['metrics'] = m
    ctx['artifacts_path'] = results.get('artifacts_path') or 'artifacts'
    plots: List[str] = []
    plot_dir = os.path.join(ctx['artifacts_path'], 'plots')
    if os.path.isdir(plot_dir):
        files = sorted([f for f in os.listdir(plot_dir) if f.lower().endswith('.png')])
        plots = [os.path.join(ctx['artifacts_path'], 'plots', f) for f in files]
    ctx['plots'] = plots
    ctx['leaderboard'] = _topn_leaderboard(results, n=10)
    ctx['warnings'] = results.get('warnings') or []
    try:
        ctx['fit_failures'] = int(results.get('fit_failures', 0))
    except Exception:
        ctx['fit_failures'] = 0
    return ctx

def build_training_report_md(results: Dict[str, Any], top_n: int = 10, verbose: bool = False) -> str:
    ctx = _extract_context(results)
    lines: List[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines += ["# TRAINING REPORT", "", f"_Generated: {now}_", ""]
    lines += ["## Summary"]
    lines += [f"**Problem:** {ctx['problem_type']}  ", f"**Target:** {ctx['target']}  ", f"**Rows / Cols:** {ctx['n_rows']} / {ctx['n_cols']}", ""]
    lines += ["## Config"]
    cv = ctx['cv'] or {}
    cv_line = ", ".join([f"{k}={v}" for k,v in cv.items()]) if cv else "—"
    lines += [f"**CV:** {cv_line}  ", f"**Primary metric:** {ctx['primary_metric'] or '—'}", ""]
    lines += ["## Best model", f"**Name:** {ctx['best_model_name']}", ""]
    met = ctx['metrics'] or {}
    if met:
        lines += ["## Metrics"]
        for k, v in met.items():
            # liczby → format, reszta → str
            try:
                lines.append(f"- **{k}**: {_safe_number(v)}")
            except Exception:
                lines.append(f"- **{k}**: {v}")
        lines.append("")
    if ctx['leaderboard']:
        lines += [f"## Leaderboard (Top {min(top_n, len(ctx['leaderboard']))})", ""]
        keys = list(ctx['leaderboard'][0].keys())
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "|".join(["---" for _ in keys]) + "|")
        for row in ctx['leaderboard'][:top_n]:
            vals = [str(row.get(k, "")) for k in keys]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
    if ctx['plots']:
        lines += ["## Plots", ""]
        for p in ctx['plots']:
            lines.append(f"- {p}")
        lines.append("")
    any_notes = bool(ctx['warnings'] or ctx['fit_failures'])
    if any_notes:
        lines += ["## Notes"]
        if ctx['fit_failures']:
            lines.append(f"- Fit failures: {ctx['fit_failures']}")
        for w in ctx['warnings']:
            lines.append(f"- {w}")
        lines.append("")
    lines += [f"**Artifacts path:** `{ctx['artifacts_path']}`"]
    if bool(verbose) and results.get('training_plan'):
        lines += ["", "<details><summary>Training plan (JSON)</summary>", "", "```json", json.dumps(results['training_plan'], indent=2, ensure_ascii=False), "```", "", "</details>"]
    return "\n".join(lines)

def save_training_report(md: str, path: str = "artifacts/TRAINING_REPORT.md") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path
