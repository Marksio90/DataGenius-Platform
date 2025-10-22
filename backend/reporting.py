from __future__ import annotations
# -*- coding: utf-8 -*-

# backend/report_generator.py

import os
import datetime
import json
from typing import Optional, List

import pandas as pd


def _fmt_float(x, ndigits: int = 5) -> str:
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return "—"


def _wrap_lines(text: str, width: int = 96) -> List[str]:
    import textwrap
    return textwrap.wrap(str(text), width=width, replace_whitespace=False)


def generate_training_report(
    artifacts_dir: str,
    results_df: Optional[pd.DataFrame],
    best: dict | None,
    plan: dict | None,
    recs: dict | None
) -> str:
    """
    Tworzy raport PDF (jeśli dostępny reportlab) z podsumowaniem treningu oraz osadza
    wykresy PNG z katalogu artifacts_dir/plots. W razie braku reportlab — fallback do .md.

    Zwraca ścieżkę do stworzonego pliku (PDF lub MD).
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    pdf_path = os.path.join(artifacts_dir, "report.pdf")
    md_path = os.path.join(artifacts_dir, "report.md")

    best = best or {}
    plan = plan or {}
    recs = recs or {}

    # ===== PDF (reportlab) =====
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        from reportlab.lib.utils import ImageReader

        width, height = A4
        margin = 2 * cm
        line_h = 0.6 * cm
        y = height - margin

        c = canvas.Canvas(pdf_path, pagesize=A4)
        c.setTitle("TMIV — Raport treningu")

        def _new_page():
            nonlocal y
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 10)

        def _need(lines: int = 1, extra_px: float = 0.0):
            """Nowa strona, jeśli brakuje miejsca."""
            nonlocal y
            if y - (lines * line_h + extra_px) < margin:
                _new_page()

        def _title(text: str):
            nonlocal y
            c.setFont("Helvetica-Bold", 14)
            _need(2)
            c.drawString(margin, y, str(text))
            y -= line_h * 1.5
            c.setFont("Helvetica", 10)

        def _text_line(text: str):
            nonlocal y
            _need(1)
            c.drawString(margin, y, str(text)[:110])
            y -= line_h

        def _text_block(lines: List[str]):
            for t in lines:
                _text_line(t)

        # Nagłówek
        _title("TMIV — Raport treningu")
        _text_line(f"Data (UTC): {datetime.datetime.utcnow().isoformat()}")

        # Najlepszy model
        _title("Najlepszy wynik")
        _text_line(f"Model: {best.get('model', '—')}")
        if "metric" in best and "cv_mean" in best:
            _text_line(f"Metryka: {best.get('metric')} = {_fmt_float(best.get('cv_mean'))}")
        else:
            _text_line("Metryka: —")

        # Rekomendacje AI
        _title("Rekomendacje AI")
        _text_line(f"Problem: {recs.get('problem', '?')}")
        _text_line(f"CV folds: {recs.get('cv_folds', '?')}")
        metrics_val = recs.get("metrics", [])
        if isinstance(metrics_val, list):
            _text_line(f"Metryki: {', '.join(map(str, metrics_val)) if metrics_val else '—'}")
        else:
            _text_line(f"Metryki: {metrics_val if metrics_val else '—'}")

        try:
            models_list = recs.get("models", [])
            model_names = [m["name"] for m in models_list if isinstance(m, dict) and "name" in m]
            _text_line(f"Modele: {', '.join(model_names) if model_names else '—'}")
        except Exception:
            _text_line("Modele: —")

        # Plan Data-Prep
        _title("Plan Data-Prep")
        steps = plan.get("steps") or []
        if steps:
            for stp in steps[:30]:
                name = (stp or {}).get("name", "step")
                detail = (stp or {}).get("detail", "")
                for ln in _wrap_lines(f"- {name}: {detail}", width=100):
                    _text_line(ln)
        else:
            _text_line("—")

        # Wyniki CV
        _title("Wyniki CV (Top 10)")
        if isinstance(results_df, pd.DataFrame) and not results_df.empty:
            try:
                top = results_df.head(10).to_dict(orient="records")
                header = "model | metric | cv_mean ± cv_std"
                _text_line(header)
                _text_line("-" * len(header))
                for r in top:
                    mean_s = _fmt_float(r.get("cv_mean"))
                    std_s = _fmt_float(r.get("cv_std"))
                    _text_line(f"{r.get('model')} | {r.get('metric')} | {mean_s} ± {std_s}")
            except Exception:
                _text_line("(nie udało się zrenderować tabeli wyników)")
        else:
            _text_line("—")

        # Wykresy
        plots_dir = os.path.join(artifacts_dir, "plots")
        pngs = sorted([p for p in os.listdir(plots_dir) if p.lower().endswith(".png")]) if os.path.isdir(plots_dir) else []

        if pngs:
            _title("Wykresy")
            gap = 0.5 * cm
            col_w = (width - 2 * margin - gap) / 2.0
            max_row_h = 7.0 * cm
            x_positions = [margin, margin + col_w + gap]

            i = 0
            while i < len(pngs):
                # rezerwuj miejsce na nowy wiersz (do 2 obrazów)
                _need(1, extra_px=(max_row_h + 0.8 * cm))
                row_h = 0.0
                for col in range(2):
                    if i >= len(pngs):
                        break
                    pth = os.path.join(plots_dir, pngs[i])
                    try:
                        img = ImageReader(pth)
                        iw, ih = img.getSize()
                        scale = min(col_w / float(iw), max_row_h / float(ih))
                        w, h = iw * scale, ih * scale
                        x = x_positions[col]
                        y_img_bottom = y - h
                        c.drawImage(img, x, y_img_bottom, width=w, height=h, preserveAspectRatio=True, mask="auto")
                        # podpis
                        c.setFont("Helvetica", 9)
                        c.drawString(x, y_img_bottom - 0.35 * cm, pngs[i])
                        c.setFont("Helvetica", 10)
                        row_h = max(row_h, h + 0.6 * cm)
                    except Exception:
                        pass
                    i += 1
                y -= (row_h + 0.5 * cm)

        # Meta (light)
        _title("Meta (light)")
        try:
            meta = {
                "best": {k: v for k, v in (best or {}).items() if k not in ("model_object", "estimator")},
                "plan_keys": list((plan or {}).keys()),
                "recs_keys": list((recs or {}).keys()),
            }
            meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
            for line in meta_json.splitlines():
                for L in _wrap_lines(line, width=100):
                    _text_line(L)
        except Exception:
            _text_line("—")

        # Stopka
        _need(1)
        c.setFont("Helvetica-Oblique", 9)
        _text_line("© TMIV — AutoML Report")
        c.showPage()
        c.save()
        return pdf_path

    except Exception:
        # ===== Fallback: Markdown =====
        try:
            lines: List[str] = []
            lines.append(f"# TMIV — Raport treningu ({datetime.datetime.utcnow().isoformat()} UTC)")
            lines.append("")
            # Best
            lines.append("## Najlepszy wynik")
            lines.append(f"- **Model:** {best.get('model', '—')}")
            if "metric" in best and "cv_mean" in best:
                lines.append(f"- **Metryka:** {best.get('metric')} = {_fmt_float(best.get('cv_mean'))}")
            lines.append("")
            # Recs
            lines.append("## Rekomendacje AI")
            lines.append(f"- Problem: {recs.get('problem','?')}")
            lines.append(f"- CV folds: {recs.get('cv_folds','?')}")
            metrics_val = recs.get("metrics", [])
            if isinstance(metrics_val, list):
                lines.append(f"- Metryki: {', '.join(map(str, metrics_val)) if metrics_val else '—'}")
            else:
                lines.append(f"- Metryki: {metrics_val if metrics_val else '—'}")
            try:
                models_list = recs.get("models", [])
                mnames = [m['name'] for m in models_list if isinstance(m, dict) and 'name' in m]
                lines.append(f"- Modele: {', '.join(mnames) if mnames else '—'}")
            except Exception:
                lines.append("- Modele: —")
            lines.append("")
            # Plan
            lines.append("## Plan Data-Prep")
            steps = plan.get("steps") or []
            if steps:
                for stp in steps[:30]:
                    lines.append(f"- {(stp or {}).get('name', 'step')}: {(stp or {}).get('detail','')}")
            else:
                lines.append("- —")
            lines.append("")
            # Results
            lines.append("## Wyniki CV (Top 10)")
            if isinstance(results_df, pd.DataFrame) and not results_df.empty:
                top = results_df.head(10).to_dict(orient="records")
                lines.append("| model | metric | cv_mean | cv_std |")
                lines.append("|-|-|-:|-:|")
                for r in top:
                    lines.append(f"| {r.get('model')} | {r.get('metric')} | {_fmt_float(r.get('cv_mean'))} | {_fmt_float(r.get('cv_std'))} |")
            else:
                lines.append("- —")
            lines.append("")
            # Plots (MD)
            plots_dir = os.path.join(artifacts_dir, "plots")
            pngs = sorted([p for p in os.listdir(plots_dir) if p.lower().endswith(".png")]) if os.path.isdir(plots_dir) else []
            if pngs:
                lines.append("## Wykresy (PNG)")
                for p in pngs[:24]:
                    lines.append(f"![{p}](plots/{p})")

            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            return md_path
        except Exception:
            return md_path
