# backend/export_explain_pdf.py
"""
Explainability PDF exporter for TMIV – Advanced ML Platform.

Creates a clean, business-friendly PDF with:
- cover (title, subtitle, meta),
- model metrics table,
- top-k feature importance,
- optional charts (ROC/PR, calibration, SHAP, etc.),
- optional notes / recommendations.

Dependencies: reportlab (required), pandas (optional).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Optional pandas
try:  # pragma: no cover
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# ReportLab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# =========================
# Helpers & dataclasses
# =========================


@dataclass
class PdfMeta:
    title: str = "Explainability Report"
    model_name: str | None = None
    dataset_name: str | None = None
    created_by: str | None = None
    problem_type: str | None = None
    run_id: str | None = None


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def _ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _fmt(v: Any) -> str:
    try:
        if v is None:
            return "-"
        if isinstance(v, float):
            # adaptive precision
            if abs(v) >= 1000 or v == 0:
                return f"{v:.2f}"
            if abs(v) >= 1:
                return f"{v:.3f}"
            return f"{v:.4f}"
        return str(v)
    except Exception:
        return str(v)


def _page_size_from_name(name: str | None):
    n = (name or "A4").upper()
    if n == "LETTER":
        return LETTER
    return A4


def _styles():
    ss = getSampleStyleSheet()
    title = ParagraphStyle(
        "TMIVTitle",
        parent=ss["Title"],
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#EAEAF2"),
        spaceAfter=12,
    )
    h1 = ParagraphStyle(
        "TMIVH1",
        parent=ss["Heading1"],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#222222"),
        spaceBefore=12,
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "TMIVH2",
        parent=ss["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#222222"),
        spaceBefore=10,
        spaceAfter=6,
    )
    p = ParagraphStyle(
        "TMIVBody",
        parent=ss["BodyText"],
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1a1b1e"),
    )
    caption = ParagraphStyle(
        "TMIVCaption",
        parent=ss["BodyText"],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#666"),
        alignment=1,  # center
        spaceBefore=4,
        spaceAfter=4,
    )
    return {"title": title, "h1": h1, "h2": h2, "p": p, "caption": caption}


def _kv_table(data: Mapping[str, Any]) -> Table:
    rows = [["Metric", "Value"]]
    for k, v in data.items():
        rows.append([str(k), _fmt(v)])
    tbl = Table(rows, hAlign="LEFT", colWidths=[6 * cm, 8 * cm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16181D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F2F3F7")]),
            ]
        )
    )
    return tbl


def _fi_table(fi: Any, top_k: int = 25) -> Table:
    # Accept pd.DataFrame with columns ["feature","importance"...] OR iterable of dicts/tuples
    rows = [["Rank", "Feature", "Importance"]]
    parsed: list[tuple[str, float]] = []

    try:
        if pd is not None and isinstance(fi, pd.DataFrame):
            cols = [c.lower() for c in fi.columns]
            if "feature" in cols and ("importance" in cols or "importance_mean" in cols):
                fcol = fi.columns[cols.index("feature")]
                if "importance" in cols:
                    icol = fi.columns[cols.index("importance")]
                else:
                    icol = fi.columns[cols.index("importance_mean")]
                tmp = fi[[fcol, icol]].dropna()
                tmp = tmp.sort_values(icol, ascending=False).head(top_k)
                parsed = [(str(r[fcol]), float(r[icol])) for _, r in tmp.iterrows()]
        if not parsed and isinstance(fi, Iterable):
            for item in fi:
                if isinstance(item, Mapping) and "feature" in item and ("importance" in item or "importance_mean" in item):
                    parsed.append((str(item["feature"]), float(item.get("importance", item.get("importance_mean", 0.0)))))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    parsed.append((str(item[0]), float(item[1])))
        parsed = parsed[:top_k]
    except Exception:
        parsed = []

    rank = 1
    for feat, imp in parsed:
        rows.append([str(rank), feat, _fmt(imp)])
        rank += 1

    if len(rows) == 1:
        rows.append(["–", "No feature importance available", "–"])

    tbl = Table(rows, hAlign="LEFT", colWidths=[2 * cm, 9 * cm, 3 * cm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16181D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("ALIGN", (0, 1), (0, -1), "RIGHT"),
                ("ALIGN", (2, 1), (2, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F2F3F7")]),
            ]
        )
    )
    return tbl


def _image_flowable(path: str | os.PathLike[str], max_width: float) -> Image | None:
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        img = ImageReader(str(p))
        w, h = img.getSize()
        scale = min(1.0, max_width / float(w))
        iw = w * scale
        ih = h * scale
        im = Image(str(p), width=iw, height=ih)
        return im
    except Exception:
        return None


def _charts_section(charts: Mapping[str, str] | Sequence[tuple[str, str]], max_width: float) -> list:
    flows: list = []
    if charts is None:
        return flows
    if isinstance(charts, Mapping):
        items = list(charts.items())
    else:
        items = list(charts)

    st = _styles()
    for title, path in items:
        im = _image_flowable(path, max_width)
        if im is None:
            continue
        flows.append(Paragraph(f"{title}", st["h2"]))
        flows.append(im)
        flows.append(Spacer(1, 0.2 * inch))
    return flows


def _page_footer(canvas: Canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.gray)
    page = canvas.getPageNumber()
    text = f"TMIV – Explainability Report • Page {page}"
    canvas.drawRightString(doc.pagesize[0] - 1.5 * cm, 1.0 * cm, text)
    canvas.restoreState()


# =========================
# Main API
# =========================


def build_pdf_report(
    output_path: str,
    metrics: Mapping[str, Any] | None = None,
    feature_importance: Any | None = None,
    charts: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
    *,
    title: str = "Explainability Report",
    model_name: str | None = None,
    dataset_name: str | None = None,
    problem_type: str | None = None,
    created_by: str | None = None,
    run_id: str | None = None,
    notes: Sequence[str] | str | None = None,
    page_size: str = "A4",
) -> str:
    """
    Build a standalone PDF explainability report.

    Parameters
    ----------
    output_path : str
        Absolute or relative path to the PDF file to be created.
    metrics : Mapping[str, Any] | None
        Model metrics (key->value).
    feature_importance : DataFrame | list | dict | None
        Any structure understood by _fi_table (see docstring).
    charts : Mapping[str, str] | Sequence[tuple[str,str]] | None
        Mapping or list of (title, image_path). Non-existing paths are skipped.
    title, model_name, dataset_name, problem_type, created_by, run_id : meta
    notes : list[str] | str | None
        Optional bullet points (recommendations/observations).
    page_size : "A4" | "LETTER"

    Returns
    -------
    str : absolute path to the generated PDF.
    """
    _ensure_dir(output_path)
    pagesize = _page_size_from_name(page_size)
    st = _styles()

    # Document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=pagesize,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.5 * cm,
        title=title,
        author=created_by or "TMIV",
    )

    story: list = []

    # ---- Cover / header block ----
    story.append(Paragraph(title, st["title"]))

    subtitle_bits = []
    if model_name:
        subtitle_bits.append(f"<b>Model:</b> {model_name}")
    if dataset_name:
        subtitle_bits.append(f"<b>Dataset:</b> {dataset_name}")
    if problem_type:
        subtitle_bits.append(f"<b>Problem:</b> {problem_type}")
    if run_id:
        subtitle_bits.append(f"<b>Run ID:</b> {run_id}")
    subtitle_bits.append(f"<b>Generated:</b> {_now_iso()}")
    if created_by:
        subtitle_bits.append(f"<b>By:</b> {created_by}")

    story.append(Paragraph(" &nbsp;•&nbsp; ".join(subtitle_bits), st["p"]))
    story.append(Spacer(1, 0.35 * inch))

    # ---- Metrics ----
    story.append(Paragraph("Model Metrics", st["h1"]))
    if metrics:
        story.append(_kv_table(metrics))
    else:
        story.append(Paragraph("No metrics provided.", st["p"]))
    story.append(Spacer(1, 0.25 * inch))

    # ---- Feature Importance ----
    story.append(Paragraph("Feature Importance (Top)", st["h1"]))
    story.append(_fi_table(feature_importance, top_k=25))
    story.append(Spacer(1, 0.25 * inch))

    # ---- Charts (if any) ----
    flows = _charts_section(charts or {}, max_width=doc.width)
    if flows:
        story.append(Paragraph("Charts", st["h1"]))
        story.extend(flows)

    # ---- Notes / Recommendations ----
    if notes:
        story.append(PageBreak())
        story.append(Paragraph("Recommendations & Notes", st["h1"]))
        if isinstance(notes, str):
            story.append(Paragraph(notes, st["p"]))
        else:
            for n in notes:
                story.append(Paragraph(f"• {n}", st["p"]))
                story.append(Spacer(1, 0.05 * inch))

    # Build
    doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)

    return str(Path(output_path).resolve())


# =========================
# Minimal CLI (manual test)
# =========================

if __name__ == "__main__":  # pragma: no cover
    # Quick demo file
    out = build_pdf_report(
        "exports/demo_explain.pdf",
        metrics={"ROC AUC": 0.8921, "F1": 0.7812, "Accuracy": 0.8435},
        feature_importance=[("age", 0.21), ("income", 0.18), ("region", 0.11)],
        charts=[],  # or {"ROC Curve": "path/to/roc.png"}
        title="TMIV – Explainability Report",
        model_name="LightGBM Classifier",
        dataset_name="avocado.csv",
        problem_type="classification",
        created_by="TMIV",
        run_id="run_demo_001",
        notes=[
            "Consider class weights or threshold tuning to improve F1.",
            "Add SHAP analysis for top features.",
            "Monitor drift monthly (PSI on key inputs).",
        ],
    )
    print("PDF written to:", out)
