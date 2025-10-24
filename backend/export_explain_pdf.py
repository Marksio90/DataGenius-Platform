"""
export_explain_pdf.py
Docstring (PL): Generuje raport PDF (ReportLab) z podstawowymi sekcjami: dane, plan, metryki, rekomendacje.
"""
from __future__ import annotations
from typing import Dict, Any, List
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

def _wrap_text(c, text: str, x: int, y: int, max_width: int, line_height: int=14):
    # Prosty wrapper textu do szerokości
    words = text.split()
    line = ""
    for w in words:
        if c.stringWidth(line + (" " if line else "") + w, "Helvetica", 10) < max_width:
            line = (line + " " + w) if line else w
        else:
            c.drawString(x, y, line)
            y -= line_height
            line = w
    if line:
        c.drawString(x, y, line)
        y -= line_height
    return y

def build_pdf(out_path: str, context: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    margin = 2*cm
    x = margin
    y = height - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, "TMIV – Explainability Report")
    y -= 20
    c.setFont("Helvetica", 10)
    y = _wrap_text(c, f"Dataset fingerprint: {context.get('fingerprint','-')}", x, y, width-2*margin)
    y = _wrap_text(c, f"Strategy: {context.get('strategy','-')}", x, y, width-2*margin)
    y = _wrap_text(c, f"Problem type: {context.get('problem_type','-')}", x, y, width-2*margin)

    c.setFont("Helvetica-Bold", 12); y -= 10; c.drawString(x, y, "Plan treningu"); y -= 16
    c.setFont("Helvetica", 10)
    y = _wrap_text(c, f"Target: {context.get('target','-')}", x, y, width-2*margin)
    y = _wrap_text(c, f"CV: {context.get('validation','-')}", x, y, width-2*margin)

    c.setFont("Helvetica-Bold", 12); y -= 10; c.drawString(x, y, "Metryki modeli"); y -= 16
    c.setFont("Helvetica", 10)
    res = context.get("results", {})
    for name, info in res.items():
        line = f"{name}: {info.get('metrics',{})} (status: {info.get('status')})"
        y = _wrap_text(c, line, x, y, width-2*margin)
        if y < 3*cm: c.showPage(); y = height - margin; c.setFont("Helvetica", 10)

    c.setFont("Helvetica-Bold", 12); y -= 10; c.drawString(x, y, "Rekomendacje"); y -= 16
    c.setFont("Helvetica", 10)
    for r in context.get("recommendations", []):
        y = _wrap_text(c, f"- {r}", x, y, width-2*margin)
        if y < 3*cm: c.showPage(); y = height - margin; c.setFont("Helvetica", 10)

    c.setFont("Helvetica", 8); y = 2*cm
    c.drawString(x, y, f"Seed: {context.get('seed','-')} • Generated: {context.get('generated_at','-')}")
    c.showPage(); c.save()
    return out_path