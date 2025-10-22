from __future__ import annotations

import os
from datetime import datetime
from typing import List, Tuple


def _natural_key(s: str) -> List:
    """Klucz do 'natural sort' (np. img2.png < img10.png)."""
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]


def _wrap_lines(lines: List[str], max_chars: int = 90) -> List[str]:
    """Proste łamanie linii tekstu (bez zależności od platypus)."""
    out: List[str] = []
    for ln in lines:
        ln = ln.replace("\t", "    ")
        while len(ln) > max_chars:
            # szukaj spacji w okolicach limitu
            cut = ln.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            out.append(ln[:cut])
            ln = ln[cut:].lstrip()
        out.append(ln)
    return out


def _collect_pngs(plots_dir: str) -> List[str]:
    if not os.path.isdir(plots_dir):
        return []
    files = [f for f in os.listdir(plots_dir) if f.lower().endswith(".png")]
    files.sort(key=_natural_key)
    return [os.path.join(plots_dir, f) for f in files]


def export_explainability_pdf(
    out_path: str = "artifacts/Explainability_Report.pdf",
    plots_dir: str = "artifacts/plots",
    md_path: str = "artifacts/TRAINING_REPORT.md",
    *,
    title: str = "Explainability Report",
    author: str | None = None,
) -> str:
    """
    Składa prosty PDF (ReportLab) z:
      • strony tytułowej (tytuł, data, autor),
      • opcjonalnego skrótu z pliku Markdown,
      • kolejnych stron z obrazkami PNG (zachowując proporcje),
      • numeracją stron.
    Zwraca ścieżkę do wygenerowanego PDF-a.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        from reportlab.lib.units import cm
    except Exception as e:
        # Zachowujemy dotychczasowe API (jasny komunikat)
        raise RuntimeError(f"reportlab is required: {e}")

    # Katalog wyjściowy
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Inicjalizacja PDF
    c = canvas.Canvas(out_path, pagesize=A4)
    W, H = A4

    # Metadane dokumentu
    try:
        c.setTitle(title)
        if author:
            c.setAuthor(author)
        c.setSubject("Model explainability and training summary")
        c.setCreator("TMIV — Ultra Pro")
    except Exception:
        pass

    # ===== Strona tytułowa =====
    c.setFont("Helvetica-Bold", 20)
    c.drawString(2 * cm, H - 2.5 * cm, title)

    c.setFont("Helvetica", 11)
    c.drawString(2 * cm, H - 3.2 * cm, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if author:
        c.drawString(2 * cm, H - 3.9 * cm, f"Author: {author}")

    # Markdown (pierwsze ~40 linii, łamanie do 90 znaków)
    y = H - 5.0 * cm
    if os.path.exists(md_path):
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                raw_lines = f.read().splitlines()[:40]
            lines = _wrap_lines(raw_lines, max_chars=90)
            c.setFont("Helvetica", 9)
            for ln in lines:
                if y < 2.5 * cm:  # nowa strona jeśli brakuje miejsca
                    _draw_footer_page_number(c, W, H)
                    c.showPage()
                    y = H - 2.5 * cm
                    c.setFont("Helvetica", 9)
                c.drawString(2 * cm, y, ln[:150])
                y -= 0.45 * cm
        except Exception:
            # brak markdown nie powinien zatrzymać generowania PDF
            pass

    _draw_footer_page_number(c, W, H)
    c.showPage()

    # ===== Obrazki =====
    files = _collect_pngs(plots_dir)
    if not files:
        # Brak obrazków – zostaw stronę informacyjną
        c.setFont("Helvetica", 12)
        c.drawString(2 * cm, H - 2.5 * cm, "No plots found.")
        c.drawString(2 * cm, H - 3.2 * cm, f"Looked in: {os.path.abspath(plots_dir)}")
        _draw_footer_page_number(c, W, H)
        c.showPage()
    else:
        for img_path in files:
            try:
                img = ImageReader(img_path)
                iw, ih = img.getSize()
                # Maksymalny obszar (ramki)
                max_w, max_h = W - 2 * cm, H - 3 * cm  # 1 cm margines z każdej strony + miejsce na tytuł
                scale = min(max_w / iw, max_h / ih) if iw > 0 and ih > 0 else 1.0
                w, h = iw * scale, ih * scale

                # Nagłówek z nazwą pliku (ucięty jeśli za długi)
                c.setFont("Helvetica", 10)
                base = os.path.basename(img_path)
                label = base if len(base) <= 90 else base[:87] + "..."
                c.drawString(2 * cm, H - 1.5 * cm, label)

                # Rysuj obraz wyśrodkowany
                x = (W - w) / 2
                y = (H - h) / 2 - 0.3 * cm
                c.drawImage(img, x, y, width=w, height=h, preserveAspectRatio=True, anchor='c')

                _draw_footer_page_number(c, W, H)
                c.showPage()
            except Exception:
                # uszkodzony plik/format – pomiń
                continue

    # Zapisz
    c.save()
    return out_path


def _draw_footer_page_number(c, W, H, *, font="Helvetica", size=8) -> None:
    """Stopka z numerem strony (rysowana przed showPage())."""
    try:
        from reportlab.lib.units import cm
        c.setFont(font, size)
        # Nie znamy numeru strony bezpośrednio, ale ReportLab wstawi poprawnie inkrementowaną stronę.
        # Tu tylko subtelna kreska i podpis.
        c.setLineWidth(0.3)
        c.line(1.5 * cm, 1.7 * cm, W - 1.5 * cm, 1.7 * cm)
        c.drawRightString(W - 1.5 * cm, 1.2 * cm, f"Page {c.getPageNumber()}")
    except Exception:
        pass
