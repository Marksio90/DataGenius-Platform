
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
import os, json

PAGE_W, PAGE_H = A4

def add_toc(c, entries):
    c.setFont("Helvetica-Bold", 14); c.drawString(72, 800, "Spis treści")
    c.setFont("Helvetica", 11)
    y = 780
    for i,(title,page) in enumerate(entries, start=1):
        c.drawString(80, y, f"{i}. {title} ...... {page}")
        y -= 16
    c.showPage()

def add_section_title(c, title):
    c.setFont("Helvetica-Bold", 16); c.drawString(72, 800, title)
    c.setFont("Helvetica", 10)

def draw_kv(c, x, y, key, val):
    c.drawString(x, y, f"- {key}: {val}")

def main():
    os.makedirs("artifacts/reports", exist_ok=True)
    path = "artifacts/reports/report.pdf"
    c = canvas.Canvas(path, pagesize=A4)

    # Prepare content pages to count
    toc = [("Metryki", 2), ("Wykresy (ROC/PR/CM)", 3), ("Feature Importance", 4), ("Rekomendacje", 5)]

    # Cover
    c.setFont("Helvetica-Bold", 18); c.drawString(72, 800, "TMIV – Explainability Report")
    c.setFont("Helvetica", 11); c.drawString(72, 782, "Wersja: 2.0 Pro  |  Generowane automatycznie")
    c.showPage()

    # TOC
    add_toc(c, toc)

    # Metrics
    add_section_title(c, "Metryki")
    y = 770
    met_path = "artifacts/models/metrics.json"
    if os.path.exists(met_path):
        metrics = json.load(open(met_path, "r", encoding="utf-8"))
        for k,v in metrics.items():
            c.drawString(80, y, f"{k}: {v}")
            y -= 14
    else:
        c.drawString(80, y, "Brak metryk – uruchom trening.")
    c.showPage()

    # Plots
    add_section_title(c, "Wykresy (ROC/PR/CM)")
    y = 760
    for name in ["roc.png", "pr.png", "cm.png"]:
        p = os.path.join("artifacts/plots", name)
        if os.path.exists(p):
            img = ImageReader(p)
            c.drawImage(img, 72, y-220, width=15*cm, height=9*cm, preserveAspectRatio=True, anchor='n')
            y -= 240
    c.showPage()

    # Feature Importance
    add_section_title(c, "Feature Importance")
    fi_csv = "artifacts/plots/feature_importance.csv"
    if os.path.exists(fi_csv):
        import csv
        y = 770
        c.setFont("Helvetica-Bold", 12); c.drawString(72, y, "Top 15 cech"); y-=18; c.setFont("Helvetica", 10)
        with open(fi_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)[:15]
        for r in rows:
            c.drawString(80, y, f"{r['feature']}: {float(r['importance']):.3f}")
            y -= 14
    else:
        c.drawString(80, 770, "Brak danych o ważności cech.")
    c.showPage()

    # Recommendations
    add_section_title(c, "Rekomendacje")
    c.setFont("Helvetica", 11)
    lines = [
        "• Dla klasyfikacji: dobierz próg decyzyjny wg kosztów FP/FN oraz sprawdź kalibrację (Brier).",
        "• Jeśli 1 cecha > 0.4 w FI (drzewa) – ostrzeżenie o over-reliance i możliwy bias.",
        "• Przy niskim R² w regresji – rozważ nieliniowy model i inżynierię cech.",
        "• Monitoruj drift (PSI/KS/JS) i odśwież model przy alarmie PSI ≥ 0.3.",
    ]
    y = 770
    for ln in lines:
        c.drawString(80, y, ln); y -= 18
    c.save()
    print(path)

if __name__ == "__main__":
    main()
