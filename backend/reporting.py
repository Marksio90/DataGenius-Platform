# -*- coding: utf-8 -*-
import os, datetime, json
from typing import Optional
import pandas as pd

def generate_training_report(artifacts_dir: str, results_df: Optional[pd.DataFrame], best: dict, plan: dict, recs: dict) -> str:
    """Tworzy PDF (jeśli reportlab) lub fallback .md z podsumowaniem treningu."""
    pdf_path = os.path.join(artifacts_dir, "report.pdf")
    md_path = os.path.join(artifacts_dir, "report.md")
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4
        y = height - 2*cm
        def line(txt):
            nonlocal y
            c.drawString(2*cm, y, str(txt)[:110])
            y -= 0.6*cm
            if y < 2*cm:
                c.showPage(); y = height - 2*cm
        line(f"TMIV — Raport treningu ({datetime.datetime.utcnow().isoformat()} UTC)")
        line(" ")
        line("Najlepszy model: {}".format(best.get('model') if best else '—'))
        line("Metryka: {} = {}".format(best.get('metric'), round(best.get('cv_mean'),5) if best else '—'))
        line(" ")
        line("Rekomendacje AI (skrócone):")
        line("• Problem: {}".format(recs.get('problem','?')))
        line("• CV folds: {}".format(recs.get('cv_folds','?')))
        line("• Metryki: {}".format(", ".join(recs.get('metrics',[]))))
        line("• Modele: {}".format(", ".join([m['name'] for m in recs.get('models',[])])))
        line(" ")
        line("Kroki Data-Prep (plan):")
        for stp in plan.get('steps',[])[:20]:
            line(" - {}: {}".format(stp.get('name'), stp.get('detail')))
        if results_df is not None and not results_df.empty:
            line(" ")
            line("Wyniki CV (Top 10): model | metric | mean")
            top = results_df.head(10).to_dict(orient='records')
            for r in top:
                line(f" - {r['model']} | {r['metric']} | {round(r['cv_mean'],5)}")
        c.showPage(); c.save()
        return pdf_path
    except Exception:
        # Fallback do .md
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# TMIV — Raport treningu ({datetime.datetime.utcnow().isoformat()} UTC)\n\n")
                f.write(f"**Najlepszy model:** {best.get('model') if best else '—'}  \n")
                f.write(f"**Metryka:** {best.get('metric')} = {round(best.get('cv_mean'),5) if best else '—'}\n\n")
                f.write("## Rekomendacje AI\n")
                f.write(f"- Problem: {recs.get('problem','?')}\n")
                f.write(f"- CV folds: {recs.get('cv_folds','?')}\n")
                f.write(f"- Metryki: {recs.get('metrics',[])}\n")
                f.write(f"- Modele: {[m['name'] for m in recs.get('models',[])]}\n\n")
                f.write("## Plan Data-Prep\n")
                for stp in plan.get('steps',[])[:20]:
                    f.write(f"- {stp.get('name')}: {stp.get('detail')}\n")
                if results_df is not None and not results_df.empty:
                    f.write("\n## Wyniki CV (Top 10)\n")
                    top = results_df.head(10).to_dict(orient='records')
                    for r in top:
                        f.write(f"- {r['model']} | {r['metric']} | {round(r['cv_mean'],5)}\n")
            return md_path
        except Exception:
            return md_path
