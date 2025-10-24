"""
ai_integration.py
Docstring (PL): Generuje rekomendacje biznesowe. Domyślnie używa deterministycznego fallbacku,
który opiera się na metrykach i typie problemu. Integracje LLM będą włączane flagą (gated).
"""
from __future__ import annotations
from typing import Dict, Any, List

def deterministic_recommendations(problem_type: str, best_model: str, metrics: Dict[str, float]) -> List[str]:
    rec = []
    if problem_type == "classification":
        auc = float(metrics.get("ROC_AUC", "nan") or "nan")
        f1 = float(metrics.get("F1", "nan") or "nan")
        acc = float(metrics.get("Accuracy", "nan") or "nan")
        ap = float(metrics.get("AveragePrecision", "nan") or "nan")
        rec.append(f"Najlepszy model: **{best_model}**. Skup się na progu decyzyjnym i kalibracji, aby poprawić F1/AP.")
        rec.append("Sprawdź balans klas i rozważ oversampling/undersampling, jeśli klasy są niezrównoważone.")
        if auc and auc < 0.75:
            rec.append("AUC < 0.75 — rozważ inżynierię cech i/lub model boostingowy.")
        if ap and ap < 0.5:
            rec.append("Average Precision niska — rozważ regularyzację i optymalizację progu.")
    else:
        rmse = float(metrics.get("RMSE", "nan") or "nan")
        mae = float(metrics.get("MAE", "nan") or "nan")
        r2 = float(metrics.get("R2", "nan") or "nan")
        rec.append(f"Najlepszy model: **{best_model}**. Przeanalizuj rozkład residuali i outliery.")
        if r2 and r2 < 0.6:
            rec.append("R² < 0.6 — rozważ nieliniowe modele (RF/GBM) lub bogatsze cechy.")
        if rmse and mae and rmse > (1.5 * mae):
            rec.append("RMSE >> MAE — wskazuje na silne outliery; rozważ robust scalers lub trimming.")
    rec.append("Zweryfikuj drift danych i jakość cech przed wdrożeniem.")
    rec.append("Zapisz konfigurację i artefakty — umożliwi replikację (ZIP/PDF w aplikacji).")
    return rec