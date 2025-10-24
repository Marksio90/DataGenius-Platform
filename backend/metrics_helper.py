"""
metrics_helper.py
Docstring (PL): Pomocnicze funkcje do normalizacji metryk oraz przygotowania danych do wykresu radarowego.
"""
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np

# Definicje: które metryki "mniej=lepiej"
LESS_IS_BETTER = {"RMSE", "MAE"}

def normalize_metrics(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Docstring (PL): Normalizacja metryk do [0,1] z uwzględnieniem kierunku optymalizacji.
    Wejście: results[model] = {"metrics": {...}}
    Wyjście: norm[metric][model] = value in [0,1]
    """
    # Zbierz unikalne metryki
    metrics = set()
    for mres in results.values():
        for k in mres.get("metrics", {}).keys():
            metrics.add(k)
    metrics = sorted(metrics)

    # Zbuduj macierze
    raw = {m: {} for m in metrics}
    for model, info in results.items():
        for m in metrics:
            v = info.get("metrics", {}).get(m, np.nan)
            raw[m][model] = float(v) if v is not None else np.nan

    # Normalizacja min-max z obsługą less-is-better
    norm = {m: {} for m in metrics}
    for m in metrics:
        vals = np.array([v for v in raw[m].values()], dtype=float)
        # ignoruj NaN w skali
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            for model in raw[m]:
                norm[m][model] = 0.0
            continue
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        rng = vmax - vmin if vmax > vmin else 1.0
        for model, v in raw[m].items():
            if not np.isfinite(v):
                nv = 0.0
            else:
                nv = (v - vmin) / rng
            # Odwrócenie jeśli mniej=lepiej
            if m in LESS_IS_BETTER:
                nv = 1.0 - nv
            norm[m][model] = float(max(0.0, min(1.0, nv)))
    return norm

def radar_series(norm: Dict[str, Dict[str, float]], models: List[str]) -> Dict[str, List[float]]:
    """
    Docstring (PL): Zwraca wektory radarowe dla poszczególnych modeli w kolejności metryk.
    """
    metrics_order = list(norm.keys())
    series = {}
    for model in models:
        series[model] = [norm[m].get(model, 0.0) for m in metrics_order]
    return {"metrics": metrics_order, "series": series}