"""
drift_detection.py
Docstring (PL): Wykrywanie driftu: PSI, JS divergence, KS test dla kolumn liczbowych (względem baseline).
"""
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    # PSI dla histogramów
    e_counts, edges = np.histogram(expected, bins=bins)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_prop = (e_counts + 1e-9) / (np.sum(e_counts) + 1e-9)
    a_prop = (a_counts + 1e-9) / (np.sum(a_counts) + 1e-9)
    return float(np.sum((a_prop - e_prop) * np.log(a_prop / e_prop)))

def _js(p: np.ndarray, q: np.ndarray, bins: int = 10) -> float:
    # JS divergence (sym. KL) na histogramach
    p_counts, edges = np.histogram(p, bins=bins)
    q_counts, _ = np.histogram(q, bins=edges)
    p = (p_counts + 1e-9) / (np.sum(p_counts) + 1e-9)
    q = (q_counts + 1e-9) / (np.sum(q_counts) + 1e-9)
    m = 0.5*(p+q)
    kl_pm = np.sum(p*np.log(p/m))
    kl_qm = np.sum(q*np.log(q/m))
    return float(0.5*(kl_pm+kl_qm))

def drift_report(baseline: pd.DataFrame, current: pd.DataFrame, bins: int = 10) -> Dict[str, Any]:
    num_cols = list(set(baseline.select_dtypes(include=[np.number]).columns) & set(current.select_dtypes(include=[np.number]).columns))
    rep = {}
    for c in num_cols:
        b = baseline[c].dropna().values
        a = current[c].dropna().values
        if len(b) < 5 or len(a) < 5:
            continue
        rep[c] = {
            "PSI": _psi(b, a, bins=bins),
            "JS": _js(b, a, bins=bins),
            "KS_pvalue": float(ks_2samp(b, a).pvalue),
        }
    return rep