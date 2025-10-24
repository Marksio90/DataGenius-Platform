
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple

def _hist(a: np.ndarray, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.zeros(bins), np.linspace(0,1,bins+1)
    hist, edges = np.histogram(a, bins=bins, range=(np.nanmin(a), np.nanmax(a) if np.nanmax(a)>np.nanmin(a) else np.nanmin(a)+1e-9))
    p = hist / max(hist.sum(), 1)
    return p, edges

def population_stability_index(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    p, edges = _hist(a, bins)
    q, _ = np.histogram(b[~np.isnan(b)], bins=edges); q = q / max(q.sum(), 1)
    eps = 1e-12
    psi = float(np.nansum((p - q) * np.log((p + eps) / (q + eps))))
    return psi

def kolmogorov_smirnov(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a[~np.isnan(a)]); b = np.sort(b[~np.isnan(b)])
    if a.size==0 or b.size==0:
        return 0.0
    # Empirical CDFs
    allv = np.concatenate([a,b]); allv.sort()
    def ecdf(x, data):
        return np.searchsorted(data, x, side='right') / data.size
    d = np.max(np.abs([ecdf(x,a)-ecdf(x,b) for x in allv]))
    return float(d)

def jensen_shannon(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    p, edges = _hist(a, bins)
    q, _ = np.histogram(b[~np.isnan(b)], bins=edges); q = q / max(q.sum(), 1)
    m = 0.5*(p+q); eps=1e-12
    kl_pm = np.nansum(p * np.log((p+eps)/(m+eps)))
    kl_qm = np.nansum(q * np.log((q+eps)/(m+eps)))
    js = 0.5*(kl_pm + kl_qm)
    return float(js)
