
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any

def local_whatif(pipeline, sample: pd.Series, changes: Dict[str, Any]) -> Dict[str, Any]:
    """Symulacja lokalna: modyfikuje wartości kolumn i zwraca proby przed/po (jeśli dostępne) lub predykcje."""
    x0 = sample.to_frame().T.copy()
    for k,v in changes.items():
        if k in x0.columns:
            x0.loc[:,k] = v
    out: Dict[str, Any] = {}
    try:
        proba0 = pipeline.predict_proba(sample.to_frame().T)[:,1][0]
        proba1 = pipeline.predict_proba(x0)[:,1][0]
        out.update({"y_prob_before": float(proba0), "y_prob_after": float(proba1), "delta": float(proba1-proba0)})
    except Exception:
        y0 = float(pipeline.predict(sample.to_frame().T)[0])
        y1 = float(pipeline.predict(x0)[0])
        out.update({"y_before": y0, "y_after": y1, "delta": float(y1-y0)})
    out["applied_changes"] = changes
    return out
