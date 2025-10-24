
from __future__ import annotations
import numpy as np
from typing import Dict, List
from sklearn.metrics import roc_curve
from .thresholding import optimize_threshold_by_cost, metrics_at_threshold

def grid_cost_thresholds(y_true, y_prob, fp_vals: List[float], fn_vals: List[float]) -> List[Dict[str, float]]:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    out = []
    for cfp in fp_vals:
        for cfn in fn_vals:
            best = optimize_threshold_by_cost(thr, tpr, fpr, {"FP": cfp, "FN": cfn})
            m = metrics_at_threshold(y_true, y_prob, best["threshold"])
            out.append({
                "cost_fp": float(cfp),
                "cost_fn": float(cfn),
                "threshold": float(best["threshold"]),
                "expected_cost": float(best["expected_cost"]),
                **{k: float(v) for k,v in m.items() if isinstance(v,(int,float))}
            })
    # sort by expected cost asc
    out.sort(key=lambda r: r["expected_cost"])
    return out
