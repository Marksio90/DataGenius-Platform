
from __future__ import annotations
import numpy as np
from typing import Dict

def optimize_threshold_by_youden(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> float:
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx])

def optimize_threshold_by_cost(thresholds: np.ndarray,
                               tpr: np.ndarray,
                               fpr: np.ndarray,
                               cost_matrix: Dict[str, float]) -> Dict[str, float]:
    cost_fp = float(cost_matrix.get("FP", 1.0))
    cost_fn = float(cost_matrix.get("FN", 1.0))
    expected = 0.5 * (fpr * cost_fp + (1.0 - tpr) * cost_fn)
    idx = int(np.argmin(expected))
    return {"threshold": float(thresholds[idx]), "expected_cost": float(expected[idx])}

def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    yhat = (y_prob >= thr).astype(int)
    tp = int(((yhat==1)&(y_true==1)).sum())
    tn = int(((yhat==0)&(y_true==0)).sum())
    fp = int(((yhat==1)&(y_true==0)).sum())
    fn = int(((yhat==0)&(y_true==1)).sum())
    acc = (tp+tn)/max(len(y_true),1)
    prec = tp/max(tp+fp,1) if (tp+fp)>0 else 0.0
    rec = tp/max(tp+fn,1) if (tp+fn)>0 else 0.0
    f1 = 2*prec*rec/max(prec+rec,1e-12) if (prec+rec)>0 else 0.0
    return {"tp":tp,"tn":tn,"fp":fp,"fn":fn,"accuracy":acc,"precision":prec,"recall":rec,"f1":f1}
