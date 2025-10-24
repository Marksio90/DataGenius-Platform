
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict

def group_metrics(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Wylicza metryki per-grupa: accuracy, TPR, FPR."""
    res: Dict[str, Dict[str, float]] = {}
    uniq = pd.Series(group).fillna("NA").astype(str).unique().tolist()
    for g in uniq:
        mask = (pd.Series(group).astype(str) == g).to_numpy()
        yt = y_true[mask]; yp = y_pred[mask]
        if yt.size == 0:
            continue
        acc = float((yt==yp).mean())
        # TPR i FPR dla binarnej klasyfikacji
        tp = int(((yp==1)&(yt==1)).sum())
        fn = int(((yp==0)&(yt==1)).sum())
        fp = int(((yp==1)&(yt==0)).sum())
        tn = int(((yp==0)&(yt==0)).sum())
        tpr = tp / max(tp+fn,1)
        fpr = fp / max(fp+tn,1)
        res[str(g)] = {"accuracy": acc, "tpr": float(tpr), "fpr": float(fpr), "n": int(yt.size)}
    return res
