
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

def calibrate_model(clf, X, y, method: str = "isotonic"):
    cal = CalibratedClassifierCV(base_estimator=clf, method=method, cv=3)
    cal.fit(X, y)
    return cal

def calibration_snapshot(y_true, y_prob, n_bins: int = 10) -> Dict[str, Any]:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist(), "bins": n_bins}
