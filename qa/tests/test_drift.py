from __future__ import annotations
import pandas as pd
import numpy as np
from backend.drift_detection import drift_report

def test_drift_simple():
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"x": rng.normal(0,1,500), "y": rng.normal(0,1,500)})
    b = pd.DataFrame({"x": rng.normal(0.5,1,500), "y": rng.normal(0,1,500)})
    rep = drift_report(a, b)
    assert "x" in rep