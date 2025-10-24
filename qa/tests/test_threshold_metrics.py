
import numpy as np
from backend.thresholding import metrics_at_threshold

def test_metrics_at_threshold_shape():
    y = np.array([0,1,0,1,1,0])
    p = np.array([0.1,0.9,0.2,0.8,0.4,0.7])
    m = metrics_at_threshold(y, p, 0.5)
    assert set(["tp","tn","fp","fn","accuracy","precision","recall","f1"]).issubset(m.keys())
