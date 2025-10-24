
import numpy as np
from backend.drift import population_stability_index, kolmogorov_smirnov, jensen_shannon

def test_drift_metrics_basic():
    a = np.random.normal(0,1,1000)
    b = np.random.normal(0,1,1000)
    psi = population_stability_index(a,b,10)
    ks = kolmogorov_smirnov(a,b)
    js = jensen_shannon(a,b,20)
    assert psi >= 0.0 and ks >= 0.0 and js >= 0.0
