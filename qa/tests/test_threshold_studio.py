
import numpy as np
from backend.threshold_studio import grid_cost_thresholds

def test_grid_cost_thresholds_orders_by_cost():
    y = np.array([0,1,0,1,1,0,0,1,0,1])
    p = np.array([0.1,0.9,0.2,0.8,0.7,0.4,0.3,0.6,0.2,0.55])
    grid = grid_cost_thresholds(y, p, [1.0,5.0], [1.0,5.0])
    assert len(grid) == 4
    costs = [row["expected_cost"] for row in grid]
    assert costs == sorted(costs)
