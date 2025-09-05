# fairbench/utils/threshold.py
import numpy as np
def tune_threshold(proba, y, grid=None):
    grid = grid or np.linspace(0.05,0.95,37)
    best_t, best = 0.5, 0.0
    for t in grid:
        acc = ( (proba>=t).astype(int) == y ).mean()
        if acc>best: best_t, best = t, acc
    return float(best_t)
