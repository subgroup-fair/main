# fairbench/utils/threshold.py
import numpy as np
import numpy as np, random, torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    
def tune_threshold(proba, y, grid=None):
    grid = grid or np.linspace(0.05,0.95,37)
    best_t, best = 0.5, 0.0
    for t in grid:
        acc = ( (proba>=t).astype(int) == y ).mean()
        if acc>best: best_t, best = t, acc
    return float(best_t)
