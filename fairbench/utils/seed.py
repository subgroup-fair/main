# fairbench/utils/seed.py
import random, numpy as np, torch
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    try: torch.manual_seed(seed); 
    except: pass
