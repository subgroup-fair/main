# fairbench/metrics/accuracy.py
import numpy as np
import numpy as np, random, torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    
def accuracy(y_true, y_pred):
    return (y_true.reshape(-1)==y_pred.reshape(-1)).mean()
