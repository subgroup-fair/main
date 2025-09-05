# fairbench/metrics/accuracy.py
import numpy as np
def accuracy(y_true, y_pred):
    return (y_true.reshape(-1)==y_pred.reshape(-1)).mean()
