# fairbench/utils/trainval.py
from sklearn.model_selection import train_test_split
def split_train_val(X, y, S, val_size=0.2, seed=42):
    return train_test_split(X, y, S, test_size=val_size, random_state=seed, stratify=y)
