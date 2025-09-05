# fairbench/datasets/toy_manyq.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ..utils.trainval import split_train_val

def load_toy_manyq(args):
    n = 20000; d = 20; q = args.q
    rng = np.random.default_rng(args.seed)
    X = rng.normal(size=(n, d)).astype(np.float32)

    # latent sensitive factors (q binary), correlated with some features
    S = (rng.normal(size=(n, q)) + 0.5*X[:, :min(d, q)][:, None]).mean(axis=1, keepdims=True)
    S = (rng.normal(size=(n, q)) > 0.0).astype(int)  # 독립적인 q도 가능
    S = pd.DataFrame(S, columns=[f"s{j}" for j in range(q)])

    # target: non-linear function + mild spurious corr with a few sensitive attrs
    logits = 0.8*X[:,0] - 0.6*X[:,1] + 0.5*X[:,2]*X[:,3] + 0.2*(S.values[:,:min(q,5)].sum(axis=1))
    p = 1/(1+np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)

    X = pd.DataFrame(X, columns=[f"x{j}" for j in range(d)])

    X_tr, X_te, y_tr, y_te, S_tr, S_te = train_test_split(X, y, S, test_size=0.2, random_state=args.seed, stratify=y)
    X_tr, X_va, y_tr, y_va, S_tr, S_va = split_train_val(X_tr, y_tr, S_tr, val_size=0.2, seed=args.seed)

    return dict(
        X_train=X_tr, y_train=y_tr, S_train=S_tr,
        X_val=X_va,   y_val=y_va,   S_val=S_va,
        X_test=X_te,  y_test=y_te,  S_test=S_te,
        type="tabular"
    )
