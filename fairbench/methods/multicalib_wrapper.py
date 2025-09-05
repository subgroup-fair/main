# fairbench/methods/multicalib_wrapper.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def _concat_rows(A, B):
    if B is None:
        return A
    if hasattr(A, "iloc"):
        return pd.concat([A, B], axis=0, ignore_index=True)
    return np.concatenate([np.asarray(A), np.asarray(B)], axis=0)

def run_multicalib(args, data):
    """
    Multicalibration wrapper (no validation use).
    - TR := (train ∪ val)
    - threshold := args.thr (default 0.5)
    """
    try:
        from multicalibration import MulticalibrationPredictor
    except Exception as e:
        print(f"[SKIP] Multicalibration not available: {e}")
        return dict(proba=None, pred=None, note=f"multicalib_skipped: {e}")

    if data.get("type") != "tabular":
        print("[SKIP] Multicalibration: only tabular supported (uses logistic base).")
        return dict(proba=None, pred=None, note="multicalib_tabular_only")

    # ----- merge TR := train (+ val if present) -----
    X_tr = _concat_rows(data["X_train"], data.get("X_val"))
    y_tr = _concat_rows(data["y_train"], data.get("y_val"))
    S_tr = _concat_rows(data["S_train"], data.get("S_val"))

    X_te, y_te, S_te = data["X_test"], data["y_test"], data["S_test"]

    # 1) Base model
    base = LogisticRegression(max_iter=1000)
    base.fit(X_tr, np.asarray(y_tr).ravel())
    p_tr = base.predict_proba(X_tr)[:, 1]
    p_te = base.predict_proba(X_te)[:, 1]

    # 2) Build subgroup index lists using TRAIN groups
    sub_tr, sub_te = [], []
    for c in S_tr.columns:
        col_tr = np.asarray(S_tr[c]).ravel()
        col_te = np.asarray(S_te[c]).ravel()
        if not set(np.unique(col_tr)).issubset({0, 1}):
            vals, cnts = np.unique(col_tr, return_counts=True)
            pos = vals[np.argmax(cnts)]
            col_tr = (col_tr == pos).astype(int)
            col_te = (col_te == pos).astype(int)
        for v in (0, 1):
            tr_idx = np.where(col_tr == v)[0].tolist()
            if len(tr_idx) == 0:
                continue
            te_idx = np.where(col_te == v)[0].tolist()
            sub_tr.append(tr_idx)
            sub_te.append(te_idx)

    if len(sub_tr) == 0:
        print("[INFO] Multicalibration: no non-empty train subgroups; returning base predictions.")
        yhat_base = (p_te >= float(getattr(args, "thr", 0.5))).astype(int)
        return dict(proba=p_te, pred=yhat_base, note="no_subgroups_base_returned_no_val")

    # 3) Hyperparameters
    mc_alpha = float(getattr(args, "mc_alpha", 0.1))
    mc_lambda = float(getattr(args, "mc_lambda", 0.1))
    mc_max_iter = int(getattr(args, "mc_max_iter", 30))
    mc_randomized = bool(getattr(args, "mc_randomized", True))
    mc_use_oracle = bool(getattr(args, "mc_use_oracle", False))
    params = dict(alpha=mc_alpha, **{"lambda": mc_lambda}, max_iter=mc_max_iter,
                  randomized=mc_randomized, use_oracle=mc_use_oracle)

    # 4) Fit multicalibrator on TRAIN and predict TEST
    mcb = MulticalibrationPredictor("HKRR")
    mcb.fit(p_tr, np.asarray(y_tr).ravel(), sub_tr, params)
    p_new = mcb.predict(p_te, sub_te)

    # 5) Fixed threshold (no val)
    thr = float(getattr(args, "thr", 0.5))
    yhat = (p_new >= thr).astype(int)
    return dict(
        proba=p_new,
        pred=yhat,
        note=f"multicalib(TR+VAL merged), thr={thr}, alpha={mc_alpha}, lambda={mc_lambda}, "
             f"max_iter={mc_max_iter}, randomized={mc_randomized}, oracle={mc_use_oracle}"
    )
