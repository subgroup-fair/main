# ===== Unfair baseline (no fairness constraints) =====
import numpy as np
import pandas as pd

import numpy as np, random, torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

def _concat_rows(A, B):
    if B is None:
        return A
    if hasattr(A, "iloc"):
        return pd.concat([A, B], axis=0, ignore_index=True)
    return np.concatenate([np.asarray(A), np.asarray(B)], axis=0)

def run_unfair(args, data):
    """
    Plain (unconstrained) baseline.
    - TRAIN := train ∪ val
    - TEST  := single classifier
    - Base estimator 선택: args.red_base in {"logreg","linear_svm","rf","mlp_clf","mlp_reg"} (기본: logreg)
    - Threshold: args.thr (기본 0.5)
    반환: dict(proba, pred, note)
    """
    if data.get("type") != "tabular":
        print("[SKIP] unfair: tabular only.")
        return dict(proba=None, pred=None, note="unfair_tabular_only")

    # --- imports (fairlearn 불필요) ---
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        print(f"[SKIP] sklearn not available: {e}")
        return dict(proba=None, pred=None, note=f"unfair_skipped: {e}")

    # --- helpers ---
    def _sigmoid_norm(z):
        z = np.asarray(z, float).ravel()
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _proba_from_estimator(h, X):
        # 가능한 경우 확률 사용, 아니면 decision_function→시그모이드, 그마저 없으면 predict→시그모이드
        if hasattr(h, "predict_proba"):
            try:
                p = h.predict_proba(X)
                return p[:, 1] if getattr(p, "ndim", 1) > 1 else np.ravel(p)
            except Exception:
                pass
        if hasattr(h, "decision_function"):
            try:
                return _sigmoid_norm(h.decision_function(X))
            except Exception:
                pass
        return _sigmoid_norm(h.predict(X))

    # ---- TRAIN / TEST 병합 ----
    X_tr = _concat_rows(data["X_train"], data.get("X_val"))
    y_tr = _concat_rows(data["y_train"], data.get("y_val"))
    X_te = data["X_test"]
    # y_te, S_* 는 여기선 불필요

    # --- args ---
    seed = int(getattr(args, "seed", 42))
    thr = 0.5
    base_name = str(getattr(args, "red_base", "logreg")).lower()

    # --- base estimator 선택 (간단 표준화 포함) ---
    if base_name == "linear_svm":
        est = make_pipeline(StandardScaler(), LinearSVC(random_state=seed))
    elif base_name == "rf":
        est = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    elif base_name == "mlp_clf":
        est = make_pipeline(StandardScaler(),
                            MLPClassifier(hidden_layer_sizes=(100,), activation="relu", max_iter=200, random_state=seed))
    elif base_name == "mlp_reg":
        est = make_pipeline(StandardScaler(),
                            MLPRegressor(hidden_layer_sizes=(100,), activation="relu", max_iter=200, random_state=seed))
    else:
        base_name = "logreg"
        est = make_pipeline(StandardScaler(),
                            LogisticRegression(solver="liblinear", max_iter=2000, random_state=seed))

    # ---- fit ----
    est.fit(X_tr, y_tr)

    # ---- predict ----
    p_te = _proba_from_estimator(est, X_te)
    p_te = np.asarray(p_te, float).ravel()
    p_te = np.nan_to_num(p_te, nan=0.5, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    yhat = (p_te >= thr).astype(int)

    note = f"unfair-baseline base={base_name}, thr={thr}, seed={seed}"
    return dict(proba=p_te, pred=yhat, note=note)


