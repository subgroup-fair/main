# fairbench/methods/gerryfair_wrapper.py
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

def _concat_rows(A, B):
    if B is None:
        return A
    if hasattr(A, "iloc"):
        return pd.concat([A, B], axis=0, ignore_index=True)
    return np.concatenate([np.asarray(A), np.asarray(B)], axis=0)

def _zsigmoid_with_stats(x, mu, sd):
    x = np.asarray(x, float).ravel()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(sd) or sd == 0.0:
        sd = 1.0
    z = (x - mu) / sd
    return 1.0 / (1.0 + np.exp(-z))

def run_gerryfair(args, data):
    """
    Clean GerryFair wrapper (validation 미사용).
    - TRAIN := train ∪ val
    - Base regressor: 'linear'(default) | 'ridge' | 'mlp'
    - Probability는 AIF360 scores 미사용. 우선 best_classifier에서 연속점수 취득, 실패 시
      gerry 예측 라벨을 타깃으로 하는 보정(calibrator) 로지스틱 회귀로 확률 생성.
    - Threshold: args.thr (default 0.5)
    Returns: dict(proba, pred, note)
    """
    try:
        from aif360.algorithms.inprocessing import GerryFairClassifier
        from aif360.datasets import StandardDataset
        from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        print(f"[SKIP] GerryFair not available: {e}")
        return dict(proba=None, pred=None, note=f"gerryfair_skipped: {e}")

    if data.get("type") != "tabular":
        print("[SKIP] GerryFair: only tabular supported.")
        return dict(proba=None, pred=None, note="gerryfair_tabular_only")

    # ---------- merge TRAIN ----------
    X_tr = _concat_rows(data["X_train"], data.get("X_val"))
    y_tr = _concat_rows(data["y_train"], data.get("y_val"))
    S_tr = _concat_rows(data["S_train"], data.get("S_val"))
    X_te, y_te, S_te = data["X_test"], data["y_test"], data["S_test"]

    # ---------- args ----------
    seed         = int(getattr(args, "seed", 0))
    base_name    = str(getattr(args, "gf_base", "linear")).lower()  # linear|ridge|mlp
    gamma        = float(getattr(args, "gamma", 0.005))
    max_iters    = int(getattr(args, "gf_max_iters", 20))
    C            = float(getattr(args, "gf_C", 1.0))
    fairness_def = str(getattr(args, "gf_fairness", "SP"))          # SP|FNR|FPR
    thr          = float(getattr(args, "thr", 0.5))

    # ---------- base regressor ----------
    if base_name == "ridge":
        predictor = Ridge()  # Ridge는 random_state 없음
    elif base_name == "mlp":
        predictor = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(100,), activation="relu",
                         max_iter=400, random_state=seed)
        )
    else:
        base_name = "linear"
        predictor = LinearRegression()

    # ---------- AIF360 dataset builders ----------
    def _build_sd(X, y, S):
        df = X.copy()
        for c in S.columns:
            df[f"sensitive_{c}"] = S[c].values
        df["target"] = np.asarray(y).ravel().astype(int)
        df = df.loc[:, ~df.columns.duplicated()].copy()
        sens_cols = [f"sensitive_{c}" for c in S.columns]
        # privileged_classes 기본값: 이진 {0,1}이면 [1], 아니면 최빈값
        privs = []
        for c in sens_cols:
            vals = pd.unique(df[c])
            if 1 in set(vals):
                privs.append([1])
            else:
                m = df[c].mode(dropna=True)
                privs.append([int(m.iloc[0]) if len(m) else int(vals[0])])
        ds = StandardDataset(
            df=df,
            label_name="target",
            favorable_classes=[1],
            protected_attribute_names=sens_cols,
            privileged_classes=privs,
        )
        return ds, sens_cols

    train_ds, sens_cols = _build_sd(X_tr, y_tr, S_tr)
    test_ds, _          = _build_sd(X_te, y_te, S_te)

    # ---------- fit GerryFair ----------
    gerry = GerryFairClassifier(
        C=C,
        printflag=False,
        max_iters=max_iters,
        gamma=gamma,
        fairness_def=fairness_def,
        predictor=predictor,
    )
    gerry.fit(train_ds)

    # ---------- labels (항상 AIF360 predict로) ----------
    yhat_te = np.asarray(gerry.predict(test_ds).labels, dtype=int).ravel()

    # ---------- robust probability construction ----------
    # 1) 가능한 경우: best_classifier/predictor에서 연속 점수 취득
    def _raw_and_kind(est, X):
        # kind: 'proba' | 'df' | 'pred'
        if hasattr(est, "predict_proba"):
            p = est.predict_proba(X)
            p = p[:, 1] if getattr(p, "ndim", 1) > 1 else np.ravel(p)
            return np.asarray(p, float).ravel(), "proba"
        if hasattr(est, "decision_function"):
            z = est.decision_function(X)
            return np.asarray(z, float).ravel(), "df"
        yhat = est.predict(X)
        return np.asarray(yhat, float).ravel(), "pred"

    def _try_estimator_probs():
        est = getattr(gerry, "best_classifier", None)
        if est is None:
            est = getattr(gerry, "predictor", None)
        if est is None:
            raise NotFittedError("No estimator available from GerryFair.")
        # TRAIN 통계 추출
        tr_vals, tr_kind = _raw_and_kind(est, train_ds.features)
        if tr_kind == "proba":
            te_vals, _ = _raw_and_kind(est, test_ds.features)
            return np.clip(te_vals, 0.0, 1.0)
        mu, sd = float(np.mean(tr_vals)), float(np.std(tr_vals))
        te_vals, _ = _raw_and_kind(est, test_ds.features)
        return _zsigmoid_with_stats(te_vals, mu, sd)

    p_te = None
    try:
        p_te = _try_estimator_probs()
    except Exception:
        # 2) fallback: gerry 라벨을 타깃으로 X→y^gerry 보정 로지스틱을 학습해서 확률 생성
        try:
            yhat_tr = np.asarray(gerry.predict(train_ds).labels, dtype=int).ravel()
            calib = LogisticRegression(solver="liblinear", max_iter=2000, class_weight="balanced")
            calib.fit(X_tr, yhat_tr)
            p_te = calib.predict_proba(X_te)[:, 1]
        except Exception:
            # 3) 최종 폴백: 라벨을 약간 부드럽게 확률화
            p_te = np.where(yhat_te > 0, 0.99, 0.01).astype(float)

    # ---------- thresholding ----------
    yhat = (p_te >= thr).astype(int)

    note = (
        f"gerryfair(clean): base={base_name}, gamma={gamma}, iters={max_iters}, "
        f"C={C}, def={fairness_def}, thr={thr}, "
        f"trainN={len(train_ds.labels)}, testN={len(test_ds.labels)}, "
        f"proba_src={'estimator' if 'calib' not in locals() else 'calibrator'}"
    )
    return dict(proba=p_te, pred=yhat, note=note)
