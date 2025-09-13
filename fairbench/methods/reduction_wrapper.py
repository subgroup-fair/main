# 1. reduction_wrapper.py 대체

# fairbench/methods/reduction_wrapper.py
import numpy as np
import pandas as pd

def _concat_rows(A, B):
    if B is None:
        return A
    if hasattr(A, "iloc"):
        return pd.concat([A, B], axis=0, ignore_index=True)
    return np.concatenate([np.asarray(A), np.asarray(B)], axis=0)


def run_reduction(args, data):
    """
    Fairlearn ExponentiatedGradient baseline (TRAIN := train ∪ val),
    **marginal-vs-overall** fairness:
      - S가 다열/범주일 때, 각 '마진'(열 또는 각 범주의 one-hot)에 대해
        X,y를 복제하고 sensitive_features를 그 마진의 0/1로 만들어
        EG가 모든 마진 제약을 동시에 보도록 함.
    TEST는 (기본) mixture=uniform 평균 확률로 예측(단일 선택도 가능).
    """
    if data.get("type") != "tabular":
        print("[SKIP] reduction: tabular only.")
        return dict(proba=None, pred=None, note="reduction_tabular_only")

    try:
        from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.base import BaseEstimator
        import inspect
    except Exception as e:
        print(f"[SKIP] fairlearn not available: {e}")
        return dict(proba=None, pred=None, note=f"reduction_skipped: {e}")

    # ---------- helpers ----------
    def _to_df(x):
        if isinstance(x, pd.DataFrame): return x.copy()
        if isinstance(x, pd.Series): return x.to_frame()
        arr = np.asarray(x)
        if arr.ndim == 1: arr = arr[:, None]
        return pd.DataFrame(arr, columns=[f"c{i}" for i in range(arr.shape[1])])

    def _make_onehot_marginals(S_df: pd.DataFrame) -> pd.DataFrame:
        """S의 각 열이 이진이면 그대로, 다치/문자면 열별로 get_dummies하여
        '열이름=범주' 형태의 one-hot 열들로 확장. 최종적으로 전부 {0,1} 컬럼."""
        out = []
        for c in S_df.columns:
            col = S_df[c]
            if col.dtype == bool:
                out.append(col.astype(int).rename(c))
            elif pd.api.types.is_numeric_dtype(col):
                # 숫자여도 {0,1} 아니면 범주로 본다
                vals = pd.unique(col.dropna())
                if set(vals).issubset({0, 1, 0.0, 1.0}):
                    out.append(col.fillna(0).astype(float).round().astype(int).rename(c))
                else:
                    dmy = pd.get_dummies(col.astype("category"), prefix=c, dummy_na=False)
                    out.append(dmy)
            else:
                dmy = pd.get_dummies(col.astype("category"), prefix=c, dummy_na=False)
                out.append(dmy)
        OH = pd.concat(out, axis=1)
        # 안전 클린업: {0,1} 강제
        OH = OH.fillna(0).clip(lower=0, upper=1).astype(int)
        # 0열만 있는(모두 0) 열은 제거
        keep = [c for c in OH.columns if OH[c].sum() > 0]
        return OH[keep] if keep else OH

    def _sigmoid_norm(z):
        z = np.asarray(z, float).ravel()
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        sd = z.std()
        if not np.isfinite(sd) or sd == 0.0:
            return 1.0 / (1.0 + np.exp(-z))
        z = (z - z.mean()) / sd
        return 1.0 / (1.0 + np.exp(-z))

    def _proba_from_estimator(h, X):
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

    # ---------- SampleWeight adapter ----------
    class SampleWeightAdapter(BaseEstimator):
        def __init__(self, estimator, step_name=None, seed=42):
            self.estimator = estimator; self.step_name = step_name; self.seed = seed
        def _final_estimator(self):
            try:
                if hasattr(self.estimator, "steps"): return self.estimator.steps[-1][1]
            except Exception: pass
            return self.estimator
        def _accepts_sample_weight(self):
            try:
                return "sample_weight" in inspect.signature(self._final_estimator().fit).parameters
            except Exception:
                return False
        def _resample_by_weight(self, X, y, sample_weight):
            w = np.asarray(sample_weight, float).ravel(); w = np.clip(w, 0.0, None)
            p = w / w.sum() if np.isfinite(w).all() and w.sum() > 0 else np.ones_like(w)/len(w)
            n_total = len(y); rng = np.random.RandomState(self.seed)
            idx = rng.choice(np.arange(n_total), size=n_total, replace=True, p=p)
            Xr = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
            yr = y.iloc[idx] if hasattr(y, "iloc") else np.asarray(y)[idx]
            return Xr, yr
        def fit(self, X, y, sample_weight=None, **fit_params):
            if sample_weight is None:
                self.estimator.fit(X, y, **fit_params); return self
            if self._accepts_sample_weight():
                if self.step_name is not None and hasattr(self.estimator, "fit"):
                    fit_params = {**fit_params, f"{self.step_name}__sample_weight": sample_weight}
                    self.estimator.fit(X, y, **fit_params)
                else:
                    try:
                        self.estimator.fit(X, y, sample_weight=sample_weight, **fit_params)
                    except TypeError:
                        Xr, yr = self._resample_by_weight(X, y, sample_weight)
                        self.estimator.fit(Xr, yr, **fit_params)
            else:
                Xr, yr = self._resample_by_weight(X, y, sample_weight)
                self.estimator.fit(Xr, yr, **fit_params)
            return self
        def predict(self, X): return self.estimator.predict(X)
        def predict_proba(self, X):
            return self.estimator.predict_proba(X) if hasattr(self.estimator, "predict_proba") else None
        def decision_function(self, X):
            return self.estimator.decision_function(X) if hasattr(self.estimator, "decision_function") else None
        def get_params(self, deep=True):
            params = {"estimator": self.estimator, "step_name": self.step_name, "seed": self.seed}
            if deep and hasattr(self.estimator, "get_params"):
                for k, v in self.estimator.get_params(deep=True).items():
                    params[f"estimator__{k}"] = v
            return params
        def set_params(self, **params):
            if "estimator" in params: self.estimator = params.pop("estimator")
            if "step_name" in params: self.step_name = params.pop("step_name")
            if "seed" in params: self.seed = params.pop("seed")
            nest = {k[len("estimator__"):]: v for k, v in params.items() if k.startswith("estimator__")}
            if nest: self.estimator.set_params(**nest)
            return self
    # ----------------------------------------

    # ---- TRAIN (merge TR+VAL) ----
    X_tr = _concat_rows(data["X_train"], data.get("X_val"))
    y_tr = _concat_rows(data["y_train"], data.get("y_val"))
    S_tr_raw = _concat_rows(data["S_train"], data.get("S_val"))

    # ---- TEST ----
    X_te, y_te, S_te = data["X_test"], data["y_test"], data["S_test"]

    # --- constraints ---
    cons_name = str(getattr(args, "red_constraint", "DP")).upper()
    constraint = EqualizedOdds() if cons_name == "EO" else DemographicParity()
    cons_name = "EO" if cons_name == "EO" else "DP"

    # --- base estimator ---
    seed = getattr(args, "seed", 42)
    base_name = str(getattr(args, "red_base", "logreg")).lower()
    if base_name == "linear_svm":
        base = LinearSVC(random_state=seed); step_name = None
    elif base_name == "rf":
        base = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=seed); step_name = None
    elif base_name == "mlp_clf":
        pipe = make_pipeline(StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(100), activation="relu", max_iter=200, random_state=seed))
        base = SampleWeightAdapter(pipe, step_name="mlpclassifier", seed=seed)
    elif base_name == "mlp_reg":
        pipe = make_pipeline(StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(100), activation="relu", max_iter=200, random_state=seed))
        base = SampleWeightAdapter(pipe, step_name="mlpregressor", seed=seed)
    else:
        base_name = "logreg"
        base = LogisticRegression(solver="liblinear", max_iter=2000, random_state=seed)
        step_name = None

    eps = float(getattr(args, "red_eps", 0.02))
    max_iter = int(getattr(args, "red_max_iter", 50))
    thr = float(getattr(args, "thr", 0.5))
    # pick = str(getattr(args, "red_pick", "mixture")).lower()  # default: mixture(평균)
    pick = 'random'

    # ====== ★ marginal-vs-overall 스택 구성 ======
    X_tr_df = _to_df(X_tr)
    y_tr_sr = y_tr if isinstance(y_tr, pd.Series) else pd.Series(np.asarray(y_tr).ravel())
    S_df = _make_onehot_marginals(_to_df(S_tr_raw))  # 각 열이 한 '마진'

    if S_df.shape[1] == 0:
        raise ValueError("No valid marginal columns found in S_train.")

    q = S_df.shape[1]
    n = len(y_tr_sr)

    # X,y 복제 스택 (numpy로 통일)
    Xn = X_tr_df.to_numpy()
    yn = y_tr_sr.to_numpy().ravel().astype(int)
    X_stack = np.vstack([Xn for _ in range(q)])                  # (q*n, d)
    y_stack = np.concatenate([yn for _ in range(q)])             # (q*n,)

    # # 각 마진의 0/1 민감변수
    # sens_list = [S_df.iloc[:, j].to_numpy().astype(int).ravel() for j in range(q)]
    # S_stack = np.concatenate(sens_list)                           # (q*n,)

    # 각 마진 j에 대해 '0'을 2*j, '1'을 2*j+1로 라벨링 → 서로 다른 그룹으로 취급
    sens_list = [S_df.iloc[:, j].to_numpy().astype(int).ravel() for j in range(q)]
    S_stack = np.concatenate([2 * j + s for j, s in enumerate(sens_list)]).astype(int)  # (q*n,)


    # 샘플 가중치: 각 복제블록에 1/q 부여(총량 보존)
    sw_stack = np.full(q * n, 1.0 / q, dtype=float)

    # ---- fit EG on STACKED TRAIN ----
    eg = ExponentiatedGradient(estimator=base, constraints=constraint, eps=eps, max_iter=max_iter)
    eg.fit(X_stack, y_stack, sensitive_features=S_stack)

    # ---- helpers for prediction ---- (위에서 정의한 _proba_from_estimator 사용)

    # ---- pick ONE or MIXTURE for TEST ----
    predictors = getattr(eg, "predictors_", None)
    if predictors is None:
        predictors = getattr(eg, "classifiers_", None)

    if predictors is None:
        # fallback: EG이 단일 추정기만 가지고 있을 때
        est = getattr(eg, "estimator_", eg)
        p_te = _proba_from_estimator(est, X_te)
        yhat = (np.asarray(p_te) >= thr).astype(int)
        note = (f"fairlearn-EG({cons_name}) single=fallback(estimator_), "
                f"eps={eps}, iters={max_iter}, base={base_name}, thr={thr}, pick={pick}, "
                f"marginals={list(S_df.columns)}")
        return dict(proba=np.asarray(p_te, float).ravel(), pred=yhat, note=note)

    # 시퀀스로 강제
    if isinstance(predictors, (list, tuple)):
        pred_list = list(predictors)
    elif isinstance(predictors, pd.Series):
        pred_list = list(predictors.tolist())
    else:
        try:
            pred_list = list(predictors)
        except Exception:
            pred_list = [predictors]

    # === mixture(평균) 또는 단일 선택 ===
    if pick in {"mixture", "wavg"}:
        # 가능 모델들의 확률 수집(되는 것만 사용)
        P_cols, used = [], []
        for i, h in enumerate(pred_list):
            try:
                p = np.asarray(_proba_from_estimator(h, X_te), float).ravel()
                if p.ndim != 1: raise ValueError("proba not 1D")
                P_cols.append(p); used.append(i)
            except Exception:
                continue
        if not used:
            # 전부 실패 → 첫 모델 fallback
            p_te = _proba_from_estimator(pred_list[0], X_te)
            yhat = (np.asarray(p_te) >= thr).astype(int)
            note = (f"fairlearn-EG({cons_name}) MIXTURE[fallback-first] "
                    f"idx=0, eps={eps}, iters={max_iter}, base={base_name}, thr={thr}, "
                    f"marginals={list(S_df.columns)}")
            return dict(proba=np.asarray(p_te, float).ravel(), pred=yhat, note=note)

        P = np.column_stack(P_cols)                       # (n_test, k_used)
        ww = np.full(len(used), 1.0/len(used), float)     # 균등 가중치
        p_te = P @ ww
        p_te = np.nan_to_num(p_te, nan=0.5, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
        yhat = (p_te >= thr).astype(int)
        note = (f"fairlearn-EG({cons_name}) MIXTURE[uniform] k={len(used)}, "
                f"eps={eps}, iters={max_iter}, base={base_name}, thr={thr}, "
                f"marginals={list(S_df.columns)}")
        return dict(proba=p_te, pred=yhat, note=note)

    # 단일 선택 규칙
    weights = getattr(eg, "weights_", None)
    try:
        w = np.asarray(weights, float).ravel()
    except Exception:
        w = None
    if w is None or w.size != len(pred_list) or w.size == 0:
        w = np.ones(len(pred_list), float)

    if pick == "last":
        best_idx = len(pred_list) - 1
    elif pick == "first":
        best_idx = 0
    elif pick == "random":
        rng = np.random.RandomState(seed)
        probs = w / w.sum() if np.isfinite(w).all() and w.sum() > 0 else None
        best_idx = int(rng.choice(np.arange(len(pred_list)), p=probs))
        # rng = np.random.RandomState(seed)
        # pos = np.flatnonzero(np.isfinite(w) & (w > 0))
        # pool = pos if pos.size > 0 else np.arange(len(pred_list))
        # best_idx = int(rng.choice(당구))
    else:  # "max_weight"
        best_idx = int(np.argmax(w))

    best_h = pred_list[best_idx]
    p_te = _proba_from_estimator(best_h, X_te)
    yhat = (np.asarray(p_te) >= thr).astype(int)
    note = (
        f"fairlearn-EG({cons_name}) SINGLE[{pick}] idx={best_idx}, "
        f"eps={eps}, iters={max_iter}, base={base_name}, thr={thr}, "
        f"marginals={list(S_df.columns)}"
    )
    return dict(proba=np.asarray(p_te, float).ravel(), pred=yhat, note=note)
