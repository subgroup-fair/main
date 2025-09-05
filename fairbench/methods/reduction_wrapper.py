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
    Fairlearn ExponentiatedGradient baseline (no validation use).
    - TRAIN := train ∪ val
    - threshold := args.thr (default 0.5)
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

    # ---- merge TRAIN ----
    X_tr = _concat_rows(data["X_train"], data.get("X_val"))
    y_tr = _concat_rows(data["y_train"], data.get("y_val"))
    S_tr = _concat_rows(data["S_train"], data.get("S_val"))

    X_te, y_te, S_te = data["X_test"], data["y_test"], data["S_test"]

    # --- constraints ---
    cons_name = str(getattr(args, "red_constraint", "DP")).upper()
    constraint = EqualizedOdds() if cons_name == "EO" else DemographicParity(); cons_name = "EO" if cons_name=="EO" else "DP"

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
        base = LogisticRegression(solver="liblinear", max_iter=2000, random_state=seed); step_name = None

    eps = float(getattr(args, "red_eps", 0.02))
    max_iter = int(getattr(args, "red_max_iter", 50))

    eg = ExponentiatedGradient(estimator=base, constraints=constraint, eps=eps, max_iter=max_iter)
    eg.fit(X_tr, y_tr, sensitive_features=S_tr)

    def _sigmoid_norm(z):
        z = np.asarray(z, float).ravel()
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        mu, sd = z.mean(), z.std()
        if not np.isfinite(sd) or sd == 0.0: sd = 1.0
        z = (z - mu) / sd
        return 1.0 / (1.0 + np.exp(-z))

    def _proba_from_estimator(h, X):
        if hasattr(h, "predict_proba") and (h.predict_proba(X) is not None):
            p = h.predict_proba(X); return p[:, 1] if p.ndim > 1 else np.ravel(p)
        if hasattr(h, "decision_function") and (h.decision_function(X) is not None):
            return _sigmoid_norm(h.decision_function(X))
        return _sigmoid_norm(h.predict(X))

    def mixture_proba(model, X):
        """EG의 혼합분류기 확률: 가중 평균(predict_proba/decision_function→sigmoid)"""
        def _sigmoid_norm(z):
            z = np.asarray(z, float).ravel()
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            mu, sd = z.mean(), z.std()
            if not np.isfinite(sd) or sd == 0.0:
                sd = 1.0
            z = (z - mu) / sd
            return 1.0 / (1.0 + np.exp(-z))

        def _proba_from_estimator(h, X_):
            # predict_proba → decision_function → predict 순으로 시도
            if hasattr(h, "predict_proba"):
                p = h.predict_proba(X_)
                return p[:, 1] if getattr(p, "ndim", 1) > 1 else np.ravel(p)
            if hasattr(h, "decision_function"):
                return _sigmoid_norm(h.decision_function(X_))
            return _sigmoid_norm(h.predict(X_))

        # --- predictors 안전하게 회수 (절대 `or` 쓰지 말기) ---
        predictors = getattr(model, "predictors_", None)
        if predictors is None:
            predictors = getattr(model, "classifiers_", None)

        # 리스트/튜플/시퀀스로 강제 변환
        if predictors is None:
            pred_list = []
        elif isinstance(predictors, (list, tuple)):
            pred_list = list(predictors)
        else:
            # pandas Series 등
            try:
                pred_list = list(predictors)
            except Exception:
                pred_list = [predictors]

        # 단일 추정기 케이스 처리
        if len(pred_list) == 0:
            est = getattr(model, "estimator_", None)
            if est is None:
                # 마지막 수단: 모델의 predict로 대체
                return np.asarray(model.predict(X), dtype=float)
            return _proba_from_estimator(est, X)

        # --- 가중치 회수 & 정규화 ---
        weights = getattr(model, "weights_", None)
        if weights is None:
            w = np.ones(len(pred_list), dtype=float) / float(len(pred_list))
        else:
            try:
                w = np.asarray(weights, dtype=float).ravel()
            except Exception:
                w = np.asarray(list(weights), dtype=float).ravel()
            if w.size != len(pred_list) or w.size == 0:
                w = np.ones(len(pred_list), dtype=float) / float(len(pred_list))

        # --- 혼합 확률 계산 ---
        preds = [ _proba_from_estimator(h, X) for h in pred_list ]
        P = np.average(np.vstack([np.ravel(p) for p in preds]), axis=0, weights=w)
        return np.ravel(P)

    # --- proba on TEST & fixed threshold ---
    p_te = mixture_proba(eg, X_te)
    thr = float(getattr(args, "thr", 0.5))
    yhat = (p_te >= thr).astype(int)

    note = f"fairlearn-EG({cons_name}) TR+VAL merged, eps={eps}, iters={max_iter}, base={base_name}, thr={thr}"
    return dict(proba=p_te, pred=yhat, note=note)
