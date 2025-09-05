# fairbench/methods/sequential_fairness_wrapper.py
import numpy as np
import pandas as pd

def _concat_rows(A, B):
    if B is None:
        return A
    if hasattr(A, "iloc"):
        return pd.concat([A, B], axis=0, ignore_index=True)
    return np.concatenate([np.asarray(A), np.asarray(B)], axis=0)

def run_sequential(args, data):
    """
    Sequential Fairness (post-processing) wrapper (no validation use).
    - TRAIN := train ∪ val
    - sequential mapping is fitted on TRAIN and applied to TEST
    - threshold := args.thr (default 0.5)
    """
    if data.get("type") != "tabular":
        print("[SKIP] sequential_fairness: tabular only.")
        return dict(proba=None, pred=None, note="seqfair_tabular_only")

    # ---------- merge TRAIN ----------
    X_tr = _concat_rows(data["X_train"], data.get("X_val"))
    y_tr = _concat_rows(data["y_train"], data.get("y_val"))
    S_tr = _concat_rows(data["S_train"], data.get("S_val"))

    X_te, y_te, S_te = data["X_test"], data["y_test"], data["S_test"]

    # ---------- hyper-params ----------
    base_name   = str(getattr(args, "sf_base", "logreg")).lower()   # logreg|rf|linear_svm
    order_str   = str(getattr(args, "sf_order", "")).strip()
    weight_mode = str(getattr(args, "sf_weights", "prop")).lower()  # prop|uniform
    grid_n      = int(getattr(args, "sf_grid", 2001))
    min_count   = int(getattr(args, "sf_min_count", 5))
    seed        = int(getattr(args, "seed", 42))
    thr         = float(getattr(args, "thr", 0.5))

    # ---------- base score model ----------
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        print(f"[SKIP] sklearn not available: {e}")
        return dict(proba=None, pred=None, note=f"seqfair_skipped: {e}")

    if base_name == "rf":
        base = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    elif base_name == "linear_svm":
        base = LinearSVC()
    else:
        base_name = "logreg"
        base = LogisticRegression(solver="liblinear", max_iter=2000, random_state=seed)

    base.fit(X_tr, np.asarray(y_tr).ravel())

    def _score(est, X):
        if hasattr(est, "predict_proba"):
            p = est.predict_proba(X); return p[:, 1] if p.ndim > 1 else p
        if hasattr(est, "decision_function"):
            z = est.decision_function(X); z = np.asarray(z, float).ravel()
            return 1.0 / (1.0 + np.exp(-z))
        yhat = est.predict(X)
        return (np.asarray(yhat, float).ravel() + 0.5) / 2.0

    s_tr0 = _score(base, X_tr)
    s_te0 = _score(base, X_te)

    # ---------- use official lib if available ----------
    _used_official = False
    attrs = [c.strip() for c in (order_str.split(",") if order_str else list(S_tr.columns))]
    try:
        from sequential_fairness import fit_sequential, transform_sequential  # type: ignore
        model_sf = fit_sequential(scores=s_tr0, sens_df=S_tr[attrs],
                                  weights=weight_mode, grid_n=grid_n,
                                  min_count=min_count, random_state=seed)
        s_te = transform_sequential(model_sf, s_te0, S_te[attrs])
        _used_official = True
    except Exception:
        # ---------- fallback: quantile-barycenter sequential ----------
        eps = 1e-9
        u_grid = np.linspace(eps, 1.0 - eps, grid_n)

        def _sorted(v):
            v = np.asarray(v, float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                return np.array([0.0, 1.0])
            return np.sort(v)

        def _ecdf_percentile(xs_sorted, x):
            n = xs_sorted.size
            r = np.searchsorted(xs_sorted, x, side="right")
            return np.clip(r / max(n, 1), 0.0 + eps, 1.0 - eps)

        def _quantiles(xs_sorted):
            return np.quantile(xs_sorted, u_grid, method="linear") if hasattr(np, "quantile") else np.percentile(xs_sorted, u_grid * 100.0)

        def _fit_step(scores, sens_col):
            s = np.asarray(scores, float).ravel()
            g = pd.Series(sens_col).reset_index(drop=True)
            keys, counts = np.unique(g.astype("category"), return_counts=True); keys = list(keys)
            if weight_mode == "uniform":
                w = np.ones(len(keys), float) / max(len(keys), 1)
            else:
                w = counts.astype(float); w = w / max(w.sum(), 1.0)
            xs_sorted = {}; q_g = []
            for k, cnt in zip(keys, counts):
                idx = (g == k).to_numpy()
                xs = _sorted(s[idx])
                if cnt < min_count:
                    xs = _sorted(s)
                xs_sorted[k] = xs
                q_g.append(_quantiles(xs))
            q_g = np.vstack(q_g)
            Q_star = np.average(q_g, axis=0, weights=w)
            s_next = s.copy()
            for k in keys:
                idx = (g == k).to_numpy()
                if not np.any(idx): continue
                u = _ecdf_percentile(xs_sorted[k], s_next[idx])
                s_next[idx] = np.interp(u, u_grid, Q_star, left=Q_star[0], right=Q_star[-1])
            return dict(keys=keys, weights=w, u_grid=u_grid, Q_star=Q_star,
                        xs_sorted=xs_sorted, global_sorted=_sorted(s), col_name=str(getattr(sens_col, "name", "attr"))), s_next

        def _apply_step(step, scores, sens_col):
            s = np.asarray(scores, float).ravel()
            g = pd.Series(sens_col).reset_index(drop=True)
            Q_star = step["Q_star"]; u_grid = step["u_grid"]
            xs_sorted = step["xs_sorted"]; keys = step["keys"]; glob = step["global_sorted"]
            for k in pd.unique(g):
                idx = (g == k).to_numpy()
                if not np.any(idx): continue
                xs = xs_sorted.get(k, glob)
                u = _ecdf_percentile(xs, s[idx])
                s[idx] = np.interp(u, u_grid, Q_star, left=Q_star[0], right=Q_star[-1])
            return s

        steps = []
        s_tr = s_tr0.copy()
        for a in attrs:
            step, s_tr = _fit_step(s_tr, S_tr[a]); steps.append(step)

        s_te = s_te0.copy()
        for (a, step) in zip(attrs, steps):
            s_te = _apply_step(step, s_te, S_te[a])

    # ---------- fixed threshold on TEST ----------
    yhat = (s_te >= thr).astype(int)
    used_backend = "official" if _used_official else "fallback"
    note = f"seq-fair(TR+VAL merged): order={','.join(attrs)} weights={weight_mode} grid={grid_n} base={base_name} backend={used_backend}, thr={thr}"
    return dict(proba=s_te, pred=yhat, note=note)
