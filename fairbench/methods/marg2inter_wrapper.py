# fairbench/methods/marg2inter_wrapper.py
import numpy as np
import pandas as pd

def run_marg2inter(args, data):
    """
    NeurIPS'22 'Bounding & Approximating Intersectional Fairness through Marginal Fairness'
    (Molina et al.)를 참고한 래퍼.
    - 분류기는 일반 supervised로 학습(기본: 로지스틱)
    - 학습 후, 테스트셋에서
        * 인터섹셔널 불공정성의 경험값 u_joint (교차그룹 기준)
        * 마지널 근사 u_I (논문식, sup_y sum_k sup_{a_k,a_k'} |log P(y|A_k)/P(y|A_k')|)
        * 독립성 위반 척도 s* (논문식 σ, σ_y 기반)
      을 계산해 `note`에 기록.
    - 공식 레포가 설치돼 있으면 우선 시도(import)하고, 실패하면 본 파일의 fallback 구현 사용.
    - 반환 형식: dict(proba, pred, note)  (기존 파이프라인과 동일)

    Args (예시):
        args.m2i_base:    'logreg'|'rf'|'linear_svm' (default 'logreg')
        args.decision_threshold: float (val로 튠 못할 때 기본 0.5)
        args.m2i_delta:   float, 확률적 경계용 δ (기록용; 기본 0.1)
        args.m2i_eps_clip: float, 확률 0 보호용 클리핑 (기본 1e-12)
        args.seed:        int
        args.m2i_repo_path: str, Git clone 경로(sys.path에 추가 시도)
    """
    if data.get("type") != "tabular":
        print("[SKIP] marg2inter: tabular only.")
        return dict(proba=None, pred=None, note="m2i_tabular_only")

    # ---------------- unpack ----------------
    X_tr, y_tr, S_tr = data["X_train"], data["y_train"], data["S_train"]
    X_va, y_va, S_va = data["X_val"],   data["y_val"],   data["S_val"]
    X_te, y_te, S_te = data["X_test"],  data["y_test"],  data["S_test"]

    # ---------------- base classifier ----------------
    base_name = str(getattr(args, "m2i_base", "logreg")).lower()
    seed = int(getattr(args, "seed", 42))
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        print(f"[SKIP] sklearn not available: {e}")
        return dict(proba=None, pred=None, note=f"m2i_skipped: {e}")

    if base_name == "rf":
        base = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                      n_jobs=-1, random_state=seed)
    elif base_name == "linear_svm":
        base = LinearSVC()
    else:
        base_name = "logreg"
        base = LogisticRegression(solver="liblinear", max_iter=2000, random_state=seed)

    y_tr_vec = np.asarray(y_tr).ravel().astype(int)
    base.fit(X_tr, y_tr_vec)

    def _score(est, X):
        if hasattr(est, "predict_proba"):
            p = est.predict_proba(X);  return p[:, 1] if p.ndim > 1 else p
        if hasattr(est, "decision_function"):
            z = est.decision_function(X); z = np.asarray(z, float).ravel()
            return 1.0 / (1.0 + np.exp(-z))
        yhat = est.predict(X)
        return (np.asarray(yhat, float).ravel() + 0.5) / 2.0

    p_va = _score(base, X_va)
    p_te = _score(base, X_te)

    # ---------------- threshold (validation) ----------------
    try:
        from ..utils.threshold import tune_threshold
        t = tune_threshold(p_va, y_va)
    except Exception:
        t_default = float(getattr(args, "decision_threshold", 0.5))
        t = np.median(p_va) if np.isfinite(p_va).all() else t_default

    yhat_te = (p_te >= t).astype(int)

    # ---------------- try official repo import ----------------
    used_backend = "fallback"
    try:
        repo_path = getattr(args, "m2i_repo_path", "")
        if repo_path:
            import sys, os
            if os.path.isdir(repo_path) and (repo_path not in sys.path):
                sys.path.append(repo_path)
        # 아래는 레포 구조에 따라 바뀔 수 있음: 설치되어 있으면 사용
        # from BoundApproxInterMargFairness import estimators as bai  # 예시
        # (실제 함수명은 레포에 맞게 수정)
        # ---- 만약 성공하면, 해당 모듈 API로 u_I/u_joint/s* 계산 ----
        # u_joint, u_I, s_star = bai.compute_everything(S_te, yhat_te)
        # used_backend = "official"
        pass
    except Exception:
        pass

    # ---------------- fallback implementation (paper-faithful) ----------------
    # 기본 설정
    delta = float(getattr(args, "m2i_delta", 0.1))
    eps_clip = float(getattr(args, "m2i_eps_clip", 1e-12))

    # 보조 유틸
    def _clip(p):
        return np.clip(np.asarray(p, float), eps_clip, 1.0 - eps_clip)

    def _counts(series):
        s = pd.Series(series)
        c = s.value_counts(dropna=False)
        return c.to_dict(), float(len(s))

    # 확률 추정
    def _emp_p(series):
        c, n = _counts(series); p = {k: v / n for k, v in c.items()}
        return p, n

    # p(y | group) 계산
    def _py_given_group(yhat, mask):
        y = np.asarray(yhat).ravel().astype(int)
        m = np.asarray(mask).ravel().astype(bool)
        if m.sum() == 0:
            return {0: 0.5, 1: 0.5}  # 중립 폴백
        p1 = float((y[m] == 1).mean())
        return {0: 1.0 - p1, 1: p1}

    # log-ratio 기반 불공정성 (논문식)
    def _u_single_attr(S_df, yhat, col):
        vals = pd.Series(S_df[col]).unique()
        out = []
        for y in (0, 1):
            best = 0.0
            for a in vals:
                for b in vals:
                    if a == b:
                        continue
                    Pa = _py_given_group(yhat, S_df[col] == a)[y]
                    Pb = _py_given_group(yhat, S_df[col] == b)[y]
                    r = abs(np.log(_clip(Pa)) - np.log(_clip(Pb)))
                    if r > best: best = r
            out.append(best)
        return float(max(out))  # sup_y

    def _u_joint(S_df, yhat):
        # 교차 그룹을 하나의 값으로 묶어 계산
        key = pd.util.hash_pandas_object(S_df, index=False).astype(int)
        vals = pd.Series(key).unique()
        out = []
        for y in (0, 1):
            best = 0.0
            for a in vals:
                for b in vals:
                    if a == b:
                        continue
                    Pa = _py_given_group(yhat, key == a)[y]
                    Pb = _py_given_group(yhat, key == b)[y]
                    r = abs(np.log(_clip(Pa)) - np.log(_clip(Pb)))
                    if r > best: best = r
            out.append(best)
        return float(max(out))

    def _u_I(S_df, yhat):
        # sup_y sum_k sup_{a_k,a'_k} |log P(y|A_k)/P(y|A'_k)|
        cols = list(S_df.columns)
        out = []
        for y in (0, 1):
            s = 0.0
            for c in cols:
                vals = pd.Series(S_df[c]).unique()
                best = 0.0
                for a in vals:
                    for b in vals:
                        if a == b: continue
                        Pa = _py_given_group(yhat, S_df[c] == a)[y]
                        Pb = _py_given_group(yhat, S_df[c] == b)[y]
                        r = abs(np.log(_clip(Pa)) - np.log(_clip(Pb)))
                        if r > best: best = r
                s += best
            out.append(s)
        return float(max(out))

    # s* 계산 (σ, σ_y 기반)
    def _s_star(S_df, yhat):
        cols = list(S_df.columns)
        # p_A(a)와 p_{A_k}(a_k)
        # 모든 조합 a = (a1,...,ak)의 경험확률
        A = S_df.copy()
        A_joint = pd.MultiIndex.from_frame(A).to_frame(index=False)
        # joint prob
        joint_counts = A.value_counts().to_dict()
        n = float(len(A))
        pA = {k: v / n for k, v in joint_counts.items()}
        # marginals for each col
        pAk = {}
        for c in cols:
            cnt = A[c].value_counts().to_dict()
            pAk[c] = {k: v / n for k, v in cnt.items()}

        # L = log p_A(a) - sum_k log p_{A_k}(a_k)
        def _L_of_tuple(tup):
            # tup는 dict key로 쓰인 (a1,...,ak)
            l = np.log(_clip(pA.get(tup, eps_clip)))
            for i, c in enumerate(cols):
                l -= np.log(_clip(pAk[c].get(tup[i], eps_clip)))
            return float(l)

        # E[L] 및 E[L^2]
        EL = 0.0; EL2 = 0.0
        for tup, p in pA.items():
            l = _L_of_tuple(tup)
            EL  += p * l
            EL2 += p * (l * l)
        var = max(EL2 - EL * EL, 0.0)
        sigma = np.sqrt(var)

        # conditional on Ŷ
        yhat = np.asarray(yhat).ravel().astype(int)
        # p(Ŷ)
        pY = {y: float((yhat == y).mean()) for y in (0, 1)}
        sigma_y_terms = []
        for y in (0, 1):
            idx = (yhat == y)
            ny = float(idx.sum())
            if ny == 0:
                continue
            # conditional joint p(a|y)
            Ay = A[idx].copy()
            pA_y = {k: v / ny for k, v in Ay.value_counts().to_dict().items()}
            # conditional marginal p(a_k|y)
            pAk_y = {}
            for c in cols:
                cnt = Ay[c].value_counts().to_dict()
                pAk_y[c] = {k: v / ny for k, v in cnt.items()}

            def _Ly(tup):
                l = np.log(_clip(pA_y.get(tup, eps_clip)))
                for i, c in enumerate(cols):
                    l -= np.log(_clip(pAk_y[c].get(tup[i], eps_clip)))
                return float(l)

            ELy = 0.0; EL2y = 0.0
            for tup, p in pA_y.items():
                ly = _Ly(tup)
                ELy  += p * ly
                EL2y += p * (ly * ly)
            vary = max(EL2y - ELy * ELy, 0.0)
            sigma_y_terms.append(pY[y] * vary)  # law of total variance의 가중 평균 사용

        sigma_y = np.sqrt(np.sum(sigma_y_terms)) if sigma_y_terms else 0.0

        s_star = (sigma ** (2.0 / 3.0) + sigma_y ** (2.0 / 3.0)) ** (3.0 / 2.0)
        return float(s_star), float(sigma), float(sigma_y)

    # 계산
    try:
        u_joint = _u_joint(S_te, yhat_te)
        uI = _u_I(S_te, yhat_te)
        s_star, sigma, sigma_y = _s_star(S_te, yhat_te)
    except Exception as e:
        u_joint = np.nan; uI = np.nan; s_star = np.nan; sigma = np.nan; sigma_y = np.nan
        print(f"[WARN] m2i metrics failed: {e}")

    # (참고) 확률적 상계 ε(δ)은 논문 상수와 항이 더 들어간다.
    # 여기서는 보고 지표로 delta와 s*만 기록. (정확한 바운드 구현은 공식 레포 API 사용 권장)

    note = (f"marg->inter (paper-based): base={base_name}, "
            f"tuned_t={t:.4f}, "
            f"u_joint={u_joint:.4f}, u_I={uI:.4f}, s*={s_star:.4f} "
            f"(sigma={sigma:.4f}, sigma_y={sigma_y:.4f}, delta={delta}), "
            f"backend={used_backend}")

    return dict(proba=p_te, pred=yhat_te, note=note)
