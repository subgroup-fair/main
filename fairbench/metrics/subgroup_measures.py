# fairbench/metrics/subgroup_measures.py
import numpy as np

def _rate(pred):
    return float(np.mean(pred)) if len(pred)>0 else np.nan

def worst_group_spd(proba, S_or_list):
    """max_g |Pr(h=1|g) - Pr(h=1)|; h는 proba>=0.5 대체 (측정 시에만)"""
    arr = (proba>=0.5).astype(int)
    overall = _rate(arr)
    worst = 0.0
    if isinstance(S_or_list, list):
        keys = list(S_or_list[0].keys())
        m_all = {k: np.array([int(s[k]) for s in S_or_list]) for k in keys}
        for k,m in m_all.items():
            a = arr[m==1]; 
            if len(a)>5: worst = max(worst, abs(_rate(a)-overall))
        return float(worst)
    else:
        for c in S_or_list.columns:
            m = S_or_list[c].values.astype(int)
            a = arr[m==1]
            if len(a)>5: worst = max(worst, abs(_rate(a)-overall))
        return float(worst)

def worst_group_fpr_gap(pred, y, S_df):
    """max_g |FPR_g - FPR_all|, FPR = P(h=1|y=0)"""
    pred = pred.reshape(-1); y=y.reshape(-1)
    mask_all = (y==0); 
    if mask_all.sum()<5: return np.nan
    fpr_all = (pred[mask_all]==1).mean()
    worst=0.0
    for c in S_df.columns:
        m = (S_df[c].values==1)&(y==0)
        if m.sum()<5: continue
        fpr = (pred[m]==1).mean()
        worst=max(worst, abs(fpr-fpr_all))
    return float(worst)

def multicalib_worst_gap(proba, y, S_df, bins=10):
    """max_g max_b |E[y|score in bin & g] - avg(score in bin & g)|"""
    proba=proba.reshape(-1); y=y.reshape(-1)
    edges = np.linspace(0,1,bins+1)
    worst=0.0
    for c in S_df.columns:
        m = S_df[c].values==1
        if m.sum()<10: continue
        for i in range(bins):
            lo,hi = edges[i], edges[i+1]
            idx = m & (proba>=lo) & (proba<hi)
            if idx.sum()<10: continue
            gap = abs(y[idx].mean() - proba[idx].mean())
            worst=max(worst, float(gap))
    return worst if worst>0 else 0.0

def mean_group_spd(proba, S_or_list, thr=0.5, min_support=5):
    """
    평균 SPD gap: mean_g |Pr(h=1|g) - Pr(h=1)|  (측정시에만 h = 1(proba>=thr))
    - tabular: S_or_list = DataFrame (0/1 컬럼)
    - image:   S_or_list = list[dict] (각 dict의 키가 민감 속성)
    """
    arr = (proba.reshape(-1) >= thr).astype(int)
    overall = float(arr.mean()) if len(arr) > 0 else np.nan
    diffs = []
    if isinstance(S_or_list, list):  # image 스타일
        keys = list(S_or_list[0].keys())
        m_all = {k: np.array([int(s[k]) for s in S_or_list]) for k in keys}
        for k, m in m_all.items():
            a = arr[m == 1]
            if a.size >= min_support:
                diffs.append(abs(float(a.mean()) - overall))
    else:  # DataFrame
        for c in S_or_list.columns:
            m = S_or_list[c].values.astype(int)
            a = arr[m == 1]
            if a.size >= min_support:
                diffs.append(abs(float(a.mean()) - overall))
    return float(np.mean(diffs)) if len(diffs) > 0 else 0.0


def mean_group_fpr_gap(pred, y, S_df, min_support=5):
    """
    평균 FPR gap: mean_g |FPR_g - FPR_all|,  FPR = P(h=1 | y=0)
    - pred: {0,1} 예측 (이미 threshold 적용된 값 사용 권장)
    - y: {0,1} 라벨
    - S_df: 각 컬럼이 0/1인 그룹 마스크
    """
    pred = pred.reshape(-1)
    y = y.reshape(-1)
    mask_all = (y == 0)
    if mask_all.sum() < min_support:
        return np.nan
    fpr_all = float((pred[mask_all] == 1).mean())

    gaps = []
    for c in S_df.columns:
        m = (S_df[c].values == 1) & (y == 0)
        if m.sum() < min_support:
            continue
        fpr = float((pred[m] == 1).mean())
        gaps.append(abs(fpr - fpr_all))
    return float(np.mean(gaps)) if len(gaps) > 0 else 0.0


def multicalib_mean_gap(proba, y, S_df, bins=10, min_support=10):
    """
    평균 multicalibration gap:
      mean_{g,b} | E[y | p in bin & g] - E[p | p in bin & g] |
    - proba ∈ [0,1], y ∈ {0,1}
    - bin마다/그룹마다 표본이 min_support 이상일 때만 포함
    """
    proba = proba.reshape(-1)
    y = y.reshape(-1)
    edges = np.linspace(0, 1, bins + 1)
    gaps = []
    for c in S_df.columns:
        m_g = (S_df[c].values == 1)
        if m_g.sum() < min_support:
            continue
        for i in range(bins):
            lo, hi = edges[i], edges[i + 1]
            idx = m_g & (proba >= lo) & (proba < hi if i < bins - 1 else proba <= hi)
            if idx.sum() < min_support:
                continue
            gaps.append(abs(float(y[idx].mean()) - float(proba[idx].mean())))
    return float(np.mean(gaps)) if len(gaps) > 0 else 0.0


# =====================================================================
# ============== 여기부터 Marginal(속성별) 메트릭 추가 =================
# =====================================================================

def _families_from_S(S_or_list):
    """
    S를 속성 family로 묶는 헬퍼.
    - DataFrame: 'attr_value' 형태의 컬럼은 'attr'를 family로 묶음.
                 '_'가 없으면 해당 컬럼명이 곧 family(이진).
    - list[dict] (image): 각 key가 하나의 이진 속성 → family[key] = [key]
    return: dict[str, list[str]]  (family -> list of columns/keys)
    """
    fam = {}
    if isinstance(S_or_list, list):
        keys = list(S_or_list[0].keys())
        for k in keys:
            fam[k] = [k]
    else:
        for c in S_or_list.columns:
            parts = str(c).split("_", 1)
            family = parts[0] if len(parts) > 1 else str(c)
            fam.setdefault(family, []).append(c)
    return fam

def _mask_from_key(S_or_list, key):
    if isinstance(S_or_list, list):
        return np.array([int(s[key]) for s in S_or_list])
    else:
        return S_or_list[key].values.astype(int)

def _spd_for_family(arr01, S_or_list, keys, min_support=5):
    """
    family 안에서의 SPD 정의:
      - keys가 1개(이진 속성): |Pr(h=1|S=1) - Pr(h=1|S=0)|
      - keys가 여러 개(원-핫 카테고리): max_{i,j} |Pr(h=1|k_i) - Pr(h=1|k_j)|
    """
    if len(keys) == 1:
        m = _mask_from_key(S_or_list, keys[0]) == 1
        n1, n0 = m.sum(), (~m).sum()
        if n1 >= min_support and n0 >= min_support:
            return abs(float(arr01[m].mean()) - float(arr01[~m].mean()))
        return np.nan
    # multi-cat
    rates = []
    for k in keys:
        m = _mask_from_key(S_or_list, k) == 1
        if m.sum() >= min_support:
            rates.append(float(arr01[m].mean()))
    if len(rates) < 2:
        return np.nan
    return float(np.max(rates) - np.min(rates))

def _fpr_for_family(pred01, y, S_or_list, keys, min_support=5):
    """
    family 안에서의 FPR 정의(위와 동일한 방식으로):
      - binary: |FPR(S=1) - FPR(S=0)|
      - multi-cat: max_{i,j} |FPR(k_i) - FPR(k_j)|
    """
    y = y.reshape(-1)
    pred01 = pred01.reshape(-1)
    mask_all = (y == 0)
    if mask_all.sum() < min_support:
        return np.nan

    def fpr_of(m):
        idx = mask_all & (m == 1)
        if idx.sum() < min_support:
            return None
        return float((pred01[idx] == 1).mean())

    if len(keys) == 1:
        m = _mask_from_key(S_or_list, keys[0])
        f1 = fpr_of(m)
        f0 = fpr_of(1 - m)
        if f1 is None or f0 is None:
            return np.nan
        return abs(f1 - f0)

    vals = []
    for k in keys:
        m = _mask_from_key(S_or_list, k)
        f = fpr_of(m)
        if f is not None:
            vals.append(f)
    if len(vals) < 2:
        return np.nan
    return float(np.max(vals) - np.min(vals))

def _mc_for_family(proba, y, S_or_list, keys, bins=10, min_support=10):
    """
    family 안에서의 multicalibration worst gap:
      - binary: max_b gap(S=1,b), gap(S=0,b) 중 최대
      - multi-cat: max_{k in keys} max_b gap(k,b)
    """
    proba = proba.reshape(-1)
    y = y.reshape(-1)
    edges = np.linspace(0, 1, bins + 1)

    def max_gap_of_mask(m):
        worst = 0.0
        for i in range(bins):
            lo, hi = edges[i], edges[i + 1]
            idx = (m == 1) & (proba >= lo) & (proba < (hi if i < bins - 1 else proba.max() + 1e-12))
            if idx.sum() < min_support:
                continue
            worst = max(worst, abs(float(y[idx].mean()) - float(proba[idx].mean())))
        return worst if worst > 0 else None

    gaps = []
    if len(keys) == 1:
        m = _mask_from_key(S_or_list, keys[0])
        g1 = max_gap_of_mask(m)
        g0 = max_gap_of_mask(1 - m)
        for g in (g1, g0):
            if g is not None:
                gaps.append(g)
    else:
        for k in keys:
            m = _mask_from_key(S_or_list, k)
            g = max_gap_of_mask(m)
            if g is not None:
                gaps.append(g)

    if len(gaps) == 0:
        return np.nan
    return float(np.max(gaps))

# -------- Marginal SPD --------
def marginal_spd_worst(proba, S_or_list, thr=0.5, min_support=5, families=None):
    arr01 = (proba.reshape(-1) >= thr).astype(int)
    fam = families or _families_from_S(S_or_list)
    vals = []
    for a, keys in fam.items():
        v = _spd_for_family(arr01, S_or_list, keys, min_support=min_support)
        if v == v:  # not NaN
            vals.append(v)
    return float(np.max(vals)) if len(vals) else 0.0

def marginal_spd_mean(proba, S_or_list, thr=0.5, min_support=5, families=None):
    arr01 = (proba.reshape(-1) >= thr).astype(int)
    fam = families or _families_from_S(S_or_list)
    vals = []
    for a, keys in fam.items():
        v = _spd_for_family(arr01, S_or_list, keys, min_support=min_support)
        if v == v:
            vals.append(v)
    return float(np.mean(vals)) if len(vals) else 0.0

# -------- Marginal FPR --------
def marginal_fpr_worst(pred, y, S_or_list, min_support=5, families=None):
    fam = families or _families_from_S(S_or_list)
    vals = []
    for a, keys in fam.items():
        v = _fpr_for_family(pred, y, S_or_list, keys, min_support=min_support)
        if v == v:
            vals.append(v)
    return float(np.max(vals)) if len(vals) else 0.0

def marginal_fpr_mean(pred, y, S_or_list, min_support=5, families=None):
    fam = families or _families_from_S(S_or_list)
    vals = []
    for a, keys in fam.items():
        v = _fpr_for_family(pred, y, S_or_list, keys, min_support=min_support)
        if v == v:
            vals.append(v)
    return float(np.mean(vals)) if len(vals) else 0.0

# -------- Marginal Multicalibration --------
def marginal_mc_worst(proba, y, S_or_list, bins=10, min_support=10, families=None):
    fam = families or _families_from_S(S_or_list)
    vals = []
    for a, keys in fam.items():
        v = _mc_for_family(proba, y, S_or_list, keys, bins=bins, min_support=min_support)
        if v == v:
            vals.append(v)
    return float(np.max(vals)) if len(vals) else 0.0

def marginal_mc_mean(proba, y, S_or_list, bins=10, min_support=10, families=None):
    fam = families or _families_from_S(S_or_list)
    vals = []
    for a, keys in fam.items():
        v = _mc_for_family(proba, y, S_or_list, keys, bins=bins, min_support=min_support)
        if v == v:
            vals.append(v)
    return float(np.mean(vals)) if len(vals) else 0.0