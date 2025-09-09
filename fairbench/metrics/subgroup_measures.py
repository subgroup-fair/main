# fairbench/metrics/subgroup_measures.py
import numpy as np
from tqdm import tqdm

## 건웅이가 넘겨준거: SPD/FPR/MC (worst, mean, marginal)
#########################################################
## 우리가 실험할 metrics (subgroup-based)
## [vv] 1) sup_mmd_dfcols: 두 분포의 MMD.                                                                  - singleton subgroup
## [vv] 2) sup_w1_dfcols: 두 분포의 WD.                                                                     - singleton subgroup
## [vv] 3) sup_mmd_over_V: 전체 vs 각 subgroup 분포 MMD의 최댓값.                                           - 𝒱 (subgroup subset)
## [vv] 4) sup_w1_over_V: 전체 vs 각 subgroup 분포 WD의 최댓값.                                             - 𝒱 (subgroup subset)

## [v] 5,6) worst/mean worst_group_spd, mean_group_spd: SPD.                                                - singleton subgroup
## [vv] 7,8) worst/mean worst_weighted_group_spd, mean_weighted_group_spd : 그룹 빈도로 가중 평균된 SPD.          - singleton subgroup
## [v] 9,10) worst/mean worst_spd_over_V,mean_spd_over_V : SPD.                                                - 𝒱 (subgroup subset)
## [vv] 11,12) worst/mean worst_weighted_spd_over_V, mean_weighted_spd_over_V: 그룹 빈도로 가중 평균된 SPD.        - 𝒱 (subgroup subset)

#########################################################
## 기존 subgroup 메트릭 (worst/mean)
## 1) worst/mean subgroup/marginal FPR gap
## 2) worst/mean subgroup/marginal multicalibration gap
#########################################################



# parameter
# min_support: 그룹에 이거보다 샘플 수 적으면 metric 계산 안할게
# sigma: MMD에 있는 RBF kernel bandwidth (None이면 샘플간 거리의 중앙값을 쓰겠다)

def _as_mask_list_from_V(V, n):
    """
    subgroup subset 이 input으로 들어오면 각 샘플이 해당 ss에 들어가는지 여부 반환
    subgroup subset 10개 들어오면 각 샘플의 10개 bool mask 출력
    V: list of group specs. 각 원소는
       - bool mask (shape (n,))
       - 정수 인덱스 배열
    return: list[np.ndarray(bool, shape (n,))]
    """
    masks = []
    for g in V:
        if isinstance(g, np.ndarray) and g.dtype == bool:
            m = g
        else:
            idx = np.asarray(g, dtype=int).reshape(-1)
            m = np.zeros(n, dtype=bool)
            m[idx] = True
        masks.append(m)
    return masks


def _rate(pred):
    return float(np.mean(pred)) if len(pred)>0 else np.nan


## [ ] 1) mmd_statistic(P,Q): 두 분포의 MMD.
## [ ] 3) sup_mmd: 전체 vs 각 subgroup 분포 MMD의 최댓값.


def _pairwise_rbf_sum(x, sigma):
    """
    ∑_{i<j} k(x_i,x_j) with RBF. (self-term 제외)
    E[k(x,x')] 계산하는애
    """
    x = x.reshape(-1, 1)
    # (x_i - x_j)^2 = (x^2) + (x^2)^T - 2 x x^T
    x2 = (x ** 2).sum(axis=1, keepdims=True)
    d2 = x2 + x2.T - 2.0 * (x @ x.T)
    K = np.exp(-d2 / (2.0 * sigma * sigma))
    # 대각을 0으로
    np.fill_diagonal(K, 0.0)
    # 상삼각 합
    return float(K.sum() / 2.0)

def _mmd2_unbiased(x, y, sigma):
    """
    Unbiased MMD^2 for RBF kernel (Gretton et al.).
    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return np.nan
    k_xx = _pairwise_rbf_sum(x, sigma)
    k_yy = _pairwise_rbf_sum(y, sigma)
    # 교항: 모든 (i in x, j in y)
    x_ = x.reshape(-1, 1)
    y_ = y.reshape(1, -1)
    d2 = (x_ ** 2) + (y_ ** 2) - 2.0 * (x_ @ y_)
    k_xy = np.exp(-d2 / (2.0 * sigma * sigma)).sum()
    mmd2 = (2.0 / (nx * (nx - 1))) * k_xx \
         + (2.0 / (ny * (ny - 1))) * k_yy \
         - (2.0 / (nx * ny)) * k_xy
    return float(max(mmd2, 0.0))

def _median_heuristic_sigma(z):
    """
    median heuristic for 1D scores.
    """
    z = np.asarray(z).reshape(-1)
    if z.size < 2:
        return 1.0
    zz = np.sort(z)
    d = np.abs(zz[1:] - zz[:-1])
    med = np.median(d)
    return float(max(med, 1e-3))

## [ ] 1) mmd_statistic(P,Q): 두 분포의 MMD.
def sup_mmd_dfcols(proba, S_df, sigma=None, min_support=5):
    """
    각 S_df[col]을 그룹으로 보고 sup MMD(그룹 vs 보완) 계산.
    """
    p = np.asarray(proba).reshape(-1)
    n = p.size
    if sigma is None:
        sigma = _median_heuristic_sigma(p)
    best = 0.0
    for c in S_df.columns:
        m = S_df[c].values.astype(int) == 1
        ns, nsc = int(m.sum()), int((~m).sum())
        if ns < min_support or nsc < min_support:
            continue
        v = _mmd2_unbiased(p[m], p[~m], sigma) ** 0.5  # MMD (not squared)
        if v > best:
            best = v
    return float(best)

## [ ] 3) sup_mmd: 전체 vs 각 subgroup 분포 MMD의 최댓값.
def sup_mmd_over_V(proba, V, min_support=5, sigma=None):
    """
    𝒱(list of groups) 위에서 sup MMD(그룹 vs 보완) 계산.
    V 원소: bool mask (n,) 또는 정수 인덱스 배열.
    """
    p = np.asarray(proba).reshape(-1)
    n = p.size
    masks = _as_mask_list_from_V(V, n)
    if sigma is None:
        sigma = _median_heuristic_sigma(p)
    best = 0.0
    iterator = tqdm(masks, desc="[Performance metric: worst_spd_over_V]") if len(masks) > 50 else masks
    for m in iterator:
        ns, nsc = int(m.sum()), int((~m).sum())  # ns = ss에 들어간 샘플 수, nsc = ss complement에 들어간 샘플 수
        if ns < min_support or nsc < min_support:
            continue
        v = _mmd2_unbiased(p[m], p[~m], sigma) ** 0.5
        if v > best:
            best = v
    return float(best)



## [ ] 2) wasserstein_statistic(P,Q): 두 분포의 WD.
## [ ] 4) sup_wd: 전체 vs 각 subgroup 분포 WD의 최댓값.


def _cdf_1d(values):
    """정렬·중복 포함한 값과 CDF 값 쌍 반환. """
    v = np.sort(values.reshape(-1))
    # 고유 지점에서의 CDF (계단함수 좌연속)
    uniq, counts = np.unique(v, return_counts=True)
    cdf = np.cumsum(counts) / float(v.size)
    return uniq, cdf

def _w1_1d(x, y):
    """
    Wasserstein-1 (1D) for empirical distributions.
    수치 적분: CDF 차이의 L1 적분.
    W1(P,Q) = ∫ |F_P(t) - F_Q(t)| dt
    W1(\hat{P}_n, \hat{Q}_m) = \sum |F_P(t_i) - F_Q(t_i)| * (t_{i+1}-t_i)
    (t_i: 적분을 summation으로 근사할때 구간 잘리는 샘플들, 원래 smoothing cdf인데 empirical cdf라 계단식)
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size == 0 or y.size == 0:
        return np.nan
    ux, Fx = _cdf_1d(x)
    uy, Fy = _cdf_1d(y)
    # 공통 격자에서 CDF 보간(계단함수)
    grid = np.unique(np.concatenate([ux, uy]))
    # Fx(grid), Fy(grid)
    Fx_on = np.interp(grid, ux, Fx, left=0.0, right=1.0)
    Fy_on = np.interp(grid, uy, Fy, left=0.0, right=1.0)
    # 적분: sum |ΔF| * Δx
    dx = np.diff(grid)
    dF = np.abs(Fx_on[:-1] - Fy_on[:-1])
    return float(np.sum(dF * dx))


## [ ] 2) wasserstein_statistic(P,Q): 두 분포의 WD.
def sup_w1_dfcols(proba, S_df, min_support=5):
    """
    각 S_df[col]을 그룹으로 보고 sup W1(그룹 vs 보완) 계산.
    """
    p = np.asarray(proba).reshape(-1)
    best = 0.0
    for c in S_df.columns:
        m = S_df[c].values.astype(int) == 1
        ns, nsc = int(m.sum()), int((~m).sum())
        if ns < min_support or nsc < min_support:
            continue
        v = _w1_1d(p[m], p[~m])
        if v == v and v > best:
            best = v
    return float(best)

## [ ] 4) sup_wd: 전체 vs 각 subgroup 분포 WD의 최댓값.
def sup_w1_over_V(proba, V, min_support=5):
    """
    𝒱(list of groups) 위에서 sup W1(그룹 vs 보완) 계산.
    """
    p = np.asarray(proba).reshape(-1)
    n = p.size
    masks = _as_mask_list_from_V(V, n)
    best = 0.0
    iterator = tqdm(masks, desc="[Performance metric: worst_spd_over_V]") if len(masks) > 50 else masks
    for m in iterator:
        ns, nsc = int(m.sum()), int((~m).sum()) # ns = ss에 들어간 샘플 수, nsc = ss complement에 들어간 샘플 수
        if ns < min_support or nsc < min_support:
            continue
        v = _w1_1d(p[m], p[~m])
        if v == v and v > best:
            best = v
    return float(best)



## [v] 5,6) worst/mean subgroup_spd: SPD.

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



## [ ] 7,8) worst/mean weighted_subgroup_spd: 그룹 빈도로 가중 평균된 SPD.


def worst_weighted_group_spd(proba, S_or_list, thr=0.5, min_support=5):
    """
    가중 최악 SPD: max_g  w_g * |Pr(h=1|g) - Pr(h=1)|
    w_g = |g| / n   (그룹 빈도)
    """
    arr = (np.asarray(proba).reshape(-1) >= float(thr)).astype(int)
    n = arr.size
    if n == 0:
        return np.nan
    overall = float(arr.mean())
    best = 0.0
    if isinstance(S_or_list, list):
        keys = list(S_or_list[0].keys())
        m_all = {k: np.array([int(s[k]) for s in S_or_list]) for k in keys}
        for k, m in m_all.items():
            m = (m == 1)
            ns = int(m.sum())
            if ns < min_support:
                continue
            w = ns / float(n)
            gap = abs(float(arr[m].mean()) - overall)
            best = max(best, w * gap)
    else:
        for c in S_or_list.columns:
            m = (S_or_list[c].values.astype(int) == 1)
            ns = int(m.sum())
            if ns < min_support:
                continue
            w = ns / float(n)
            gap = abs(float(arr[m].mean()) - overall)
            best = max(best, w * gap)
    return float(best)


def mean_weighted_group_spd(proba, S_or_list, thr=0.5, min_support=5):
    """
    가중 평균 SPD: (∑_g w_g * |Pr(h=1|g) - Pr(h=1)|) / (∑_g w_g)
    w_g = |g| / n   (그룹 빈도)
    """
    arr = (np.asarray(proba).reshape(-1) >= float(thr)).astype(int)
    n = arr.size
    if n == 0:
        return np.nan
    overall = float(arr.mean())
    num = 0.0
    den = 0.0
    if isinstance(S_or_list, list):  # image 스타일
        keys = list(S_or_list[0].keys())
        m_all = {k: np.array([int(s[k]) for s in S_or_list]) for k in keys}
        for k, m in m_all.items():
            m = (m == 1)
            ns = int(m.sum())
            if ns < min_support:
                continue
            w = ns / float(n)
            num += w * abs(float(arr[m].mean()) - overall)
            den += w
    else:  # DataFrame
        for c in S_or_list.columns:
            m = (S_or_list[c].values.astype(int) == 1)
            ns = int(m.sum())
            if ns < min_support:
                continue
            w = ns / float(n)
            num += w * abs(float(arr[m].mean()) - overall)
            den += w
    if den == 0.0:
        return 0.0
    return float(num / den)



## [v] 9,10) worst/mean subgroup_subset_spd / dp_sup_V,dp_mean_V : SPD.

def worst_spd_over_V(proba, V, thr=0.5, min_support=5):
    """
    𝒱 위에서 DP sup: max_G |Pr(h=1|G) - Pr(h=1)|.
    """
    arr = (np.asarray(proba).reshape(-1) >= float(thr)).astype(int)
    n = arr.size
    overall = float(arr.mean()) if n > 0 else np.nan
    masks = _as_mask_list_from_V(V, n)
    best = 0.0
    iterator = tqdm(masks, desc="[Performance metric: worst_spd_over_V]") if len(masks) > 50 else masks
    for m in iterator:
        ns = int(m.sum())
        if ns < min_support:
            continue
        gap = abs(float(arr[m].mean()) - overall)
        if gap > best:
            best = gap
    return float(best)

def mean_spd_over_V(proba, V, thr=0.5, min_support=5):
    """
    𝒱 위에서 DP 평균 격차.
    """
    arr = (np.asarray(proba).reshape(-1) >= float(thr)).astype(int)
    n = arr.size
    overall = float(arr.mean()) if n > 0 else np.nan
    masks = _as_mask_list_from_V(V, n)
    diffs = []
    iterator = tqdm(masks, desc="[Performance metric: mean_spd_over_V]") if len(masks) > 50 else masks
    for m in iterator:
        ns = int(m.sum())
        if ns < min_support:
            continue
        diffs.append(abs(float(arr[m].mean()) - overall))
    return float(np.mean(diffs)) if len(diffs) > 0 else 0.0





## [v] 11,12) worst/mean weighted_subgroup_subset_spd: 그룹 빈도로 가중 평균된 SPD

def worst_weighted_spd_over_V(proba, V, thr=0.5, min_support=5):
    """
    가중 최악 SPD over 𝒱: max_G  w_G * |Pr(h=1|G) - Pr(h=1)|
    w_G = |G| / n
    """
    arr = (np.asarray(proba).reshape(-1) >= float(thr)).astype(int)
    n = arr.size
    if n == 0:
        return np.nan
    overall = float(arr.mean())
    masks = _as_mask_list_from_V(V, n)
    best = 0.0
    iterator = tqdm(masks, desc="[Performance metric: worst_weighted_spd_over_V]") if len(masks) > 50 else masks
    for m in iterator:
        ns = int(m.sum())
        if ns < min_support:
            continue
        w = ns / float(n)
        gap = abs(float(arr[m].mean()) - overall)
        best = max(best, w * gap)
    return float(best)


def mean_weighted_spd_over_V(proba, V, thr=0.5, min_support=5):
    """
    가중 평균 SPD over 𝒱: (∑_G w_G * |Pr(h=1|G) - Pr(h=1)|) / (∑_G w_G)
    w_G = |G| / n
    """
    arr = (np.asarray(proba).reshape(-1) >= float(thr)).astype(int)
    n = arr.size
    if n == 0:
        return np.nan
    overall = float(arr.mean())
    masks = _as_mask_list_from_V(V, n)
    num = 0.0
    den = 0.0
    iterator = tqdm(masks, desc="[Performance metric: mean_weighted_spd_over_V]") if len(masks) > 50 else masks
    for m in iterator:
        ns = int(m.sum())
        if ns < min_support:
            continue
        w = ns / float(n)
        num += w * abs(float(arr[m].mean()) - overall)
        den += w
    if den == 0.0:
        return 0.0
    return float(num / den)





#########################################################
## 기존의 metrics
#########################################################



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