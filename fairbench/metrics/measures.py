from __future__ import annotations
import pandas as pd
from itertools import combinations

#####################################
import numpy as np, random, torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
######################################################

def _wasserstein1d_ot(a, b):
    """
    1D p=1 Wasserstein
    """
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size == 0 or b.size == 0:
        return np.nan
    import ot
    if hasattr(ot, "wasserstein_1d"):
        return float(ot.wasserstein_1d(np.sort(a), np.sort(b), p=1))
    aw = np.ones(len(a)) / len(a)
    bw = np.ones(len(b)) / len(b)
    M = ot.dist(a.reshape(-1,1), b.reshape(-1,1), metric="euclidean")  # |x - y|
    return float(ot.emd2(aw, bw, M))

def supipm_wasserstein(proba, S_or_list, min_support=5):
    """
    sup_W W1( P(f(X)|S in W), P(f(X)|S not in W) )
    """
    arr = np.asarray(proba).reshape(-1)
    supv = 0.0
    if isinstance(S_or_list, list):
        keys = list(S_or_list[0].keys())
        for k in keys:
            m = np.array([int(s[k]) for s in S_or_list])
            a = arr[m==1]; b = arr
            if len(a) > min_support and len(b) > min_support:
                supv = max(supv, _wasserstein1d_ot(a, b))
        return float(supv)
    else:
        for c in S_or_list.columns:
            m = S_or_list[c].values.astype(int)
            a = arr[m==1]; b = arr
            if len(a) > min_support and len(b) > min_support:
                supv = max(supv, _wasserstein1d_ot(a, b))
        return float(supv)


# column family
def _families_from_S(S: pd.DataFrame) -> dict[str, list[str]]:
    fam = {}
    for c in S.columns:
        parts = str(c).split("_", 1)
        family = parts[0] if len(parts) > 1 else str(c)
        fam.setdefault(family, []).append(c)
    return fam

# 1D Wasserstein-1
def _w1_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    if x.size == 0 or y.size == 0:
        return 0.0
    x = np.sort(x); y = np.sort(y)
    grid = np.unique(np.concatenate([x, y]))
    Fx = np.searchsorted(x, grid, side="right") / float(x.size)
    Fy = np.searchsorted(y, grid, side="right") / float(y.size)
    dx = np.diff(grid)
    dF = np.abs(Fx[:-1] - Fy[:-1])
    return float(np.sum(dF * dx))

# SPD
def _partition_aggregate_kearns(proba01: np.ndarray, S: pd.DataFrame, cols: list[str],
                                min_support: int = 1) -> float:
    n = proba01.size
    if n == 0 or len(cols) == 0:
        return 0.0
    overall = float(proba01.mean())
    A = S[cols].values.astype(int)
    uniq, inv = np.unique(A, axis=0, return_inverse=True)
    total = 0.0
    for gid in range(uniq.shape[0]):
        m = (inv == gid)
        ns = int(m.sum())
        if ns < min_support:
            continue
        w = ns / float(n)
        gap = abs(float(proba01[m].mean()) - overall)
        total += w * gap
    return float(total)

def _partition_aggregate_w1(proba: np.ndarray, S: pd.DataFrame, cols: list[str],
                            min_support: int = 1) -> float:
    n = proba.size
    if n == 0 or len(cols) == 0:
        return 0.0
    all_scores = np.asarray(proba, float).reshape(-1)
    A = S[cols].values.astype(int)
    uniq, inv = np.unique(A, axis=0, return_inverse=True)
    total = 0.0
    for gid in range(uniq.shape[0]):
        m = (inv == gid)
        ns = int(m.sum())
        if ns < min_support:
            continue
        w = ns / float(n)
        total += w * _w1_1d(all_scores[m], all_scores) 
    return float(total)

# 1st marginal
def marginal_kearns_order1_worst(proba: np.ndarray, S: pd.DataFrame,
                                 thr: float = 0.5, min_support: int = 1) -> float:
    proba01 = (np.asarray(proba, float).reshape(-1) >= float(thr)).astype(int)
    fam = _families_from_S(S)
    vals = []
    for _, cols1 in fam.items():
        vals.append(_partition_aggregate_kearns(proba01, S, cols1, min_support))
    return float(max(vals) if len(vals) else 0.0)

# (2) 1st marginal Wasserstein
def marginal_wasserstein_order1_worst(proba: np.ndarray, S: pd.DataFrame,
                                      min_support: int = 1) -> float:
    scores = np.asarray(proba, float).reshape(-1)
    fam = _families_from_S(S)
    vals = []
    for _, cols1_1 in fam.items():
        vals.append(_partition_aggregate_w1(scores, S, cols1_1, min_support))
    return float(max(vals) if len(vals) else 0.0)

# (3) 2nd marginal
def marginal_kearns_order2_worst(proba: np.ndarray, S: pd.DataFrame,
                                 thr: float = 0.5, min_support: int = 1) -> float:
    proba01 = (np.asarray(proba, float).reshape(-1) >= float(thr)).astype(int)
    fam = _families_from_S(S)
    fam_names = list(fam.keys())
    vals = []
    for fam_pair in combinations(fam_names, 2):
        cols2 = sum([fam[f] for f in fam_pair], [])
        # print("fam_pair: ", cols2)
        vals.append(_partition_aggregate_kearns(proba01, S, cols2, min_support))
    return float(max(vals) if len(vals) else 0.0)

# (4) 3rd marginal
def marginal_kearns_order3_worst(proba: np.ndarray, S: pd.DataFrame,
                                 thr: float = 0.5, min_support: int = 1) -> float:
    proba01 = (np.asarray(proba, float).reshape(-1) >= float(thr)).astype(int)
    fam = _families_from_S(S)
    fam_names = list(fam.keys())
    vals = []
    for fam_trip in combinations(fam_names, 3):
        
        cols3 = sum([fam[f] for f in fam_trip], [])
        vals.append(_partition_aggregate_kearns(proba01, S, cols3, min_support))
    return float(max(vals) if len(vals) else 0.0)

# (5) subgroup fairness
def kearns_subgroup_worst(proba: np.ndarray, S: pd.DataFrame,
                          thr: float = 0.5, min_support: int = 1) -> float:
    proba01 = (np.asarray(proba, float).reshape(-1) >= float(thr)).astype(int)
    n = proba01.size
    if n == 0:
        return 0.0
    overall = float(proba01.mean())
    A = S.values.astype(int)
    uniq, inv = np.unique(A, axis=0, return_inverse=True)
    best = 0.0
    for gid in range(uniq.shape[0]):
        m = (inv == gid)
        ns = int(m.sum())
        if ns < min_support:
            continue
        w = ns / float(n)
        gap = abs(float(proba01[m].mean()) - overall)
        best = max(best, w * gap)
    return float(best)
