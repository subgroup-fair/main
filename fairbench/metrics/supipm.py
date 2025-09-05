# fairbench/metrics/supipm.py
import numpy as np

def _mmd2_rbf(x, y, sigma=0.5):
    # x,y: 1D probabilities
    def k(a,b): 
        a=a.reshape(-1,1); b=b.reshape(1,-1)
        return np.exp(-((a-b)**2)/(2*sigma**2))
    Kxx = k(x,x).mean(); Kyy = k(y,y).mean(); Kxy = k(x,y).mean()
    return Kxx + Kyy - 2*Kxy

def _wasserstein1d_ot(a, b):
    """
    1D p=1 Wasserstein (EMD). 우선 ot.wasserstein_1d, 없으면 ot.emd2, 둘 다 실패하면 numpy fallback.
    """
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size == 0 or b.size == 0:
        return np.nan
    import ot
    # 빠른 1D 전용 구현이 있으면 그걸 사용
    if hasattr(ot, "wasserstein_1d"):
        # 균등 가중치 (None) + p=1
        return float(ot.wasserstein_1d(np.sort(a), np.sort(b), p=1))
    # 일반 EMD (선형계획)로 계산: 비용행렬 |x-y|
    aw = np.ones(len(a)) / len(a)
    bw = np.ones(len(b)) / len(b)
    M = ot.dist(a.reshape(-1,1), b.reshape(-1,1), metric="euclidean")  # |x - y|
    return float(ot.emd2(aw, bw, M))

def supipm_wasserstein(proba, S_or_list, min_support=5):
    """
    sup_W W1( P(f(X)|S in W), P(f(X)|S not in W) )
    - f(X)=proba (1D 점수/확률)
    - tabular: S_or_list = DataFrame(0/1 컬럼)
    - image:   S_or_list = list[dict] (각 dict에 키별 0/1)
    """
    arr = np.asarray(proba).reshape(-1)
    supv = 0.0
    if isinstance(S_or_list, list):
        keys = list(S_or_list[0].keys())
        for k in keys:
            m = np.array([int(s[k]) for s in S_or_list])
            a = arr[m==1]; b = arr[m==0]
            if len(a) > min_support and len(b) > min_support:
                supv = max(supv, _wasserstein1d_ot(a, b))
        return float(supv)
    else:
        for c in S_or_list.columns:
            m = S_or_list[c].values.astype(int)
            a = arr[m==1]; b = arr[m==0]
            if len(a) > min_support and len(b) > min_support:
                supv = max(supv, _wasserstein1d_ot(a, b))
        return float(supv)

def supipm_rbf(proba, S_or_list, sigma=0.5):
    """
    sup_W MMD_k( P(f(X)|S in W), P(f(X)|S not in W) )
    - 여기서는 singleton subgroup만 sup (실전에서는 조합 W로 확장 가능)
    - tabular: S: DataFrame(여러 열의 0/1)
    - image:   S_or_list: list of dict
    """
    if isinstance(S_or_list, list):  # image
        # 각 민감 키에 대해 0/1 마스크 생성
        keys = list(S_or_list[0].keys())
        supv = 0.0
        arr = proba.reshape(-1)
        for k in keys:
            m = np.array([int(s[k]) for s in S_or_list])
            a = arr[m==1]; b = arr[m==0]
            if len(a)>5 and len(b)>5:
                supv = max(supv, np.sqrt(max(_mmd2_rbf(a,b,sigma), 0)))
        return float(supv)
    else:
        supv = 0.0; arr = proba.reshape(-1)
        for c in S_or_list.columns:
            m = S_or_list[c].values.astype(int)
            a = arr[m==1]; b = arr[m==0]
            if len(a)>5 and len(b)>5:
                supv = max(supv, np.sqrt(max(_mmd2_rbf(a,b,sigma), 0)))
        return float(supv)
