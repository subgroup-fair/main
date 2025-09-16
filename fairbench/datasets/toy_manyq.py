# # fairbench/datasets/toy_manyq.py
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from ..utils.trainval import split_train_val

# def load_toy_manyq(args):
#     n = 2000; d = 60; q = args.q
#     assert d > q, f"d must be larger than q, but got d={d} <= q={q}"
#     rng = np.random.default_rng(args.seed)
#     X = rng.normal(size=(n, d)).astype(np.float32)

#     # latent sensitive factors (q binary), correlated with some features
#     S = (rng.normal(size=(n, q)) + 0.5*X[:, :min(d, q)][:, None]).mean(axis=1, keepdims=True)
#     S = (rng.normal(size=(n, q)) > 0.0).astype(int)  # 독립적인 q도 가능
#     S = pd.DataFrame(S, columns=[f"s{j}" for j in range(q)])

#     # target: non-linear function + mild spurious corr with a few sensitive attrs
#     logits = 0.8*X[:,0] - 0.6*X[:,1] + 0.5*X[:,2]*X[:,3] + 0.2*(S.values[:,:min(q,5)].sum(axis=1))
#     p = 1/(1+np.exp(-logits))
#     y = (rng.uniform(size=n) < p).astype(int)

#     X = pd.DataFrame(X, columns=[f"x{j}" for j in range(d)])

#     X_tr, X_te, y_tr, y_te, S_tr, S_te = train_test_split(X, y, S, test_size=0.2, random_state=args.seed, stratify=y)
#     X_tr, X_va, y_tr, y_va, S_tr, S_va = split_train_val(X_tr, y_tr, S_tr, val_size=0.2, seed=args.seed)

#     return dict(
#         X_train=X_tr, y_train=y_tr, S_train=S_tr,
#         X_val=X_va,   y_val=y_va,   S_val=S_va,
#         X_test=X_te,  y_test=y_te,  S_test=S_te,
#         type="tabular"
#     )

# fairbench/datasets/toy_manyq.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ..utils.trainval import split_train_val

def load_toy_manyq(args):
    """
    q=10, 서브그룹 2^q=1024개를 모두 생성.
    각 서브그룹 샘플 수를 10~20 사이(기본: 16개 균등)로 설정해 거의 균등한 민감 분포를 만들고,
    라벨은 (i) 모든 마진에서 base rate 차이가 나도록 선형 항을 주고,
          (ii) 고차 상호작용(예: 3-way product)으로 교차 서브그룹 격차를 크게 만든다.
    """
    # ---- 하이퍼/파라미터 ----
    q = int(getattr(args, "q", 10))
    assert q == 10, f"이 토이는 q=10에 맞춰 설계됨 (지금 q={q})"
    seed = int(getattr(args, "seed", 42))
    rng = np.random.default_rng(seed)

    # 서브그룹당 샘플 수 설정: 기본은 균등 16개. (원하면 Uniform[10,20]로)
    per_group = getattr(args, "toy_per_group", 16)  # 정수면 균등, None이면 랜덤
    min_pg = int(getattr(args, "toy_min_per_group", 10))
    max_pg = int(getattr(args, "toy_max_per_group", 20))

    # 피처 차원
    d = int(getattr(args, "d", 64))  # q + 여분

    # ---- 1024개 서브그룹 코드(이진 10비트) + 그룹별 샘플 수 ----
    G = 1 << q  # 1024
    codes = np.array([[(i >> j) & 1 for j in range(q)] for i in range(G)], dtype=int)  # (1024, q)

    if per_group is None:
        counts = rng.integers(low=min_pg, high=max_pg + 1, size=G)  # 10~20
    else:
        counts = np.full(G, int(per_group), dtype=int)  # 균등 16

    N = int(counts.sum())
    S = np.repeat(codes, counts, axis=0)  # (N, q)
    S_df = pd.DataFrame(S, columns=[f"s{j}" for j in range(q)])

    # ---- 피처 X: 일부는 S와 상관을 갖게 만들어 예측 신호 제공 ----
    X = rng.normal(size=(N, d)).astype(np.float32)
    s_pm = 2 * S - 1  # {0,1} -> {-1,+1}
    # 처음 q개 피처를 민감속성과 상관 있도록 주입
    noise = 0.25 * rng.normal(size=(N, q))
    X[:, :q] = s_pm + noise

    # ---- 라벨 생성: (i) 모든 마진에서 base-rate 차이 + (ii) 고차 상호작용(3-way) ----
    # (i) 각 S_j에 대해 양의 가중치 -> 모든 마진에서 불공정 보장
    w_s = rng.uniform(0.35, 0.60, size=q)  # 각 마진 base-rate shift
    term_linear_S = (s_pm @ w_s)  # (N,)

    # (ii) 3-way 상호작용 여러 개 -> marginal 완화 후에도 교차 서브그룹에서 큰 격차
    n_inter = int(getattr(args, "toy_num_interactions", 12))
    term_inter = np.zeros(N)
    for _ in range(n_inter):
        idx = rng.choice(q, size=3, replace=False)
        coef = rng.uniform(0.8, 1.2) * rng.choice([-1, 1])
        term_inter += coef * np.prod(s_pm[:, idx], axis=1)

    # (iii) X 기반 신호(약간): 초반 6개 피처 사용
    kx = min(d, 6)
    w_x = rng.normal(scale=0.6, size=kx)
    term_x = X[:, :kx] @ w_x

    # 로짓 합성 + 스케일/센터링
    logits = term_linear_S + term_inter + term_x
    logits = (logits - logits.mean()) / (logits.std() + 1e-8) * 1.5  # 과포화 방지
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=N) < p).astype(int)

    # ---- pandas 포맷 ----
    X_df = pd.DataFrame(X, columns=[f"x{j}" for j in range(d)])

    # ---- split ----
    X_tr, X_te, y_tr, y_te, S_tr, S_te = train_test_split(
        X_df, y, S_df, test_size=0.2, random_state=seed, stratify=y
    )
    X_tr, X_va, y_tr, y_va, S_tr, S_va = split_train_val(
        X_tr, y_tr, S_tr, val_size=0.2, seed=seed
    )

    # ---- 메타(확인용) ----
    grp_counts = S_df.groupby(list(S_df.columns)).size()
    meta = dict(
        q=q,
        groups=G,
        N=N,
        per_group_stats=dict(min=int(grp_counts.min()), max=int(grp_counts.max()), mean=float(grp_counts.mean())),
        interactions=n_inter,
        design="all-margins-unfair + high-order interactions",
    )

    return dict(
        X_train=X_tr, y_train=y_tr, S_train=S_tr,
        X_val=X_va,   y_val=y_va,   S_val=S_va,
        X_test=X_te,  y_test=y_te,  S_test=S_te,
        type="tabular",
        dataset="toy_manyq",
        meta=meta,
    )
