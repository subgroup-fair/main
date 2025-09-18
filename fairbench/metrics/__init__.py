# fairbench/metrics/__init__.py
import numpy as np
from .accuracy import accuracy
from .supipm import supipm_rbf, supipm_wasserstein
from .subgroup_measures import (
    worst_group_spd, mean_group_spd,
    worst_group_fpr_gap, mean_group_fpr_gap,
    multicalib_worst_gap, multicalib_mean_gap,
    # ↓↓↓ marginal metrics 추가
    marginal_spd_worst, marginal_spd_mean,
    marginal_fpr_worst, marginal_fpr_mean,
    marginal_mc_worst, marginal_mc_mean,
    # ===== [추가] 12개 메트릭용 함수들 =====
    # 1–2) singleton supIPM
    sup_mmd_dfcols, sup_w1_dfcols,
    # 3–4) V supIPM
    sup_mmd_over_V, sup_w1_over_V,
    # 7–8) weighted SPD (singleton)
    worst_weighted_group_spd, mean_weighted_group_spd,
    # 9–10) DP over V (sup/mean)
    worst_spd_over_V, mean_spd_over_V,
    # 11–12) weighted SPD over V
    worst_weighted_spd_over_V, mean_weighted_spd_over_V,
)
from tqdm.auto import tqdm

import numpy as np, random, torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# === [ADD] imports (파일 상단 근처) ===
import numpy as np
from itertools import product  # tqdm, combinations 등 기존 import 유지

# ===== NEW: Random packing unions over disjoint base subgroups =====
def _random_pack_unions_from_counts(counts, T, M=64, seed=0, max_len=None, max_cols=None, log_prefix="V"):
    """
    (1) 관측된 디스조인트 셀 인덱스를 무작위로 나열
    (2) 앞에서부터 누적합이 T에 도달할 때까지 합쳐 하나의 subset으로 저장 (여집합도 T 이상)
    (3) 리스트 끝까지 반복하여 여러 subset 생성
    (4) 위 과정을 M회 반복
    (5) 중복 제거하여 반환
    (6) 핵심 로그를 print로 남김
    """
    import numpy as np

    N = int(np.sum(counts))
    K = len(counts)
    if K == 0 or T <= 0:
        print(f"[{log_prefix}-pack] early-exit: K={K}, T={T}")
        return []

    rng = np.random.default_rng(seed if seed is not None else 0)
    seen = set()
    out = []

    print(f"[{log_prefix}-pack] N={N}, K={K}, T={T}, M={M}, max_len={max_len}, max_cols={max_cols}")

    for rep in range(int(M)):
        perm = rng.permutation(K)
        i = 0
        made_this_round = 0

        while i < K:
            cum = 0
            members = []
            # 누적합이 T 도달할 때까지 앞에서부터 채움
            while i < K and cum < T:
                j = int(perm[i])
                c = int(counts[j])
                if c > 0:            # 관측된 셀만 사용
                    members.append(j)
                    cum += c
                i += 1
                if (max_len is not None) and (len(members) >= int(max_len)) and (cum < T):
                    # 길이 제한 때문에 더 못 채우면 중단
                    break

            # T 도달하면 후보로 채택 (여집합도 T 이상일 때)
            if cum >= T:
                if (N - cum) >= T:
                    key = tuple(sorted(members))  # 순서 불변 키
                    if key not in seen:
                        seen.add(key)
                        out.append(key)
                        made_this_round += 1
                        if (max_cols is not None) and (len(out) >= int(max_cols)):
                            print(f"[{log_prefix}-pack] max_cols hit: {len(out)}")
                            return out
            else:
                # 남은 걸로 T 도달 불가 → 종료
                continue

        print(f"[{log_prefix}-pack] rep {rep+1}/{M}: new={made_this_round}, cum={len(out)}")

    print(f"[{log_prefix}-pack] done. unique unions={len(out)}")
    return out


# # === [ADD] Apriori 유틸: counts -> minimal unions ===
# def _apriori_min_cover_from_counts(counts, T, max_len=None):
#     """
#     각 서브그룹의 데이터 개수 보고 T넘는 최소 union 찾음
#     counts: list[int]  # 관측된 디스조인트 셀의 크기
#     T: int             # 최소 지지도(ceil(gamma*N))
#     max_len: Optional[int]  # 합집합 크기 제한
#     return: list[tuple]     # '최소(minimal)' 합집합의 인덱스 튜플들
#     """
#     Lk = []  # list[(tuple(idx), sum_count)]
#     results = []  # list[tuple(idx)]
#     for i, c in enumerate(counts):
#         if c >= T:
#             results.append((i,))
#         elif c > 0:
#             Lk.append(((i,), c))
#     k = 1
#     def keyset(L): return {key for (key, _) in L}
#     while Lk and (max_len is None or k < max_len):
#         k += 1
#         prev_keys = keyset(Lk)
#         Lk_sorted = sorted(Lk, key=lambda x: x[0])
#         Ck = []
#         for ai in tqdm(range(len(Lk_sorted))):
#             A, sumA = Lk_sorted[ai]
#             for bi in range(ai+1, len(Lk_sorted)):
#                 B, _ = Lk_sorted[bi]
#                 if A[:-1] != B[:-1]:
#                     break
#                 cand = A + (B[-1],)  # 사전식 확장
#                 ok = True
#                 for r in range(len(cand)):
#                     if (cand[:r] + cand[r+1:]) not in prev_keys:
#                         ok = False; break
#                 if not ok: continue
#                 sumC = sumA + counts[B[-1]]
#                 Ck.append((cand, sumC))
#         nextL = []
#         for cand, s in Ck:
#             if s >= T:
#                 results.append(cand)  # 처음 T 넘겼으니 '최소'
#             else:
#                 nextL.append((cand, s))
#         Lk = nextL
#     return results

# === [ADD] 비트패킹 + 마스크 생성 (DataFrame) ===
def _bitpack_cells_from_df_for_metrics(S_df):
    """
    S_df: (N x q) {0,1} DataFrame
    return: masks: List[np.ndarray(bool)], counts: List[int]
    """
    cols = list(S_df.columns)
    if len(cols) == 0 or len(S_df) == 0:
        return [], []
    B = np.stack([S_df[c].astype(np.int8).values for c in cols], axis=1)  # (N,q)
    codes = np.zeros(len(S_df), dtype=np.int64)
    for j in range(B.shape[1]):
        codes |= (B[:, j].astype(np.int64) << j)
    uniq, inv = np.unique(codes, return_inverse=True)
    masks, counts = [], []
    for k in range(len(uniq)):
        m = (inv == k)
        masks.append(m)
        counts.append(int(m.sum()))
    return masks, counts

# === [ADD] 비트패킹 + 마스크 생성 (list[dict]) ===
def _bitpack_cells_from_S_list_for_metrics(S_list, key_order=None):
    """
    S_list: List[dict] with binary values
    key_order: 고정 키 순서(재현성)
    """
    if len(S_list) == 0:
        return [], [], []
    keys = key_order or list(S_list[0].keys())
    N = len(S_list)
    codes = np.zeros(N, dtype=np.int64)
    for j, k in enumerate(keys):
        bit = np.array([int(bool(s[k])) for s in S_list], dtype=np.int64)
        codes |= (bit << j)
    uniq, inv = np.unique(codes, return_inverse=True)
    masks, counts = [], []
    for uidx in range(len(uniq)):
        m = (inv == uidx)
        masks.append(m)
        counts.append(int(m.sum()))
    return masks, counts, keys

# === [CHANGED] (이름은 유지) Apriori 기반 V 빌더 (DataFrame 버전) → 내부에서 랜덤 패킹 사용
def build_V_apriori_from_S_df(S_df, T, agg_max_len=4, agg_max_cols=2048, agg_repeat=64, seed=None):
    # import pdb; pdb.set_trace()
    N = len(S_df)
    if N == 0 or S_df.shape[1] == 0 or T <= 0:
        return [], []
    base_masks, counts = _bitpack_cells_from_df_for_metrics(S_df)
    alive = sum(c >= T for c in counts)
    print("원자 셀 단계에서 살아남은 개수 =", alive)
    if len(base_masks) == 0:
        return [], []

    # [변경] unions = _apriori_min_cover_from_counts(counts, T, max_len=agg_max_len)
    unions = _random_pack_unions_from_counts(
        counts, T, M=int(agg_repeat), seed=seed, max_len=agg_max_len, max_cols=agg_max_cols, log_prefix="V"
    )

    V, choices = [], []
    for idx_tuple in unions:
        um = np.zeros(N, dtype=bool)
        for i_local in idx_tuple:
            um |= base_masks[i_local]
        pos = int(um.sum()); neg = N - pos
        if pos >= T and neg >= T:
            V.append(um)
            choices.append(("cells", tuple(idx_tuple), f"pos={pos}", f"neg={neg}"))
            if len(V) >= int(agg_max_cols):
                break
    print(f"[V-pack] built |V|={len(V)} (after pos/neg>=T and max_cols)")
    return V, choices

# === [CHANGED] (이름은 유지) Apriori 기반 V 빌더 (list[dict] 버전) → 내부에서 랜덤 패킹 사용
def build_V_apriori_from_S_list(S_list, T, agg_max_len=4, agg_max_cols=2048, key_order=None, agg_repeat=64, seed=None):
    N = len(S_list)
    # import pdb;pdb.set_trace()
    if N == 0 or T <= 0:
        return [], []
    masks, counts, _ = _bitpack_cells_from_S_list_for_metrics(S_list, key_order=key_order)
    alive = sum(c >= T for c in counts)
    print("원자 셀 단계에서 살아남은 개수 =", alive)
    if len(masks) == 0:
        return [], []

    # [변경] unions = _apriori_min_cover_from_counts(counts, T, max_len=agg_max_len)
    unions = _random_pack_unions_from_counts(
        counts, T, M=int(agg_repeat), seed=seed, max_len=agg_max_len, max_cols=agg_max_cols, log_prefix="V"
    )

    V, choices = [], []
    for idx_tuple in unions:
        um = np.zeros(N, dtype=bool)
        for i_local in idx_tuple:
            um |= masks[i_local]
        pos = int(um.sum()); neg = N - pos
        if pos >= T and neg >= T:
            V.append(um)
            choices.append(("cells", tuple(idx_tuple), f"pos={pos}", f"neg={neg}"))
            if len(V) >= int(agg_max_cols):
                break
    print(f"[V-pack] built |V|={len(V)} (after pos/neg>=T and max_cols)")
    return V, choices

# ===== [NEW] k-차(정확히 k개 열 AND) 교차집단 마스크 생성 =====
def _apriori_exact_k_unions_from_df(S_df, T, k, max_cols=None, log_prefix="afk"):
    """
    S_df: (N x q) {0,1} DataFrame
    T   : subset & complement 최소 지원 수
    k   : 정확히 k차 교차집단만 생성 (열 k개 AND)
    max_cols: 생성할 최대 개수 (None이면 제한 없음)
    return: List[np.ndarray(bool)]
    """
    import numpy as np
    from itertools import combinations

    N, q = len(S_df), S_df.shape[1]
    if N == 0 or q == 0 or T <= 0 or k <= 0 or k > q:
        print(f"[{log_prefix}] skip: N={N}, q={q}, T={T}, k={k}")
        return []

    X = S_df.values.astype(np.int8)  # (N,q) in {0,1}
    out = []
    kept = 0
    for comb in combinations(range(q), k):
        m = X[:, comb].all(axis=1)      # 교차집단: 열 k개 AND
        cnt = int(m.sum()); comp = N - cnt
        if cnt >= T and comp >= T:
            out.append(m)
            kept += 1
            if (max_cols is not None) and (kept >= int(max_cols)):
                print(f"[{log_prefix}] reached max_cols={max_cols}; stop early")
                break
    print(f"[{log_prefix}] exact-k={k} kept={kept}")
    return out


# ===== [NEW] 1~q차 전체(또는 max_order까지) 교차집단 마스크 생성 =====
from ..methods.dr_subgroup_subset_random import _apriori_forward_unions_from_df

#####################################
from tqdm import tqdm

######################################################
def compute_metrics(args, data, pred_pack):
    proba = pred_pack.get("proba", None)
    pred = pred_pack.get("pred", None)
    test_dr = pred_pack.get("test_dr", None)
    report = dict(dataset=getattr(args, "dataset", ""),
                  method=getattr(args, "method", ""),
                  seed=getattr(args, "seed", None))
    report["V_stats"] = pred_pack.get("V_stats", [])
    thr = getattr(args, "thr", 0.5)
    min_support = int(getattr(args, "min_support", 1))
    mc_bins = getattr(args, "mc_bins", 10)
    mc_min_support = int(getattr(args, "mc_min_support", 1))
    agg_max_cols = int(getattr(args, "agg_max_cols", 2048))


    n_low = getattr(args, "n_low_test", None)
    # n_low_frac = getattr(args, "n_low_frac", None)
    if data["type"] == "tabular":
        y = data.get("y_test", None); S = data.get("S_test", None)
        N = (len(S) if hasattr(S, "shape") else (len(y) if y is not None else 0))
    else:
        ys, Ss = [], []
        for _, yb, Sb in data["test_loader"]:
            ys.append(yb.numpy()); Ss += Sb
        y = np.concatenate(ys) if len(ys) > 0 else None
        S = Ss; N = len(S)


    # score dist 뽑아보는 용

    # from pathlib import Path; import numpy as np
    # # if proba is not None:
    # outdir = Path("../scores3"); outdir.mkdir(parents=True, exist_ok=True)
    # stem = f"scores_{getattr(args,'dataset','data')}_{getattr(args,'method','meth')}_seed{getattr(args,'seed','NA')}"
    # np.save(outdir / f"{stem}.npy", np.asarray(proba).ravel())
    # tmp = f"truey_{getattr(args,'dataset','data')}_{getattr(args,'method','meth')}_seed{getattr(args,'seed','NA')}"
    # np.save(outdir / f"{tmp}.npy", np.asarray(y).ravel())
    # tmp = f"trues_{getattr(args,'dataset','data')}_{getattr(args,'method','meth')}_seed{getattr(args,'seed','NA')}"
    # # S → 2D 행렬로 통일
    # # import pdb; pdb.set_trace()
    # if hasattr(S, "values"):  # pandas DataFrame
    #     S_mat = S.values
    # elif isinstance(S, list) and len(S) > 0 and isinstance(S[0], dict):  # list[dict]
    #     keys = list(S[0].keys())
    #     S_mat = np.array([[int(bool(si[k])) for k in keys] for si in S], dtype=np.int64)
    # else:  # numpy or 기타
    #     S_mat = np.asarray(S)
    #     if S_mat.ndim == 1:
    #         S_mat = S_mat.reshape(-1, 1)

    # # 행 단위 유니크 조합 → 1..K 그룹 ID
    # _, inv = np.unique(S_mat, axis=0, return_inverse=True)
    # group_id = (inv + 1).astype(np.int64)  # 1-based
    # print(group_id)

    # np.save(outdir / f"{tmp}.npy", group_id)
    
    # report["proba_path"] = str(outdir / f"{stem}.npy")



    n_low = int(np.ceil(N / 32))
    T = n_low
    # if (n_low_frac is not None) and (n_low_frac > 0) and N > 0:
    #     T = int(np.ceil(float(n_low_frac) * N))
    # else:
    #     base_low = int(n_low) if n_low is not None else 0
    #     # T = max(int(min_support), base_low)
    print("min support: ", min_support, ", n_low:", n_low, "/", {N})
    # min support는 marginal fair, subgroup fair에서 min_support 이상인 것만 사용
    # n_low는 subgroup subset fair에서 subset 내부 원소 개수가 n_low 이상인 것을 사용

    V_mode_list = [[1,2,3]]
    V, V_choices = None, None

    print("[Performance] V counting ...")
    for V_mode in V_mode_list:
        V = None
        print("[Performance]: ", V_mode)
        if len(V_mode) == 1:
        # ===== [NEW] 정확히 k차만 고려한 supIPM =====

            masks_np = _apriori_exact_k_unions_from_df(
                S, T, k=V_mode[0], max_cols=agg_max_cols, log_prefix=f"afk{V_mode[0]}"
            )
            if len(masks_np) == 0:
                V, V_choices = [], []
            else:
                V = [m.astype(bool) for m in masks_np]
                V_choices = [("af_exact_k", V_mode[0], f"pos={int(m.sum())}", f"neg={int(len(m)-m.sum())}") for m in masks_np[:min(15, len(masks_np))]]

        elif len(V_mode) > 1:
            # ===== [NEW] 1~K차(또는 af_max_order까지) 모두 고려한 supIPM =====
            masks_np = _apriori_forward_unions_from_df(
                S, T, max_order=V_mode[-1], max_cols=agg_max_cols, log_prefix="Test_V"
            )
            if len(masks_np) == 0:
                V, V_choices = [], []
            else:
                V = [m.astype(bool) for m in masks_np]
                V_choices = [("af_all", f"len={len(masks_np)}", f"max_order={V_mode[-1]}", f"T={T}")]
        
        report[f"V_mode_{V_mode}"] = V_choices
        print("=== sup_mmd_over_V, sup_w1_over_V START ===")
        if (proba is not None) and (V is not None):
            try:
                # report[f"sup_mmd_over_V_{V_mode}"] = float(sup_mmd_over_V(proba, V, min_support=n_low)) 
                report[f"sup_w1_over_V_{V_mode}"] = float(sup_w1_over_V(proba, V, min_support=n_low))
            except Exception: report[f"sup_w1_over_V_{V_mode}"] = np.nan
        else:
            report[f"sup_mmd_over_V_{V_mode}"] = np.nan; report[f"sup_w1_over_V_{V_mode}"] = np.nan
        print(f"[metric] sup_w1_over_V = {report[f'sup_w1_over_V_{V_mode}']}")


        print("=== DP over V (sup/mean) START ===")
        if (proba is not None) and (V is not None):
            try: report[f"worst_spd_over_V_{V_mode}"] = float(worst_spd_over_V(proba, V, thr=thr, min_support=n_low))
            except Exception: report[f"worst_spd_over_V_{V_mode}"] = np.nan
            try: report[f"mean_spd_over_V_{V_mode}"] = float(mean_spd_over_V(proba, V, thr=thr, min_support=n_low))
            except Exception: report[f"mean_spd_over_V_{V_mode}"] = np.nan
        else:
            report[f"worst_spd_over_V_{V_mode}"] = np.nan; report[f"mean_spd_over_V_{V_mode}"] = np.nan
        print(f"[metric] worst_spd_over_V = {report[f'worst_spd_over_V_{V_mode}']}")
        print(f"[metric] mean_spd_over_V = {report[f'mean_spd_over_V_{V_mode}']}")

        

        print("=== Weighted SPD over V START ===")
        if (proba is not None) and (V is not None):
            try: report[f"worst_weighted_spd_over_V_{V_mode}"] = float(worst_weighted_spd_over_V(proba, V, thr=thr, min_support=n_low))
            except Exception: report[f"worst_weighted_spd_over_V_{V_mode}"] = np.nan
            try: report[f"mean_weighted_spd_over_V_{V_mode}"] = float(mean_weighted_spd_over_V(proba, V, thr=thr, min_support=n_low))
            except Exception: report[f"mean_weighted_spd_over_V_{V_mode}"] = np.nan
        else:
            report[f"worst_weighted_spd_over_V_{V_mode}"] = np.nan; report[f"mean_weighted_spd_over_V_{V_mode}"] = np.nan
        print(f"[metric] worst_weighted_spd_over_V = {report[f'worst_weighted_spd_over_V_{V_mode}']}")
        print(f"[metric] mean_weighted_spd_over_V = {report[f'mean_weighted_spd_over_V_{V_mode}']}")


    # V랑 상관없는 것들
    try: report[f"test_dr"] = float(test_dr)
    except Exception: report[f"test_dr"] = np.nan
    
    print("=== accuracy START ===")
    report["accuracy"] = accuracy(y, pred) if (y is not None and pred is not None) else np.nan
    print(f"[metric] accuracy = {report['accuracy']}")
    print("=== accuracy END ===")


    print("=== supIPM(overall) START ===")
    if proba is not None and S is not None:
        try: report["supipm_rbf"] = float(supipm_rbf(proba, S))
        except Exception: report["supipm_rbf"] = np.nan
        try: report["supipm_w1"] = float(supipm_wasserstein(proba, S, min_support=min_support))
        except Exception: report["supipm_w1"] = np.nan
    else:
        report["supipm_rbf"] = np.nan; report["supipm_w1"] = np.nan
    print(f"[metric] supipm_rbf = {report['supipm_rbf']}")
    print(f"[metric] supipm_w1 = {report['supipm_w1']}")
    print("=== supIPM(overall) END ===")


    # print("=== sup_mmd_dfcols, sup_w1_dfcols START ===")
    # if (data["type"] == "tabular") and (proba is not None) and (S is not None):
    #     try: report["sup_mmd_dfcols"] = float(sup_mmd_dfcols(proba, S, min_support=min_support))
    #     except Exception: report["sup_mmd_dfcols"] = np.nan
    #     try: report["sup_w1_dfcols"] = float(sup_w1_dfcols(proba, S, min_support=min_support))
    #     except Exception: report["sup_w1_dfcols"] = np.nan
    # else:
    #     report["sup_mmd_dfcols"] = np.nan; report["sup_w1_dfcols"] = np.nan
    # print(f"[metric] sup_mmd_dfcols = {report['sup_mmd_dfcols']}")
    # print(f"[metric] sup_w1_dfcols = {report['sup_w1_dfcols']}")


    print("=== SPD(singleton) START ===")
    if proba is not None and S is not None:
        try: report["spd_worst"] = float(worst_group_spd(proba, S))
        except Exception: report["spd_worst"] = np.nan
        try: report["spd_mean"] = float(mean_group_spd(proba, S, thr=thr, min_support=min_support))
        except Exception: report["spd_mean"] = np.nan
    else:
        report["spd_worst"] = np.nan; report["spd_mean"] = np.nan
    print(f"[metric] worst_group_spd = {report['spd_worst']}")
    print(f"[metric] mean_group_spd = {report['spd_mean']}")


    print("=== Weighted SPD(singleton) START ===")
    if proba is not None and S is not None:
        try: report["worst_weighted_group_spd"] = float(worst_weighted_group_spd(proba, S, thr=thr, min_support=min_support))
        except Exception: report["worst_weighted_group_spd"] = np.nan
        try: report["mean_weighted_group_spd"] = float(mean_weighted_group_spd(proba, S, thr=thr, min_support=min_support))
        except Exception: report["mean_weighted_group_spd"] = np.nan
    else:
        report["worst_weighted_group_spd"] = np.nan; report["mean_weighted_group_spd"] = np.nan
    print(f"[metric] worst_weighted_group_spd = {report['worst_weighted_group_spd']}")
    print(f"[metric] mean_weighted_group_spd = {report['mean_weighted_group_spd']}")




    print("=== marginal metric START ===")
    if (data["type"] == "tabular") and (pred is not None) and (y is not None) and (S is not None):
        try: report["fpr_worst"] = float(worst_group_fpr_gap(pred, y, S))
        except Exception: report["fpr_worst"] = np.nan
        try: report["fpr_mean"] = float(mean_group_fpr_gap(pred, y, S, min_support=min_support))
        except Exception: report["fpr_mean"] = np.nan
    else:
        report["fpr_worst"] = np.nan; report["fpr_mean"] = np.nan
    if (data["type"] == "tabular") and (proba is not None) and (y is not None) and (S is not None):
        try: report["mc_worst"] = float(multicalib_worst_gap(proba, y, S, bins=mc_bins))
        except Exception: report["mc_worst"] = np.nan
        try: report["mc_mean"] = float(multicalib_mean_gap(proba, y, S, bins=mc_bins, min_support=mc_min_support))
        except Exception: report["mc_mean"] = np.nan
    else:
        report["mc_worst"] = np.nan; report["mc_mean"] = np.nan
    if proba is not None and S is not None:
        try: report["marg_spd_worst"] = float(marginal_spd_worst(proba, S, thr=thr, min_support=min_support))
        except Exception: report["marg_spd_worst"] = np.nan
        try: report["marg_spd_mean"] = float(marginal_spd_mean(proba, S, thr=thr, min_support=min_support))
        except Exception: report["marg_spd_mean"] = np.nan
    else:
        report["marg_spd_worst"] = np.nan; report["marg_spd_mean"] = np.nan
    if (pred is not None) and (y is not None) and (S is not None):
        try: report["marg_fpr_worst"] = float(marginal_fpr_worst(pred, y, S, min_support=min_support))
        except Exception: report["marg_fpr_worst"] = np.nan
        try: report["marg_fpr_mean"] = float(marginal_fpr_mean(pred, y, S, min_support=min_support))
        except Exception: report["marg_fpr_mean"] = np.nan
    else:
        report["marg_fpr_worst"] = np.nan; report["marg_fpr_mean"] = np.nan
    if (proba is not None) and (y is not None) and (S is not None):
        try: report["marg_mc_worst"] = float(marginal_mc_worst(proba, y, S, bins=mc_bins, min_support=mc_min_support))
        except Exception: report["marg_mc_worst"] = np.nan
        try: report["marg_mc_mean"] = float(marginal_mc_mean(proba, y, S, bins=mc_bins, min_support=mc_min_support))
        except Exception: report["marg_mc_mean"] = np.nan
    else:
        report["marg_mc_worst"] = np.nan; report["marg_mc_mean"] = np.nan


    return report
