# fairbench/methods/dr.py
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from ..utils.threshold import tune_threshold
import logging, time
from tqdm.auto import tqdm
from fairbench.utils.logging_utils import Timer, mem_str
# ===== NEW: Apriori-based unions over disjoint base subgroups =====
from itertools import combinations
from ..utils.mlp import MLP

import numpy as np, random, torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


def _apriori_forward_unions_from_df(
    S_df, T, max_order=3, max_cols=None, log_prefix="af",
    prune_on_comp=False,  # ← 기본 False: comp<T 로는 프루닝 안 함(안전)
    debug=False, debug_max_lines=200,
):
    """
    1차(원자)에서 최소지지 T를 통과한 패턴(기본: cnt>=T)만 사용해서
    2차, 3차 ... 교차 패턴을 생성. 최종 유지 조건은 cnt>=T & comp>=T.
    """
    import numpy as np
    from itertools import combinations, product

    N, q = len(S_df), S_df.shape[1]
    if N == 0 or q == 0 or T <= 0:
        print(f"[{log_prefix}] skip: N={N}, q={q}, T={T}")
        return []

    X = S_df.values.astype(np.int32)  # (N, q) in {0,1}
    q_max = q if (max_order is None or max_order <= 0) else min(int(max_order), q)

    out, kept = [], 0

    print(f"[Apriori-forward] Max order: {q_max}")
    print(f"[{log_prefix}] N={N}, q={q}, T={T}, max_cols={max_cols}, prune_on_comp={prune_on_comp}")

    # ---- L1: 원자 패턴(각 컬럼 j에 대해 bit 0/1) 준비 + 생존 여부 계산 ----
    atoms_mask = [[None, None] for _ in range(q)]  # atoms_mask[j][b] -> (N,) bool
    alive_cnt  = np.zeros((q, 2), dtype=bool)
    alive_comp = np.zeros((q, 2), dtype=bool)

    for j in range(q):
        col = X[:, j]
        for b in (0, 1):
            m = (col == b)
            cnt = int(m.sum()); comp = N - cnt
            atoms_mask[j][b] = m
            alive_cnt[j, b]  = (cnt  >= T)
            alive_comp[j, b] = (comp >= T)
            print(f"[{log_prefix}] L1 j={j}, b={b}, cnt={cnt}, comp={comp}, "
                 f"alive_cnt={alive_cnt[j,b]}, alive_comp={alive_comp[j,b]}")

    # L1 생존 기준: 기본은 cnt만(안전). 필요하면 comp도 함께 요구 가능.
    alive_atom = alive_cnt & alive_comp if prune_on_comp else alive_cnt

    # j별로 살아있는 bit 목록
    alive_by_attr = {j: [b for b in (0,1) if alive_atom[j, b]] for j in range(q)}
    attrs = [j for j, bs in alive_by_attr.items() if len(bs) > 0]
    if len(attrs) == 0:
        print(f"[{log_prefix}] no alive atoms at L1; return empty")
        return []

    # ---- order = 1: L1에서 최종 조건(cnt/comp)까지 만족하는 것만 out에 추가 ----
    for j in attrs:
        for b in alive_by_attr[j]:
            m = atoms_mask[j][b]
            cnt = int(m.sum()); comp = N - cnt
            if cnt >= T and comp >= T:
                out.append(m)
                kept += 1
                print(f"[{log_prefix}] comb=({j},), pat=({b},), cnt={cnt}, comp={comp}")
                if max_cols is not None and kept >= int(max_cols):
                    print(f"[{log_prefix}] reached max_cols={max_cols}; stop early (kept={kept})")
                    return out
    print(f"[{log_prefix}] order=1 done (kept so far={kept})")

    # ---- order >= 2: L1에서 살아남은 bit만 사용해 패턴 생성 ----
    for k in range(2, q_max + 1):
        print(f"[Apriori-forward] Order: {k}")
        for comb in combinations(attrs, k):               # k개의 속성 선택
            pools = [alive_by_attr[j] for j in comb]      # 각 속성에서 허용되는 bit만
            for pat in product(*pools):                   # 허용 bit 조합만 생성
                # 원자 마스크들의 AND
                m = np.ones(N, dtype=bool)
                for j, b in zip(comb, pat):
                    m &= atoms_mask[j][b]
                    if not m.any():                        # 조기 중단
                        break
                if not m.any():
                    continue
                cnt = int(m.sum()); comp = N - cnt
                if cnt >= T and comp >= T:
                    out.append(m)
                    kept += 1
                    print(f"[{log_prefix}] comb={comb}, pat={pat}, cnt={cnt}, comp={comp}")
                    if max_cols is not None and kept >= int(max_cols):
                        print(f"[{log_prefix}] reached max_cols={max_cols}; stop early (kept={kept})")
                        return out
        print(f"[{log_prefix}] order={k} done (kept so far={kept})")

    print(f"[{log_prefix}] done. total kept={kept}")
    if debug and debug_max_lines is not None and dbg_lines >= int(debug_max_lines):
        print(f"[{log_prefix}] (debug) output truncated at {dbg_lines} lines")
    return out


#     # ===== NEW: Apriori-forward (1차→2차→... 교차집단; 모든 카테고리 패턴 포함) =====
# def _apriori_forward_unions_from_df(S_df, T, max_order=3, max_cols=None, log_prefix="af"):
#     """
#     S_df의 이진 컬럼들에 대해 1차(단일)→2차→... 순으로,
#     각 선택된 컬럼 집합의 모든 0/1 패턴(2^k)을 교차집단으로 생성한다.
#     유지 조건: support cnt >= T 그리고 complement comp >= T.
#     - T: 최소 지지
#     - max_order: 최대 차수(k), None/<=0이면 q까지
#     - max_cols: 생성 상한
#     return: List[np.ndarray(bool)] (각 교차 마스크)
#     """
#     import numpy as np
#     from itertools import combinations, product

#     N, q = len(S_df), S_df.shape[1]
#     if N == 0 or q == 0 or T <= 0:
#         print(f"[{log_prefix}] skip: N={N}, q={q}, T={T}")
#         return []

#     X = S_df.values.astype(np.int32)  # (N, q) in {0,1}
#     q_max = q if (max_order is None or max_order <= 0) else min(int(max_order), q)

#     out = []
#     kept = 0
#     print(f"[Apriori-forward] Max order: {q_max}")

#     for k in range(1, q_max + 1):
#         print(f"[Apriori-forward] Order: {k}")
#         for comb in combinations(range(q), k):
#             sub = X[:, comb]  # (N, k)
#             # 모든 카테고리 패턴(0/1)의 교차 생성
#             for pat in product((0, 1), repeat=k):
#                 m = (sub == np.array(pat, dtype=np.int32)).all(axis=1)
#                 cnt = int(m.sum()); comp = N - cnt
#                 if cnt >= T and comp >= T:
#                     print(f"[{log_prefix}] comb={comb}, pat={pat}, cnt={cnt}, comp={comp}")
#                     out.append(m)
#                     kept += 1
#                     if (max_cols is not None) and (kept >= int(max_cols)):
#                         print(f"[{log_prefix}] reached max_cols={max_cols}; stop early")
#                         return out
#         print(f"[{log_prefix}] order={k} done (kept so far={kept})")

#     print(f"[{log_prefix}] done. total kept={kept}")
#     return out


# # ===== NEW: Apriori-forward (1차→2차→... 교차집단) =====
# def _apriori_forward_unions_from_df(S_df, T, max_order=3, max_cols=None, log_prefix="af"):
#     max_order=3
#     print(f"[Apriori-forward] Max order: {max_order}")
#     """
#     S_df의 이진 컬럼들을 1차(단일 컬럼) → 2차(AND) → 3차(AND) ... 순서로
#     교차집단 마스크를 만들고, 지지도/여지도(T) 조건을 만족하는 것만 반환.
#     - T: subset size >= T 그리고 complement >= T
#     - max_order: 몇 차까지(없으면 q 전체)
#     - max_cols: 최대 개수 제한
#     return: List[np.ndarray(bool)]  (각 교차집단 마스크)
#     """
#     import numpy as np
#     from itertools import combinations

#     N, q = len(S_df), S_df.shape[1]
#     if N == 0 or q == 0 or T <= 0:
#         print(f"[{log_prefix}] skip: N={N}, q={q}, T={T}")
#         return []

#     X = S_df.values.astype(np.int32)  # (N, q) in {0,1}
#     q_max = q if (max_order is None or max_order <= 0) else min(int(max_order), q)

#     out = []
#     kept = 0
#     for k in range(1, q_max + 1):                       # 1차 → 2차 → ...
#         print(f"[Apriori-forward] Order: {k}")
#         for comb in combinations(range(q), k):
#             m = X[:, comb].all(axis=1)                  # 교차집단: AND
#             cnt = int(m.sum()); comp = N - cnt
#             if cnt >= T and comp >= T:
#                 print(f"[{log_prefix}] comb={comb}, cnt={cnt}, comp={comp}")
#                 out.append(m)
#                 kept += 1
#                 if (max_cols is not None) and (kept >= int(max_cols)):
#                     print(f"[{log_prefix}] reached max_cols={max_cols}; stop early")
#                     return out
#         print(f"[{log_prefix}] order={k} done (kept so far={kept})")

#     print(f"[{log_prefix}] done. total kept={kept}")
#     return out

# ===== NEW: Random packing unions over disjoint base subgroups =====
def _random_pack_unions(counts, T, M=64, seed=None, max_len=None, max_cols=None):
    """
    (1) 관측된 디스조인트 셀들을 무작위로 나열하고
    (2) 앞에서부터 합쳐가며 누적 카운트가 T에 도달하면 그 묶음을 하나의 subgroup-subset으로 저장
    (3) 리스트 끝까지 진행, 이 과정을 M회 반복
    (4) 중복 제거 후 반환
    - complement(= N - union_size)도 T 이상인 것만 유지
    """
    import numpy as np
    N = int(np.sum(counts))
    K = len(counts)
    if K == 0 or T <= 0:
        print(f"[pack] skip: K={K}, T={T}")
        return []

    rng = np.random.default_rng(seed)
    seen = set()
    out = []
    print(f"[pack] N={N}, K={K}, T={T}, M={M}, max_len={max_len}, max_cols={max_cols}, seed={seed}")

    for rep in range(int(M)):
        perm = rng.permutation(K)
        print(f"[pack][rep={rep+1}/{int(M)}])")
        i = 0
        block_idx = 0
        while i < K:
            start_i = i
            cum = 0
            members = []
            # (2) 누적이 T 도달할 때까지 앞에서부터 채움
            while i < K and cum < T:
                j = int(perm[i])
                c = int(counts[j])
                if c > 0:            # 0 카운트(비관측) 셀은 건너뜀
                    members.append(j)
                    cum += c
                i += 1
                if (max_len is not None) and (len(members) >= int(max_len)) and (cum < T):
                    # 길이 제한이 있으면 여기서 더 못 채움
                    # print(f"[pack][rep={rep+1}]  max_len hit (len={len(members)}) before reaching T; stop block")
                    break

            # (3) 누적이 T에 도달한 경우만 후보
            if cum >= T:
                block_idx += 1
                comp = N - cum
                valid = (comp >= T)
                print(f"[pack][rep={rep+1}] block#{block_idx}: "
                          f"members={members} | counts={[int(counts[m]) for m in members]} "
                          f"| size={cum} | comp={comp} | valid={'Y' if valid else 'N'}")
                if (N - cum) >= T:   # complement ≥ T
                    key = tuple(sorted(members))   # 순서 불변 canonical key
                    ############## 중복제거하는곳!!!!!!!!!##############
                    if key not in seen:
                        seen.add(key)
                        out.append(key)
                        # print(f"[pack][rep={rep+1}]  -> kept (new)")
                        if (max_cols is not None) and (len(out) >= int(max_cols)):
                            print(f"[pack] reached max_cols={max_cols}; stop early")
                            return out
            else:
                # 남은 걸로는 더 이상 T 도달 불가 → 종료
                # print(f"[pack][rep={rep+1}] stop: remaining insufficient (cum={cum} < T) "
                        #   f"from i={start_i}..{i-1}")
                # 이 rep을 종료하지 말고, 현재 i 지점부터 "다음 블록"을 계속 시도
                continue

    print(f"[pack] done: unique_unions={len(out)}")
    return out


### 모든 경우의 수 다 고려하는 거
# ===== NEW: Enumerate ALL subgroup-subsets (unions) over observed disjoint cells =====
def _enumerate_all_unions(counts, T, max_len=None, max_cols=None):
    """
    모든 셀 조합을 열거해(1..K 또는 max_len까지) 지지도/여지 조건을 만족하는
    합집합만 반환한다.
      - counts: 각 관측 셀의 크기 리스트 (길이 K)
      - T: 최소 지지도(합집합 size >= T) & complement >= T 조건
      - max_len: 합집합에 포함될 셀 수의 상한(없으면 K)
      - max_cols: 반환할 최대 개수(없으면 전부)
    반환: List[Tuple[int,...]]  (셀 인덱스 조합을 정렬된 튜플로)
    주의: 조합 수가 2^K - 1로 폭증하므로 K가 크면 비용이 큼.
    """
    import numpy as np
    from itertools import combinations

    N = int(np.sum(counts))
    K = len(counts)
    if K == 0 or T <= 0:
        print(f"[enum] skip: K={K}, T={T}")
        return []

    Lmax = K if (max_len is None) else int(max_len)
    Lmax = max(1, min(Lmax, K))

    out = []
    # 길이 1 -> Lmax까지 순서대로(짧은 길이 우선) 순회
    for L in range(1, Lmax + 1):
        for comb in combinations(range(K), L):
            union_size = int(sum(int(counts[i]) for i in comb))
            comp = N - union_size
            if union_size >= T and comp >= T:
                out.append(tuple(comb))
                if (max_cols is not None) and (len(out) >= int(max_cols)):
                    print(f"[enum] reached max_cols={max_cols}; stop early")
                    return out
    print(f"[enum] done: unique_unions={len(out)} (K={K}, Lmax={Lmax})")
    return out



def _bitpack_cells_from_df(S_df, device="cuda"):
    """
    S_df: (N x q) 이진 DataFrame
    return: 
      - masks: list[torch.BoolTensor] # 각 '관측된' 셀의 마스크 (디스조인트)
      - counts: list[int]             # 각 셀의 크기
      - order: list[int]              # 셀 인덱스(비트패턴)를 오름차순으로
    """
    # import pdb;pdb.set_trace()
    import numpy as np, torch
    cols = list(S_df.columns)
    if len(cols) == 0 or len(S_df) == 0:
        return [], [], []
    # bitpack: code = sum(S[:,j] << j)
    B = np.stack([S_df[c].astype(np.int8).values for c in cols], axis=1) # (N,q)
    codes = np.zeros(len(S_df), dtype=np.int64)
    for j in range(B.shape[1]):
        codes |= (B[:, j].astype(np.int64) << j)
    uniq, inv = np.unique(codes, return_inverse=True) # 관측된 셀만
    N = len(S_df)
    masks, counts, order = [], [], []
    for k, u in enumerate(uniq.tolist()):
        m = torch.from_numpy((inv == k)).to(device)
        cnt = int(m.sum().item())
        masks.append(m)
        counts.append(cnt)
        order.append(int(u))
    return masks, counts, order

# ===== NEW: pretty-print helpers for unions (tabular) =====
def _fmt_bits_from_code(code: int, cols):
    q = len(cols)
    bits = [(code >> j) & 1 for j in range(q)]
    return "{" + ", ".join(f"{cols[j]}={bits[j]}" for j in range(q)) + "}"

def _log_unions_tabular(unions, counts, order_codes, cols, N, max_show=10, tag="pack"):
    """
    unions: List[Tuple[int,...]]  # _random_pack_unions가 돌려준 셀 인덱스 튜플
    counts: List[int]              # 셀별 카운트
    order_codes: List[int]         # 셀의 비트코드(컬럼 순서 기반)
    cols: List[str]                # 민감속성 컬럼명 순서
    """
    K = len(unions)
    print(f"[{tag}] unions_proposed={K} (show up to {max_show})")
    show = min(max_show, K)
    for r in range(show):
        idx_tuple = unions[r]
        # cell_codes = [order_codes[i] for i in idx_tuple]
        cell_counts = [int(counts[i]) for i in idx_tuple]
        pos = sum(cell_counts); neg = N - pos
        # cells_human = "[" + ", ".join(_fmt_bits_from_code(c, cols) for c in cell_codes) + "]"
        # print(f"[{tag}][#{r+1}] cells={idx_tuple} | size={pos} (comp={neg}) | members={cells_human} | counts={cell_counts}")
        print(f"[{tag}][#{r+1}] cells={idx_tuple} | size={pos} (comp={neg}) | counts={cell_counts}")

    if K > show:
        print(f"[{tag}] ... {K-show} more")

# (서명 변경) agg_repeat 추가  ▼▼▼
def build_C_from_unions_df(S_df, device="cuda",
                            n_low=None, n_low_frac=None,
                            agg_max_len=300, agg_max_cols=2048,
                            af_max_order=3,
                            unions_mode="pack",
                            include_marginals_always=False):  # NEW
    """
    디스조인트 셀(관측된 것들) 위에서 Apriori로 '최소 합집합'을 만들고, 
    각 합집합(Union)의 마스크 컬럼을 C에 쌓아 반환.
    - 지지도 임계: T = ceil(gamma * N) (gamma = n_low_frac 또는 n_low/N)
    - 추가로 complement 크기도 T 이상인 것만 유지(편향/퇴화 방지)
    """
    import torch, numpy as np
    

    # 개수 세기
    N = len(S_df)
    if N == 0 or S_df.shape[1] == 0:
        return torch.zeros((N, 0), device=device)
    if (n_low_frac is not None) and (n_low_frac > 0):
        T = int(np.ceil(float(n_low_frac) * N))
    else:
        T = int(n_low or 0)
    
    print(f"[tabular] n low frac: {n_low_frac}, n low: {n_low}, T: {T}")
    
    # 민감속성중 개수 안되는거 탈락
    base_masks, counts, order = _bitpack_cells_from_df(S_df, device=device)
    alive = sum(c >= T for c in counts)
    print(f"탈락 수: {sum(c < T for c in counts)}/{len(counts)}, T = {T}")
    print(f"[tabular] 원자 셀 단계에서 살아남은 개수 = {alive}")
    if len(base_masks) == 0:
        return torch.zeros((N, 0), device=device)

    # # union 
    # if unions_mode == "all":
    #     unions = _enumerate_all_unions(
    #         counts, T, max_len=agg_max_len, max_cols=agg_max_cols
    #     )
        
    if unions_mode == "apriori_forward":
        # af_max_order = 3
        # 교차집단 마스크 직접 생성 (단일→2차→... AND)    
        # ✅ base_masks / counts에서 T 이상인 '살아남은 원자'만 추려서 Apriori에 투입
        # alive_idx = [i for i, c in enumerate(counts) if c >= T]
        # if len(alive_idx) == 0:
        #     return torch.zeros((N, 0), device=device)

        # # base_masks: (N, K) torch → numpy (N, K_alive)
        # if isinstance(base_masks, torch.Tensor):
        #     base_np = (base_masks[:, alive_idx] > 0.5).float().detach().cpu().numpy()
        # else:
        #     # 리스트/배열로 올 경우도 대응
        #     base_np = np.stack([base_masks[i].detach().cpu().numpy().astype(np.float32).ravel()
        #                         for i in alive_idx], axis=1)
        # alive_cols = [S_df.columns[i] for i, c in enumerate(counts) if c >= T and i < S_df.shape[1]]
        # S_alive = S_df[alive_cols]
        masks_np = _apriori_forward_unions_from_df(
            S_df, T, max_order=af_max_order, max_cols=agg_max_cols, log_prefix="Train_V"
        )
        # masks_np = _apriori_forward_unions_from_df(
        #     S_df, T, max_order=af_max_order, max_cols=agg_max_cols, log_prefix="Train_V"
        # )
        if len(masks_np) == 0:
            return torch.zeros((N, 0), device=device)

        cols = []
        for m_np in masks_np:
            v = torch.tensor(m_np.astype(np.float32), device=device).view(-1, 1)
            pos = int(v.sum().item()); neg = N - pos
            if pos >= T and neg >= T:
                cols.append(v)
            if len(cols) >= int(agg_max_cols):
                break

        if len(cols) == 0:
            return torch.zeros((N, 0), device=device)

        print(f"[apriori_forward] N={N}, q={S_df.shape[1]}, T={T}, max_order={af_max_order}, "
            f"max_cols={agg_max_cols}, kept={len(cols)}")
        return torch.cat(cols, dim=1)
    
    # else:  # default: random packing
    #     unions = _random_pack_unions(
    #         counts, T, M=int(agg_repeat), max_len=agg_max_len, max_cols=agg_max_cols
    #     )

    # cols_list = list(S_df.columns)
    # _log_unions_tabular(unions, counts, order, cols_list, N, max_show=10, tag="pack->C")  # NEW

    # if len(unions) == 0:
    #     return torch.zeros((N, 0), device=device)
    # cols = []
    # for idx_tuple in unions:
    #     umask = None
    #     for i_local in idx_tuple:
    #         m = base_masks[i_local]
    #         umask = m if umask is None else (umask | m)
    #     pos = int(umask.sum().item())
    #     neg = N - pos
    #     if pos >= T and neg >= T:
    #         cols.append(umask.float().view(-1, 1))
    #     if len(cols) >= int(agg_max_cols):
    #         break
    
    # # === NEW: 항상 마진널 그룹(S_j==1)도 포함 ===
    # marg_cols = []
    # for c in S_df.columns:
    #     v = torch.tensor(S_df[c].values.astype(np.int32),
    #                      dtype=torch.float32, device=device).view(-1, 1)
    #     pos = int(v.sum().item()); neg = N - pos
    #     if pos >= T and neg >= T:     # 기존 T/complement 조건 유지
    #         marg_cols.append(v)

    # # <<< 여기서 선택적으로 포함
    # if include_marginals_always and len(marg_cols) > 0:
    #     all_cols = marg_cols + cols
    # else:
    #     all_cols = cols

    # if len(all_cols) == 0:
    #     return torch.zeros((N, 0), device=device)
    
    # print(f"[tabular] N={N}, q={S_df.shape[1]}, T={T}, agg_repeat={agg_repeat}, "
    #       f"max_len={agg_max_len}, max_cols={agg_max_cols}")
    # print(f"[tabular] 관측 셀 K={len(counts)}, alive(>=T)={alive}, dead(<T)={len(counts)-alive}")
    # return torch.cat(all_cols, dim=1)

# log = logging.getLogger("fair")



# ----- DR 서브루틴 -----
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,8), nn.ReLU(), 
            nn.Linear(8,16), nn.ReLU(), 
            nn.Linear(16,32), nn.ReLU(), 
            nn.Linear(32,32), nn.ReLU(), 
            nn.Linear(32,16), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(16,8), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(8,1),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.net(z)

# def artanh_corr(yv, gz, eps=1e-6):
#     y_c = yv - yv.mean()
#     g_c = gz - gz.mean()
#     y_std = torch.sqrt((y_c**2).mean() + eps); g_std = torch.sqrt((g_c**2).mean() + eps)
#     corr = (y_c*g_c).mean()/(y_std*g_std + eps)
#     corr_abs = torch.clamp(torch.abs(corr), 0.0, 1.0-1e-6)
#     return torch.atanh(corr_abs)

def artanh_corr(yv, gz, eps=1e-6):
    y_c = yv - yv.mean()
    g_c = gz - gz.mean()
    corr = (y_c*g_c).mean()/(y_c**2).mean() 
    return torch.log(1 + torch.abs(corr))

import torch, math

# def mmd_loss(x, y, sigma=1.0, n_features=500, device=None):
#     """
#     RFF 기반 MMD 근사
#     - x, y: (N, d) 또는 (N,) 텐서
#     - sigma: RBF 커널 bandwidth
#     - n_features: 랜덤 Fourier feature 수
#     """
#     if x.dim() == 1:
#         x = x.unsqueeze(1)
#     if y.dim() == 1:
#         y = y.unsqueeze(1)
#     if device is None:
#         device = x.device

#     N, d = x.shape
#     M, _ = y.shape

#     # 랜덤 주파수와 위상
#     W = torch.randn(d, n_features, device=device) / sigma
#     b = 2 * math.pi * torch.rand(n_features, device=device)

#     # RFF 매핑
#     Zx = torch.sqrt(torch.tensor(2.0 / n_features, device=device)) * torch.cos(x @ W + b)
#     Zy = torch.sqrt(torch.tensor(2.0 / n_features, device=device)) * torch.cos(y @ W + b)

#     # MMD^2 근사
#     return (Zx.mean(0) - Zy.mean(0)).pow(2).sum()

def build_C_tensor(S_df, args, device="cuda", n_low=None, n_low_frac=None,
                   apriori_union=True, agg_max_len=4, agg_max_cols=2048):
    N = len(S_df)
    if N == 0 or S_df.shape[1] == 0:
        return torch.zeros((N, 0), device=device)
    if apriori_union:
        return build_C_from_unions_df(
            S_df, device=device, n_low=n_low, n_low_frac=n_low_frac,
            agg_max_len=agg_max_len, agg_max_cols=agg_max_cols,
            af_max_order=int(getattr(args, "af_max_order", 3)),
            unions_mode=str(getattr(args, "union_mode", "pack"))  # NEW
        )
    if (n_low_frac is not None) and (n_low_frac > 0):
        min_support = int(np.ceil(float(n_low_frac) * N))
    else:
        min_support = int(n_low or 0)
    cols = list(S_df.columns)
    mats = []
    for c in cols:
        v = torch.tensor(S_df[c].values.astype(np.int32),
                         dtype=torch.float32, device=device)
        pos = int(v.sum().item()); neg = int(N - pos)
        if pos >= min_support and neg >= min_support:
            mats.append(v.view(-1, 1))
    if len(mats) == 0:
        return torch.zeros((N, 0), device=device)
    return torch.cat(mats, dim=1)

# def _train_tabular(args, data, device):
#     X_tr = torch.tensor(data["X_train"].values, dtype=torch.float32, device=device)
#     y_tr = torch.tensor(data["y_train"], dtype=torch.float32, device=device)
#     X_va = torch.tensor(data["X_val"].values, dtype=torch.float32, device=device)
#     y_va = torch.tensor(data["y_val"], dtype=torch.float32, device=device)
#     S_tr = data["S_train"]; S_va = data["S_val"]
#     n, d = X_tr.shape
#     # f = MLP(d).to(device)
#     print("d_x: ", d)

#     f = MLP(d).to(device)

#     g = Discriminator().to(device)
#     n_low = getattr(args, "n_low", None)
#     n_low_frac = getattr(args, "n_low_frac", None)
#     use_ap = bool(getattr(args, "agg_apriori", True) or getattr(args, "apriori_union", False))
#     agg_max_len = int(getattr(args, "agg_max_len", 4))
#     agg_max_cols = int(getattr(args, "agg_max_cols", 2048))
#     with torch.no_grad():
#         Ctr = build_C_tensor(
#             S_tr, args, device=device, n_low=n_low, n_low_frac=n_low_frac,
#             apriori_union=use_ap, agg_max_len=agg_max_len, agg_max_cols=agg_max_cols
#         )
#     v = nn.Parameter(torch.randn(Ctr.shape[1], device=device))
#     if Ctr.shape[1] > 0:
#         v.data = v / (v.norm() + 1e-12)
#     opt_f = optim.Adam(f.parameters(), lr=args.lr, weight_decay=1e-7)
#     opt_g = optim.Adam(g.parameters(), lr=args.lr*3)
#     opt_v = optim.Adam([v], lr=args.lr*10)
#     loss_bce = nn.BCEWithLogitsLoss()
#     best_val = float("inf"); best_acc = -1.0; best_f = None
#     thr = float(getattr(args, "thr", 0.5))
#     for ep in tqdm(range(args.epochs)):
#         f.train(); g.train()
#         logit = f(X_tr)
#         cls = loss_bce(logit, y_tr)
#         if args.lambda_fair == 0.0 or Ctr.shape[1] == 0:
#             total = cls
#             opt_f.zero_grad(); total.backward(); opt_f.step()
#         else:
#             with torch.no_grad():
#                 v_unit = v / (v.norm() + 1e-12)
#                 yv = (Ctr @ v_unit).detach()
#             gz = g(logit.unsqueeze(1)).squeeze(-1)
#             dr = artanh_corr(yv, gz)
#             total = cls + args.lambda_fair * dr
#             opt_f.zero_grad(); total.backward(retain_graph=True); opt_f.step()
#             for _ in range(3):
#                 logit_det = f(X_tr).detach()
#                 v_unit = v / (v.norm() + 1e-12)
#                 yv = (Ctr @ v_unit)
#                 gz = g(logit_det.unsqueeze(1)).squeeze(-1)
#                 dr = artanh_corr(yv, gz)
#                 opt_g.zero_grad(); (-dr).backward(retain_graph=True); opt_g.step()
#                 opt_v.zero_grad(); (-dr).backward(); opt_v.step()
#             with torch.no_grad():
#                 v.data = v.data / (v.data.norm() + 1e-12)
#         f.eval()
#         with torch.no_grad():
#             logit_va = f(X_va)
#             prob_va = torch.sigmoid(logit_va)
#             pred_va = (prob_va >= thr).float()
#             val_acc = (pred_va == y_va).float().mean().item()
#             val_loss = loss_bce(logit_va, y_va).item()
#             if (val_acc > best_acc) or (np.isclose(val_acc, best_acc) and val_loss < best_val):
#                 best_acc = val_acc; best_val = val_loss
#                 best_f = {k: v_.detach().cpu().clone() for k, v_ in f.state_dict().items()}
#     if best_f is not None:
#         f.load_state_dict({k: v_.to(device) for k, v_ in best_f.items()})


    
#     # [ADD-TS] Temperature scaling on validation
#     with torch.no_grad():
#         logit_va = f(X_va)                       # [N_val]
#     bce = nn.BCEWithLogitsLoss()

#     # logT로 최적화해 T>0 보장 (T = exp(logT))
#     logT = torch.zeros((), device=device, requires_grad=True)
#     optT = torch.optim.Adam([logT], lr=0.05)

#     for _ in range(200):  # 가볍게 200스텝
#         T = torch.exp(logT) + 1e-6
#         loss_T = bce(logit_va / T, y_va)        # NLL 최소화
#         optT.zero_grad()
#         loss_T.backward()
#         optT.step()

#     T = float(torch.exp(logT).item())

#     # (선택) 너무 극단적 T 방지: 클리핑
#     T = float(np.clip(T, 0.25, 4.0))

#     # calibration 적용한 확률로 임계값 튜닝
#     with torch.no_grad():
#         prob_va_cal = torch.sigmoid(logit_va / T).cpu().numpy()

#     thr_metric = getattr(args, "thr_metric", "accuracy")  # 'accuracy'|'f1'|'youden' 등
#     thr = float(tune_threshold(prob_va_cal, y_va.cpu().numpy()))
#     log.info(f"[TS] fitted T={T:.4f}; [THR] tuned metric={thr_metric}, thr={thr:.4f}")

#     # === Test with calibrated probs ===
#     f.eval()
#     with torch.no_grad():
#         logit_te = f(torch.tensor(
#             data["X_test"].values, dtype=torch.float32, device=device
#         ))
#         p_test = torch.sigmoid(logit_te / T).cpu().numpy()

#     yhat = (p_test >= thr).astype(int)

#     f.eval()
#     with torch.no_grad():
#         p_test = torch.sigmoid(f(torch.tensor(
#             data["X_test"].values, dtype=torch.float32, device=device
#         ))).cpu().numpy()
#         yhat = (p_test >= thr).astype(int)
#     return dict(proba=p_test, pred=yhat)

# def _train_image(args, data, device):
#     f = SmallConvNet().to(device)
#     g = Discriminator().to(device)
#     opt_f = optim.Adam(f.parameters(), lr=args.lr, weight_decay=1e-4)
#     opt_g = optim.Adam(g.parameters(), lr=args.lr*3)
#     loss_bce = nn.BCEWithLogitsLoss()
#     thr = float(getattr(args, "thr", 0.5))
#     best_val = float("inf"); best_f = None
#     def batch_C(S_list, n_low_frac=None, n_low=None, 
#             apriori_union=False, agg_max_len=4, agg_max_cols=512,
#             agg_repeat=64):  # NEW
#         import numpy as np, torch
#         keys = list(data["meta"]["sens_list"])
#         N = len(S_list)
#         if N == 0: return torch.zeros((0,0), device=device)
#         if (n_low_frac is not None) and (n_low_frac > 0):
#             T = int(np.ceil(float(n_low_frac) * N))
#         else:
#             T = int(n_low or 0)
#         codes = np.zeros(N, dtype=np.int64)
#         for j, k in enumerate(keys):
#             bit = np.array([int(bool(s[k])) for s in S_list], dtype=np.int64)
#             codes |= (bit << j)
#         uniq, inv = np.unique(codes, return_inverse=True)
#         masks = []; counts = []
#         for uidx in range(len(uniq)):
#             m = torch.from_numpy((inv == uidx)).to(device)
#             masks.append(m); counts.append(int(m.sum().item()))
#         # import pdb; pdb.set_trace()
#         alive = sum(c >= T for c in counts)
#         print("원자 셀 단계에서 살아남은 개수 =", alive)
#         # Apriori unions
#         if apriori_union:
#             print("DR) Apriori 기반 합집합 생성 중...")
#             # unions = _apriori_min_cover(counts, T, max_len=agg_max_len)
#             unions = _random_pack_unions(
#                 counts, T, M=int(agg_repeat), max_len=agg_max_len, max_cols=agg_max_cols
#             )
#             cols = []
#             for idx_tuple in unions:
#                 um = None
#                 for i_local in idx_tuple:
#                     m = masks[i_local]
#                     um = m if um is None else (um | m)
#                 pos = int(um.sum().item()); neg = N - pos
#                 if pos >= T and neg >= T:
#                     cols.append(um.float().view(-1,1))
#                 if len(cols) >= int(agg_max_cols): break
#             return torch.cat(cols, dim=1) if len(cols)>0 else torch.zeros((N,0), device=device)
#         Ms = []
#         for m in masks:
#             cnt = int(m.sum().item()); neg = N - cnt
#             if cnt >= T and neg >= T:
#                 Ms.append(m.float().view(-1,1))
#         return torch.cat(Ms, dim=1) if len(Ms)>0 else torch.zeros((N,0), device=device)
#     for ep in range(args.epochs):
#         f.train(); g.train()
#         for x,y,S in data["train_loader"]:
#             x=x.to(device); y=y.float().to(device)
#             logit = f(x); cls = loss_bce(logit, y)
#             if args.lambda_fair == 0.0:
#                 opt_f.zero_grad(); cls.backward(); opt_f.step(); continue
#             Cb = batch_C(
#                 S,
#                 n_low_frac=getattr(args, "n_low_frac", None),
#                 n_low=getattr(args, "n_low", None),
#                 apriori_union=bool(getattr(args, "agg_apriori", True) or getattr(args, "apriori_union", False)),
#                 agg_max_len=int(getattr(args, "agg_max_len", 4)),
#                 agg_max_cols=int(getattr(args, "agg_max_cols", 512)),
#                 agg_repeat=int(getattr(args, "agg_repeat", 64)),  # NEW
#             )
#             if Cb.shape[1] == 0:
#                 opt_f.zero_grad(); cls.backward(); opt_f.step(); continue
#             with torch.no_grad():
#                 v_ = torch.randn(Cb.shape[1], device=device)
#                 v_ = v_ / (v_.norm() + 1e-12)
#                 yv = (Cb @ v_)
#             gz = g(logit.unsqueeze(1)).squeeze(-1)
#             dr = artanh_corr(yv, gz)
#             total = cls + args.lambda_fair * dr
#             opt_f.zero_grad(); total.backward(retain_graph=True); opt_f.step()
#             opt_g.zero_grad(); (-dr).backward(); opt_g.step()
#         f.eval(); val_losses = []
#         with torch.no_grad():
#             for x,y,S in data["val_loader"]:
#                 x=x.to(device); y=y.float().to(device)
#                 logit = f(x)
#                 val_losses.append(loss_bce(logit, y).item())
#         val_loss = float(np.mean(val_losses)) if len(val_losses)>0 else float("inf")
#         if val_loss < best_val:
#             best_val = val_loss
#             best_f = {k: v_.detach().cpu().clone() for k, v_ in f.state_dict().items()}
#     if best_f is not None:
#         f.load_state_dict({k: v_.to(device) for k, v_ in best_f.items()})

#     f.eval(); pt=[]
#     with torch.no_grad():
#         for x,y,S in data["test_loader"]:
#             p = torch.sigmoid(f(x.to(device))).cpu().numpy()
#             pt.append(p)
#     pt = np.concatenate(pt)
#     yhat = (pt >= thr).astype(int)
#     return dict(proba=pt, pred=yhat)

# def run_dr_subgroup_subset_random(args, data):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     if data["type"] == "tabular":
#         return _train_tabular(args, data, device)
#     elif data["type"] == "image":
#         return _train_image(args, data, device)
#     else:
#         raise ValueError(data["type"])
