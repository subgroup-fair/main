# fairbench/methods/dr.py
import numpy as np, torch, torch.nn as nn
import random
from itertools import combinations, product

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# nlow above
def build_required_atomic_masks(S_df, T, max_cols=None, log_prefix="supgroup>T"):

    N, q = len(S_df), S_df.shape[1]
    if N == 0 or q == 0 or T <= 0:
        print(f"[{log_prefix}] skip: N={N}, q={q}, T={T}")
        return []

    X = S_df.values.astype(np.int8) 
    codes = np.zeros(N, dtype=np.uint64)
    for j in range(q):
        codes |= (X[:, j].astype(np.uint64) << j)

    uniq, inv = np.unique(codes, return_inverse=True)
    out = []
    seen = set()
    kept = 0

    for uid, code in enumerate(uniq):
        m = (inv == uid)
        cnt = int(m.sum())
        if cnt >= T:
            key = m.tobytes()
            if key not in seen:
                seen.add(key)
                out.append(m)
                kept += 1
                print(f"[{log_prefix}] atomic code={int(code)}, cnt={cnt}  -> kept")
                if (max_cols is not None) and (kept >= int(max_cols)):
                    print(f"[{log_prefix}] reached max_cols={max_cols}; stop early (kept={kept})")
                    break
        else:
            pass

    print(f"[{log_prefix}] required atomic masks kept={kept}")
    return out


def concat_masks_unique(required_masks, generated_masks, log_prefix="subgroup+order"):
    out = []
    seen = set()
    for m in (required_masks or []):
        mb = np.asarray(m, dtype=bool).ravel()
        key = mb.tobytes()
        if key not in seen:
            seen.add(key)
            out.append(mb)
    kept_req = len(out)

    for m in (generated_masks or []):
        mb = np.asarray(m, dtype=bool).ravel()
        key = mb.tobytes()
        if key not in seen:
            seen.add(key)
            out.append(mb)

    print(f"[{log_prefix}] concat: required={kept_req}, total={len(out)} (unique)")
    return out


def _apriori_forward_unions_from_df(
    S_df, T, max_order=[1,2,3], max_cols=None, log_prefix="Training order",
    prune_on_comp=True
):
    N, q = len(S_df), S_df.shape[1]
    if N == 0 or q == 0 or T <= 0:
        print(f"[{log_prefix}] skip: N={N}, q={q}, T={T}")
        return []

    X = S_df.values.astype(np.int32)  # (N, q) in {0,1}
    q_max = q 

    
    if isinstance(max_order, (list, tuple, set)):
        orders = sorted({int(k) for k in max_order if int(k) >= 1})
        if len(orders) == 0:
            print(f"[{log_prefix}] no valid orders in {max_order}; return empty")
            return []
        q_max = min(max(orders), q)
    else:
        q_max = q if (max_order is None or max_order <= 0) else min(int(max_order), q)
        orders = list(range(1, q_max + 1))

    out, kept = [], 0

    print(f"[Apriori-forward] Max order: {q_max}")
    print(f"[{log_prefix}] N={N}, q={q}, T={T}, max_cols={max_cols}, prune_on_comp={prune_on_comp}")

    atoms_mask = [[None, None] for _ in range(q)]  
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

    alive_atom = alive_cnt & alive_comp if prune_on_comp else alive_cnt

    alive_by_attr = {j: [b for b in (0,1) if alive_atom[j, b]] for j in range(q)}
    attrs = [j for j, bs in alive_by_attr.items() if len(bs) > 0]
    if len(attrs) == 0:
        print(f"[{log_prefix}] no alive atoms at L1; return empty")
        return []

    # order = 1
    if 1 in orders:
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
    else:
        print(f"[{log_prefix}] order=1 skipped")

    for k in orders:
        if k < 2:
            continue
        if k > q_max:
            continue
        # order >= 2
        for k in range(2, q_max + 1):
            print(f"[Apriori-forward] Order: {k}")
            for comb in combinations(attrs, k):               
                pools = [alive_by_attr[j] for j in comb]     
                for pat in product(*pools):                  
                    m = np.ones(N, dtype=bool)
                    for j, b in zip(comb, pat):
                        m &= atoms_mask[j][b]
                        if not m.any():                    
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

    return out




# all select
def _enumerate_all_unions(counts, T, max_len=None, max_cols=None):

    N = int(np.sum(counts))
    K = len(counts)
    if K == 0 or T <= 0:
        print(f"[enum] skip: K={K}, T={T}")
        return []

    Lmax = K if (max_len is None) else int(max_len)
    Lmax = max(1, min(Lmax, K))

    out = []
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
    S_df: (N x q)
    """
    cols = list(S_df.columns)
    if len(cols) == 0 or len(S_df) == 0:
        return [], [], []
    B = np.stack([S_df[c].astype(np.int8).values for c in cols], axis=1)
    codes = np.zeros(len(S_df), dtype=np.int64)
    for j in range(B.shape[1]):
        codes |= (B[:, j].astype(np.int64) << j)
    uniq, inv = np.unique(codes, return_inverse=True) 
    N = len(S_df)
    masks, counts, order = [], [], []
    for k, u in enumerate(uniq.tolist()):
        m = torch.from_numpy((inv == k)).to(device)
        cnt = int(m.sum().item())
        masks.append(m)
        counts.append(cnt)
        order.append(int(u))
    return masks, counts, order


def build_C_from_unions_df(S_df, device="cuda",
                            n_low=None, n_low_frac=None,
                            agg_max_len=300, agg_max_cols=2048,
                            af_max_order=[1,2,3],
                            unions_mode="pack"): 

    # count
    N = len(S_df)
    if N == 0 or S_df.shape[1] == 0:
        return torch.zeros((N, 0), device=device)
    if (n_low_frac is not None) and (n_low_frac > 0):
        T = int(np.ceil(float(n_low_frac) * N))
    else:
        T = int(n_low or 0)
    print(f"[tabular] n low frac: {n_low_frac}, n low: {n_low}, T: {T}")
    
    # alive
    base_masks, counts, order = _bitpack_cells_from_df(S_df, device=device)
    alive = sum(c >= T for c in counts)
    print(f"[tabular] alive = {alive}")
    print("[max_order]: ", af_max_order)
    if len(base_masks) == 0:
        return torch.zeros((N, 0), device=device)

        
    if unions_mode == "apriori_forward":
        V = build_required_atomic_masks(S_df, T, max_cols=agg_max_cols)
        if af_max_order:
            masks_np = _apriori_forward_unions_from_df(
                S_df, T, max_order=af_max_order, max_cols=agg_max_cols, log_prefix="Train_V"
            )

            V = concat_masks_unique(V, masks_np)
            if len(masks_np) == 0:
                return torch.zeros((N, 0), device=device)

        cols = []
        for m_np in V:
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



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.net(z)

def artanh_corr(yv, gz, eps=1e-6):
    y_c = yv - yv.mean()
    g_c = gz - gz.mean()
    corr = (y_c*g_c).mean()/(y_c**2).mean() 
    return torch.log(1 + torch.abs(corr))


## main
def build_C_tensor(S_df, args, device="cuda", n_low=None, n_low_frac=None,
                   apriori_union=True, agg_max_len=4, agg_max_cols=2048):
    N = len(S_df)
    if N == 0 or S_df.shape[1] == 0:
        return torch.zeros((N, 0), device=device)
    if apriori_union:
        return build_C_from_unions_df(
            S_df, device=device, n_low=n_low, n_low_frac=n_low_frac,
            agg_max_len=agg_max_len, agg_max_cols=agg_max_cols,
            af_max_order= getattr(args, "af_max_order", [1,2,3]),
            unions_mode=str(getattr(args, "union_mode", "pack"))
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



def _bitpack_codes_df(S_df, key_order=None):
    if key_order is None:
        key_order = list(S_df.columns)
    else:
        missing = [c for c in key_order if c not in S_df.columns]
        if missing:
            raise ValueError(f"S_df is missing columns from train: {missing}")
    X = np.stack([S_df[c].astype(np.int8).to_numpy() for c in key_order], axis=1) 
    if not set(np.unique(X)).issubset({0, 1}):
        X = np.rint(np.clip(X, 0, 1)).astype(np.int8)

    codes = np.zeros(X.shape[0], dtype=np.int64)
    for j in range(X.shape[1]):
        codes |= (X[:, j].astype(np.int64) << j)
    return codes, key_order


def build_C_te_from_Ctr(S_tr, C_tr, S_te, device="cuda"):
    if isinstance(C_tr, torch.Tensor):
        Ctr_np = (C_tr.detach().cpu().numpy() > 0.5)
    else:
        Ctr_np = (np.asarray(C_tr) > 0.5)

    n_tr, K = Ctr_np.shape
    if K == 0:
        return torch.zeros((len(S_te), 0), dtype=torch.float32, device=device)

    codes_tr, key_order = _bitpack_codes_df(S_tr, key_order=None)
    codes_te, _ = _bitpack_codes_df(S_te, key_order=key_order)


    uniq_codes, inv_tr = np.unique(codes_tr, return_inverse=True)
    U = len(uniq_codes)
    cols_for_code = np.zeros((U, K), dtype=bool)

    for r in range(n_tr):
        cols_for_code[inv_tr[r]] |= Ctr_np[r]

    code2idx = {int(c): i for i, c in enumerate(uniq_codes)}
    idx_te = np.fromiter((code2idx.get(int(c), -1) for c in codes_te), count=len(codes_te), dtype=np.int64)

    Cte_np = np.zeros((len(S_te), K), dtype=np.float32)
    seen_mask = (idx_te >= 0)
    if np.any(seen_mask):
        Cte_np[seen_mask] = cols_for_code[idx_te[seen_mask]].astype(np.float32)

    return torch.tensor(Cte_np, dtype=torch.float32, device=device)