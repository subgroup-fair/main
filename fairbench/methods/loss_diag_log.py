# -*- coding: utf-8 -*-
import numpy as np
import logging

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

log = logging.getLogger("fair")

# ---------- 작은 유틸 ----------
def _to_np(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def c_to_masks_bool(C_torch):
    """C (N x M) -> [M개의 bool mask]"""
    if C_torch is None:
        return []
    C = (C_torch > 0.5).detach().cpu().numpy().astype(bool)
    return [C[:, j] for j in range(C.shape[1])]

def masks_from_S_df(S_df):
    """마지널 subgroup 마스크(각 컬럼별)"""
    out = []
    N = len(S_df)
    for c in S_df.columns:
        v = S_df[c].astype(int).values.astype(bool)
        if v.sum() == 0 or v.sum() == N:
            continue
        out.append(v)
    return out

def _summ_by_masks(vec, masks, kind="subgroup", topk=25):
    N = vec.shape[0]
    rows = []
    for i, m in enumerate(masks):
        m = np.asarray(m, bool).reshape(-1)
        if m.size != N:
            continue
        sup = int(m.sum()); comp = N - sup
        if sup == 0 or comp == 0:
            continue
        rows.append((i, sup, comp, float(vec[m].mean())))
    rows.sort(key=lambda r: r[3], reverse=True)
    return rows[: max(1, topk)]

# ---------- per-sample 벡터 만들기 ----------
def per_sample_bce_vec(logits, y):
    """분류 BCE (N,)"""
    if torch is None:
        raise RuntimeError("PyTorch 필요")
    loss = nn.BCEWithLogitsLoss(reduction="none")(logits, y)
    return _to_np(loss).reshape(-1)

# fairbench/methods/loss_diag_log.py


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def fair_vec_from_corr(yv, gz, w=None, eps=1e-6):
    """
    yv, gz : (N,) 또는 (N,1), torch/np 무엇이든 OK
    반환: (N,) numpy, 각 샘플의 정규화된 상관 공헌도
    """
    y = _to_np(yv).reshape(-1)
    g = _to_np(gz).reshape(-1)
    if w is None:
        w = np.ones_like(y, dtype=float)
    else:
        w = _to_np(w).reshape(-1).astype(float)
    w = w / (w.sum() + eps)

    y_c = y - (w * y).sum()
    g_c = g - (w * g).sum()

    # 공분산의 샘플별 기여를 정규화
    denom = np.sqrt((w * y_c**2).sum()) * np.sqrt((w * g_c**2).sum()) + eps
    contrib = np.abs(w * y_c * g_c) / denom
    return contrib  # numpy (N,)

def cls_total_vec(bce_vec, fair_vec, lam: float):
    b = np.asarray(bce_vec, dtype=float).reshape(-1)
    f = np.asarray(fair_vec, dtype=float).reshape(-1)
    assert b.shape == f.shape, f"shape mismatch: bce={b.shape}, fair={f.shape}"
    return b + float(lam) * f


def fair_vec_from_mh_bce(bce_mat, heads=None, reduction="mean"):
    """
    멀티헤드 BCE 감사자일 때 공정성 벡터 (N,)
    bce_mat: (N, M_active) per-sample BCE
    heads: 사용 헤드 인덱스 (Top-K 등)
    reduction: 'mean'|'sum'
    """
    L = _to_np(bce_mat)
    if L.ndim == 1:
        L = L[:, None]
    if heads is not None and len(heads) > 0:
        L = L[:, list(heads)]
    v = L.mean(axis=1) if reduction == "mean" else L.sum(axis=1)
    return v.astype(float)

# ---------- 로깅 본체 ----------
def log_loss_scalar(epoch, split, disc, cls_total, cls_bce, cls_fair):
    log.info(f"[LOSS][{split}] ep={epoch} | disc={disc:.6f} | cls_total={cls_total:.6f} | cls_bce={cls_bce:.6f} | cls_fair={cls_fair:.6f}")

def _log_one_breakdown(title, vec, rows):
    v = vec
    log.info(f"[BREAKDOWN] {title} | N={len(v)} | mean={np.mean(v):.6f} | std={np.std(v):.6f} | min={np.min(v):.6f} | max={np.max(v):.6f}")
    for rank, (idx, sup, comp, m) in enumerate(rows, 1):
        log.info(f"  #{rank:02d} id={idx} | support={sup} | comp={comp} | mean={m:.6f}")

def log_loss_breakdowns(epoch, split, name, vec,
                        S_masks=None, V_masks=None, topk=25):
    v = _to_np(vec).reshape(-1).astype(float)
    if S_masks:
        rowsS = _summ_by_masks(v, S_masks, kind="subgroup", topk=topk)
        _log_one_breakdown(f"{name} by SUBGROUP (top{len(rowsS)}) ep={epoch} split={split}", v, rowsS)
    if V_masks:
        rowsV = _summ_by_masks(v, V_masks, kind="subset", topk=topk)
        _log_one_breakdown(f"{name} by SUBSET (top{len(rowsV)}) ep={epoch} split={split}", v, rowsV)
