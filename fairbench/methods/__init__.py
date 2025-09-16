# fairbench/methods/__init__.py
# fairbench/methods/dr.py
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from ..utils.threshold import tune_threshold
from tqdm.auto import tqdm
# ===== NEW: Apriori-based unions over disjoint base subgroups =====
from ..utils.mlp import MLP, Linear

from .dr import run_dr
from .dr_subgroup_subset import run_dr_subgroup_subset
from .dr_subgroup_subset_3q import run_dr_subgroup_subset_3q
from .dr_subgroup_subset_random import run_dr_subgroup_subset_random
from .gerryfair_wrapper import run_gerryfair
from .multicalib_wrapper import run_multicalib
from .seq_wrapper import run_sequential
from .reduction_wrapper import run_reduction
from .unfair_wrapper import run_unfair
from .reg_wrapper import run_reg

# dr.py 상단
from .loss_diag_log import (
    log_loss_scalar, log_loss_breakdowns,
    per_sample_bce_vec, fair_vec_from_corr, fair_vec_from_mh_bce, cls_total_vec,
    c_to_masks_bool, masks_from_S_df
)

# NEW: train/val에서 subset 마스크(C) 만들 때 사용
from .dr_subgroup_subset_random import build_C_tensor
# NEW: DRFair 타입 체크(있으면 정밀 로그, 없으면 완화 로그)
from ..utils.fair_losses import DRFair

import numpy as np, random, torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# 상단 임포트에 추가
try:
    from ..utils.fair_losses import get_fair_loss
    from ..utils.fair_losses import NoneFair
except Exception:
    from fairbench.utils.fair_losses import get_fair_loss

try:
    from ..utils.mlp import MLP
except Exception:
    from fairbench.utils.mlp import MLP

# # ---- unified hparam helpers ----
# def _get_param(args, keys, default):
#     """args에서 keys 중 먼저 발견되는 값을 반환. 없으면 default."""
#     for k in keys:
#         if hasattr(args, k):
#             v = getattr(args, k)
#             if v is not None:
#                 return v
#     return default

# def _get_fair_weight(a) -> float:
#     """메서드별 공정성 람다 키를 한 곳에서 처리"""
#     return float(_get_param(a, ("lambda_fair", "mf_lambda", "gf_C", "reg_lambda"), 0.0))

# def _get_lr(a) -> float:
#     """모델 학습용 lr: lr > mf_lr > gf_lr_model"""
#     return float(_get_param(a, ("lr", "mf_lr", "gf_lr_model"), 1e-3))

# def _get_lr_group(a, base_lr: float) -> float:
#     """어드버서리/그룹용 lr: gf_lr_group 없으면 3*base_lr"""
#     return float(_get_param(a, ("gf_lr_group",), 3.0*base_lr))

# def _get_lr_v(a, base_lr: float) -> float:
#     """v(보조변수)용 lr: 지정 없으면 10*base_lr"""
#     return float(_get_param(a, ("v_lr",), 10.0*base_lr))

# def _get_wd(a) -> float:
#     """weight decay: wd > mf_wd > 기본"""
#     return float(_get_param(a, ("wd", "mf_wd"), 1e-6))

# def _get_epochs(a) -> int:
#     """epochs: epochs > mf_epochs > gf_max_iters"""
#     return int(_get_param(a, ("epochs", "mf_epochs", "gf_max_iters"), 200))

# def _get_thr(a) -> float:
#     return float(_get_param(a, ("thr",), 0.5))



def _train_tabular(args, data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = getattr(args, "base_model")

    X_tr = torch.tensor(data["X_train"].values, dtype=torch.float32, device=device)
    y_tr = torch.tensor(data["y_train"], dtype=torch.float32, device=device)
    X_va = torch.tensor(data["X_val"].values, dtype=torch.float32, device=device)
    y_va = torch.tensor(data["y_val"], dtype=torch.float32, device=device)

     # 데이터 텐서 만든 직후에 추가 ▼
    S_tr = data["S_train"]
    S_va = data["S_val"]

    # subgroup masks (마지널)
    S_masks_tr = masks_from_S_df(S_tr)
    S_masks_val = masks_from_S_df(S_va)

    # subset masks (C 열 → bool 마스크들)
    # train 은 DRFair일 때 fair.Ctr를 그대로 쓰는 게 가장 안전
    V_masks_tr = []
    V_masks_val = []

    n, d = X_tr.shape
    if base_model == "mlp":
        f = MLP(d).to(device)
    elif base_model == "linear":
        f = Linear(d).to(device)

    
    def _get_fair_weight(args):
        method = getattr(args, "method", None)
        if method == "gerryfair":
            return float(getattr(args, "gf_C", 0.0))
        elif method == "reg":
            return float(getattr(args, "mf_lambda", 0.0))
        else:
            # 기본은 lambda_fair
            return float(getattr(args, "lambda_fair", 0.0))

        # return 0.0

        # --- unified hparam getters (NEW) ---
    def _get_param(args, keys, default):
        for k in keys:
            if hasattr(args, k):
                v = getattr(args, k)
                if v is not None:
                    return v
        return default

    def _get_lr(a) -> float:
        # lr > mf_lr > gf_lr_model
        # return float(_get_param(a, ("lr", "mf_lr", "gf_lr_model"), 1e-3))
        return float(1e-3)

    def _get_wd(a) -> float:
        # wd > mf_wd
        # return float(_get_param(a, ("wd", "mf_wd"), 1e-6))
        return 1e-6

    def _get_epochs(a) -> int:
        # epochs > mf_epochs > gf_max_iters
        # return int(_get_param(a, ("epochs", "mf_epochs", "gf_max_iters"), 200))
        return 200


    def _get_thr(a) -> float:
        # return float(_get_param(a, ("thr",), 0.5))
        return 0.5

    # import pdb;pdb.set_trace()
    fair_weight = _get_fair_weight(args)  # <-- NEW

    # ★ 공정성 패널티 모듈: lam<=0이면 NoneFair로 단락 (불필요한 준비 비용도 회피)
    fair = get_fair_loss(args, data, fair_weight, device) if fair_weight > 0 else NoneFair()  # <-- NEW

    # ▼▼▼ NEW: subset 마스크 준비 (DRFair이면 train은 fair.Ctr로, val은 build_C_tensor로)
    if isinstance(fair, DRFair) and getattr(fair, "Ctr", None) is not None and fair.Ctr.numel() > 0:
        V_masks_tr = c_to_masks_bool(fair.Ctr)  # train용 subset
        with torch.no_grad():
            C_val = build_C_tensor(
                S_va, args, device=device,
                n_low=getattr(args, "n_low", None),
                n_low_frac=getattr(args, "n_low_frac", None),
                apriori_union=bool(getattr(args, "agg_apriori", True) or getattr(args, "apriori_union", False)),
                agg_max_len=int(getattr(args, "agg_max_len", 400)),   # ← 400으로
                agg_max_cols=int(getattr(args, "agg_max_cols", 256)),
            )
        V_masks_val = c_to_masks_bool(C_val)
    else:
        # DRFair가 아니면 subset 로깅은 생략하거나 필요시 마지널만 사용
        V_masks_tr, V_masks_val = [], []


    # opt_f = torch.optim.Adam(f.parameters(), lr=args.lr, weight_decay=1e-7)
    lr = _get_lr(args)
    wd = _get_wd(args)
    epochs = _get_epochs(args)
    opt_f = torch.optim.Adam(f.parameters(), lr=lr, weight_decay=wd)
    loss_bce = nn.BCEWithLogitsLoss()

    # best_val = float("inf")
    # best_acc = -1.0
    # best_f = None

       # --- [ADD] Early-Stop/Checkpoint 설정
    metric = getattr(args, "early_metric", "cls_total")  # {'cls_total','cls_bce','loss_bce','acc'}
    patience = int(getattr(args, "early_patience", 10))
    mode = "min" if metric in ("cls_total","cls_bce","loss_bce","loss") else "max"
    best_score = float("inf") if mode=="min" else -float("inf")
    best_epoch = -1
    no_improve = 0
    best_f = None

    thr = _get_thr(args)

    warmup = int(getattr(args, "fair_warmup_epochs", 10)) # <-- NEW 1004
    decay_start = int(0.85 * epochs)

    for ep in tqdm(range(epochs)):
        f.train()
        logits = f(X_tr)                                   # (N,)
        probs  = torch.sigmoid(logits).clamp(0.0, 1.0)

        cls = loss_bce(logits, y_tr)
        if fair_weight > 0:   
            if ep < warmup:
                lam_t = fair_weight * (ep / max(1,warmup))
            elif ep >= decay_start:
                lam_t = fair_weight * (1 - (ep - decay_start) / max(1, epochs - decay_start))
            else:
                lam_t = fair_weight
                
            pen = fair.penalty(logits, probs)
            total = cls + lam_t * pen                             # <-- NEW 1004
            # pen = fair.penalty(logits, probs)              # 스칼라
            # total = cls + fair_weight * pen
        else:
            total = cls

        opt_f.zero_grad()
        total.backward()
        opt_f.step()

        # adversary / v 등은 lam>0일 때만 업데이트
        if fair_weight > 0:                                # <-- NEW
            fair.after_f_step(f, X_tr)

        # ----- validation -----
        f.eval()
        with torch.no_grad():
            logit_va = f(X_va)  
            prob_va  = torch.sigmoid(logit_va)
            pred_va = (prob_va >= thr).float()
            val_acc = (pred_va == y_va).float().mean().item()
            val_loss = loss_bce(logit_va, y_va).item()
            bce_vec_val = per_sample_bce_vec(logit_va, y_va)       # numpy (N,)
            if isinstance(fair, DRFair) and getattr(fair, "Ctr", None) is not None and len(V_masks_val) > 0:
                v_unit  = fair._v_unit()
                if 'C_val' in locals() and C_val.shape[1] == fair.Ctr.shape[1]:
                    yv_val = (C_val @ v_unit)                      # torch (N,)
                else:
                    if len(V_masks_val) > 0:
                        memb = torch.tensor(
                            np.stack(V_masks_val, axis=1).astype(np.float32), device=device
                        ).sum(dim=1)
                        yv_val = (memb - memb.mean()) / (memb.std() + 1e-6)
                    else:
                        yv_val = torch.zeros_like(logit_va)

                gz_val = fair.g(prob_va.unsqueeze(1)).squeeze(-1)  # torch (N,)
                fair_vec_val = fair_vec_from_corr(yv_val, gz_val)  # numpy (N,)
                disc_vec_val = fair_vec_val.copy()
            else:
                pen_val = float(fair.penalty(logit_va, prob_va).detach().cpu().item()) if fair_weight > 0 else 0.0
                fair_vec_val = np.full_like(bce_vec_val, fill_value=pen_val/max(1, bce_vec_val.shape[0]), dtype=float)
                disc_vec_val = fair_vec_val.copy()

            lam_use = lam_t if (fair_weight > 0) else 0.0
            total_vec_val = cls_total_vec(bce_vec_val, fair_vec_val, lam=lam_use) # ← 스케줄 람다 사용

            # 스칼라 로그
            # disc_s_val     = float(np.mean(disc_vec_val))
            cls_bce_s_val  = float(np.mean(bce_vec_val))
            # cls_fair_s_val = float(np.mean(fair_vec_val) * lam_use )
            cls_tot_s_val  = float(np.mean(total_vec_val))

            # log_loss_scalar(ep, "val",   disc_s_val, cls_tot_s_val, cls_bce_s_val, cls_fair_s_val)
            # log_loss_breakdowns(ep, "val", "disc",      disc_vec_val,  S_masks_val, V_masks_val, topk=25)
            # log_loss_breakdowns(ep, "val", "cls_bce",   bce_vec_val,   S_masks_val, V_masks_val, topk=25)
            # log_loss_breakdowns(ep, "val", "cls_fair",  fair_vec_val,  S_masks_val, V_masks_val, topk=25)
            # log_loss_breakdowns(ep, "val", "cls_total", total_vec_val, S_masks_val, V_masks_val, topk=25)
        
        
        # --- [ADD][CHG] 체크포인트 판단을 여기(VAL 스칼라 계산 직후)로 이동
        if metric == "cls_total":
            cur = cls_tot_s_val
        elif metric == "cls_bce":
            cur = cls_bce_s_val
        elif metric in ("loss_bce","loss","bce"):
            cur = val_loss
        elif metric in ("acc","accuracy"):
            cur = val_acc
        else:
            cur = cls_tot_s_val  # 안전 디폴트

        improved = (cur < best_score - 1e-8) if mode=="min" else (cur > best_score + 1e-8)
        if improved:
            best_score = cur
            best_epoch = ep
            best_f = {k: v_.detach().cpu().clone() for k, v_ in f.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # --- [ADD] 얼리스톱
        if no_improve >= patience:
            print(f"[EarlyStop] metric={metric}, best@ep={best_epoch}, score={best_score:.6f}")
            break

        # # ---------- TRAIN per-sample ----------
        # with torch.no_grad():
        #     logit_tr_full = f(X_tr)
        #     bce_vec_tr = per_sample_bce_vec(logit_tr_full, y_tr) 
        #     if isinstance(fair, DRFair) and getattr(fair, "Ctr", None) is not None and fair.Ctr.numel() > 0:
        #         V_masks_tr = c_to_masks_bool(fair.Ctr)
        #         v_unit = fair._v_unit()
        #         yv_tr  = (fair.Ctr @ v_unit)                       # torch
        #         gz_tr  = fair.g(torch.sigmoid(logit_tr_full).unsqueeze(1)).squeeze(-1)
        #         fair_vec_tr = fair_vec_from_corr(yv_tr, gz_tr)     # numpy
        #         disc_vec_tr = fair_vec_tr.copy()
        #     else:
        #         pen_tr = float(fair.penalty(logit_tr_full, torch.sigmoid(logit_tr_full)).detach().cpu().item()) if fair_weight > 0 else 0.0
        #         fair_vec_tr = np.full_like(bce_vec_tr, fill_value=pen_tr/max(1, bce_vec_tr.shape[0]), dtype=float)
        #         disc_vec_tr = fair_vec_tr.copy()

        #     total_vec_tr  = cls_total_vec(bce_vec_tr,  fair_vec_tr,  lam=lam_use)  # ← 동일 람다

        #     disc_s_tr     = float(np.mean(disc_vec_tr))
        #     cls_bce_s_tr  = float(np.mean(bce_vec_tr))
        #     cls_fair_s_tr = float(np.mean(fair_vec_tr) * lam_use)
        #     cls_tot_s_tr  = float(np.mean(total_vec_tr))

        #     log_loss_scalar(ep, "train", disc_s_tr, cls_tot_s_tr, cls_bce_s_tr, cls_fair_s_tr)
        #     log_loss_breakdowns(ep, "train", "disc",      disc_vec_tr,  S_masks_tr, V_masks_tr, topk=25)
        #     log_loss_breakdowns(ep, "train", "cls_bce",   bce_vec_tr,   S_masks_tr, V_masks_tr, topk=25)
        #     log_loss_breakdowns(ep, "train", "cls_fair",  fair_vec_tr,  S_masks_tr, V_masks_tr, topk=25)
        #     log_loss_breakdowns(ep, "train", "cls_total", total_vec_tr, S_masks_tr, V_masks_tr, topk=25)
     
        

    if best_f is not None:
        f.load_state_dict({k: v_.to(device) for k, v_ in best_f.items()})

    # ---------------- Temperature scaling + threshold 튜닝 ----------------
    with torch.no_grad():
        logit_va = f(X_va)
    bce = nn.BCEWithLogitsLoss()
    logT = torch.zeros((), device=device, requires_grad=True)
    optT = torch.optim.Adam([logT], lr=0.05)
    for _ in range(200):
        T = torch.exp(logT) + 1e-6
        loss_T = bce(logit_va / T, y_va)
        optT.zero_grad(); loss_T.backward(); optT.step()
    T = float(torch.exp(logT).item())
    T = float(np.clip(T, 0.25, 4.0))
    with torch.no_grad():
        prob_va_cal = torch.sigmoid(logit_va / T).cpu().numpy()
    thr_metric = getattr(args, "thr_metric", "accuracy")
    thr = float(tune_threshold(prob_va_cal, y_va.cpu().numpy()))
    print(f"[TS] fitted T={T:.4f}; [THR] tuned metric={thr_metric}, thr={thr:.4f}")

    f.eval()
    with torch.no_grad():
        logit_te = f(torch.tensor(data["X_test"].values, dtype=torch.float32, device=device))
        p_test = torch.sigmoid(logit_te / T).cpu().numpy()
    yhat = (p_test >= thr).astype(int)
    return dict(proba=p_test, pred=yhat)


def run_method(args, data):
    # if args.method == "dr":
    #     return run_dr(args, data)
    # if args.method == "dr_subgroup_subset":
    #     return run_dr_subgroup_subset(args, data)
    # if args.method == "dr_subgroup_subset_3q":
    #     return run_dr_subgroup_subset_3q(args, data)

    if args.method == "dr_subgroup_subset_random":
        return _train_tabular(args, data)
    if args.method == "gerryfair":
        return _train_tabular(args, data)
    if args.method == 'reg':
        return _train_tabular(args, data)
    

    if args.method == "multicalib":
        return run_multicalib(args, data)
    if args.method == "sequential":
        return run_sequential(args, data)
    if args.method == "reduction":
        return run_reduction(args, data)
    if args.method == "unfair":
        return run_unfair(args, data)
    raise ValueError(args.method)