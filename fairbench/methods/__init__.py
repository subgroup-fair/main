# fairbench/methods/__init__.py
# fairbench/methods/dr.py
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from ..utils.threshold import tune_threshold
from tqdm.auto import tqdm
# ===== NEW: Apriori-based unions over disjoint base subgroups =====
from ..utils.mlp import MLP

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

    X_tr = torch.tensor(data["X_train"].values, dtype=torch.float32, device=device)
    y_tr = torch.tensor(data["y_train"], dtype=torch.float32, device=device)
    X_va = torch.tensor(data["X_val"].values, dtype=torch.float32, device=device)
    y_va = torch.tensor(data["y_val"], dtype=torch.float32, device=device)

    n, d = X_tr.shape
    f = MLP(d).to(device)

    # --- 메서드별 람다 키 통일해서 읽기 ---
    def _get_fair_weight(args):  # <-- NEW
        method = getattr(args, "method")
        if method == "gerryfair": 
            return getattr(args, "gf_C")
        elif method == "reg":
            return getattr(args, "mf_lambda")
        elif method == "dr_subgroup_subset_random":
            return getattr(args, "lambda_fair")
            

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

    # opt_f = torch.optim.Adam(f.parameters(), lr=args.lr, weight_decay=1e-7)
    lr = _get_lr(args)
    wd = _get_wd(args)
    epochs = _get_epochs(args)
    opt_f = torch.optim.Adam(f.parameters(), lr=lr, weight_decay=wd)
    loss_bce = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_acc = -1.0
    best_f = None
    thr = _get_thr(args)

    warmup = int(getattr(args, "fair_warmup_epochs", 30)) # <-- NEW 1004
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
            prob_va = torch.sigmoid(logit_va)
            pred_va = (prob_va >= thr).float()
            val_acc = (pred_va == y_va).float().mean().item()
            val_loss = loss_bce(logit_va, y_va).item()

        if (val_acc > best_acc) or (np.isclose(val_acc, best_acc) and val_loss < best_val):
            best_acc = val_acc
            best_val = val_loss
            best_f = {k: v_.detach().cpu().clone() for k, v_ in f.state_dict().items()}

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