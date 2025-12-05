import numpy as np, torch, torch.nn as nn, torch.optim as optim
from tqdm.auto import tqdm

from ..utils.mlp import MLP, Linear
from ..metrics import marginal_kearns_order1_worst
from .dr_wrapper import build_C_tensor
from ..utils.fair_losses import DRFair

import numpy as np, random, torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

try:
    from ..utils.fair_losses import get_fair_loss
    from ..utils.fair_losses import NoneFair
except Exception:
    from fairbench.utils.fair_losses import get_fair_loss

try:
    from ..utils.mlp import MLP
except Exception:
    from fairbench.utils.mlp import MLP


def tune_threshold(proba, y, grid=None):
    grid = grid or np.linspace(0.05,0.95,37)
    best_t, best = 0.5, 0.0
    for t in grid:
        acc = ( (proba>=t).astype(int) == y ).mean()
        if acc>best: best_t, best = t, acc
    return float(best_t)



def _train_tabular(args, data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = getattr(args, "base_model")

    X_tr = torch.tensor(data["X_train"].values, dtype=torch.float32, device=device)
    y_tr = torch.tensor(data["y_train"], dtype=torch.float32, device=device)
    X_va = torch.tensor(data["X_val"].values, dtype=torch.float32, device=device)
    y_va = torch.tensor(data["y_val"], dtype=torch.float32, device=device)

    S_tr = data["S_train"]
    S_va = data["S_val"]


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
            return float(getattr(args, "lambda_fair", 0.0))


    def _get_lr(a) -> float:
        return float(1e-3)
    def _get_wd(a) -> float:
        return 1e-6
    def _get_epochs(a) -> int:
        return 200
    def _get_thr(a) -> float:
        return 0.5

    fair_weight = _get_fair_weight(args) 
    fair = get_fair_loss(args, data, fair_weight, device) if fair_weight > 0 else NoneFair()

    if isinstance(fair, DRFair) and getattr(fair, "Ctr", None) is not None and fair.Ctr.numel() > 0:
        with torch.no_grad():
            C_val = build_C_tensor(
                S_va, args, device=device,
                n_low=getattr(args, "n_low", None),
                n_low_frac=getattr(args, "n_low_frac", None),
                apriori_union=bool(getattr(args, "agg_apriori", True) or getattr(args, "apriori_union", False)),
                agg_max_len=int(getattr(args, "agg_max_len", 400)), 
                agg_max_cols=int(getattr(args, "agg_max_cols", 256)),
            )

    lr = _get_lr(args)
    wd = _get_wd(args)
    epochs = _get_epochs(args)
    opt_f = torch.optim.Adam(f.parameters(), lr=lr, weight_decay=wd)
    loss_bce = nn.BCEWithLogitsLoss()

    metric = "acc"
    validation_method = "all_select"
    
    print(f"[val] metric={metric}, strategy={validation_method}")
    
    patience = int(getattr(args, "early_patience", 10))
    mode = "min" if metric in ("cls_total","cls_bce","loss_bce","loss") else "max"
    best_score = float("inf") if mode=="min" else -float("inf")
    best_epoch = -1
    no_improve = 0
    best_f = None

    thr = _get_thr(args)

    warmup = int(getattr(args, "fair_warmup_epochs", 10))
    decay_start = int(0.85 * epochs)

    # training history
    history = {
        "epoch": [],          
        "cls": [],            # train BCE
        "pen": [],            # lambda
        "mp1": [],            
        "total": [],          # train total = cls + lam*pen
    }

    for ep in tqdm(range(epochs)):
        f.train()
        logits = f(X_tr)                                  
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
            total = cls + lam_t * pen                             
            print(f"[loss] cls={cls.item():.4f}, pen={pen.item():.4f}, "
                    f"λ={lam_t:.4f} → total={total.item():.4f}")

        else:
            total = cls
        
        if args.method :
            history["epoch"].append(int(ep))
            history["cls"].append(float(cls.detach().cpu().item()))
            if fair_weight > 0:
                history["pen"].append(float(pen.detach().cpu().item()))
                history["mp1"].append(float(float(marginal_kearns_order1_worst(probs.cpu().detach().numpy(), S_tr, thr=thr, min_support=1))))
                history["total"].append(float(total.detach().cpu().item()))
            else:
                history["pen"].append(float("nan"))
                history["mp1"].append(0.0)
                history["total"].append(float(total.detach().cpu().item()))

        opt_f.zero_grad()
        total.backward()
        opt_f.step()

        if fair_weight > 0:                              
            fair.after_f_step(f, X_tr)

        # validation
        f.eval()
        with torch.no_grad():
            logit_va = f(X_va)

            need_prob   = metric in ("acc", "accuracy", "cls_total") or (fair_weight > 0 and metric == "cls_total")
            need_loss   = metric in ("loss_bce", "loss", "bce", "cls_bce", "cls_total")
            need_acc    = metric in ("acc", "accuracy")
            N_va = y_va.shape[0]

            if need_prob:
                prob_va = torch.sigmoid(logit_va)

            if need_acc:
                pred_va = (prob_va >= thr).float()
                val_acc = (pred_va == y_va).float().mean().item()

            if need_loss:
                val_loss = loss_bce(logit_va, y_va).item()

            if metric == "cls_total":
                cls_bce_s_val = float(val_loss) 
                if fair_weight > 0:
                    pen_val = float(fair.penalty(logit_va, prob_va).detach().cpu().item())
                    lam_use = lam_t
                    cls_tot_s_val = cls_bce_s_val + lam_use * (pen_val / max(1, N_va))
                else:
                    lam_use = 0.0
                    cls_tot_s_val = cls_bce_s_val

            elif metric == "cls_bce":
                cls_bce_s_val = float(val_loss)

            # checkpoint
            if metric == "cls_total":
                cur = cls_tot_s_val
            elif metric == "cls_bce":
                cur = cls_bce_s_val
            elif metric in ("loss_bce", "loss", "bce"):
                cur = val_loss
            elif metric in ("acc", "accuracy"):
                cur = val_acc
            else:
                cls_bce_s_val = float(val_loss)
                if fair_weight > 0:
                    prob_va = torch.sigmoid(logit_va) if not need_prob else prob_va
                    pen_val = float(fair.penalty(logit_va, prob_va).detach().cpu().item())
                    lam_use = lam_t
                    cls_tot_s_val = cls_bce_s_val + lam_use * (pen_val / max(1, N_va))
                else:
                    cls_tot_s_val = cls_bce_s_val
                cur = cls_tot_s_val

        improved = (cur < best_score - 1e-8) if mode=="min" else (cur > best_score + 1e-8)
        if improved and (ep > 150):
            best_score = cur
            best_epoch = ep
            best_f = {k: v_.detach().cpu().clone() for k, v_ in f.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        
        if validation_method == "early_stop":
            if no_improve >= patience:
                print(f"[EarlyStop] metric={metric}, best@ep={best_epoch}, score={best_score:.6f}")
                break
        elif validation_method == "all_select":
            pass
        else:
            if no_improve >= patience:
                print(f"[EarlyStop:default] metric={metric}, best@ep={best_epoch}, score={best_score:.6f}")
                break
    
    if best_f is not None:
        f.load_state_dict(best_f, strict=True)
    print(f"[val] select best@ep={best_epoch} (metric={metric}, score={best_score:.6f}, mode={mode}, strategy={validation_method})")


    # Temperature & threshold
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
        test_fair_loss = 0
        if isinstance(fair, DRFair) and fair_weight > 0:                
            pen_te = fair.penalty_test(logit_te, data["X_test"])             
            N_te = float(data["X_test"].shape[0])
            test_fair_loss = float(pen_te.detach().cpu().item() / max(1.0, N_te))  

    yhat = (p_test >= thr).astype(int)
    return dict(proba=p_test, pred=yhat, test_dr=test_fair_loss, train_proba = probs) 

def run_method(args, data):
    if args.method == "dr":
        return _train_tabular(args, data)
    raise ValueError(args.method)