# fairbench/methods/sp_gd.py
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from ..utils.mlp import MLP

import numpy as np, random, torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# # ---------------- Models ----------------
# class Linear(nn.Module):
#     """logit = w^T x + b"""
#     def __init__(self, d):
#         super().__init__()
#         self.net = nn.Linear(d, 1)
#     def forward(self, x):  # returns logits
#         return self.net(x).squeeze(-1)

# def _act(name: str):
#     name = str(name).lower()
#     return {
#         "relu": nn.ReLU(),
#         "gelu": nn.GELU(),
#         "tanh": nn.Tanh(),
#         "leaky_relu": nn.LeakyReLU(0.1),
#         "elu": nn.ELU(),
#         "silu": nn.SiLU(),
#     }.get(name, nn.ReLU())

# class MLP(nn.Module):
#     """logit = MLP(x) with configurable hidden layers"""
#     def __init__(self, d_in, hidden=(128, 64), act="relu", dropout=0.0, bn=False):
#         super().__init__()
#         layers = []
#         prev = d_in
#         for h in hidden:
#             layers.append(nn.Linear(prev, h))
#             if bn:
#                 layers.append(nn.BatchNorm1d(h))
#             layers.append(_act(act))
#             if dropout and dropout > 0:
#                 layers.append(nn.Dropout(dropout))
#             prev = h
#         layers.append(nn.Linear(prev, 1))
#         self.net = nn.Sequential(*layers)
#     def forward(self, x):  # (N,)
#         return self.net(x).squeeze(-1)

class LinGroup(nn.Module):
    """g(x_s) = sigmoid( temp * (u^T x_s + c) )  -- differentiable subgroup"""
    def __init__(self, d_s, temp=50.0):
        super().__init__()
        self.lin = nn.Linear(d_s, 1, bias=True)
        self.temp = float(temp)
    def forward(self, S):
        return torch.sigmoid(self.temp * self.lin(S).squeeze(-1))  # in [0,1]

# ------------- helpers -------------
def _to_tensor(X, device):
    if hasattr(X, "values"):
        X = X.values
    return torch.tensor(np.asarray(X, dtype=np.float32), device=device)

def _ensure_val(data):
    # 없으면 train을 val로 재사용
    if data.get("X_val") is None:
        data["X_val"] = data["X_train"]
        data["y_val"] = data["y_train"]
        data["S_val"] = data["S_train"]
    return data

def _fairness_terms(p, gprob, eps=1e-8):
    """
    p: (N,) acceptance probs of classifier on TRAIN
    gprob: (N,) soft subgroup membership in [0,1]
    returns alpha, sp_base, sp_group, weighted_disparity
    """
    alpha = gprob.mean()
    sp_base = p.mean()
    denom = gprob.sum().clamp_min(eps)
    sp_grp = (p * gprob).sum() / denom
    wdisp = alpha * torch.abs(sp_grp - sp_base)
    return alpha, sp_base, sp_grp, wdisp

def _size_barrier(alpha, min_prop, sharp=200.0):
    """
    soft barrier for min_prop ≤ alpha ≤ 1 - min_prop
    (온도(sharp) 큼 → 급격한 장벽)
    """
    sp = nn.Softplus(beta=sharp)
    return sp(min_prop - alpha) + sp(min_prop - (1.0 - alpha))

# ------------- core trainer (tabular) -------------
def _train_sp_tabular(args, data, device):
    """
    Min_f  BCE(f) + lam * softplus_beta( max_g α|SP_g - SP| ) / beta
      - gamma는 완전히 무시 (조기 종료/힌지 없음)
      - lam=0이면 정확히 BCE만 학습 (그룹/패널티/업데이트 모두 스킵)
      - ascent 루프/재시작 없음: 매 epoch에 adversary를 -penalty로 1스텝만 업데이트
      - best epoch: validation ACC 우선, 동률 시 BCE 낮은 쪽 (dr.py와 동일)
    """
    data = _ensure_val(data)

    X_tr = _to_tensor(data["X_train"], device)
    y_tr = torch.tensor(np.asarray(data["y_train"], dtype=np.float32), device=device)
    X_va = _to_tensor(data["X_val"], device)
    y_va = torch.tensor(np.asarray(data["y_val"], dtype=np.float32), device=device)
    S_tr = _to_tensor(data["S_train"], device)
    S_va = _to_tensor(data["S_val"], device)
    X_te = _to_tensor(data["X_test"], device)

    N, d_x = X_tr.shape
    d_s = S_tr.shape[1]

    # ----- hyperparams -----
    lr_f       = float(getattr(args, "gf_lr_model", 1e-3))
    lr_g       = float(getattr(args, "gf_lr_group", 1e-2))
    epochs     = int(getattr(args, "gf_max_iters", 200))
    lam        = float(getattr(args, "gf_C", 10.0))            # penalty weight (공정성 강도)
    # gamma는 사용하지 않음
    min_prop   = float(getattr(args, "gf_min_group_prop", 0.01))
    thr        = float(getattr(args, "thr", 0.5))
    temp_g     = float(getattr(args, "gf_temp_group", 50.0))   # group sigmoid temp (steep)
    beta_sp    = float(getattr(args, "gf_softplus_temp", 200.0)) # softplus temperature (매우 큼)

    # ----- classifier -----
    # f = Linear(d_x).to(device)
    print("d_x: ", d_x)
    f = MLP(d_x).to(device)
    opt_f = optim.Adam(f.parameters(), lr=lr_f, weight_decay=1e-6)
    bce = nn.BCEWithLogitsLoss()

    # ----- adversary(group): lam==0이면 아예 안 씀 -----
    use_fair = (lam > 0.0) and (d_s > 0)
    if use_fair:
        g = LinGroup(d_s, temp=temp_g).to(device)
        opt_g = optim.Adam(g.parameters(), lr=lr_g)
        softplus_sp = nn.Softplus(beta=beta_sp)

    best_acc = -1.0
    best_bce = float("inf")
    best_f = None

    for ep in tqdm(range(1, epochs + 1)):
        f.train()
        logits = f(X_tr)                          # (N,)
        p = torch.sigmoid(logits).clamp(0.0, 1.0) # acceptance probs

        # ---- adversary 1-step (if lam>0) ----
        if use_fair:
            opt_g.zero_grad()
            gprob = g(S_tr).clamp(0.0, 1.0)
            alpha, _, _, wdisp_g = _fairness_terms(p.detach(), gprob)   # p 고정
            # 패널티: softplus_beta(wdisp) / beta  (max-g 근사 강하게)
            pen_g = softplus_sp(wdisp_g) / beta_sp
            # 그룹 크기 장벽
            barrier = _size_barrier(alpha, min_prop, sharp=beta_sp)
            # adversary는 (penalty - barrier)를 최대화 ⇒ -(... )를 최소화
            loss_g = -(pen_g - barrier)
            loss_g.backward()
            opt_g.step()

        # ---- classifier update ----
        opt_f.zero_grad()
        cls = bce(logits, y_tr)
        if use_fair:
            gprob = g(S_tr).clamp(0.0, 1.0).detach()           # g 고정
            _, _, _, wdisp = _fairness_terms(p, gprob)         # f만 미분
            pen = softplus_sp(wdisp) / beta_sp
            loss = cls + lam * pen
        else:
            loss = cls
        loss.backward()
        opt_f.step()

        # ---- validation (ACC 우선, tie는 BCE) ----
        f.eval()
        with torch.no_grad():
            logit_va = f(X_va)
            p_va = torch.sigmoid(logit_va)
            pred_va = (p_va >= thr).float()
            acc_va = (pred_va == y_va).float().mean().item()
            bce_va = bce(logit_va, y_va).item()
        if (acc_va > best_acc) or (np.isclose(acc_va, best_acc) and bce_va < best_bce):
            best_acc = acc_va
            best_bce = bce_va
            best_f = {k: v.detach().cpu().clone() for k, v in f.state_dict().items()}

    # ---- restore best f ----
    if best_f is not None:
        f.load_state_dict({k: v.to(device) for k, v in best_f.items()})

    # ---- test ----
    f.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(f(X_te)).cpu().numpy().reshape(-1)
    yhat = (p_te >= thr).astype(int)
    return dict(proba=p_te, pred=yhat)

# ------------- (optional) vision stub 동일 형식 -------------
def _train_sp_image(args, data, device):
    raise NotImplementedError("SP-GD is implemented for tabular data only.")

# ------------- public API -------------
def run_gerryfair(args, data):
    """
    SP-ERM (식 2,3의 SP 버전) 경사하강 래퍼.
    - model: Linear
    - gamma는 완전히 무시. 공정성은 lam(=gf_C)로만 조절.
    - lam=0이면 정확히 제약 없는 BCE 학습.
    - best epoch: validation ACC 우선, 동률 시 BCE 낮은 쪽 (dr.py와 동일)
    Returns dict(proba, pred)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if data["type"] == "tabular":
        return _train_sp_tabular(args, data, device)
    elif data["type"] == "image":
        return _train_sp_image(args, data, device)
    else:
        raise ValueError(f"Unsupported data type: {data['type']}")
