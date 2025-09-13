# fairbench/methods/marginal_ce.py
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
from ..utils.mlp import MLP, Linear

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
#         self.fc = nn.Linear(d, 1)
#     def forward(self, x):  # (N,)
#         return self.fc(x).squeeze(-1)

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

# ---------------- helpers ----------------
def _to_tensor(X, device):
    if hasattr(X, "values"):  # pandas
        X = X.values
    return torch.tensor(np.asarray(X, dtype=np.float32), device=device)

def _ensure_val(data):
    """val이 없으면 train을 재사용"""
    if data.get("X_val") is None:
        data["X_val"] = data["X_train"]
        data["y_val"] = data["y_train"]
        data["S_val"] = data["S_train"]
    return data

def _to_df(x):
    if isinstance(x, pd.DataFrame): return x.copy()
    if isinstance(x, pd.Series):    return x.to_frame()
    arr = np.asarray(x)
    if arr.ndim == 1: arr = arr[:, None]
    return pd.DataFrame(arr, columns=[f"s{i}" for i in range(arr.shape[1])])

def _build_marginal_groups(S):
    """
    S(DataFrame) -> G in {0,1}^{N x K}, group_names(list)
    - 이진 열은 0/1 두 값 모두 컬럼 생성
    - 범주형은 카테고리별 원핫
    - 지원도 0인 마진은 제거
    """
    Sdf = _to_df(S)
    cols, names = [], []
    for c in Sdf.columns:
        col = Sdf[c]
        if col.dtype == bool:
            cat = pd.Categorical(col.astype(int), categories=[0, 1])
            dmy = pd.get_dummies(cat, prefix=c, drop_first=False)
        elif pd.api.types.is_numeric_dtype(col):
            uniq = set(pd.unique(col.dropna()))
            if uniq.issubset({0, 1, 0.0, 1.0}):
                cat = pd.Categorical(col.fillna(0).astype(int), categories=[0, 1])
                dmy = pd.get_dummies(cat, prefix=c, drop_first=False)
            else:
                dmy = pd.get_dummies(col.astype("category"), prefix=c, drop_first=False)
        else:
            dmy = pd.get_dummies(col.astype("category"), prefix=c, drop_first=False)
        cols.append(dmy.values.astype(np.float32))
        names.extend(dmy.columns.tolist())

    if not cols:
        return np.zeros((len(Sdf), 0), dtype=np.float32), []

    G = np.concatenate(cols, axis=1)  # (N, K_raw)
    supp = G.sum(axis=0)
    keep = supp > 0.5                 # 지원 0인 마진 제거
    G = G[:, keep]
    names = [names[i] for i in np.where(keep)[0]]
    return G.astype(np.float32, copy=False), names

# ---------------- core trainer (tabular only) ----------------
def run_reg(args, data):
    """
    Train classifier by:
        Loss = BCE + lam * sum_g | P(hatY=1 | g) - P(hatY=1) |
    where g runs over *marginal groups* (male/female, black/white, old/young, ...).

    Returns: dict(proba, pred)
    """
    assert data.get("type") == "tabular", "marginal_ce: tabular only."
    data = _ensure_val(data)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- tensors -----
    X_tr = _to_tensor(data["X_train"], device)
    y_tr = torch.tensor(np.asarray(data["y_train"], dtype=np.float32), device=device)
    X_va = _to_tensor(data["X_val"],   device)
    y_va = torch.tensor(np.asarray(data["y_val"],   dtype=np.float32), device=device)
    X_te = _to_tensor(data["X_test"],  device)

    # ----- marginal groups from S_train -----
    G_np, group_names = _build_marginal_groups(data["S_train"])
    G = torch.tensor(G_np, device=device)  # (N, K)
    K = G.shape[1]

    N, d_x = X_tr.shape

    # ----- hyperparams -----
    lr        = float(getattr(args, "mf_lr", 1e-3))
    wd        = float(getattr(args, "mf_wd", 1e-6))
    epochs    = int(getattr(args, "mf_epochs", 200))
    lam       = float(getattr(args, "mf_lambda", 0.0))    # fairness weight
    thr       = float(getattr(args, "thr", 0.5))
    seed      = int(getattr(args, "seed", 42))
    base      = str(getattr(args, "mf_base", "mlp")).lower()  # "linear" | "mlp"

    # MLP 옵션
    # hidden_str = str(getattr(args, "mf_hidden", "128,64"))
    # hidden = tuple(int(h) for h in hidden_str.split(",") if str(h).strip() != "")
    # act      = str(getattr(args, "mf_act", "relu"))
    # dropout  = float(getattr(args, "mf_dropout", 0.0))
    # bn       = bool(getattr(args, "mf_bn", False))
    hidden = [100]
    act = 'relu'
    dropout = 0.0
    bn = False

    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ----- model/opt -----
    if base == "mlp":
        # model = MLP(d_x, hidden=hidden, act=act, dropout=dropout, bn=bn).to(device)
        model = MLP(d_x).to(device)
        print("d_x: ", d_x)
        
    else:
        base = "linear"
        model = Linear(d_x).to(device)
    
    # import pdb;pdb.set_trace()

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss()

    # opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # bce = nn.BCEWithLogitsLoss()

    def _fairness_from_p(p: torch.Tensor) -> torch.Tensor:
        """
        p: (N,) sigmoid(logits)
        returns sum_g | P(hatY=1 | g) - P(hatY=1) |
        """
        if K == 0:
            return torch.zeros((), device=device)
        base_pos = p.mean()                   # P(hatY=1)
        supp = G.sum(dim=0)                   # (K,)
        valid = supp > 0.5
        if not torch.any(valid):
            return torch.zeros((), device=device)
        Gv = G[:, valid]                      # (N, Kv)
        suppv = supp[valid]                   # (Kv,)
        grp = (Gv.T @ p) / suppv              # P(hatY=1 | g)
        return torch.abs(grp - base_pos).sum()

    # ----- train with val selection (ACC 우선, tie→BCE) -----
    best = {"acc": -1.0, "bce": float("inf"), "state": None}
    for _ in tqdm(range(epochs)):
        model.train()
        logits = model(X_tr)                  # (N,)
        p = torch.sigmoid(logits).clamp(0.0, 1.0)
        # loss = bce(logits, y_tr) + lam * _fairness_from_p(p)
        loss = bce(logits, y_tr) if lam == 0.0 else bce(logits, y_tr) + lam * _fairness_from_p(p)


        opt.zero_grad()
        loss.backward()
        opt.step()

        # validate
        model.eval()
        with torch.no_grad():
            logit_va = model(X_va)
            p_va = torch.sigmoid(logit_va)
            pred_va = (p_va >= thr).float()
            acc_va = (pred_va == y_va).float().mean().item()
            bce_va = bce(logit_va, y_va).item()
        if (acc_va > best["acc"]) or (np.isclose(acc_va, best["acc"]) and bce_va < best["bce"]):
            best["acc"], best["bce"] = acc_va, bce_va
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best["state"] is not None:
        model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})

    # ----- test -----
    model.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(model(X_te)).cpu().numpy().reshape(-1)
    yhat = (p_te >= thr).astype(int)
    return dict(proba=p_te, pred=yhat)