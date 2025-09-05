# fairbench/methods/dr.py
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from ..utils.threshold import tune_threshold
import logging, time
from tqdm.auto import tqdm
from fairbench.utils.logging_utils import Timer, mem_str

log = logging.getLogger("fair")

# ----- 모델들 -----
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)   # logits
        )
    def forward(self, x): return self.net(x).squeeze(-1)

class Linear(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Linear(d, 1)
    def forward(self, x): return self.net(x).squeeze(-1)

class SmallConvNet(nn.Module):
    """CelebA용 경량 CNN (SequentialFairness류의 단순 ConvNet 기조)"""
    def __init__(self, n_classes=1):
        super().__init__()
        ch=[32,64,128,128]
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
        self.features = nn.Sequential(
            block(3, ch[0]),
            block(ch[0], ch[1]),
            block(ch[1], ch[2]),
            block(ch[2], ch[3]),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch[3], n_classes))
    def forward(self, x): return self.head(self.features(x)).squeeze(-1)

# ----- DR 서브루틴 -----
class Discriminator(nn.Module):
    """g(z): scalar"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU(), nn.Linear(16,1))
    def forward(self, z): return self.net(z)

def artanh_corr(yv, gz, eps=1e-6):
    y_c = yv - yv.mean()
    g_c = gz - gz.mean()
    y_std = torch.sqrt((y_c**2).mean() + eps); g_std = torch.sqrt((g_c**2).mean() + eps)
    corr = (y_c*g_c).mean()/(y_std*g_std + eps)
    corr_abs = torch.clamp(torch.abs(corr), 0.0, 1.0-1e-6)
    return torch.atanh(corr_abs)

def build_C_tensor(S_df, device="cpu", n_low=None, n_low_frac=None):
    """
    각 이진 민감 컬럼에 대해, 양/음 클래스가 모두 최소지지 이상인 것들만 선택.
    - n_low_frac가 주어지면 ceil(frac * N)을 임계로 사용 (우선순위 높음)
    - 그렇지 않으면 n_low(개수) 사용
    """
    N = len(S_df)
    if N == 0 or S_df.shape[1] == 0:
        return torch.zeros((N, 0), device=device)

    if (n_low_frac is not None) and (n_low_frac > 0):
        min_support = int(np.ceil(float(n_low_frac) * N))
    else:
        min_support = int(n_low or 0)

    cols = list(S_df.columns)
    mats = []
    for c in cols:
        v = torch.tensor(S_df[c].values.astype(np.int32), dtype=torch.float32, device=device)
        pos = int(v.sum().item()); neg = int(N - pos)
        if pos >= min_support and neg >= min_support:
            mats.append(v.view(-1, 1))
    if len(mats) == 0:
        return torch.zeros((N, 0), device=device)
    return torch.cat(mats, dim=1)

def _train_tabular(args, data, device):
    X_tr = torch.tensor(data["X_train"].values, dtype=torch.float32, device=device)
    y_tr = torch.tensor(data["y_train"], dtype=torch.float32, device=device)
    X_va = torch.tensor(data["X_val"].values, dtype=torch.float32, device=device)
    y_va = torch.tensor(data["y_val"], dtype=torch.float32, device=device)
    S_tr = data["S_train"]; S_va = data["S_val"]

    n, d = X_tr.shape
    # f = MLP(d).to(device)
    f = Linear(d).to(device)
    g = Discriminator().to(device)

    # 기존: n_low = getattr(args, "n_low", 100)
    n_low = getattr(args, "n_low", None)
    n_low_frac = getattr(args, "n_low_frac", None)  # NEW

    with torch.no_grad():
        Ctr = build_C_tensor(S_tr, device=device, n_low=n_low, n_low_frac=n_low_frac)
        v = nn.Parameter(torch.randn(Ctr.shape[1], device=device))
        v.data = v / (v.norm() + 1e-12)

    opt_f = optim.Adam(f.parameters(), lr=args.lr, weight_decay=1e-7)
    opt_g = optim.Adam(g.parameters(), lr=args.lr*3)
    opt_v = optim.Adam([v], lr=args.lr*10)

    loss_bce = nn.BCEWithLogitsLoss()

    best_val = float("inf")   # tie-breaker 용
    best_acc = -1.0           # ★ ACC 기준 선택
    best_f = None             # f의 베스트 가중치만 저장(테스트엔 f만 필요)
    thr = float(getattr(args, "thr", 0.5))  # 고정 임계값

    for ep in tqdm(range(args.epochs)):
        # ===== train step =====
        f.train(); g.train()

        logit = f(X_tr)                 # [N]
        cls = loss_bce(logit, y_tr)     # BCE(logits, targets)

        if args.lambda_fair == 0.0 or Ctr.shape[1] == 0:
            total = cls
            opt_f.zero_grad(); total.backward(); opt_f.step()
        else:
            with torch.no_grad():
                v_unit = v / (v.norm() + 1e-12)
            yv = (Ctr @ v_unit).detach()                # [N]
            gz = g(logit.unsqueeze(1)).squeeze(-1)      # [N]
            dr = artanh_corr(yv, gz)                    # 스칼라
            total = cls + args.lambda_fair * dr
            opt_f.zero_grad(); total.backward(retain_graph=True); opt_f.step()

            # adversary k-step
            for _ in range(3):
                logit_det = f(X_tr).detach()
                v_unit = v / (v.norm() + 1e-12)
                yv = (Ctr @ v_unit)
                gz = g(logit_det.unsqueeze(1)).squeeze(-1)
                dr = artanh_corr(yv, gz)
                opt_g.zero_grad(); (-dr).backward(retain_graph=True); opt_g.step()
                opt_v.zero_grad(); (-dr).backward(); opt_v.step()
                with torch.no_grad():
                    v.data = v.data / (v.data.norm() + 1e-12)

        # ===== validation step (★ ACC로 선택, tie는 BCE로) =====
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

    # ===== best model 복원 & 테스트 =====
    if best_f is not None:
        f.load_state_dict({k: v_.to(device) for k, v_ in best_f.items()})

    f.eval()
    with torch.no_grad():
        p_test = torch.sigmoid(f(torch.tensor(
            data["X_test"].values, dtype=torch.float32, device=device
        ))).cpu().numpy()
    yhat = (p_test >= thr).astype(int)

    return dict(proba=p_test, pred=yhat)


def _train_image(args, data, device):
    f = SmallConvNet().to(device)
    g = Discriminator().to(device)
    opt_f = optim.Adam(f.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_g = optim.Adam(g.parameters(), lr=args.lr*3)
    loss_bce = nn.BCEWithLogitsLoss()

    thr = float(getattr(args, "thr", 0.5))  # 고정 임계값
    best_val = float("inf")
    best_f = None

    def batch_C(S_list, n_low_frac=None, n_low=None):
        keys = list(data["meta"]["sens_list"])
        N = len(S_list)
        if (n_low_frac is not None) and (n_low_frac > 0):
            min_support = int(np.ceil(float(n_low_frac) * N))
        else:
            min_support = int(n_low or 0)
        Ms = []
        for k in keys:
            m = torch.tensor([s[k] for s in S_list], dtype=torch.float32, device=device).view(-1,1)
            cnt = int(m.sum().item()); neg = N - cnt
            if cnt >= min_support and neg >= min_support:
                Ms.append(m)
        return torch.cat(Ms, dim=1) if len(Ms) > 0 else torch.zeros((N,0), device=device)


    for ep in range(args.epochs):
        # ===== train =====
        f.train(); g.train()
        for x,y,S in data["train_loader"]:
            x=x.to(device); y=y.float().to(device)
            logit = f(x)
            cls = loss_bce(logit, y)

            if args.lambda_fair == 0.0:
                opt_f.zero_grad(); cls.backward(); opt_f.step(); continue

            # Cb = batch_C(S, n_low=16)
            Cb = batch_C(S, n_low_frac=getattr(args, "n_low_frac", None), n_low=getattr(args, "n_low", None))
            if Cb.shape[1] == 0:
                opt_f.zero_grad(); cls.backward(); opt_f.step(); continue

            with torch.no_grad():
                v_ = torch.randn(Cb.shape[1], device=device)
                v_ = v_ / (v_.norm() + 1e-12)

            yv = (Cb @ v_)                        # [B]
            gz = g(logit.unsqueeze(1)).squeeze(-1)
            dr = artanh_corr(yv, gz)

            total = cls + args.lambda_fair * dr
            opt_f.zero_grad(); total.backward(retain_graph=True); opt_f.step()
            opt_g.zero_grad(); (-dr).backward(); opt_g.step()

        # ===== validation (BCE만으로 선택) =====
        f.eval()
        val_losses = []
        with torch.no_grad():
            for x,y,S in data["val_loader"]:
                x=x.to(device); y=y.float().to(device)
                logit = f(x)
                val_losses.append(loss_bce(logit, y).item())
        val_loss = float(np.mean(val_losses)) if len(val_losses)>0 else float("inf")

        if val_loss < best_val:
            best_val = val_loss
            best_f = {k: v_.detach().cpu().clone() for k, v_ in f.state_dict().items()}

    # ===== best model 복원 & 테스트 =====
    if best_f is not None:
        f.load_state_dict({k: v_.to(device) for k, v_ in best_f.items()})

    f.eval()
    pt=[]
    with torch.no_grad():
        for x,y,S in data["test_loader"]:
            p = torch.sigmoid(f(x.to(device))).cpu().numpy()
            pt.append(p)
    pt = np.concatenate(pt)
    yhat = (pt >= thr).astype(int)
    return dict(proba=pt, pred=yhat)

def run_dr(args, data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if data["type"] == "tabular":
        return _train_tabular(args, data, device)
    elif data["type"] == "image":
        return _train_image(args, data, device)
    else:
        raise ValueError(data["type"])
