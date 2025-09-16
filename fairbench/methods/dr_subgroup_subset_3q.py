# fairbench/methods/dr.py
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from ..utils.threshold import tune_threshold
import logging, time
from tqdm.auto import tqdm
from fairbench.utils.logging_utils import Timer, mem_str
from itertools import product, combinations

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

def artanh_corr_old(yv, gz, eps=1e-6):
    y_c = yv - yv.mean()
    g_c = gz - gz.mean()
    y_std = torch.sqrt((y_c**2).mean() + eps); g_std = torch.sqrt((g_c**2).mean() + eps)
    corr = (y_c*g_c).mean()/(y_std*g_std + eps)
    corr_abs = torch.clamp(torch.abs(corr), 0.0, 1.0-1e-6)
    return torch.atanh(corr_abs)

def artanh_corr(yv, gz, eps=1e-6):
    print("new r^2")
    ## y_c = v^t*c, g_c = g(f(x,s))
    y_c = yv - yv.mean()
    g_c = gz - gz.mean()

    y_std = torch.sqrt((y_c**2).mean() + eps)
    # g_std = torch.sqrt((g_c**2).mean() + eps)

    rr = (y_c*g_c).mean()/(y_std + eps)
    corr_abs = torch.clamp(torch.abs(rr/2), 0.0, 1.0-1e-6)

    return torch.atanh(corr_abs)



def _build_V_3style_from_S_df(S_df, min_support=1):
    """
    3^q 스타일(일반화: 각 family에서 {ALL} ∪ {각 카테고리==1} ∪ (이진이면 {col==0}) 중 하나 선택)
    → 모든 family 선택을 AND로 결합 → 𝒱 생성.
    - 공집합(전부 ALL)은 제외
    - min_support 미만은 프루닝
    반환:
      V: bool mask 리스트
      choices: 각 V[k]에 대한 per-family 선택(spec) 리스트(사람이 읽을 수 있도록 기록)
    """
    n = len(S_df)
    if n == 0:
        return [], []

    # family 묶기
    fam = {}
    for c in S_df.columns:
        parts = str(c).split("_", 1)
        f = parts[0] if len(parts) > 1 else str(c)
        fam.setdefault(f, []).append(c)

    arrays = {c: S_df[c].values.astype(int) for c in S_df.columns}

    # family별 옵션 세트 구성
    # - 항상 ('ALL', None)
    # - 원핫 multi-cat: 각 ('col1', col)
    # - 이진(컬럼 1개): ('col1', col) + ('col0', col)
    fam_opts = []
    fam_names = []
    for f, cols in fam.items():
        opts = [("ALL", None)]
        for col in cols:
            opts.append(("col1", col))
        if len(cols) == 1:
            col = cols[0]
            opts.append(("col0", col))
        fam_opts.append(opts)
        fam_names.append(f)

    # 전체 조합 수 = ∏_family |opts|
    total = 1
    for opts in fam_opts:
        total *= len(opts)

    V, choices = [], []
    seen = set()
    with tqdm(total=total, desc=f"[DR-V] build 3^q-style", leave=False) as pbar:
        for pick in product(*fam_opts):
            pbar.update(1)
            # 전부 ALL(= 전체 데이터) 선택은 스킵
            if all(tag == "ALL" for tag, _ in pick):
                continue

            m = np.ones(n, dtype=bool)
            ok = True
            for (tag, col) in pick:
                if tag == "ALL":
                    continue
                colv = arrays[col]
                if tag == "col1":
                    m &= (colv == 1)
                else:  # 'col0'
                    m &= (colv == 0)
                if not m.any():
                    ok = False
                    break
            if not ok:
                continue

            ns = int(m.sum())
            if ns < int(min_support):
                continue

            key = m.tobytes()
            if key in seen:
                continue
            seen.add(key)

            # 선택 사양을 사람이 읽기 쉽게 기록
            readable = []
            for (fam_name, (tag, col)) in zip(fam_names, pick):
                if tag == "ALL":
                    readable.append(f"{fam_name}=ALL")
                elif tag == "col1":
                    readable.append(f"{fam_name}=={col}")
                else:
                    readable.append(f"{fam_name}==NOT({col})")  # binary complement
            V.append(m)
            choices.append(tuple(readable))

    return V, choices


def _build_V_and_C_from_S_df(S_df, device="cpu", n_low=None, n_low_frac=None):
    """
    S_df → 원자 → 모든 합집합 V (min_support 프루닝) → C (N x |V|)
    각 열은 c_{im} = 1[i ∈ G_m]. (양/음 최소지지 둘 다 만족하는 열만 사용)
    + V_stats: 각 G_m의 샘플 수 통계 저장
    """
    N = len(S_df)
    if (n_low_frac is not None) and (n_low_frac > 0):
        min_support = int(np.ceil(float(n_low_frac) * N))
    else:
        min_support = int(n_low or 0)


    print("min_suppoert:", min_support)

    V, choices = _build_V_3style_from_S_df(S_df, min_support=min_support)
    # import pdb; pdb.set_trace()
    log.info(f"[DR-V] 3^q-style |V|={len(V)} (min_support={min_support})")
    if len(V) == 0:
        return [], torch.zeros((N,0), device=device), []

    # 두 편(그룹/보완) 모두 최소지지 만족하는 열만 선택 + 통계 저장
    cols = []
    V_stats = []   # ★ 추가: 각 subgroup-subset별 샘플 수 기록
    kept = 0
    for m in V:
        ns = int(m.sum()); nsc = N - ns
        if ns >= min_support and nsc >= min_support:
            cols.append(torch.tensor(m.astype(np.float32), device=device).view(-1,1))
            V_stats.append(ns)
            kept += 1
    # import pdb; pdb.set_trace()
    if kept == 0:
        C = torch.zeros((N,0), device=device)
    else:
        C = torch.cat(cols, dim=1)

    log.info(f"[DR-V] C built with {C.shape[1]} columns (filtered by both-sides support)")
    log.info(f"[DR-V] stats example (first 3): {V_stats[:3] if len(V_stats)>0 else []}")
    return V, C, choices, V_stats



# ----- tabular 학습 -----
def train_tabular(args, data, device):
    X_tr = torch.tensor(data["X_train"].values, dtype=torch.float32, device=device)
    y_tr = torch.tensor(data["y_train"], dtype=torch.float32, device=device)
    X_va = torch.tensor(data["X_val"].values, dtype=torch.float32, device=device)
    y_va = torch.tensor(data["y_val"], dtype=torch.float32, device=device)
    S_tr = data["S_train"]; S_va = data["S_val"]

    n, d = X_tr.shape
    # f = MLP(d).to(device)
    f = Linear(d).to(device)
    g = Discriminator().to(device)

    n_low = getattr(args, "n_low", None)
    n_low_frac = getattr(args, "n_low_frac", None)
    print("========================================================")
    print("n_low:", n_low, ", n_low_frac:", n_low_frac)
    print("========================================================")

    # === NEW: subgroup-subset V와 C (N x |V|) 구성
    with torch.no_grad():
        V_tr, Ctr, choices, V_stats = _build_V_and_C_from_S_df(
            S_tr, device=device, n_low=n_low, n_low_frac=n_low_frac
        )

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
    adv_k = int(getattr(args, "adv_steps", 3))  # (옵션) adversarial step 조절

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
            yv = (Ctr @ v_unit).detach()               # [N]
            gz = g(logit.unsqueeze(1)).squeeze(-1)     # [N]
            dr = artanh_corr(yv, gz)                   # 스칼라
            total = cls + args.lambda_fair * dr
            opt_f.zero_grad(); total.backward(retain_graph=True); opt_f.step()

            # adversary k-step
            for _ in range(adv_k):
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

    return dict(proba=p_test, pred=yhat, V_stats = V_stats, choices = choices)


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

def run_dr_subgroup_subset_3q(args, data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if data["type"] == "tabular":
        print("=== DR-subgroup-subset-3q (tabular) ===")
        print("=== DR-subgroup-subset-3q (tabular) ===")
        print("=== DR-subgroup-subset-3q (tabular) ===")
        return train_tabular(args, data, device)
    elif data["type"] == "image":
        return _train_image(args, data, device)
    else:
        raise ValueError(data["type"])
