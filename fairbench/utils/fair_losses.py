# (맨 위 import들 근처에 추가)
import numpy as np
import pandas as pd
# fairbench/utils/fair_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 필요한 유틸/모듈 임포트 (프로젝트 경로에 맞춰 하나만 쓰면 됨)
try:
    from ..methods.dr_subgroup_subset_random import build_C_tensor  # 이미 쓰던 함수 경로
except Exception:
    from fairbench.utils.group_builder import build_C_tensor

try:
    from ..methods.dr_subgroup_subset_random import Discriminator, artanh_corr  # 예시: DR에서 쓰던 것
except Exception:
    # 기존에 Discriminator/ artanh_corr 가 있던 파일 경로로 바꿔주세요.
    from fairbench.methods.dr_utils import Discriminator, artanh_corr

import numpy as np, random, torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


# ---------- 공통 인터페이스 ----------
class BaseFair:
    """훈련 루프에서 호출할 페어니스 패널티 모듈의 공통 인터페이스."""
    def penalty(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """f 업데이트용 패널티 스칼라 텐서 리턴."""
        return torch.zeros((), device=logits.device)

    def after_f_step(self, f, X_tr: torch.Tensor) -> None:
        """필요하면 adversary, 보조변수 업데이트."""
        return

# === 아래 헬퍼 2개를 BaseFair 클래스 아래 아무 곳에 추가 ===
def _to_df(x):
    if isinstance(x, pd.DataFrame): return x.copy()
    if isinstance(x, pd.Series):    return x.to_frame()
    arr = np.asarray(x)
    if arr.ndim == 1: arr = arr[:, None]
    return pd.DataFrame(arr, columns=[f"s{i}" for i in range(arr.shape[1])])

def _build_marginal_groups(S):
    """
    S(DataFrame/ndarray) -> G in {0,1}^{N x K} (각 marginal의 one-hot/이진)
    - 이진은 0/1 모두 생성, 다값은 원-핫
    - 지원 0인 마진은 제거
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


# ---------- 아무 것도 안 하는(=unfair) 버전 ----------
class NoneFair(BaseFair):
    pass

# <-- NEW 1004
class DRFair(BaseFair):
    def __init__(self, args, data, device):
        self.device = device
        self.args = args

        S_tr = data["S_train"]
        n_low      = getattr(args, "n_low", None)
        n_low_frac = getattr(args, "n_low_frac", None)
        use_ap     = bool(getattr(args, "agg_apriori", True) or getattr(args, "apriori_union", False))
        agg_max_len  = int(getattr(args, "agg_max_len", 4))
        agg_max_cols = int(getattr(args, "agg_max_cols", 2048))

        with torch.no_grad():
            C = build_C_tensor(
                S_tr, args, device=device,
                n_low=n_low, n_low_frac=n_low_frac,
                apriori_union=use_ap, agg_max_len=agg_max_len, agg_max_cols=agg_max_cols
            )  # [N, M]

        # [NEW] 컬럼 정규화(평균0, L2=1) → 상관 안정화, 희소성 영향 완화
        if C.numel() > 0:
            C = C - C.mean(dim=0, keepdim=True)
            C = C / (C.norm(dim=0, keepdim=True) + 1e-12)
        self.Ctr = C

        # [NEW] 하이퍼
        self.gamma  = float(getattr(args, "fair_conf_gamma", 1.0))     # 0~2 추천
        self.margin = float(getattr(args, "fair_margin", 0.0))         # 0.0~0.03 추천
        self.adv_steps = int(getattr(args, "fair_adv_steps", 3))

        # v, g
        self.v = nn.Parameter(torch.randn(self.Ctr.shape[1], device=device))
        if self.Ctr.shape[1] > 0:
            self.v.data = self.v / (self.v.norm() + 1e-12)

        self.g = Discriminator().to(device)
        base_lr = float(getattr(args, "lr", 1e-3))
        self.opt_g = torch.optim.Adam(self.g.parameters(), lr=3.0 * base_lr)
        self.opt_v = torch.optim.Adam([self.v], lr=10.0 * base_lr)

    def _v_unit(self):
        return self.v / (self.v.norm() + 1e-12)

    def _conf_weight(self, logits: torch.Tensor) -> torch.Tensor:
        # w = |sigmoid(logit) - 0.5|^gamma
        p = torch.sigmoid(logits)
        w = (0.5 - (p - 0.5).abs()).clamp_min(0.0).pow(self.gamma)  # 경계↑, 확신↓

        # w = (p - 0.5).abs().pow(self.gamma).detach()
        return w + 1e-6  # 완전 0 방지

    def penalty(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        if self.Ctr.shape[1] == 0:
            return torch.zeros((), device=logits.device)

        with torch.no_grad():
            v_unit = self._v_unit()
            yv = (self.Ctr @ v_unit).detach()      # (N,)
        gz = self.g(logits.unsqueeze(1)).squeeze(-1)   # (N,)

        # [NEW] confidence 가중 상관 + margin hinge
        w = self._conf_weight(logits)
        corr = artanh_corr(w * yv, w * gz)
        pen = F.relu(corr.abs() - self.margin)
        return pen

    def after_f_step(self, f, X_tr: torch.Tensor) -> None:
        if self.Ctr.shape[1] == 0:
            return

        for _ in range(self.adv_steps):
            with torch.no_grad():
                logit_det = f(X_tr).detach()
            v_unit = self._v_unit()
            yv = (self.Ctr @ v_unit)
            gz = self.g(logit_det.unsqueeze(1)).squeeze(-1)
            # [NEW] 동일 가중으로 maximize
            w = self._conf_weight(logit_det)
            dr = artanh_corr(w * yv, w * gz)

            self.opt_g.zero_grad()
            (-dr).backward(retain_graph=True)
            self.opt_g.step()

            self.opt_v.zero_grad()
            (-dr).backward()
            self.opt_v.step()

        with torch.no_grad():
            self.v.data = self.v.data / (self.v.data.norm() + 1e-12)


# # ---------- DR(너가 쓰던 v, g, Ctr) 버전 ----------
# class DRFair(BaseFair):
#     def __init__(self, args, data, device):
#         self.device = device
#         self.args = args

#         # --- 데이터에서 그룹 텐서 만들기 ---
#         S_tr = data["S_train"]
#         n_low      = getattr(args, "n_low", None)
#         n_low_frac = getattr(args, "n_low_frac", None)
#         use_ap     = bool(getattr(args, "agg_apriori", True) or getattr(args, "apriori_union", False))
#         agg_max_len  = int(getattr(args, "agg_max_len", 4))
#         agg_max_cols = int(getattr(args, "agg_max_cols", 2048))

#         with torch.no_grad():
#             self.Ctr = build_C_tensor(
#                 S_tr, args, device=device,
#                 n_low=n_low, n_low_frac=n_low_frac,
#                 apriori_union=use_ap, agg_max_len=agg_max_len, agg_max_cols=agg_max_cols
#             )  # shape: [N, M]

#         # --- v, g, optimizer 셋업 ---
#         self.v = nn.Parameter(torch.randn(self.Ctr.shape[1], device=device))
#         if self.Ctr.shape[1] > 0:
#             self.v.data = self.v / (self.v.norm() + 1e-12)

#         self.g = Discriminator().to(device)
#         self.opt_g = torch.optim.Adam(self.g.parameters(), lr=getattr(args, "lr", 1e-3) * 3.0)
#         self.opt_v = torch.optim.Adam([self.v], lr=getattr(args, "lr", 1e-3) * 10.0)

#     def _v_unit(self):
#         return self.v / (self.v.norm() + 1e-12)

#     def penalty(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
#         """
#         f를 업데이트할 때 같이 미분되는 페널티.
#         logits: (N,), probs: sigmoid(logits) (N,)
#         """
#         if self.Ctr.shape[1] == 0:
#             return torch.zeros((), device=logits.device)

#         with torch.no_grad():
#             v_unit = self._v_unit()
#             yv = (self.Ctr @ v_unit).detach()                 # stop-grad for yv

#         gz = self.g(logits.unsqueeze(1)).squeeze(-1)          # (N,)
#         dr = artanh_corr(yv, gz)                              # 스칼라
#         return dr

#     def after_f_step(self, f, X_tr: torch.Tensor) -> None:
#         """
#         f 스텝 이후 adversary(g), v를 maximize 쪽으로 몇 스텝 업데이트.
#         """
#         if self.Ctr.shape[1] == 0:
#             return

#         for _ in range(3):
#             with torch.no_grad():
#                 logit_det = f(X_tr).detach()
#             v_unit = self._v_unit()
#             yv = (self.Ctr @ v_unit)
#             gz = self.g(logit_det.unsqueeze(1)).squeeze(-1)
#             dr = artanh_corr(yv, gz)                          # maximize


#             self.opt_g.zero_grad()
#             (-dr).backward(retain_graph=True)
#             self.opt_g.step()

#             self.opt_v.zero_grad()
#             (-dr).backward()
#             self.opt_v.step()

#         with torch.no_grad():
#             self.v.data = self.v.data / (self.v.data.norm() + 1e-12)

# === DRFair 아래에 추가: Marginal CE(=Reg) 패널티 ===
class RegFair(BaseFair):
    """
    sum_g | P(hatY=1 | g) - P(hatY=1) |  (마진별 편차의 합)
    - adversary 없음 → after_f_step() 불필요
    - lam 스케일은 트레이너에서 곱함
    """
    def __init__(self, args, data, device, min_support=0.5):
        self.device = device
        self.min_support = float(min_support)
        G_np, _ = _build_marginal_groups(data["S_train"])
        self.G = torch.tensor(G_np, device=device)  # (N, K)

    def penalty(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        if self.G.numel() == 0:
            return torch.zeros((), device=probs.device)
        base_pos = probs.mean()               # P(hatY=1)
        supp = self.G.sum(dim=0)              # (K,)
        valid = supp > self.min_support
        if not torch.any(valid):
            return torch.zeros((), device=probs.device)
        Gv = self.G[:, valid]                 # (N, Kv)
        suppv = supp[valid]                   # (Kv,)
        grp = (Gv.T @ probs) / suppv          # P(hatY=1 | g)
        return torch.abs(grp - base_pos).sum()
    
    
# === 여기부터 추가: SP-GD(=gerryfair) 패널티 ===
class SPFair(BaseFair):
    """
    Soft Demographic Parity (SP-GD) 스타일 페어니스 패널티.
    penalty()는  softplus_beta( α * |SP_g - SP| ) / beta  를 반환.
    after_f_step()에서 adversary g(S)를 (penalty - barrier) 최대화 방향으로 1~여러 스텝 업데이트.
    """
    def __init__(self, args, data, device):
        self.device = device

        # hyperparams (gerryfair 계열 네이밍 유지)
        self.lr_g     = float(getattr(args, "gf_lr_group", 1e-2))
        self.temp_g   = float(getattr(args, "gf_temp_group", 50.0))
        self.beta_sp  = float(getattr(args, "gf_softplus_temp", 200.0))  # 큰 값 권장
        self.min_prop = float(getattr(args, "gf_min_group_prop", 0.01))

        # S_train 텐서 저장
        S_tr = data["S_train"]
        if hasattr(S_tr, "values"):
            S_tr = S_tr.values
        S_tr = torch.tensor(np.asarray(S_tr, dtype=np.float32), device=device)
        self.S_tr = S_tr
        self.d_s = S_tr.shape[1] if S_tr.ndim == 2 else 0

        # adversary g: 선형 + steep sigmoid
        self.g = nn.Linear(self.d_s, 1, bias=True).to(device)
        self.opt_g = torch.optim.Adam(self.g.parameters(), lr=self.lr_g)
        self.softplus_sp = nn.Softplus(beta=self.beta_sp)

    # 내부 유틸: g(S) in [0,1]
    def _gprob(self, S):
        z = self.g(S).squeeze(-1)
        return torch.sigmoid(self.temp_g * z).clamp(0.0, 1.0)

    # 내부 유틸: α, SP, SP_g, α|SP_g - SP|
    @staticmethod
    def _fair_terms(p, gprob, eps=1e-8):
        alpha   = gprob.mean()
        sp_base = p.mean()
        denom   = gprob.sum().clamp_min(eps)
        sp_grp  = (p * gprob).sum() / denom
        wdisp   = alpha * torch.abs(sp_grp - sp_base)
        return alpha, sp_base, sp_grp, wdisp

    def penalty(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        f 업데이트에 들어갈 스칼라 패널티(softplus/β 포함)를 반환.
        - g는 고정(미분 안 보냄)으로 취급하기 위해 gprob는 detach() 하지 않지만,
          after_f_step에서 g를 따로 최적화하므로 여기선 g에 gradient가 필요 없음.
        """
        if self.d_s == 0:
            return torch.zeros((), device=logits.device)

        with torch.no_grad():
            gprob = self._gprob(self.S_tr)                 # (N,)
        _, _, _, wdisp = self._fair_terms(probs, gprob)    # (스칼라)
        pen = self.softplus_sp(wdisp) / self.beta_sp
        return pen

    def after_f_step(self, f, X_tr: torch.Tensor) -> None:
        """
        f 한 스텝 후, g를 (penalty - size barrier) 최대화하도록 1~여러 스텝 업데이트.
        """
        if self.d_s == 0:
            return

        # p는 고정, g만 미분
        with torch.no_grad():
            logits_det = f(X_tr).detach()
            p_det = torch.sigmoid(logits_det).clamp(0.0, 1.0)

        # g 업데이트
        self.opt_g.zero_grad()
        gprob = self._gprob(self.S_tr)                     # grad to g
        alpha, _, _, wdisp_g = self._fair_terms(p_det, gprob)
        pen_g   = self.softplus_sp(wdisp_g) / self.beta_sp
        # 그룹 크기 장벽(min_prop 이하/이상 방지)
        barrier = nn.Softplus(beta=self.beta_sp)(self.min_prop - alpha) \
                + nn.Softplus(beta=self.beta_sp)(self.min_prop - (1.0 - alpha))
        # g는 (pen_g - barrier) 최대화 ⇒ 음수 부호로 최소화
        loss_g = -(pen_g - barrier)
        loss_g.backward()
        self.opt_g.step()



def get_fair_loss(args, data, fair_weight, device) -> BaseFair:
    
    # lam = float(getattr(args, "lambda_fair", getattr(args, "mf_lambda", 0.0)))
    if fair_weight <= 0:
        print("fair loss: None fair")
        return NoneFair()

    key = getattr(args, "method", "unfair")
    key = str(key).lower()

    if key.startswith("dr"):
        print("fair loss: dr fair")
        return DRFair(args, data, device)
    if key.startswith("gerry") or key.startswith("sp"):
        print("fair loss: gerry fair")
        return SPFair(args, data, device)   # ★ 추가
    if key.startswith("reg") or key.startswith("marginal"):
        print("fair loss: marginal fair")
        return RegFair(args, data, device)

    return NoneFair()

