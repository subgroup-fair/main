# (맨 위 import들 근처에 추가)
import numpy as np
import pandas as pd
# fairbench/utils/fair_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 필요한 유틸/모듈 임포트 (프로젝트 경로에 맞춰 하나만 쓰면 됨)
from ..methods.dr_subgroup_subset_random import build_C_tensor  # 이미 쓰던 함수 경로
from ..methods.dr_subgroup_subset_random import Discriminator, artanh_corr  # 예시: DR에서 쓰던 것


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
        self._v_prev = None  # ← 이전 v 저장해 드리프트(코사인) 확인용
        self.device = device
        self.args = args

        S_tr = data["S_train"]
        # (A) n_low_frac 기본값 적용: args에 없으면 1% 사용
        n_low_frac_eff = getattr(args, "n_low_frac", None)
        if n_low_frac_eff is None:
            n_low_frac_eff = float(getattr(args, "n_low_frac_default", 0.01))  # 1% 기본
        n_low      = getattr(args, "n_low", None)
        use_ap     = bool(getattr(args, "agg_apriori", True) or getattr(args, "apriori_union", False))
        agg_max_len  = int(getattr(args, "agg_max_len", 4))
        agg_max_cols = int(getattr(args, "agg_max_cols", 2048))
        

        with torch.no_grad():
            # (B) 마지널 항상 포함 옵션 전달
            include_marg = bool(getattr(args, "include_marginals_always", False))
            C = build_C_tensor(
                S_tr, args, device=device,
                n_low=n_low, n_low_frac=n_low_frac_eff,
                apriori_union=use_ap, agg_max_len=agg_max_len, agg_max_cols=agg_max_cols
            )  # shape: [N, M]


        # # [NEW] 컬럼 정규화(평균0, L2=1) → 상관 안정화, 희소성 영향 완화
        # if C.numel() > 0:
        #     C = C - C.mean(dim=0, keepdim=True)
        #     C = C / (C.norm(dim=0, keepdim=True) + 1e-12)
        self.Ctr = C

        # # [NEW] 하이퍼
        # self.gamma  = float(getattr(args, "fair_conf_gamma",0.5))     # 0~2 추천
        # self.margin = float(getattr(args, "fair_margin", 0.0))         # 0.0~0.03 추천
        self.adv_steps = int(getattr(args, "fair_adv_steps", 3))
        # self.dr_clip  = float(getattr(args, "fair_dr_clip", 3.0))  # <<< ADD: 폭주 방지


        # v, g
        self.v = nn.Parameter(torch.randn(self.Ctr.shape[1], device=device))
        if self.Ctr.shape[1] > 0:
            self.v.data = self.v / (self.v.norm() + 1e-12)

        self.g = Discriminator().to(device)
        base_lr = float(getattr(args, "lr", 1e-3))
        self.opt_g = torch.optim.Adam(self.g.parameters(), lr=1.5 * base_lr)
        self.opt_v = torch.optim.Adam([self.v], lr=4.0 * base_lr)

    # utils/fair_losses.py  (DRFair 안)
    def diag_bundle(self, logits: torch.Tensor):
        """
        브레이크다운/시각화용 진단 패키지 반환.
        - C: [N, M], yv, gz, w (모두 detach)
        """
        with torch.no_grad():
            v_unit = self._v_unit()
            yv = (self.Ctr @ v_unit).detach()
            gz = self.g(torch.sigmoid(logits).unsqueeze(1)).squeeze(-1).detach()
        return {"C": self.Ctr, "yv": yv, "gz": gz}

    # def _v_unit(self):
    #     return self.v / (self.v.norm() + 1e-12)
    def _v_unit(self):
        v = self.v
        v_unit = v / (v.norm(p=2) + 1e-12)
        return v_unit
    
    # def _v_unit(self):
    #     """v를 simplex 위로 정규화"""
    #     v = self.v
    #     v_unit = torch.softmax(v, dim=0)
    #     return v_unit

    # def _conf_weight(self, logits: torch.Tensor) -> torch.Tensor:
    #     # w = |sigmoid(logit) - 0.5|^gamma
    #     p = torch.sigmoid(logits)
    #     w = (0.5 - (p - 0.5).abs()).clamp_min(0.0).pow(self.gamma)  # 경계↑, 확신↓

    #     # w = (p - 0.5).abs().pow(self.gamma).detach()
    #     return w + 1e-6  # 완전 0 방지
    
    def penalty(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        f 업데이트용 패널티 (train에서는 Ctr 사용, val/test에서는 g만 사용).
        """
        N = logits.shape[0]

        # 평가 모드(val/test): 입력 크기가 train과 다르면 self.Ctr 못 씀
        if N != self.Ctr.shape[0]:
            gz = self.g(torch.sigmoid(logits).unsqueeze(1)).squeeze(-1)
            # 여기서는 C가 없으니 corr 대신 단순 분산 측정 (fallback)
            return (gz.std() * 0.0).to(logits.device)  # 그냥 0 반환 (평가용)
        
        # ---- 훈련 모드 (train과 크기 같을 때) ----
        if self.Ctr.shape[1] == 0:
            return torch.zeros((), device=logits.device)

        with torch.no_grad():
            v_unit = self._v_unit()
            yv = (self.Ctr @ v_unit).detach()
        gz = self.g(torch.sigmoid(logits).unsqueeze(1)).squeeze(-1)

        corr = artanh_corr(yv, gz)

        return corr


    
    def after_f_step(self, f, X_tr: torch.Tensor) -> None:
        if self.Ctr.shape[1] == 0:
            return
        for _ in range(self.adv_steps):
            with torch.no_grad():
                logit_det = f(X_tr).detach()

            v_unit = self._v_unit()
            yv = (self.Ctr @ v_unit)
            gz = self.g(torch.sigmoid(logit_det).unsqueeze(1)).squeeze(-1)
            # dr = artanh_corr(w * yv, w * gz)
            dr = artanh_corr(yv, gz)

            self.opt_g.zero_grad()
            self.opt_v.zero_grad()
            (-dr).backward()
            self.opt_g.step()
            self.opt_v.step()

            # with torch.no_grad():
            #     self.v.copy_(self.v / (self.v.norm() + 1e-12))


# # === DRFair 아래에 추가: Marginal CE(=Reg) 패널티 ===
# class RegFair(BaseFair):
#     """
#     sum_g | P(hatY=1 | g) - P(hatY=1) |  (마진별 편차의 합)
#     - adversary 없음 → after_f_step() 불필요
#     - lam 스케일은 트레이너에서 곱함
#     """
#     def __init__(self, args, data, device, min_support=0.1):
#         self.device = device
#         self.min_support = float(min_support)
#         G_np, _ = _build_marginal_groups(data["S_train"])
#         self.G = torch.tensor(G_np, device=device)  # (N, K)

#     def penalty(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
#         if self.G.numel() == 0:
#             return torch.zeros((), device=probs.device)
#         base_pos = probs.mean()               # P(hatY=1)
#         supp = self.G.sum(dim=0)              # (K,)
#         valid = supp > self.min_support
#         if not torch.any(valid):
#             return torch.zeros((), device=probs.device)
#         Gv = self.G[:, valid]                 # (N, Kv)
#         suppv = supp[valid]                   # (Kv,)
#         grp = (Gv.T @ probs) / suppv          # P(hatY=1 | g)
#         return torch.abs(grp - base_pos).sum()

# === DRFair 아래에 추가했던 RegFair를 이렇게 수정 ===
class RegFair(BaseFair):
    """
    sum_g | P(hatY=1 | g) - P(hatY=1) |
    - adversary 없음 → after_f_step() 불필요
    """
    def __init__(self, args, data, device, min_support=0.5):
        self.device = device
        # train/val 각각의 마스크를 생성
        Gtr_np, _ = _build_marginal_groups(data["S_train"])
        Gva_np, _ = _build_marginal_groups(data["S_val"])
        self.G_tr = torch.tensor(Gtr_np, device=device, dtype=torch.float32)  # (N_tr, K)
        self.G_va = torch.tensor(Gva_np, device=device, dtype=torch.float32)  # (N_val, K)
        # min_support: 0<val<1이면 비율, 그 외는 개수로 해석
        self.min_support = float(min_support)

    def after_f_step(self, f, X_tr):
        # adversary 없음 → no-op
        return

    def _penalty_from_G(self, probs: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        if G.numel() == 0:
            return torch.zeros((), device=probs.device)
        base_pos = probs.mean()                 # P(hatY=1)
        supp = G.sum(dim=0)                     # (K,)
        # min_support 해석: 비율 또는 개수
        thr = self.min_support * G.shape[0] if (0.0 < self.min_support < 1.0) else self.min_support
        valid = supp >= max(1.0, thr)
        if not torch.any(valid):
            return torch.zeros((), device=probs.device)
        Gv = G[:, valid]                        # (N, Kv)
        suppv = torch.clamp(supp[valid], min=1.0)  # (Kv,)
        grp = (Gv.T @ probs) / suppv            # (Kv,)
        return torch.abs(grp - base_pos).sum()

    def penalty(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        N = probs.shape[0]
        # probs 길이에 따라 train/val 마스크 자동 선택
        if N == self.G_tr.shape[0]:
            G = self.G_tr
        elif N == self.G_va.shape[0]:
            G = self.G_va
        else:
            # (예: test 등) 안전하게 0 리턴 또는 원하는 쪽 선택
            return torch.zeros((), device=probs.device)
        return self._penalty_from_G(probs, G)

    
# # fair_losses.py 내 SPFairAll를 다음처럼 교체 (핵심만)
# class SPFairAll(BaseFair):
#     """
#     penalty = softmax-approx(max_g |E[p] - E[p|g]|) over all 2^q subgroups
#     - adversary 없음
#     - O(N + 2^q)
#     """
#     def __init__(self, args, data, device):
#         self.device = device

#         # 공통 q, M
#         S_tr = data["S_train"]
#         q = S_tr.shape[1]
#         self.q = q
#         self.M = 1 << q
#         self.tau = 0.01
#         self.min_prop = 0.0
#         self.agg = "softmax"
#         self.use_alpha = True

#         # 비트 가중치 (1,2,4,...)
#         self._w = (2 ** torch.arange(self.q, device=device, dtype=torch.long)).view(1, -1)

#         def _make_gid(S_df):
#             if S_df is None:
#                 return None, 0
#             S = S_df.values if hasattr(S_df, "values") else np.asarray(S_df)
#             S = torch.tensor(S, dtype=torch.float32, device=device)
#             S = (S > 0.5).to(torch.long)           # (N, q)
#             gid = (S * self._w).sum(dim=1)         # (N,), in [0, 2^q)
#             return gid, S.shape[0]

#         self.group_id_tr, self.N_tr = _make_gid(data.get("S_train"))
#         self.group_id_va, self.N_va = _make_gid(data.get("S_val"))
#         self.group_id_te, self.N_te = _make_gid(data.get("S_test"))

#     def _aggregate(self, delta):
#         if self.agg == "max":
#             return delta.max()
#         if self.agg == "softmax":
#             w = torch.softmax(delta / self.tau, dim=0)
#             return (w * delta).sum()
#         return self.tau * torch.logsumexp(delta / self.tau, dim=0)

#     def after_f_step(self, f, X_tr):
#         return  # no-op

#     def _penalty_with_gid(self, p: torch.Tensor, gid: torch.Tensor, N_all: int) -> torch.Tensor:
#         # p: (N,)  gid: (N,)
#         if gid is None or gid.numel() == 0:
#             return torch.zeros((), device=p.device, dtype=p.dtype)

#         base = p.mean()
#         sum_p = torch.zeros(self.M, device=p.device, dtype=p.dtype)
#         cnt   = torch.zeros(self.M, device=p.device, dtype=p.dtype)

#         sum_p.index_add_(0, gid, p)                   # ★ 길이 일치!
#         cnt.index_add_(0, gid, torch.ones_like(p))

#         mean_g = sum_p / cnt.clamp_min(1.0)
#         delta  = (mean_g - base).abs()

#         if self.min_prop > 0.0:
#             mask = (cnt >= self.min_prop * float(N_all)).to(delta.dtype)
#             delta = delta * mask

#         if self.use_alpha:
#             alpha = cnt / max(1.0, float(N_all))
#             delta = alpha * delta

#         return self._aggregate(delta)

#     def penalty(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
#         N = probs.shape[0]
#         if N == self.N_tr:
#             return self._penalty_with_gid(probs, self.group_id_tr, self.N_tr)
#         if N == self.N_va:
#             return self._penalty_with_gid(probs, self.group_id_va, self.N_va)
#         if N == self.N_te:
#             return self._penalty_with_gid(probs, self.group_id_te, self.N_te)
#         # 모르는 split이면 안전하게 0 (또는 여기서 on-the-fly로 gid 만들어도 됨)
#         return torch.zeros((), device=probs.device, dtype=probs.dtype)


class SPFairAll(BaseFair):
    """
    penalty = softmax-approx(max_g |E[p] - E[p|g]|) over all 2^q subgroups
    - adversary 없음
    - O(N + 2^q) 시간/메모리
    """
    def __init__(self, args, data, device):
        self.device = device
        S = data["S_train"]
        S = S.values if hasattr(S, "values") else np.asarray(S)
        S = torch.tensor(S, dtype=torch.float32, device=device)
        # 보장: 0/1 이진
        S = (S > 0.5).to(torch.long)

        self.N, self.q = S.shape
        self.M = 1 << self.q
        # 비트 가중치 (1,2,4,...)
        w = (2 ** torch.arange(self.q, device=device, dtype=torch.long)).view(1, -1)
        self.group_id = (S * w).sum(dim=1)  # (N,) in [0, 2^q)

        # 하이퍼
        self.tau = 0.01  # LSE/softmax 온도(작을수록 max 근사)
        self.min_prop = 0.0  # 너무 작은 그룹 무시(비율)
        self.agg = "softmax"  # "softmax" | "max"
        self.use_alpha = True  # α_g 가중 사용 여부

    def _aggregate(self, delta):
        if self.agg == "max":
            return delta.max()
        if self.agg == "softmax":
            w = torch.softmax(delta / self.tau, dim=0)
            return (w * delta).sum()
        # default: log-sum-exp (stable softmax max)
        return self.tau * torch.logsumexp(delta / self.tau, dim=0)

    def penalty(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        # probs = sigmoid(logits) 가 이미 넘어온다고 가정 (혹은 여기서 torch.sigmoid(logits))
        p = probs  # (N,)
        base = p.mean()

        # 그룹별 합/카운트 (미분 가능한 index_add_)
        sum_p = torch.zeros(self.M, device=self.device, dtype=p.dtype)
        cnt   = torch.zeros(self.M, device=self.device, dtype=p.dtype)
        sum_p.index_add_(0, self.group_id, p)
        cnt.index_add_(0, self.group_id, torch.ones_like(p))

        # 평균, 격차
        mean_g = sum_p / cnt.clamp_min(1.0)
        delta  = (mean_g - base).abs()

        # 너무 작은 그룹 제거
        if self.min_prop > 0.0:
            mask = (cnt >= self.min_prop * self.N).to(delta.dtype)
            delta = delta * mask  # 또는 delta[mask==0] = 0

        # α_g 가중 (선택)
        if self.use_alpha:
            alpha = cnt / max(1.0, float(self.N))
            delta = alpha * delta

        return self._aggregate(delta)

    def after_f_step(self, f, X_tr):  # 적대자 없음
        return

# class SPFairAll(BaseFair):
#     """
#     penalty = softmax-approx(max_g |E[p] - E[p|g]|) over all 2^q subgroups
#     - adversary 없음
#     - O(N + 2^q) 시간/메모리
#     """
#     def __init__(self, args, data, device):
#         self.device = device
#         S = data["S_train"]
#         S = S.values if hasattr(S, "values") else np.asarray(S)
#         S = torch.tensor(S, dtype=torch.float32, device=device)
#         # 보장: 0/1 이진
#         S = (S > 0.5).to(torch.long)

#         self.N, self.q = S.shape
#         self.M = 1 << self.q
#         # 비트 가중치 (1,2,4,...)
#         w = (2 ** torch.arange(self.q, device=device, dtype=torch.long)).view(1, -1)
#         self.group_id = (S * w).sum(dim=1)  # (N,) in [0, 2^q)

#         # 하이퍼
#         self.tau = 0.01  # LSE/softmax 온도(작을수록 max 근사)
#         self.min_prop = 0.0  # 너무 작은 그룹 무시(비율)
#         self.agg = "softmax"  # "softmax" | "max"
#         self.use_alpha = True  # α_g 가중 사용 여부

#     def _aggregate(self, delta):
#         if self.agg == "max":
#             return delta.max()
#         if self.agg == "softmax":
#             w = torch.softmax(delta / self.tau, dim=0)
#             return (w * delta).sum()
#         # default: log-sum-exp (stable softmax max)
#         return self.tau * torch.logsumexp(delta / self.tau, dim=0)

#     def penalty(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
#         # probs = sigmoid(logits) 가 이미 넘어온다고 가정 (혹은 여기서 torch.sigmoid(logits))
#         p = probs  # (N,)
#         base = p.mean()

#         # 그룹별 합/카운트 (미분 가능한 index_add_)
#         sum_p = torch.zeros(self.M, device=self.device, dtype=p.dtype)
#         cnt   = torch.zeros(self.M, device=self.device, dtype=p.dtype)
#         sum_p.index_add_(0, self.group_id, p)
#         cnt.index_add_(0, self.group_id, torch.ones_like(p))

#         # 평균, 격차
#         mean_g = sum_p / cnt.clamp_min(1.0)
#         delta  = (mean_g - base).abs()

#         # 너무 작은 그룹 제거
#         if self.min_prop > 0.0:
#             mask = (cnt >= self.min_prop * self.N).to(delta.dtype)
#             delta = delta * mask  # 또는 delta[mask==0] = 0

#         # α_g 가중 (선택)
#         if self.use_alpha:
#             alpha = cnt / max(1.0, float(self.N))
#             delta = alpha * delta

#         return self._aggregate(delta)

#     def after_f_step(self, f, X_tr):  # 적대자 없음
#         return

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
        # return SPFair(args, data, device)   # ★ 추가
        return SPFairAll(args, data, device)   # ★ 추가
    if key.startswith("reg") or key.startswith("marginal"):
        print("fair loss: marginal fair")
        return RegFair(args, data, device)

    return NoneFair()