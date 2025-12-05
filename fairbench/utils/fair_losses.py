import numpy as np
import torch
import torch.nn as nn

from ..methods.dr_wrapper import build_C_tensor, build_C_te_from_Ctr  
from ..methods.dr_wrapper import Discriminator, artanh_corr 


import random, torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


# base class
class BaseFair:
    def penalty(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=logits.device)

    def after_f_step(self, f, X_tr: torch.Tensor) -> None:
        return

# lambda = 0
class NoneFair(BaseFair):
    pass

# lambda > 0
class DRFair(BaseFair):
    def __init__(self, args, data, device):
        self._v_prev = None 
        self.device = device
        self.args = args
        
        self.S_tr = data["S_train"]
        n_low_frac_eff = getattr(args, "n_low_frac", None)
        if n_low_frac_eff is None:
            n_low_frac_eff = float(getattr(args, "n_low_frac_default", 0.01))  # 1% 기본
        n_low      = getattr(args, "n_low", None)
        use_ap     = bool(getattr(args, "agg_apriori", True) or getattr(args, "apriori_union", False))
        agg_max_len  = int(getattr(args, "agg_max_len", 4))
        agg_max_cols = int(getattr(args, "agg_max_cols", 2048))
        

        with torch.no_grad():
            C = build_C_tensor(
                self.S_tr, args, device=device,
                n_low=n_low, n_low_frac=n_low_frac_eff,
                apriori_union=use_ap, agg_max_len=agg_max_len, agg_max_cols=agg_max_cols
            )  
        self.Ctr = C
        self.adv_steps = int(getattr(args, "fair_adv_steps", 3))

        # v, g
        self.v = nn.Parameter(torch.randn(self.Ctr.shape[1], device=device))
        if self.Ctr.shape[1] > 0:
            self.v.data = self.v / (self.v.norm() + 1e-12)

        self.g = Discriminator().to(device)
        base_lr = float(getattr(args, "lr", 1e-3))
        self.opt_g = torch.optim.Adam(self.g.parameters(), lr=1.5 * base_lr)
        self.opt_v = torch.optim.Adam([self.v], lr=4.0 * base_lr)

    def diag_bundle(self, logits: torch.Tensor):
        with torch.no_grad():
            v_unit = self._v_unit()
            yv = (self.Ctr @ v_unit).detach()
            gz = self.g(torch.sigmoid(logits).unsqueeze(1)).squeeze(-1).detach()
        return {"C": self.Ctr, "yv": yv, "gz": gz}


    def _v_unit(self):
        v = self.v
        v_unit = v / (v.norm(p=2) + 1e-12)
        return v_unit
    
    
    def penalty(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        if self.Ctr.shape[1] == 0:
            return torch.zeros((), device=logits.device)

        with torch.no_grad():
            v_unit = self._v_unit()
            yv = (self.Ctr @ v_unit).detach()
        gz = self.g(torch.sigmoid(logits).unsqueeze(1)).squeeze(-1)
        corr = artanh_corr(yv, gz)

        return corr

    def penalty_test(self, logits: torch.Tensor, S_te: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            C_te = build_C_te_from_Ctr(self.S_tr, self.Ctr, S_te, device="cuda")
            v_unit = self._v_unit()
            yv = (C_te @ v_unit).detach()
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


def get_fair_loss(args, data, fair_weight, device) -> BaseFair:
    
    if fair_weight <= 0:
        print("fair loss: None fair")
        return NoneFair()

    key = getattr(args, "method", "unfair")
    key = str(key).lower()

    if key.startswith("dr"):
        print("fair loss: dr fair")
        return DRFair(args, data, device)

    return NoneFair()