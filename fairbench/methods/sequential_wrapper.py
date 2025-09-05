# fairbench/methods/sequential_wrapper.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .dr import SmallConvNet

# ---- 미분가능한 soft-SPD: |E[p|g]-E[p]| 의 그룹별 최대 ----
def _soft_spd_max(prob: torch.Tensor, S_list, min_support: int = 5, device: str = "cpu") -> torch.Tensor:
    """
    prob: (B,) 시그모이드 확률, torch.Tensor (requires_grad=True)
    S_list: list[dict], 각 dict는 민감 속성의 0/1
    반환: batch 내 worst-group soft SPD (스칼라 텐서)
    """
    keys = list(S_list[0].keys())
    p = prob.view(-1)  # (B,)
    spds = []
    for k in keys:
        # 배치별 0/1 마스크를 텐서로 (grad는 마스크에 필요 없음)
        m = torch.tensor([int(s[k]) for s in S_list], device=device, dtype=torch.float32)
        n1 = int(m.sum().item()); n0 = int((1.0 - m).sum().item())
        if n1 >= min_support and n0 >= min_support:
            r_g   = (p * m).sum() / (m.sum().clamp_min(1.0))  # E[p|g]
            r_all = p.mean()                                  # E[p]
            spds.append(torch.abs(r_g - r_all))
    if len(spds) == 0:
        return torch.zeros([], device=device)
    return torch.stack(spds).max()

# ---- 스케줄러: penalty 계수 시간 가중 ----
def _penalty_weight(ep: int, epochs: int, base: float, sched: str = "const", warmup: int = 0) -> float:
    """
    sched in {const, linear, cosine, exp}
    - const: base
    - linear: warmup 이후 선형 상승 → base
    - cosine: 0→base 코사인 상승
    - exp: 0→base 지수형 상승(완만 시작)
    """
    if base <= 0:
        return 0.0
    e = ep + 1
    if sched == "linear":
        if e <= warmup:
            return 0.0
        frac = min(1.0, max(0.0, (e - warmup) / max(1, epochs - warmup)))
        return float(base * frac)
    elif sched == "cosine":
        import math
        frac = min(1.0, max(0.0, e / max(1, epochs)))
        return float(base * 0.5 * (1.0 - math.cos(math.pi * frac)))
    elif sched == "exp":
        frac = min(1.0, max(0.0, e / max(1, epochs)))
        return float(base * ((10.0 ** frac - 1.0) / (10.0 - 1.0)))
    else:  # const
        return float(base)

def run_sequential(args, data):
    # 이 구현은 이미지(예: CelebA) 전용
    if data["type"] != "image":
        print("[SKIP] SequentialFairness: CelebA(image) only.")
        return dict(proba=None, pred=None, note="sequential_image_only")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    f = SmallConvNet().to(device)
    opt = optim.Adam(f.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_bce = nn.BCEWithLogitsLoss()

    # === penalty 계수: --seq_lambda가 있으면 우선, 없으면 --seq_alpha 사용 ===
    base_coef = getattr(args, "seq_lambda", None)
    if base_coef is None:
        base_coef = getattr(args, "seq_alpha", 0.1)  # 기존 인자 재활용
    sched = getattr(args, "seq_sched", "const")      # 선택: const|linear|cosine|exp
    warm  = int(getattr(args, "seq_warmup", 0))     # linear warmup 스텝(에폭 단위)
    min_support = int(getattr(args, "min_support", 5))

    for ep in range(args.epochs):
        f.train()
        coef_ep = _penalty_weight(ep, args.epochs, float(base_coef), sched, warm)
        for x, y, S in data["train_loader"]:
            x = x.to(device)
            y = y.float().to(device)
            logit = f(x)

            # 분류 손실
            cls = loss_bce(logit, y)

            # 미분가능한 soft SPD 페널티 (배치 기반)
            p = torch.sigmoid(logit)  # grad 흐름 유지!
            pen = _soft_spd_max(p.squeeze(), S, min_support=min_support, device=device)

            total = cls + coef_ep * pen
            opt.zero_grad()
            total.backward()
            opt.step()

        if (ep + 1) % max(1, getattr(args, "log_interval", 50)) == 0:
            # 가벼운 로그 (필요시 logger로 교체)
            print(f"[seq] ep={ep+1}/{args.epochs}  coef={coef_ep:.4f}  pen≈{pen.item():.4f}")

    # === validation으로 threshold 튜닝 ===
    f.eval()
    from ..utils.threshold import tune_threshold
    pv, yv = [], []
    with torch.no_grad():
        for x, y, S in data["val_loader"]:
            p = torch.sigmoid(f(x.to(device))).cpu().numpy()
            pv.append(p); yv.append(y.numpy())
    pv = np.concatenate(pv).reshape(-1); yv = np.concatenate(yv).reshape(-1)
    t = tune_threshold(pv, yv)

    # === test 예측 ===
    pt, yt = [], []
    with torch.no_grad():
        for x, y, S in data["test_loader"]:
            p = torch.sigmoid(f(x.to(device))).cpu().numpy()
            pt.append(p); yt.append(y.numpy())
    pt = np.concatenate(pt).reshape(-1)
    yhat = (pt >= t).astype(int)
    return dict(proba=pt, pred=yhat)
