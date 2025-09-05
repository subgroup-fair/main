# fairbench/metrics/__init__.py
import numpy as np
from .accuracy import accuracy
from .supipm import supipm_rbf, supipm_wasserstein
from .subgroup_measures import (
    worst_group_spd, mean_group_spd,
    worst_group_fpr_gap, mean_group_fpr_gap,
    multicalib_worst_gap, multicalib_mean_gap,
    # ↓↓↓ marginal metrics 추가
    marginal_spd_worst, marginal_spd_mean,
    marginal_fpr_worst, marginal_fpr_mean,
    marginal_mc_worst,  marginal_mc_mean,
)

def compute_metrics(args, data, pred_pack):
    proba = pred_pack.get("proba", None)
    pred  = pred_pack.get("pred",  None)

    report = dict(dataset=getattr(args, "dataset", ""),
                  method=getattr(args, "method", ""),
                  seed=getattr(args, "seed", None))

    thr = getattr(args, "thr", 0.5)
    min_support = getattr(args, "min_support", 5)
    mc_bins = getattr(args, "mc_bins", 10)
    mc_min_support = getattr(args, "mc_min_support", 10)

    # --- 정답 y 수집 ---
    if data["type"] == "tabular":
        y = data.get("y_test", None)
        S = data.get("S_test", None)
    else:
        ys, Ss = [], []
        for _, yb, Sb in data["test_loader"]:
            ys.append(yb.numpy())
            Ss += Sb
        y = np.concatenate(ys) if len(ys) > 0 else None
        S = Ss  # 이미지: list[dict]

    # --- Accuracy ---
    report["accuracy"] = accuracy(y, pred) if (y is not None and pred is not None) else np.nan

    # --- supIPM (RBF-MMD & W1) ---
    if proba is not None and S is not None:
        try:
            report["supipm_rbf"] = float(supipm_rbf(proba, S))
        except Exception:
            report["supipm_rbf"] = np.nan
        try:
            report["supipm_w1"] = float(supipm_wasserstein(proba, S, min_support=min_support))
        except Exception:
            report["supipm_w1"] = np.nan
    else:
        report["supipm_rbf"] = np.nan
        report["supipm_w1"]  = np.nan

    # --- Subgroup SPD (worst & mean) ---
    if proba is not None and S is not None:
        try:
            report["spd_worst"] = float(worst_group_spd(proba, S))
        except Exception:
            report["spd_worst"] = np.nan
        try:
            report["spd_mean"] = float(mean_group_spd(proba, S, thr=thr, min_support=min_support))
        except Exception:
            report["spd_mean"] = np.nan
    else:
        report["spd_worst"] = np.nan
        report["spd_mean"]  = np.nan

    # --- Subgroup FPR gap (worst & mean) — tabular만 계산 ---
    if (data["type"] == "tabular") and (pred is not None) and (y is not None) and (S is not None):
        try:
            report["fpr_worst"] = float(worst_group_fpr_gap(pred, y, S))
        except Exception:
            report["fpr_worst"] = np.nan
        try:
            report["fpr_mean"] = float(mean_group_fpr_gap(pred, y, S, min_support=min_support))
        except Exception:
            report["fpr_mean"] = np.nan
    else:
        report["fpr_worst"] = np.nan
        report["fpr_mean"]  = np.nan

    # --- Subgroup Multicalibration gap (worst & mean) — tabular만 계산 ---
    if (data["type"] == "tabular") and (proba is not None) and (y is not None) and (S is not None):
        try:
            report["mc_worst"] = float(multicalib_worst_gap(proba, y, S, bins=mc_bins))
        except Exception:
            report["mc_worst"] = np.nan
        try:
            report["mc_mean"] = float(multicalib_mean_gap(proba, y, S, bins=mc_bins, min_support=mc_min_support))
        except Exception:
            report["mc_mean"] = np.nan
    else:
        report["mc_worst"] = np.nan
        report["mc_mean"]  = np.nan

    # =========================
    # ==== Marginal metrics ===
    # =========================

    # Marginal SPD (속성별)
    if proba is not None and S is not None:
        try:
            report["marg_spd_worst"] = float(marginal_spd_worst(proba, S, thr=thr, min_support=min_support))
        except Exception:
            report["marg_spd_worst"] = np.nan
        try:
            report["marg_spd_mean"]  = float(marginal_spd_mean(proba, S, thr=thr, min_support=min_support))
        except Exception:
            report["marg_spd_mean"]  = np.nan
    else:
        report["marg_spd_worst"] = np.nan
        report["marg_spd_mean"]  = np.nan

    # Marginal FPR (속성별) — pred & y 필요 (이미지/탭ular 모두 지원)
    if (pred is not None) and (y is not None) and (S is not None):
        try:
            report["marg_fpr_worst"] = float(marginal_fpr_worst(pred, y, S, min_support=min_support))
        except Exception:
            report["marg_fpr_worst"] = np.nan
        try:
            report["marg_fpr_mean"]  = float(marginal_fpr_mean(pred, y, S, min_support=min_support))
        except Exception:
            report["marg_fpr_mean"]  = np.nan
    else:
        report["marg_fpr_worst"] = np.nan
        report["marg_fpr_mean"]  = np.nan

    # Marginal Multicalibration (속성별) — proba & y 필요
    if (proba is not None) and (y is not None) and (S is not None):
        try:
            report["marg_mc_worst"] = float(marginal_mc_worst(proba, y, S, bins=mc_bins, min_support=mc_min_support))
        except Exception:
            report["marg_mc_worst"] = np.nan
        try:
            report["marg_mc_mean"]  = float(marginal_mc_mean(proba, y, S, bins=mc_bins, min_support=mc_min_support))
        except Exception:
            report["marg_mc_mean"]  = np.nan
    else:
        report["marg_mc_worst"] = np.nan
        report["marg_mc_mean"]  = np.nan

    return report
