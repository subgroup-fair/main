# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd

from .measures import (
    supipm_wasserstein,
    marginal_kearns_order1_worst,
    marginal_wasserstein_order1_worst,
    marginal_kearns_order2_worst,
    marginal_kearns_order3_worst,
    kearns_subgroup_worst,
)

#####################################
import numpy as np, random, torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
######################################################

def accuracy(y_true, y_pred):
    return (y_true.reshape(-1)==y_pred.reshape(-1)).mean()


def compute_metrics(args, data, pred_pack):
    proba = pred_pack.get("proba", None)
    pred = pred_pack.get("pred", None)
    test_dr = pred_pack.get("test_dr", None)
    report = dict(dataset=getattr(args, "dataset", ""),
                  method=getattr(args, "method", ""),
                  seed=getattr(args, "seed", None))
    report["V_stats"] = pred_pack.get("V_stats", [])
    thr = getattr(args, "thr", 0.5)
    min_support = int(getattr(args, "min_support",1 ))

    if data["type"] == "tabular":
        y = data.get("y_test", None); S = data.get("S_test", None)
    else:
        ys, Ss = [], []
        for _, yb, Sb in data["test_loader"]:
            ys.append(yb.numpy()); Ss += Sb
        y = np.concatenate(ys) if len(ys) > 0 else None
        S = Ss

    ##########################################################
    print("min support: ", min_support)
    try: report[f"test_dr"] = float(test_dr)
    except Exception: report[f"test_dr"] = np.nan
    
    print("=== accuracy START ===")
    report["accuracy"] = accuracy(y, pred) if (y is not None and pred is not None) else np.nan
    print(f"[metric] accuracy = {report['accuracy']}")
    print("=== accuracy END ===")

    
    print("=== supIPM(overall) START ===")
    if proba is not None and S is not None:
        try: report["supipm_w1"] = float(supipm_wasserstein(proba, S, min_support=min_support))
        except Exception: report["supipm_w1"] = np.nan
    else:
        report["supipm_w1"] = np.nan
    print(f"[metric] supipm_w1 = {report['supipm_w1']}")
    print("=== supIPM(overall) END ===")


    print("=== Marginal 1st order START ===")
    if proba is not None and S is not None:
        try: report["marginal_kearns_order1_worst"] = float(marginal_kearns_order1_worst(proba, S, thr=thr, min_support=min_support))
        except Exception: report["marginal_kearns_order1_worst"] = np.nan
        try: report["marginal_wasserstein_order1_worst"] = float(marginal_wasserstein_order1_worst(proba, S, min_support=min_support))
        except Exception: report["marginal_wasserstein_order1_worst"] = np.nan
    else:
        report["marginal_kearns_order1_worst"] = np.nan; report["marginal_wasserstein_order1_worst"] = np.nan
    print(f"[metric] marginal_kearns_order1_worst = {report['marginal_kearns_order1_worst']}")
    print(f"[metric] marginal_wasserstein_order1_worst = {report['marginal_wasserstein_order1_worst']}")


    print("=== Marginal 2,3 order START ===")
    if proba is not None and S is not None:
        try: report["marginal_kearns_order2_worst"] = float(marginal_kearns_order2_worst(proba, S, thr=thr, min_support=min_support))
        except Exception: report["marginal_kearns_order2_worst"] = np.nan
        try: report["marginal_kearns_order3_worst"] = float(marginal_kearns_order3_worst(proba, S, thr=thr, min_support=min_support))
        except Exception: report["marginal_kearns_order3_worst"] = np.nan
    else:
        report["marginal_kearns_order2_worst"] = np.nan; report["marginal_kearns_order3_worst"] = np.nan


    print("=== Subgroup START ===")
    if proba is not None and S is not None:
        try: report["kearns_subgroup_worst"] = float(kearns_subgroup_worst(proba, S, thr=thr, min_support=min_support))
        except Exception: report["kearns_subgroup_worst"] = np.nan
    else:
        report["kearns_subgroup_worst"] = np.nan


    return report


def compute_metrics_train(args, data, train_pred_pack):
    proba = train_pred_pack.get("train_proba", None).cpu().detach().numpy()
    report = dict(dataset=getattr(args, "dataset", ""),
                  method=getattr(args, "method", ""),
                  seed=getattr(args, "seed", None))
    thr = getattr(args, "thr", 0.5)
    min_support = int(getattr(args, "min_support", 1))

    if data["type"] == "tabular":
        y = data.get("y_train", None); S = data.get("S_train", None)
    else:
        ys, Ss = [], []
        for _, yb, Sb in data["train_loader"]:
            ys.append(yb.numpy()); Ss += Sb
        y = np.concatenate(ys) if len(ys) > 0 else None
        S = Ss

    ##########################################################
    print("min support: ", min_support)
    pred = (np.asarray(proba, float).reshape(-1) >= float(thr)).astype(int)
    
    print("=== accuracy START ===")
    report["accuracy"] = accuracy(y, pred) if (y is not None and pred is not None) else np.nan
    print(f"[metric] accuracy = {report['accuracy']}")
    print("=== accuracy END ===")

    print("=== Marginal 1st order START ===")
    if proba is not None and S is not None:
        report["marginal_kearns_order1_worst"] = float(marginal_kearns_order1_worst(proba, S, thr=thr, min_support=min_support))
        report["kearns_subgroup_worst"] = float(kearns_subgroup_worst(proba, S, thr=thr, min_support=min_support))
    else:
        report["marginal_kearns_order1_worst"] = np.nan; report["marginal_wasserstein_order1_worst"] = np.nan
    print(f"[metric] marginal_kearns_order1_worst = {report['marginal_kearns_order1_worst']}")
    print(f"[metric] kearns_subgroup_worst = {report['kearns_subgroup_worst']}")


    return report