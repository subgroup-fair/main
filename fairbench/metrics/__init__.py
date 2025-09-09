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

        # ===== [추가] 12개 메트릭용 함수들 =====
    # 1–2) singleton supIPM
    sup_mmd_dfcols, sup_w1_dfcols,
    # 3–4) V supIPM
    sup_mmd_over_V, sup_w1_over_V,
    # 7–8) weighted SPD (singleton)
    worst_weighted_group_spd, mean_weighted_group_spd,
    # 9–10) DP over V (sup/mean)
    worst_spd_over_V, mean_spd_over_V,
    # 11–12) weighted SPD over V
    worst_weighted_spd_over_V, mean_weighted_spd_over_V,
)

from itertools import product, combinations
from tqdm import tqdm

def _build_V_3style_from_S_df(S_df, min_support=1):
    """
    family마다 {ALL} ∪ {각 col==1} ∪ (이진이면 {col==0}) 중 하나 선택 → AND → V 생성
    - 공집합(전부 ALL) 제외
    - support < min_support 프루닝
    반환:
      V: bool mask 리스트
      choices: 각 V[k]의 family별 선택 설명(tuple[str])
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

    # family별 옵션
    fam_opts = []
    fam_names = []
    for f, cols in fam.items():
        opts = [("ALL", None)]
        for col in cols:
            opts.append(("col1", col))
        if len(cols) == 1:
            opts.append(("col0", cols[0]))  # binary complement
        fam_opts.append(opts)
        fam_names.append(f)

    # 조합 열거
    V, choices = [], []
    seen = set()
    total = 1
    for opts in fam_opts:
        total *= len(opts)

    for pick in product(*fam_opts):
        # 전부 ALL이면 스킵
        if all(tag == "ALL" for tag, _ in pick):
            continue

        m = np.ones(n, dtype=bool)
        ok = True
        for tag, col in pick:
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

        if int(m.sum()) < int(min_support):
            continue

        key = m.tobytes()
        if key in seen:
            continue
        seen.add(key)

        # 읽기 쉬운 선택 기록
        readable = []
        for fam_name, (tag, col) in zip(fam_names, pick):
            if tag == "ALL":
                readable.append(f"{fam_name}=ALL")
            elif tag == "col1":
                readable.append(f"{fam_name}=={col}")
            else:
                readable.append(f"{fam_name}==NOT({col})")
        V.append(m)
        choices.append(tuple(readable))

    return V, choices


# [ADD] 3^q 스타일 V 생성 (DataFrame 버전)
def _build_V_3style_from_S_list(S_list, min_support=1):
    """
    각 key에 대해 {ALL, key==1, key==0} 중 하나 선택 → AND → V 생성
    - 공집합(전부 ALL) 제외
    - support < min_support 프루닝
    """
    if len(S_list) == 0:
        return [], []
    keys = list(S_list[0].keys())
    n = len(S_list)
    arrs = {k: np.array([int(d[k]) for d in S_list]) for k in keys}

    fam_opts = []
    for k in keys:
        fam_opts.append([("ALL", None), ("key1", k), ("key0", k)])

    V, choices = [], []
    seen = set()
    total = 1
    for opts in fam_opts:
        total *= len(opts)

    for pick in product(*fam_opts):
        if all(tag == "ALL" for tag, _ in pick):
            continue

        m = np.ones(n, dtype=bool)
        ok = True
        for tag, k in pick:
            if tag == "ALL":
                continue
            v = arrs[k]
            if tag == "key1":
                m &= (v == 1)
            else:
                m &= (v == 0)
            if not m.any():
                ok = False
                break
        if not ok:
            continue

        if int(m.sum()) < int(min_support):
            continue

        keyb = m.tobytes()
        if keyb in seen:
            continue
        seen.add(keyb)

        # 선택 기록
        readable = []
        for (tag, k) in pick:
            if tag == "ALL":
                readable.append(f"{k}=ALL")
            elif tag == "key1":
                readable.append(f"{k}==1")
            else:
                readable.append(f"{k}==0")
        V.append(m)
        choices.append(tuple(readable))

    return V, choices

def compute_metrics(args, data, pred_pack):
    proba = pred_pack.get("proba", None)
    pred  = pred_pack.get("pred",  None)

    report = dict(dataset=getattr(args, "dataset", ""),
                  method=getattr(args, "method", ""),
                  seed=getattr(args, "seed", None))
    report["V_stats"] = pred_pack.get("V_stats", [])

    thr = getattr(args, "thr", 0.5)
    min_support = getattr(args, "min_support", 1)
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

    print("performance) V counting (3^q-style)...")
    try:
        if isinstance(S, list) and len(S) > 0 and isinstance(S[0], dict):
            V, V_choices = _build_V_3style_from_S_list(S, min_support=min_support)
        elif hasattr(S, "columns"):
            V, V_choices = _build_V_3style_from_S_df(S, min_support=min_support)
        else:
            V, V_choices = None, None
    except Exception:
        V, V_choices = None, None

    # report["V_count"] = (len(V) if isinstance(V, list) else np.nan)
    # print(f"V count (3^q): {report['V_count']}")
    # report["V_choices"] = V_choices if V_choices is not None else []


    print("=== accuracy START ===")
    report["accuracy"] = accuracy(y, pred) if (y is not None and pred is not None) else np.nan
    print(f"[metric] accuracy = {report['accuracy']}")
    print("=== accuracy END ===")

    print("=== supIPM(overall) START ===")
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
    
    print(f"[metric] supipm_rbf = {report['supipm_rbf']}")
    print(f"[metric] supipm_w1  = {report['supipm_w1']}")
    print("=== supIPM(overall) END ===")



    # ===== (A) 1–2) supIPM — 싱글톤 기준 =====
    print("=== sup_mmd_dfcols, sup_w1_dfcols START ===")
    if (data["type"] == "tabular") and (proba is not None) and (S is not None):
        try:
            report["sup_mmd_dfcols"] = float(sup_mmd_dfcols(proba, S, min_support=min_support))
        except Exception:
            report["sup_mmd_dfcols"] = np.nan
        try:
            report["sup_w1_dfcols"] = float(sup_w1_dfcols(proba, S, min_support=min_support))
        except Exception:
            report["sup_w1_dfcols"] = np.nan
    else:
        report["sup_mmd_dfcols"] = np.nan
        report["sup_w1_dfcols"]  = np.nan

    print(f"[metric] sup_mmd_dfcols = {report['sup_mmd_dfcols']}")
    print(f"[metric] sup_w1_dfcols  = {report['sup_w1_dfcols']}")

    print("=== sup_mmd_over_V, sup_w1_over_V START ===")

    # ===== (B) 3–4) supIPM — 𝒱 기준 =====
    if (proba is not None) and (V is not None):
        try:
            report["sup_mmd_over_V"] = float(sup_mmd_over_V(proba, V, min_support=min_support))
        except Exception:
            report["sup_mmd_over_V"] = np.nan
        try:
            report["sup_w1_over_V"] = float(sup_w1_over_V(proba, V, min_support=min_support))
        except Exception:
            report["sup_w1_over_V"] = np.nan
    else:
        report["sup_mmd_over_V"] = np.nan
        report["sup_w1_over_V"]  = np.nan
    
    print(f"[metric] sup_mmd_dfcols = {report['sup_mmd_over_V']}")
    print(f"[metric] sup_w1_dfcols  = {report['sup_w1_over_V']}")

    print("=== worst_group_spd, mean_group_spd START ===")
    # --- 5–6) Subgroup SPD (worst & mean) — 싱글톤 ---
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

    print(f"[metric] worst_group_spd = {report['spd_worst']}")
    print(f"[metric] mean_group_spd  = {report['spd_mean']}")


    print("=== worst_weighted_group_spd, mean_weighted_group_spd START ===")
    # ===== (C) 7–8) Weighted SPD — 싱글톤 =====
    if proba is not None and S is not None:
        try:
            report["worst_weighted_group_spd"] = float(
                worst_weighted_group_spd(proba, S, thr=thr, min_support=min_support)
            )
        except Exception:
            report["worst_weighted_group_spd"] = np.nan
        try:
            report["mean_weighted_group_spd"] = float(
                mean_weighted_group_spd(proba, S, thr=thr, min_support=min_support)
            )
        except Exception:
            report["mean_weighted_group_spd"] = np.nan
    else:
        report["worst_weighted_group_spd"] = np.nan
        report["mean_weighted_group_spd"]  = np.nan

    print(f"[metric] mean_weighted_group_spd = {report['worst_weighted_group_spd']}")
    print(f"[metric] mean_weighted_group_spd  = {report['mean_weighted_group_spd']}")
    print("=== SPD(singleton) END ===")

    print("=== worst_weighted_group_spd, mean_weighted_group_spd START ===")
    # ===== (D) 9–10) DP over 𝒱 (sup/mean) =====
    if (proba is not None) and (V is not None):
        try:
            report["worst_spd_over_V"] = float(worst_spd_over_V(proba, V, thr=thr, min_support=min_support))
        except Exception:
            report["worst_spd_over_V"] = np.nan
        try:
            report["mean_spd_over_V"] = float(mean_spd_over_V(proba, V, thr=thr, min_support=min_support))
        except Exception:
            report["mean_spd_over_V"] = np.nan
    else:
        report["worst_spd_over_V"]  = np.nan
        report["mean_spd_over_V"] = np.nan

    print(f"[metric] worst_spd_over_V = {report['worst_spd_over_V']}")
    print(f"[metric] mean_spd_over_V  = {report['mean_spd_over_V']}")
    print("=== Weighted SPD(singleton) END ===")

    print("=== DP over V START ===")
    # ===== (E) 11–12) Weighted SPD over 𝒱 =====
    if (proba is not None) and (V is not None):
        try:
            report["worst_weighted_spd_over_V"] = float(
                worst_weighted_spd_over_V(proba, V, thr=thr, min_support=min_support)
            )
        except Exception:
            report["worst_weighted_spd_over_V"] = np.nan
        try:
            report["mean_weighted_spd_over_V"] = float(
                mean_weighted_spd_over_V(proba, V, thr=thr, min_support=min_support)
            )
        except Exception:
            report["mean_weighted_spd_over_V"] = np.nan
    else:
        report["worst_weighted_spd_over_V"] = np.nan
        report["mean_weighted_spd_over_V"]  = np.nan

    print(f"[metric] worst_weighted_spd_over_V  = {report['worst_weighted_spd_over_V']}")
    print(f"[metric] mean_weighted_spd_over_V = {report['mean_weighted_spd_over_V']}")





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
