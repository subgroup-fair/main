# fairbench/datasets/adult.py
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .shrink import shrink_smallest_by_global_frac
import numpy as np, random, torch


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

def _fetch_adult(folder):
    path = folder + "adult.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    lower = {c.lower(): c for c in df.columns}

    def alias(old_keys, new_key):
        for k in old_keys:
            if k in lower:
                src = lower[k]
                if new_key not in df.columns:
                    df.rename(columns={src: new_key}, inplace=True)
                return

    alias(["gender", "sex"], "sex")
    alias(["educational-num", "education_num", "educational_num", "education-num"], "education-num")
    alias(["hours_per_week", "hoursperweek", "hours-per-week"], "hours-per-week")
    alias(["capital_gain", "capitalgain", "capital-gain"], "capital-gain")
    alias(["capital_loss", "capitalloss", "capital-loss"], "capital-loss")
    alias(["native_country", "native-country"], "native-country")
    alias(["marital_status", "marital-status"], "marital-status")

    y = (df["income"].astype(str).str.strip() == ">50K").astype(int).values
    X = df.drop(columns=["income"]).copy()

    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = (
            X[c].astype(str).str.strip()
                .replace("?", np.nan)
                .fillna("Unknown")
        )
    return X, y


def _extended_sensitive_from_adult_df(X: pd.DataFrame) -> pd.DataFrame:
    def has(c): return c in X.columns
    S = {}

    if has("sex"):
        s = X["sex"].astype(str).str.strip()
        S["sex_Male"] = (s == "Male").astype(int)
    if has("race"):
        r = X["race"].astype(str).str.strip()
        S["race_White"] = (r == "White").astype(int) 
    if has("age"):
        age = pd.to_numeric(X["age"], errors="coerce")
        S["age_ge_40"] = (age >= 40).astype(int)  
    if has("marital-status"):
        ms = X["marital-status"].astype(str).str.strip()
        S["married"] = ms.isin({"Married-civ-spouse", "Married-AF-spouse"}).astype(int) 
    if has("education-num"):
        edu = pd.to_numeric(X["education-num"], errors="coerce").fillna(0)
        S["edu_bachelor_plus"] = (edu >= 13).astype(int)
    if has("native-country"):
        nc = X["native-country"].astype(str).str.strip()
        S["native_US"] = (nc == "United-States").astype(int)
    if has("workclass"):
        wc = X["workclass"].astype(str).str.strip()
        S["work_gov"]     = wc.isin({"Federal-gov","State-gov","Local-gov"}).astype(int)
        S["work_private"] = (wc == "Private").astype(int)
    if has("occupation"):
        occ = X["occupation"].astype(str).str.strip()
        S["occ_white_collar"] = occ.isin({
            "Prof-specialty","Exec-managerial","Tech-support"
        }).astype(int)
        S["occ_blue_collar"]  = occ.isin({
            "Craft-repair","Machine-op-inspct","Handlers-cleaners",
            "Farming-fishing","Transport-moving","Priv-house-serv"
        }).astype(int)
    if has("hours-per-week"):
        hpw = pd.to_numeric(X["hours-per-week"], errors="coerce").fillna(0)
        S["hours_ge_50"] = (hpw >= 50).astype(int)
    if has("capital-gain"):
        cg = pd.to_numeric(X["capital-gain"], errors="coerce").fillna(0)
        S["cap_gain_pos"] = (cg > 0).astype(int)
    if has("capital-loss"):
        cl = pd.to_numeric(X["capital-loss"], errors="coerce").fillna(0)
        S["cap_loss_pos"] = (cl > 0).astype(int)
    if has("relationship"):
        rel = X["relationship"].astype(str).str.strip()
        S["rel_husband"] = (rel == "Husband").astype(int)

    S_df = pd.DataFrame(S).astype(int)
    if S_df.shape[1] == 0 and has("sex"):
        S_df = pd.DataFrame({"sex_Male": (X["sex"].astype(str).str.strip() == "Male").astype(int)})
    return S_df


def _filter_sensitive_by_keys(S_df: pd.DataFrame, sens_keys) -> pd.DataFrame:

    if sens_keys is None:
        return S_df

    if isinstance(sens_keys, str):
        tokens = [t.strip() for t in sens_keys.split(",") if t.strip()]
    elif isinstance(sens_keys, (list, tuple, set)):
        tokens = [str(t).strip() for t in sens_keys if str(t).strip()]
    else:
        return S_df
    if not tokens:
        return S_df

    cols = list(S_df.columns)
    cols_lc = {c.lower(): c for c in cols} 

    def _starts_with_icase(c: str, pref: str) -> bool:
        return c.lower().startswith(pref.lower())

    raw2prefix_icase = {
        "sex": "sex_",
        "gender": "sex_",
        "race": "race_",
        "age": "age_",
        "marital-status": "married",  
        "marital": "married",
        "married": "married",
        "education-num": "edu_",
        "native-country": "native_",
        "workclass": "work_",
        "occupation": "occ_",
        "hours-per-week": "hours_",
        "capital-gain": "cap_gain_",
        "capital-loss": "cap_loss_",
        "relationship": "rel_",
    }

    include, exclude = set(), set()

    def expand(tok: str):
        neg = tok.startswith("-")
        t = tok[1:] if neg else tok
        t_lc = t.lower()
        matched = set()

        if t_lc in cols_lc:
            matched.add(cols_lc[t_lc])

        if t.endswith("*"):
            pref = t[:-1]
            matched |= {c for c in cols if _starts_with_icase(c, pref)}

        if t_lc in raw2prefix_icase:
            pref = raw2prefix_icase[t_lc]
            if pref.endswith("_"):  
                matched |= {c for c in cols if _starts_with_icase(c, pref)}
            else:                  
                if pref.lower() in cols_lc:
                    matched.add(cols_lc[pref.lower()])

        if t_lc.startswith("regex:"):
            pat = t[6:]
            try:
                rgx = re.compile(pat, re.IGNORECASE)
                matched |= {c for c in cols if rgx.search(c)}
            except re.error:
                pass

        return ("exclude", matched) if neg else ("include", matched)

    for tok in tokens:
        kind, m = expand(tok)
        if kind == "include":
            include |= m
        else:
            exclude |= m

    selected = (include or set(cols)) - exclude

    toks_lc = [t.lower().lstrip("-") for t in tokens]
    wants_sex = any(t in ("sex", "gender", "sex_*", "gender_*") for t in toks_lc)
    if wants_sex and not any(c.lower().startswith("sex_") for c in selected):
        if "sex_Male" in cols:
            selected.add("sex_Male")
        else:
            for c in cols:
                if c.lower() == "sex_male":
                    selected.add(c)
                    break

    if not selected:
        if "sex_Male" in cols:
            selected = {"sex_Male"}
        else:
            selected = {cols[0]} if cols else set()

    ordered = [c for c in cols if c in selected]

    print(f"[sens_keys] tokens={tokens}")
    print(f"[sens_keys] selected={ordered}")

    return S_df[ordered]


def _infer_raw_cols_from_S(selected_S_cols, X: pd.DataFrame) -> list:
    raws = set()
    for c in selected_S_cols:
        if c.startswith("sex_"): raws.add("sex")
        elif c.startswith("race_"): raws.add("race")
        elif c.startswith("age_"): raws.add("age")
        elif c == "married": raws.add("marital-status")
        elif c.startswith("edu_"): raws.add("education-num")
        elif c.startswith("native_"): raws.add("native-country")
        elif c.startswith("work_"): raws.add("workclass")
        elif c.startswith("occ_"): raws.add("occupation")
        elif c.startswith("hours_"): raws.add("hours-per-week")
        elif c.startswith("cap_gain_"): raws.add("capital-gain")
        elif c.startswith("cap_loss_"): raws.add("capital-loss")
        elif c.startswith("rel_"): raws.add("relationship")
    return [c for c in raws if c in X.columns]


def load_adult(args):
    X, y = _fetch_adult(folder = args.data_dir)
    S = _extended_sensitive_from_adult_df(X)

    sens_keys = getattr(args, "sens_keys", None)
    S = _filter_sensitive_by_keys(S, sens_keys)

    print(f"[adult] S columns after filter = {list(S.columns)}")

    mode = getattr(args, "x_sensitive", "concat")

    sens_raw_all = [c for c in [
        "sex","race","age","marital-status",
    ] if c in X.columns]

    if getattr(args, "sens_keys", None):
        raw_to_drop = _infer_raw_cols_from_S(S.columns, X)
    else:
        raw_to_drop = sens_raw_all

    if mode == "drop":
        X_for_fe = X.drop(columns=raw_to_drop, errors="ignore")
    elif mode == "concat":
        try:
            raw_to_drop = _infer_raw_cols_from_S(S.columns, X)
        except Exception:
            raw_to_drop = sens_raw_all
        X_for_fe = X.drop(columns=[c for c in raw_to_drop if c in X.columns])
    else:
        X_for_fe = X.copy()

    cat = X_for_fe.select_dtypes(include=["object"]).columns.tolist()
    num = X_for_fe.select_dtypes(exclude=["object"]).columns.tolist()

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = pd.DataFrame(enc.fit_transform(X_for_fe[cat]))
    X_cat.columns = enc.get_feature_names_out(cat)

    scaler = StandardScaler()
    X_num = pd.DataFrame(scaler.fit_transform(X_for_fe[num]), columns=num)

    Xp = pd.concat([X_num, X_cat], axis=1)

    if mode == "concat":
        Xp = pd.concat([Xp.reset_index(drop=True), S.reset_index(drop=True)], axis=1)
        Xp = Xp.loc[:, ~Xp.columns.duplicated()]

    

    # split
    seed = getattr(args, "seed", 42)
    X_tr, X_te, y_tr, y_te, S_tr, S_te = train_test_split(
        Xp, y, S, test_size=0.2, random_state=seed, stratify=y
    )
    X_tr, X_va, y_tr, y_va, S_tr, S_va = train_test_split(
        X_tr, y_tr, S_tr, test_size=0.2, random_state=seed, stratify=y_tr
    )
    

    print(f"[adult] final S_train cols = {list(S_tr.columns)}")

    return dict(
        X_train=X_tr, y_train=y_tr, S_train=S_tr,
        X_val=X_va,   y_val=y_va,   S_val=S_va,
        X_test=X_te,  y_test=y_te,  S_test=S_te,
        type="tabular",
        dataset="adult", 
        meta=dict(
            x_sensitive_mode=mode,
            sens_keys=(sens_keys if sens_keys is not None else "default"),
            used_S_cols=list(S.columns),
            dropped_cols=(raw_to_drop if mode=="drop" else []),
        ),
    )

def load_sparse_adult(args):
    
    X, y = _fetch_adult(folder = args.data_dir)
    S = _extended_sensitive_from_adult_df(X)

    sens_keys = getattr(args, "sens_keys", None)
    S = _filter_sensitive_by_keys(S, sens_keys)

    print(f"[adult-sparse] S columns after filter = {list(S.columns)}")

    mode = getattr(args, "x_sensitive", "concat")

    sens_raw_all = [c for c in ["sex","race","age","marital-status"] if c in X.columns]
    if getattr(args, "sens_keys", None):
        raw_to_drop = _infer_raw_cols_from_S(S.columns, X)
    else:
        raw_to_drop = sens_raw_all

    if mode == "drop":
        X_for_fe = X.drop(columns=raw_to_drop, errors="ignore")
    elif mode == "concat":
        try:
            raw_to_drop = _infer_raw_cols_from_S(S.columns, X)
        except Exception:
            raw_to_drop = sens_raw_all
        X_for_fe = X.drop(columns=[c for c in raw_to_drop if c in X.columns])
    else:
        X_for_fe = X.copy()

    cat = X_for_fe.select_dtypes(include=["object"]).columns.tolist()
    num = X_for_fe.select_dtypes(exclude=["object"]).columns.tolist()

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = pd.DataFrame(enc.fit_transform(X_for_fe[cat]))
    X_cat.columns = enc.get_feature_names_out(cat)

    scaler = StandardScaler()
    X_num = pd.DataFrame(scaler.fit_transform(X_for_fe[num]), columns=num)

    Xp = pd.concat([X_num, X_cat], axis=1)

    if mode == "concat":
        Xp = pd.concat([Xp.reset_index(drop=True), S.reset_index(drop=True)], axis=1)
        Xp = Xp.loc[:, ~Xp.columns.duplicated()]

    seed = int(getattr(args, "seed", 42))
    sparse_seed = int(getattr(args, "sparse_seed", seed))
    n_groups_target = int(getattr(args, "sparse_n_groups", 5))
    sparse_min = int(getattr(args, "sparse_min", 40))
    sparse_max = int(getattr(args, "sparse_max", 60))
    group_mode = str(getattr(args, "sparse_group_mode", "joint")).lower()

    assert sparse_min >= 1 and sparse_max >= sparse_min, "sparse_min/max 설정 확인"

    if S.shape[1] == 0:
        print("[adult-sparse][WARN] S has 0 columns; skipping sparsification.")
        S_sparse = S.copy()
        Xp_sparse = Xp.copy()
        y_sparse = np.asarray(y)
        sparsified_info = {"applied": False, "selected_groups": []}
    else:
        rng = np.random.default_rng(sparse_seed)

        if group_mode != "joint":
            print(f"[adult-sparse][WARN] unsupported sparse_group_mode={group_mode}; fallback to 'joint'")
            group_mode = "joint"


        grp_counts = S.groupby(list(S.columns)).size()             
        candidates = grp_counts[grp_counts > sparse_max]
        n_cand = int(candidates.shape[0])

        if n_cand == 0:
            print("[adult-sparse][WARN] no subgroup has support > sparse_max; skipping sparsification.")
            selected_keys = []
        else:
            pick = min(n_groups_target, n_cand)
            cand_keys = list(candidates.index) 
            sel_idx = rng.choice(np.arange(n_cand), size=pick, replace=False)
            selected_keys = [cand_keys[i] for i in sel_idx]

        drop_mask = np.zeros(len(S), dtype=bool)
        indices_map = S.groupby(list(S.columns)).groups 

        selected_summary = []
        for key in selected_keys:
            idxs = np.asarray(indices_map[key], dtype=int)
            before = idxs.size
            target = int(rng.integers(low=sparse_min, high=sparse_max+1))
            keep = np.sort(rng.choice(idxs, size=target, replace=False))
            to_drop = np.setdiff1d(idxs, keep, assume_unique=False)
            drop_mask[to_drop] = True
            selected_summary.append({"key": key, "before": int(before), "after": int(target)})

        kept_mask = ~drop_mask
        if drop_mask.any():
            Xp_sparse = Xp.iloc[kept_mask].reset_index(drop=True)
            S_sparse  = S.iloc[kept_mask].reset_index(drop=True)
            y_sparse  = np.asarray(y)[kept_mask]
        else:
            Xp_sparse = Xp.copy()
            S_sparse  = S.copy()
            y_sparse  = np.asarray(y)

        print(f"[adult-sparse] sparsified {len(selected_keys)} subgroups; "
              f"dropped {int(drop_mask.sum())} rows; final N={Xp_sparse.shape[0]}")

        sparsified_info = {
            "applied": True,
            "selected_groups": selected_summary,
            "sparse_min": sparse_min,
            "sparse_max": sparse_max,
            "n_groups_target": n_groups_target,
        }

    # shrink
    shrink_frac = 0.1
    shrink_seed = 42
    print("seed, shrink seed: ", seed, ", ", shrink_seed)
    if shrink_frac != 1.0:
        Xp_sparse, y_sparse, S_sparse, shrink_info = shrink_smallest_by_global_frac(
            Xp_sparse, y_sparse, S_sparse, frac=shrink_frac, seed=shrink_seed
        )
    else:
        shrink_info = {"shrink_applied": False}

    # split
    X_tr, X_te, y_tr, y_te, S_tr, S_te = train_test_split(
        Xp_sparse, y_sparse, S_sparse, test_size=0.2, random_state=seed, stratify=y_sparse
    )
    X_tr, X_va, y_tr, y_va, S_tr, S_va = train_test_split(
        X_tr, y_tr, S_tr, test_size=0.2, random_state=seed, stratify=y_tr
    )

    print(f"[adult-sparse] final S_train cols = {list(S_tr.columns)}")

    return dict(
        X_train=X_tr, y_train=y_tr, S_train=S_tr,
        X_val=X_va,   y_val=y_va,   S_val=S_va,
        X_test=X_te,  y_test=y_te,  S_test=S_te,
        type="tabular",
        dataset="adult_sparse",
        meta=dict(
            x_sensitive_mode=mode,
            sens_keys=(sens_keys if sens_keys is not None else "default"),
            used_S_cols=list(S.columns),
            dropped_cols=(raw_to_drop if mode=="drop" else []),
            sparsified=sparsified_info,
            shrink_info=shrink_info,
        ),
    )


