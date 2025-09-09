# fairbench/datasets/adult.py
import pandas as pd, numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ..utils.trainval import split_train_val
from .shrink import shrink_smallest_by_global_frac

def _fetch_adult(folder):
    import pandas as pd, numpy as np

    path = folder + "adult.csv"
    df = pd.read_csv(path)

    # 0) 가벼운 정리
    df.columns = [c.strip() for c in df.columns]

    # 1) 컬럼 별칭 → 표준명으로 통일 (소문자 비교)
    lower = {c.lower(): c for c in df.columns}

    def alias(old_keys, new_key):
        # old_keys: 소문자 후보들, new_key: 최종 컬럼명(정확히 이 이름을 쓰게 됨)
        for k in old_keys:
            if k in lower:
                src = lower[k]
                if new_key not in df.columns:
                    df.rename(columns={src: new_key}, inplace=True)
                # 이미 new_key가 있으면 그대로 두고 종료
                return

    # 성별: gender -> sex
    alias(["gender", "sex"], "sex")
    # 교육연차: educational-num -> education-num
    alias(["educational-num", "education_num", "educational_num", "education-num"], "education-num")
    # 근로시간: hours_per_week/hoursperweek -> hours-per-week
    alias(["hours_per_week", "hoursperweek", "hours-per-week"], "hours-per-week")
    # 자본이득/손실: 변형 → 표준 하이픈 이름
    alias(["capital_gain", "capitalgain", "capital-gain"], "capital-gain")
    alias(["capital_loss", "capitalloss", "capital-loss"], "capital-loss")
    # 출신국: native_country -> native-country
    alias(["native_country", "native-country"], "native-country")
    # 혼인상태: marital_status -> marital-status
    alias(["marital_status", "marital-status"], "marital-status")

    # 2) 타깃/특징 분리
    y = (df["income"].astype(str).str.strip() == ">50K").astype(int).values
    X = df.drop(columns=["income"]).copy()

    # 3) 문자열 정리
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

    # --- required binaries ---
    if has("sex"):
        s = X["sex"].astype(str).str.strip()
        S["sex_Male"] = (s == "Male").astype(int)

    if has("race"):
        r = X["race"].astype(str).str.strip()
        S["race_White"] = (r == "White").astype(int)  # White vs not-White (single binary)

    if has("age"):
        age = pd.to_numeric(X["age"], errors="coerce")
        S["age_ge_40"] = (age >= 40).astype(int)  # ≥40 vs <40 (single binary)

    if has("marital-status"):
        ms = X["marital-status"].astype(str).str.strip()
        S["married"] = ms.isin({"Married-civ-spouse", "Married-AF-spouse"}).astype(int)  # married vs not

    # --- other binaries (이미 모두 0/1 형태) ---
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
    """
    sens_keys로 S_df 컬럼 선택 (대소문자 무시 + 동의어 지원 + 강제 포함 보정).
    지원:
      - 정확 이름 (예: 'sex_Male')
      - 접두사* (예: 'sex_*')
      - 원본 키/동의어 (예: 'sex','gender' → 'sex_*'; 'marital-status','marital','married' → 'married')
      - 제외: '-race_*'
      - 특수: 'all'/'*' → 전체 유지
    """
    import re

    if sens_keys is None:
        return S_df

    # 문자열/리스트 모두 허용
    if isinstance(sens_keys, str):
        tokens = [t.strip() for t in sens_keys.split(",") if t.strip()]
    elif isinstance(sens_keys, (list, tuple, set)):
        tokens = [str(t).strip() for t in sens_keys if str(t).strip()]
    else:
        return S_df
    if not tokens:
        return S_df

    cols = list(S_df.columns)
    cols_lc = {c.lower(): c for c in cols}  # lc → original

    def _starts_with_icase(c: str, pref: str) -> bool:
        return c.lower().startswith(pref.lower())

    # 동의어/원본키 → 접두사/단일명 (소문자 키)
    raw2prefix_icase = {
        "sex": "sex_",
        "gender": "sex_",
        "race": "race_",
        "age": "age_",
        "marital-status": "married",   # 단일명
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

        # (1) 정확 명칭
        if t_lc in cols_lc:
            matched.add(cols_lc[t_lc])

        # (2) 접두사*
        if t.endswith("*"):
            pref = t[:-1]
            matched |= {c for c in cols if _starts_with_icase(c, pref)}

        # (3) 원본 키/동의어 → 접두사/단일명
        if t_lc in raw2prefix_icase:
            pref = raw2prefix_icase[t_lc]
            if pref.endswith("_"):  # 접두사
                matched |= {c for c in cols if _starts_with_icase(c, pref)}
            else:                   # 단일명 (예: married)
                if pref.lower() in cols_lc:
                    matched.add(cols_lc[pref.lower()])

        # (4) regex:
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

    # 🔒 강제 포함 보정: 사용자가 sex/gender를 요청했는데 매칭이 0개라면 'sex_Male'을 포함
    toks_lc = [t.lower().lstrip("-") for t in tokens]
    wants_sex = any(t in ("sex", "gender", "sex_*", "gender_*") for t in toks_lc)
    if wants_sex and not any(c.lower().startswith("sex_") for c in selected):
        if "sex_Male" in cols:
            selected.add("sex_Male")
        else:
            # 혹시 소문자 변형으로 존재할 때도 케어
            for c in cols:
                if c.lower() == "sex_male":
                    selected.add(c)
                    break

    if not selected:
        # 안전 fallback
        if "sex_Male" in cols:
            selected = {"sex_Male"}
        else:
            selected = {cols[0]} if cols else set()

    ordered = [c for c in cols if c in selected]

    # 디버그 로그
    print(f"[sens_keys] tokens={tokens}")
    print(f"[sens_keys] selected={ordered}")

    return S_df[ordered]


def _infer_raw_cols_from_S(selected_S_cols, X: pd.DataFrame) -> list:
    """
    선택된 S 컬럼들로부터 어떤 '원본(raw) 컬럼'을 사용했는지 추정.
    x_sensitive='drop'일 때 해당 raw 컬럼만 제거하도록 사용.
    """
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

    # 🔎 확인 로그 (반드시 찍힘)
    print(f"[adult] S columns after filter = {list(S.columns)}")

    # (선택) 기대 키를 줬다면 보장용 어설션 (원하면 주석 해제)
    # expected = {"sex_Male","race_White","age_ge_40","married"}
    # if sens_keys and all(k in str(sens_keys).lower() for k in ["sex","race","age","marital"]):
    #     assert set(S.columns) == expected, f"Expected {expected}, got {set(S.columns)}"

    mode = getattr(args, "x_sensitive", "drop")

    sens_raw_all = [c for c in [
        "sex","race","age","marital-status",
    ] if c in X.columns]

    if getattr(args, "sens_keys", None):
        raw_to_drop = _infer_raw_cols_from_S(S.columns, X)
    else:
        raw_to_drop = sens_raw_all

    if mode == "drop":
        # 💡 sens_keys를 썼다면 raw_to_drop만 드롭하는 게 일관적
        X_for_fe = X.drop(columns=raw_to_drop, errors="ignore")
    elif mode == "concat":
        try:
            raw_to_drop = _infer_raw_cols_from_S(S.columns, X)
        except Exception:
            raw_to_drop = sens_raw_all
        X_for_fe = X.drop(columns=[c for c in raw_to_drop if c in X.columns])
    else:
        X_for_fe = X.copy()

    # --- FE ---
    cat = X_for_fe.select_dtypes(include=["object"]).columns.tolist()
    num = X_for_fe.select_dtypes(exclude=["object"]).columns.tolist()

    enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
    X_cat = pd.DataFrame(enc.fit_transform(X_for_fe[cat]))
    X_cat.columns = enc.get_feature_names_out(cat)

    scaler = StandardScaler()
    X_num = pd.DataFrame(scaler.fit_transform(X_for_fe[num]), columns=num)

    Xp = pd.concat([X_num, X_cat], axis=1)

    if mode == "concat":
        Xp = pd.concat([Xp.reset_index(drop=True), S.reset_index(drop=True)], axis=1)
        Xp = Xp.loc[:, ~Xp.columns.duplicated()]

    # ===== 전체 shrink 후 split =====
    seed = getattr(args, "seed", 42)
    shrink_frac = float(getattr(args, "shrink_smallest_frac", 1.0))
    shrink_seed = getattr(args, "shrink_seed", seed)
    if shrink_frac != 1.0:
        # ✅ S_df → S로 수정
        Xp, y, S, shrink_info = shrink_smallest_by_global_frac(
            Xp, y, S, frac=shrink_frac, seed=shrink_seed
        )
    else:
        shrink_info = {"shrink_applied": False}

    # --- split ---
    X_tr, X_te, y_tr, y_te, S_tr, S_te = train_test_split(
        Xp, y, S, test_size=0.2, random_state=seed, stratify=y
    )
    X_tr, X_va, y_tr, y_va, S_tr, S_va = split_train_val(
        X_tr, y_tr, S_tr, val_size=0.2, seed=seed
    )

    print(f"[adult] final S_train cols = {list(S_tr.columns)}")

    return dict(
        X_train=X_tr, y_train=y_tr, S_train=S_tr,
        X_val=X_va,   y_val=y_va,   S_val=S_va,
        X_test=X_te,  y_test=y_te,  S_test=S_te,
        type="tabular",
        dataset="adult",  # 있으면 CSV 경로도 깔끔
        meta=dict(
            x_sensitive_mode=mode,
            sens_keys=(sens_keys if sens_keys is not None else "default"),
            used_S_cols=list(S.columns),
            dropped_cols=(raw_to_drop if mode=="drop" else []),
        ),
    )

