# fairbench/datasets/dutch.py
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ..utils.trainval import split_train_val
from .shrink import shrink_smallest_by_global_frac

# 민감 변수 프리셋 (전처리본 별 열 이름이 다를 수 있어 후보를 넓게)
_PRESET_SENS = {
    "auto": [
        "sex", "gender",
        "age",
        "household_size", "hh_size",
        "education_num", "education_level", "education",
        "ethnicity", "citizenship",
        "marital_status", "partner",
        "income", "income_high",
    ],
    "basic": ["sex", "age", "education_num", "income"],
}

def _build_sensitive_df_from_df(df: pd.DataFrame, sens_keys, sens_thresh=0.5):
    """
    df: 전체 데이터프레임
    sens_keys: None/'auto'/'basic' 또는 'col1,col2,...' 문자열/리스트
    sens_thresh: 수치형 이진화 분위수 (0~1), 기본 0.5(중앙값)
    return: (S_df, used_raw_cols)
    """
    if sens_keys is None:
        keys = _PRESET_SENS["auto"]
    elif isinstance(sens_keys, str):
        key = sens_keys.strip().lower()
        if key in _PRESET_SENS:
            keys = _PRESET_SENS[key]
        else:
            keys = [s.strip() for s in sens_keys.split(",") if s.strip()]
    else:
        keys = [str(s) for s in sens_keys]

    cols = set(df.columns)
    S, used = {}, []

    for k in keys:
        if k not in cols:
            continue
        col = df[k]
        uniq = pd.unique(col.dropna())

        # 1) 이미 0/1 이진? (숫자로 안전하게 해석될 때만)
        uniq_num = pd.to_numeric(pd.Series(uniq, dtype="object"), errors="coerce")
        if len(uniq) <= 2 and not uniq_num.isna().any() and set(uniq_num.astype(int)).issubset({0, 1}):
            S[k if k.endswith(("_bin", "_bool")) else f"{k}_bin"] = pd.to_numeric(col, errors="coerce").fillna(0).astype(int)
            used.append(k)
            continue

        # 2) object 이면서 정확히 2개 카테고리 → 안전한 문자열 이진 매핑
        if len(uniq) == 2 and col.dtype == "object":
            col_l = col.astype(str).str.strip().str.lower()
            tokens = set(col_l.unique())
            pos_candidates = {"1", "true", "yes", "y", "male", "m"}
            inter = tokens & pos_candidates
            if len(inter) > 0:
                pos = list(inter)[0]
            else:
                # fallback: 등장 빈도 최다값을 1로
                pos = col_l.value_counts().idxmax()
            S[f"{k}_bin"] = (col_l == pos).astype(int)
            used.append(k)
            continue

        # 3) 수치형 → 분위수 임계 초과를 1로
        if np.issubdtype(col.dtype, np.number):
            col_num = pd.to_numeric(col, errors="coerce")
            thr = float(col_num.quantile(sens_thresh))
            S[f"{k}_high"] = (col_num > thr).astype(int)
            used.append(k)
            continue

        # 4) 그 외(object, 다범주) → 최빈값 one-vs-rest
        if col.dtype == "object":
            col_s = col.astype(str).str.strip()
            mv = col_s.mode(dropna=True)
            if len(mv) > 0:
                top = str(mv.iloc[0])
                S[f"{k}_is_{top}"] = (col_s == top).astype(int)
                used.append(k)
            continue

    S_df = pd.DataFrame(S).astype(int) if len(S) > 0 else pd.DataFrame()
    return S_df, used

def load_dutch(args):
    """
    필요 파일:
      data/raw/dutch.csv  (예: https://github.com/tailequy/fairness_dataset/... )
    옵션(args):
      --x_sensitive {drop,keep,concat}   (default: drop)
      --sens_keys   <preset or comma list>  (default: auto)
      --sens_thresh <float in 0..1> (default: 0.5)
      --dutch_path  <path> (default: data/raw/dutch.csv)
    """
    path = getattr(args, "data_dir", "data/raw/dutch.csv")
    df = pd.read_csv(path + "dutch.csv")

    seed = getattr(args, "seed", 42)
    x_mode = getattr(args, "x_sensitive", "drop")
    sens_keys = getattr(args, "sens_keys", "auto")
    sens_thresh = float(getattr(args, "sens_thresh", 0.5))

    # (1) 타겟 결정
    if "occupation_label" in df.columns:
        target_col = "occupation_label"
        y = df[target_col].astype(int).values
    elif "income_high" in df.columns:
        target_col = "income_high"
        y = df[target_col].astype(int).values
    else:
        target_col = df.columns[-1]
        y_col = df[target_col]
        # 이진으로 캐스팅 가능하면 그대로, 아니면 중앙값 기준 이진화
        y_try = pd.to_numeric(y_col, errors="coerce")
        if y_try.dropna().isin([0, 1]).all():
            y = y_try.fillna(0).astype(int).values
        else:
            if np.issubdtype(y_col.dtype, np.number):
                thr = float(pd.to_numeric(y_col, errors="coerce").quantile(0.5))
                y = (pd.to_numeric(y_col, errors="coerce") > thr).astype(int).values
            else:
                # 범주일 경우 최빈값 vs 나머지
                top = y_col.astype(str).str.strip().mode(dropna=True)
                top = str(top.iloc[0]) if len(top) > 0 else ""
                y = (y_col.astype(str).str.strip() == top).astype(int).values

    # (2) 민감 S 생성
    S_df, used_sens_raw = _build_sensitive_df_from_df(df, sens_keys=sens_keys, sens_thresh=sens_thresh)
    if S_df.shape[1] == 0:
        if "sex" in df.columns:
            S_df = pd.DataFrame({"sex_bin": pd.to_numeric(df["sex"], errors="coerce").fillna(0).astype(int)})
            used_sens_raw = used_sens_raw + ["sex"]
        elif "gender" in df.columns:
            S_df = pd.DataFrame({"gender_bin": pd.to_numeric(df["gender"], errors="coerce").fillna(0).astype(int)})
            used_sens_raw = used_sens_raw + ["gender"]
        else:
            S_df = pd.DataFrame({"dummy_s": np.zeros(len(df), dtype=int)})

    # (3) X 원본 구성
    X_raw = df.drop(columns=[target_col], errors="ignore")

    # x_sensitive 처리
    if x_mode == "drop":
        X_for_fe = X_raw.drop(columns=used_sens_raw, errors="ignore")
    else:
        X_for_fe = X_raw.copy()

    # (4) 수치/범주 전처리
    num_cols = X_for_fe.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_for_fe.select_dtypes(exclude=[np.number]).columns.tolist()

    # 수치 결측 보간
    X_num = X_for_fe[num_cols].copy()
    if len(num_cols) > 0:
        X_num = X_num.fillna(X_num.median(numeric_only=True))

    # 범주 원핫
    if len(cat_cols) > 0:
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_cat_arr = enc.fit_transform(X_for_fe[cat_cols].astype(str))
        X_cat = pd.DataFrame(X_cat_arr, columns=enc.get_feature_names_out(cat_cols))
        Xp = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    else:
        Xp = X_num.reset_index(drop=True)

    # 안전 가드: 숫자만 남기고 결측 보간
    Xp = Xp.apply(pd.to_numeric, errors="coerce")
    Xp = Xp.fillna(Xp.median(numeric_only=True))

    # 스케일링
    scaler = StandardScaler()
    Xp = pd.DataFrame(scaler.fit_transform(Xp), columns=Xp.columns)

    # concat 모드면 S를 X에 붙여서 f(x,s)
    if x_mode == "concat":
        Xp = pd.concat([Xp.reset_index(drop=True), S_df.reset_index(drop=True)], axis=1)

    # (4) 전처리 완료: Xp, y, S_df 준비됨

    # ===== NEW: 전체에서 축소 후 split =====
    shrink_frac = float(getattr(args, "shrink_smallest_frac", 1.0))
    shrink_seed = getattr(args, "shrink_seed", seed)
    if shrink_frac != 1.0:
        Xp, y, S_df, shrink_info = shrink_smallest_by_global_frac(
            Xp, y, S_df, frac=shrink_frac, seed=shrink_seed
        )
    else:
        shrink_info = {"shrink_applied": False}

    # (5) split
    X_tr, X_te, y_tr, y_te, S_tr, S_te = train_test_split(
        Xp, y, S_df, test_size=0.2, random_state=seed, stratify=y
    )
    X_tr, X_va, y_tr, y_va, S_tr, S_va = split_train_val(
        X_tr, y_tr, S_tr, val_size=0.2, seed=seed
    )

    return dict(
        X_train=X_tr, y_train=y_tr, S_train=S_tr,
        X_val=X_va,   y_val=y_va,   S_val=S_va,
        X_test=X_te,  y_test=y_te,  S_test=S_te,
        type="tabular",
        meta=dict(
            target_col=target_col,
            x_sensitive_mode=x_mode,
            used_sensitive_raw=used_sens_raw,
            sens_keys=(sens_keys if isinstance(sens_keys, str) else ",".join(sens_keys)),
            sens_thresh=sens_thresh,
        )
    )
