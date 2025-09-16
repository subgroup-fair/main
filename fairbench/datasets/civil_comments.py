# fairbench/datasets/civilcomments_wilds.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from ..utils.trainval import split_train_val
from .shrink import shrink_smallest_by_global_frac

import numpy as np, random, torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# --- 민감 변수 프리셋(프로젝트 전반과 일관성 유지를 위해 둠; 본 파일에서는 직접 구성) ---
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

def _build_civilcomments_S(df: pd.DataFrame) -> pd.DataFrame:
    """
    CivilComments(WILDS) 가공 규칙에 맞춰 민감 속성 매트릭스 S (0/1) 생성.
    - Gender:  male, female, gen_other(= transgender, other_gender, heterosexual, homosexual_gay_or_lesbian, bisexual, other_sexual_orientation, LGBTQ 중 1개 이상)
               정확히 한 카테고리만 1인 행만 유지
    - Religion: christian, jewish, muslim, rel_other(= hindu, buddhist, atheist, other_religion, other_religions 중 1개 이상)
               정확히 한 카테고리만 1인 행만 유지
    - Race: black, white, asian, race_other(= latino, other_race_or_ethnicity, asian_latino_etc 중 1개 이상)
               정확히 한 카테고리만 1인 행만 유지
    """
    # 열 존재 여부를 안전하게 체크
    def safe_cols(cands):
        return [c for c in cands if c in df.columns]

    # Gender
    gender_col = safe_cols(['male', 'female', 'gen_other'])
    gender_etc = safe_cols([
        'transgender', 'other_gender',
        'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual',
        'other_sexual_orientation', 'LGBTQ'
    ])

    # 없는 경우를 대비해 0 컬럼 채우기
    for c in ['male','female','gen_other'] + gender_etc:
        if c not in df.columns:
            df[c] = 0

    df['gen_other'] = (df[gender_etc].sum(axis=1) > 0).astype(int)
    gender_col = ['male','female','gen_other']
    df_gender_mask = ((df[gender_col] > 0).sum(axis=1) == 1)
    # 이후 이 마스크는 최종 필터에 반영

    # Religion
    religion_col = safe_cols(['christian', 'jewish', 'muslim', 'rel_other'])
    religion_etc = safe_cols(['hindu', 'buddhist', 'atheist', 'other_religion', 'other_religions'])
    for c in ['christian','jewish','muslim','rel_other'] + religion_etc:
        if c not in df.columns:
            df[c] = 0
    df['rel_other'] = (df[religion_etc].sum(axis=1) > 0).astype(int)
    religion_col = ['christian','jewish','muslim','rel_other']
    df_religion_mask = ((df[religion_col] > 0).sum(axis=1) == 1)

    # Race
    race_col = safe_cols(['black', 'white', 'asian', 'race_other'])
    race_etc = safe_cols(['latino', 'other_race_or_ethnicity', 'asian_latino_etc'])
    for c in ['black','white','asian','race_other'] + race_etc:
        if c not in df.columns:
            df[c] = 0
    df['race_other'] = (df[race_etc].sum(axis=1) > 0).astype(int)
    race_col = ['black','white','asian','race_other']
    df_race_mask = ((df[race_col] > 0).sum(axis=1) == 1)

    # 세 그룹 모두 정확히 1-hot인 행만 유지
    mask = df_gender_mask & df_religion_mask & df_race_mask
    dfS = df.loc[mask, gender_col + religion_col + race_col].copy()
    dfS = (dfS > 0).astype(int)

    return dfS, mask, dict(
        gender_cols=gender_col,
        religion_cols=religion_col,
        race_cols=race_col
    )

def _prepare_X(
    df: pd.DataFrame,
    drop_cols: list
) -> pd.DataFrame:
    """
    (라벨/민감변수) 외의 나머지를 특징으로 변환.
    - 수치: 중앙값 대치 → 스케일링
    - 범주: 원핫. 단, 극고유도(object nunique > 1000) 및 매우 긴 텍스트(예: 'comment_text')는 과적합/메모리 폭주 방지를 위해 제외
    - 모든 특징이 소거되는 경우 bias(상수 1) 특성 추가
    """
    X_for_fe = df.drop(columns=drop_cols, errors="ignore").copy()

    # 매우 긴 텍스트 컬럼 자동 제외
    high_card_obj = []
    for c in X_for_fe.select_dtypes(include=["object"]).columns.tolist():
        nunq = X_for_fe[c].nunique(dropna=True)
        if nunq > 1000 or c.lower() in {"comment_text", "text", "raw_text"}:
            high_card_obj.append(c)
    if high_card_obj:
        X_for_fe = X_for_fe.drop(columns=high_card_obj, errors="ignore")

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
        X_cat = pd.DataFrame(X_cat_arr, columns=enc.get_feature_names_out(cat_cols), index=X_for_fe.index)
        X_raw = pd.concat([X_num, X_cat], axis=1)
    else:
        X_raw = X_num

    # 안전 가드
    X_raw = X_raw.apply(pd.to_numeric, errors="coerce")
    X_raw = X_raw.fillna(X_raw.median(numeric_only=True))

    # 아무 특징도 없으면 bias 추가
    if X_raw.shape[1] == 0:
        X_raw = pd.DataFrame({"bias": np.ones(len(X_raw), dtype=float)}, index=X_for_fe.index)

    # 스케일링
    scaler = StandardScaler()
    Xp = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns, index=X_raw.index)
    return Xp

def load_civilcomments(args):
    """
    필요 파일:
      /data/share/wilds/civilcomments/processed_data/all_sensitive_data.csv
      (기본값: args.data_dir 로 지정된 폴더 안의 'all_sensitive_data.csv')

    옵션(args):
      --data_dir <dir>            (default: "/data/share/wilds/civilcomments/processed_data/")
      --seed <int>                (default: 42)
      --x_sensitive {drop,keep,concat}   (default: concat)
      --tox_threshold <float>     (default: 0.4)  # toxicity 이진화 임계값
      --shrink_smallest_frac <float in (0,1]> (default: 1.0)
      --shrink_seed <int>         (default: seed)
    """
    # 경로/인자
    data_dir = getattr(args, "data_dir", "/data/share/toxic/")
    path = data_dir.rstrip("/") + "/all_data_with_identities.csv"

    seed = getattr(args, "seed", 42)
    x_mode = getattr(args, "x_sensitive", "concat")
    tox_thr = float(getattr(args, "tox_threshold", 0.4))

    # 로드
    df0 = pd.read_csv(path)

    # --- 라벨 생성 ---
    # label_col = ['toxicity']
    if "toxicity" not in df0.columns:
        raise KeyError("Input CSV must contain a 'toxicity' column.")
    y_bin = (df0["toxicity"] > tox_thr).astype(int)

    # --- 민감 S 구성(및 유효 행 필터) ---
    S_df, mask_valid, sens_groups = _build_civilcomments_S(df0)
    df = df0.loc[mask_valid].reset_index(drop=True)
    y = y_bin.loc[mask_valid].to_numpy()

    # --- X 구성 ---
    # S 구성에 사용한 모든 민감열 + 라벨은 제외하고 특징 생성
    sens_used_cols = sens_groups["gender_cols"] + sens_groups["religion_cols"] + sens_groups["race_cols"]
    drop_for_X = ["toxicity"] + sens_used_cols
    Xp = _prepare_X(df, drop_cols=drop_for_X)

    # x_sensitive 처리
    if x_mode == "drop":
        X_use = Xp
    elif x_mode == "keep":
        # keep은 X는 S 쓰지 않고 그대로, S는 별도로 반환
        X_use = Xp
    elif x_mode == "concat":
        # f(x,s)
        X_use = pd.concat([Xp.reset_index(drop=True),
                           S_df.reset_index(drop=True)], axis=1)
    else:
        raise ValueError(f"Unknown x_sensitive mode: {x_mode}")

    # ===== (선택) 소표본 축소 =====
    shrink_frac = float(getattr(args, "shrink_smallest_frac", 1.0))
    shrink_seed = getattr(args, "shrink_seed", seed)
    if shrink_frac != 1.0:
        X_use, y, S_df, shrink_info = shrink_smallest_by_global_frac(
            X_use, y, S_df, frac=shrink_frac, seed=shrink_seed
        )
    else:
        shrink_info = {"shrink_applied": False}

    # ===== Split: train/val/test =====
    X_tr, X_te, y_tr, y_te, S_tr, S_te = train_test_split(
        X_use, y, S_df, test_size=0.2, random_state=seed, stratify=y
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
            target_col="toxicity",
            tox_threshold=tox_thr,
            x_sensitive_mode=x_mode,
            sens_groups=sens_groups,
            n_rows_raw=len(df0),
            n_rows_valid=len(df),
            shrink_info=shrink_info,
            note="Rows retained only if exactly-one-hot within each of gender/religion/race."
        )
    )
