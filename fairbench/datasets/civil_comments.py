# fairbench/datasets/civilcomments_wilds.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
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
    religion_col = safe_cols(['christian', 'rel_other'])
    religion_etc = safe_cols(['hindu', 'jewish', 'muslim', 'buddhist', 'atheist', 'other_religion', 'other_religions'])
    for c in ['christian','rel_other'] + religion_etc:
        if c not in df.columns:
            df[c] = 0
    df['rel_other'] = (df[religion_etc].sum(axis=1) > 0).astype(int)
    religion_col = ['christian','rel_other']
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

    # ✅ 우리가 쓰는 민감열(keep)과 나머지(제거 대상)까지 메타로 반환
    return dfS, mask, dict(
        gender_cols=gender_col,
        religion_cols=religion_col,
        race_cols=race_col,
        # 아래 3개는 “우리가 쓰지 않는 민감 원천열들”
        gender_extra=gender_etc,
        religion_extra=religion_etc,
        race_extra=race_etc,
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
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
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

# def load_civilcomments(args):
#     """
#     필요 파일:
#       /data/share/wilds/civilcomments/processed_data/all_sensitive_data.csv
#       (기본값: args.data_dir 로 지정된 폴더 안의 'all_sensitive_data.csv')

#     옵션(args):
#       --data_dir <dir>            (default: "/data/share/wilds/civilcomments/processed_data/")
#       --seed <int>                (default: 42)
#       --x_sensitive {drop,keep,concat}   (default: concat)
#       --tox_threshold <float>     (default: 0.4)  # toxicity 이진화 임계값
#       --shrink_smallest_frac <float in (0,1]> (default: 1.0)
#       --shrink_seed <int>         (default: seed)
#     """
#     # 경로/인자
#     data_dir = getattr(args, "data_dir", "/data/share/toxic/")
#     path = data_dir.rstrip("/") + "/all_data_with_identities.csv"

#     seed = getattr(args, "seed", 42)
#     x_mode = getattr(args, "x_sensitive", "concat")
#     tox_thr = float(getattr(args, "tox_threshold", 0.4))

#     # 로드
#     df0 = pd.read_csv(path)

#     # --- 라벨 생성 ---
#     # label_col = ['toxicity']
#     if "toxicity" not in df0.columns:
#         raise KeyError("Input CSV must contain a 'toxicity' column.")
#     y_bin = (df0["toxicity"] > tox_thr).astype(int)

#    # --- 민감 S 구성(및 유효 행 필터) ---
#     S_df, mask_valid, sens_groups = _build_civilcomments_S(df0)

#     df = df0.loc[mask_valid].reset_index(drop=True)
#     y  = (df0["toxicity"] > float(getattr(args, "tox_threshold", 0.4))).loc[mask_valid].astype(int).to_numpy()

#     # ✅ 우리가 사용하지 않는 민감 속성 원천열을 전부 제거
#     drop_unused_sens = (
#         sens_groups.get("gender_extra", [])
#         + sens_groups.get("religion_extra", [])
#         + sens_groups.get("race_extra", [])
#     )
#     # 혹시 있을지 모를 기타 정체성 열 패턴이 있으면 여기서 추가로 합쳐도 됨
#     df = df.drop(columns=drop_unused_sens, errors="ignore")

#     # --- X 구성 ---
#     sens_used_cols = sens_groups["gender_cols"] + sens_groups["religion_cols"] + sens_groups["race_cols"]
#     drop_for_X = ["toxicity"] + sens_used_cols  # 원천 X에서는 민감열 제외(모드에 따라 나중에 S_df만 붙임)
#     Xp = _prepare_X(df, drop_cols=drop_for_X)

#     # ===== [로버타 임베딩 결합] =====
#     import os
#     npy_path = os.path.join(getattr(args, "data_dir", "/data/share/toxic/"), "distillroberta_base_all_features.npy")
#     if os.path.exists(npy_path):
#         rb = np.load(npy_path)  # (N_raw, D) 또는 (N_valid, D)
#         n_raw, n_valid = len(df0), int(mask_valid.sum())
#         if rb.shape[0] == n_raw:
#             rb = rb[np.asarray(mask_valid.values, dtype=bool)]
#         elif rb.shape[0] == n_valid:
#             pass
#         else:
#             raise ValueError(f"Shape mismatch for Roberta feats: {rb.shape[0]} vs raw={n_raw}, valid={n_valid}")

#         from sklearn.preprocessing import StandardScaler
#         rb = StandardScaler().fit_transform(np.asarray(rb, dtype=np.float32))
#         X_rb = pd.DataFrame(rb, columns=[f"rb_{i}" for i in range(rb.shape[1])], index=df.index)
#         Xp = pd.concat([Xp.reset_index(drop=True), X_rb.reset_index(drop=True)], axis=1)
#     else:
#         print(f"[civil] roberta .npy not found: {npy_path} (skip)")

#     # x_sensitive 모드 처리
#     x_mode = getattr(args, "x_sensitive", "concat")
#     if x_mode == "drop":
#         X_use = Xp
#     elif x_mode == "keep":
#         X_use = Xp
#     elif x_mode == "concat":
#         # ✅ 여기서만 S_df를 붙여 사용. 원천 민감열은 이미 df에서 제거됨
#         X_use = pd.concat([Xp.reset_index(drop=True), S_df.reset_index(drop=True)], axis=1)
#     else:
#         raise ValueError(f"Unknown x_sensitive mode: {x_mode}")

#     print(f"[civil] X_use shape = {X_use.shape}  (x_sensitive='{x_mode}', S appended={x_mode=='concat'})")

    
#     # # ===== (선택) 소표본 축소 =====
#     # shrink_frac = float(getattr(args, "shrink_smallest_frac", 1.0))
#     # shrink_seed = getattr(args, "shrink_seed", seed)
#     # if shrink_frac != 1.0:
#     #     X_use, y, S_df, shrink_info = shrink_smallest_by_global_frac(
#     #         X_use, y, S_df, frac=shrink_frac, seed=shrink_seed
#     #     )
#     # else:
#     #     shrink_info = {"shrink_applied": False}

#     # ===== Split: train/val/test =====
#     X_tr, X_te, y_tr, y_te, S_tr, S_te = train_test_split(
#         X_use, y, S_df, test_size=0.2, random_state=seed, stratify=y
#     )
#     X_tr, X_va, y_tr, y_va, S_tr, S_va = split_train_val(
#         X_tr, y_tr, S_tr, val_size=0.2, seed=seed
#     )

#     return dict(
#         X_train=X_tr, y_train=y_tr, S_train=S_tr,
#         X_val=X_va,   y_val=y_va,   S_val=S_va,
#         X_test=X_te,  y_test=y_te,  S_test=S_te,
#         type="tabular",
#         meta=dict(
#             target_col="toxicity",
#             tox_threshold=tox_thr,
#             x_sensitive_mode=x_mode,
#             sens_groups=sens_groups,
#             n_rows_raw=len(df0),
#             n_rows_valid=len(df),
#             note="Rows retained only if exactly-one-hot within each of gender/religion/race.",
#             external_features=dict(path=npy_path, used=os.path.exists(npy_path))
#         )
#     )


def load_civilcomments(args):
    """
    캐시 CSV를 우선 사용. 없으면 전처리 후 생성.
    캐시 경로 기본값: /data/share/civilcomments.csv  (args.civil_cache_csv로 변경 가능)
    """
    # ---- 설정/경로 ----
    data_dir = getattr(args, "data_dir", "/data/share/FairLLM/civil/")
    raw_path = os.path.join(data_dir, "all_data_with_identities.csv")
    cache_csv = getattr(args, "civil_cache_csv", "/data/share/FairLLM/civil/civilcomments.csv")

    seed = getattr(args, "seed", 42)
    x_mode = getattr(args, "x_sensitive", "concat")
    tox_thr = float(getattr(args, "tox_threshold", 0.4))

    # S 열 이름(우리가 쓰는 것만)
    S_KEEP = [
        "male","female","gen_other",
        "christian","jewish","muslim","rel_other",
        "black","white","asian","race_other",
    ]

    # =========================
    # 1) 캐시가 있으면 바로 사용
    # =========================
    if os.path.exists(cache_csv):
        dfc = pd.read_csv(cache_csv)
        if "y" not in dfc.columns:
            raise KeyError(f"[civil] cached CSV missing 'y' column: {cache_csv}")

        S_cols = [c for c in S_KEEP if c in dfc.columns]
        if not S_cols:
            raise KeyError(f"[civil] cached CSV has no sensitive cols among {S_KEEP}")

        y = dfc["y"].astype(int).to_numpy()
        S_df = dfc[S_cols].astype(int).copy()
        Xp = dfc.drop(columns=["y"] + S_cols, errors="ignore").copy()

        print(f"[civil][cache] loaded: {cache_csv}")
        print(f"[civil][cache] rows={len(dfc):,}, Xp.shape={Xp.shape}, S.shape={S_df.shape}, y=1 rate={y.mean():.4f}")

    # =========================
    # 2) 없으면 전처리 후 캐시 저장
    # =========================
    else:
        # --- 로드 ---
        df0 = pd.read_csv(raw_path)
        print(f"[civil] raw: rows={len(df0):,}, cols={len(df0.columns)} (path={raw_path})")

        # --- 라벨 ---
        if "toxicity" not in df0.columns:
            raise KeyError("Input CSV must contain a 'toxicity' column.")
        y_bin = (df0["toxicity"] > tox_thr).astype(int)
        print(f"[civil] raw y=1 rate = {y_bin.mean():.4f}")

        # --- S / 유효행 마스크 ---
        S_df, mask_valid, sens_groups = _build_civilcomments_S(df0)
        n_valid = int(mask_valid.sum())
        print(f"[civil] valid rows (exact-1hot per group) = {n_valid:,} / {len(df0):,}  ({n_valid/len(df0):.2%})")

        # 사용 S만 필터
        S_df = S_df[[c for c in S_KEEP if c in S_df.columns]].astype(int)
        print(f"[civil] S kept = {list(S_df.columns)} (q={S_df.shape[1]})")

        # --- 유효행 선택 ---
        df = df0.loc[mask_valid].reset_index(drop=True)
        y  = y_bin.loc[mask_valid].to_numpy()
        print(f"[civil] after filter: rows={len(df):,}, y=1 rate={y.mean():.4f}")

        # 우리가 쓰지 않는 원천 민감열 제거
        drop_unused_sens = (
            sens_groups.get("gender_extra", [])
            + sens_groups.get("religion_extra", [])
            + sens_groups.get("race_extra", [])
        )
        if drop_unused_sens:
            df = df.drop(columns=drop_unused_sens, errors="ignore")

        # --- X(base) 생성 (원핫/스케일) ---
        sens_used_cols = S_df.columns.tolist()
        drop_for_X = ["toxicity"] + sens_used_cols
        Xp = _prepare_X(df, drop_cols=drop_for_X)
        print(f"[civil] X(base) shape = {Xp.shape}")

        # --- RoBERTa 임베딩 결합 ---
        npy_path = os.path.join(data_dir, "distillroberta_base_all_features.npy")
        if os.path.exists(npy_path):
            rb = np.load(npy_path)
            n_raw, n_valid = len(df0), int(mask_valid.sum())
            if rb.shape[0] == n_raw:
                rb = rb[np.asarray(mask_valid.values, dtype=bool)]
                src = "raw-aligned"
            elif rb.shape[0] == n_valid:
                src = "valid-aligned"
            else:
                raise ValueError(f"Roberta feats row mismatch: {rb.shape[0]} vs raw={n_raw}, valid={n_valid}")
            from sklearn.preprocessing import StandardScaler
            rb = StandardScaler().fit_transform(np.asarray(rb, dtype=np.float32))
            X_rb = pd.DataFrame(rb, columns=[f"rb_{i}" for i in range(rb.shape[1])], index=df.index)
            Xp = pd.concat([Xp.reset_index(drop=True), X_rb.reset_index(drop=True)], axis=1)
            print(f"[civil] +Roberta({src}) D={X_rb.shape[1]} → X shape={Xp.shape}")
        else:
            print(f"[civil] Roberta .npy not found: {npy_path} (skip)")

        # --- 캐시 CSV 저장 ---
        df_cache = pd.concat(
            [pd.DataFrame({"y": y}), S_df.reset_index(drop=True), Xp.reset_index(drop=True)],
            axis=1
        )
        os.makedirs(os.path.dirname(cache_csv) or ".", exist_ok=True)
        df_cache.to_csv(cache_csv, index=False)
        print(f"[civil] cached to CSV: {cache_csv} (rows={len(df_cache):,}, cols={len(df_cache.columns)})")

    # =========== X_use 구성 (훈련 입력) ===========
    if x_mode == "concat":
        X_use = pd.concat([Xp.reset_index(drop=True), S_df.reset_index(drop=True)], axis=1)
        s_appended = True
    else:
        X_use = Xp
        s_appended = False
    print(f"[civil] X_use shape = {X_use.shape}  (x_sensitive='{x_mode}', S appended={s_appended})")

    # ===== Split =====
    X_tr, X_te, y_tr, y_te, S_tr, S_te = train_test_split(
        X_use, y, S_df, test_size=0.2, random_state=seed, stratify=y
    )
    X_tr, X_va, y_tr, y_va, S_tr, S_va = split_train_val(
        X_tr, y_tr, S_tr, val_size=0.2, seed=seed
    )
    print(f"[civil] split: X_tr={X_tr.shape}, X_va={X_va.shape}, X_te={X_te.shape}")
    print(f"[civil] y=1 rate: train={y_tr.mean():.4f}, val={y_va.mean():.4f}, test={y_te.mean():.4f}")
    print(f"[civil] S_train cols = {list(S_tr.columns)}")

    return dict(
        X_train=X_tr, y_train=y_tr, S_train=S_tr,
        X_val=X_va,   y_val=y_va,   S_val=S_va,
        X_test=X_te,  y_test=y_te,  S_test=S_te,
        type="tabular",
        meta=dict(
            target_col="toxicity",
            tox_threshold=tox_thr,
            x_sensitive_mode=x_mode,
            n_rows_cached=len(X_use),
            cache_csv=cache_csv,
        ),
    )


def load_civilcomments2(args):
    """
    캐시 CSV를 우선 사용. 없으면 전처리 후 생성.
    캐시 경로 기본값: /data/share/civilcomments.csv  (args.civil_cache_csv로 변경 가능)
    """
    # ---- 설정/경로 ----
    data_dir = getattr(args, "data_dir", "/data/share/FairLLM/civil/")
    raw_path = os.path.join(data_dir, "all_data_with_identities.csv")
    cache_csv = getattr(args, "civil_cache_csv", "/data/share/FairLLM/civil/civilcomments2.csv")

    seed = getattr(args, "seed", 42)
    x_mode = getattr(args, "x_sensitive", "concat")
    tox_thr = float(getattr(args, "tox_threshold", 0.4))

    # S 열 이름(우리가 쓰는 것만)
    S_KEEP = [
        "male","female","gen_other",
        "christian","jewish","muslim","rel_other",
        "black","white","asian","race_other",
    ]

    # =========================
    # 1) 캐시가 있으면 바로 사용
    # =========================
    if os.path.exists(cache_csv):
        dfc = pd.read_csv(cache_csv)
        if "y" not in dfc.columns:
            raise KeyError(f"[civil] cached CSV missing 'y' column: {cache_csv}")

        S_cols = [c for c in S_KEEP if c in dfc.columns]
        if not S_cols:
            raise KeyError(f"[civil] cached CSV has no sensitive cols among {S_KEEP}")

        y = dfc["y"].astype(int).to_numpy()
        S_df = dfc[S_cols].astype(int).copy()
        Xp = dfc.drop(columns=["y"] + S_cols, errors="ignore").copy()

        print(f"[civil][cache] loaded: {cache_csv}")
        print(f"[civil][cache] rows={len(dfc):,}, Xp.shape={Xp.shape}, S.shape={S_df.shape}, y=1 rate={y.mean():.4f}")

    # =========================
    # 2) 없으면 전처리 후 캐시 저장
    # =========================
    else:
        # --- 로드 ---
        df0 = pd.read_csv(raw_path)
        print(f"[civil] raw: rows={len(df0):,}, cols={len(df0.columns)} (path={raw_path})")

        # --- 라벨 ---
        if "toxicity" not in df0.columns:
            raise KeyError("Input CSV must contain a 'toxicity' column.")
        y_bin = (df0["toxicity"] > tox_thr).astype(int)
        print(f"[civil] raw y=1 rate = {y_bin.mean():.4f}")

        # --- S / 유효행 마스크 ---
        S_df, mask_valid, sens_groups = _build_civilcomments_S(df0)
        n_valid = int(mask_valid.sum())
        print(f"[civil] valid rows (exact-1hot per group) = {n_valid:,} / {len(df0):,}  ({n_valid/len(df0):.2%})")

        # 사용 S만 필터
        S_df = S_df[[c for c in S_KEEP if c in S_df.columns]].astype(int)
        print(f"[civil] S kept = {list(S_df.columns)} (q={S_df.shape[1]})")

        # --- 유효행 선택 ---
        df = df0.loc[mask_valid].reset_index(drop=True)
        y  = y_bin.loc[mask_valid].to_numpy()
        print(f"[civil] after filter: rows={len(df):,}, y=1 rate={y.mean():.4f}")

        # 우리가 쓰지 않는 원천 민감열 제거
        drop_unused_sens = (
            sens_groups.get("gender_extra", [])
            + sens_groups.get("religion_extra", [])
            + sens_groups.get("race_extra", [])
        )
        if drop_unused_sens:
            df = df.drop(columns=drop_unused_sens, errors="ignore")

        # --- X(base) 생성 (원핫/스케일) ---
        sens_used_cols = S_df.columns.tolist()
        drop_for_X = ["toxicity"] + sens_used_cols
        Xp = _prepare_X(df, drop_cols=drop_for_X)
        print(f"[civil] X(base) shape = {Xp.shape}")

        # --- RoBERTa 임베딩 결합 ---
        npy_path = os.path.join(data_dir, "roberta_base_train_features.npy")
        if os.path.exists(npy_path):
            rb = np.load(npy_path)
            n_raw, n_valid = len(df0), int(mask_valid.sum())
            if rb.shape[0] == n_raw:
                rb = rb[np.asarray(mask_valid.values, dtype=bool)]
                src = "raw-aligned"
            elif rb.shape[0] == n_valid:
                src = "valid-aligned"
            else:
                raise ValueError(f"Roberta feats row mismatch: {rb.shape[0]} vs raw={n_raw}, valid={n_valid}")
            from sklearn.preprocessing import StandardScaler
            rb = StandardScaler().fit_transform(np.asarray(rb, dtype=np.float32))
            X_rb = pd.DataFrame(rb, columns=[f"rb_{i}" for i in range(rb.shape[1])], index=df.index)
            Xp = pd.concat([Xp.reset_index(drop=True), X_rb.reset_index(drop=True)], axis=1)
            print(f"[civil] +Roberta({src}) D={X_rb.shape[1]} → X shape={Xp.shape}")
        else:
            print(f"[civil] Roberta .npy not found: {npy_path} (skip)")

        # --- 캐시 CSV 저장 ---
        df_cache = pd.concat(
            [pd.DataFrame({"y": y}), S_df.reset_index(drop=True), Xp.reset_index(drop=True)],
            axis=1
        )
        os.makedirs(os.path.dirname(cache_csv) or ".", exist_ok=True)
        df_cache.to_csv(cache_csv, index=False)
        print(f"[civil] cached to CSV: {cache_csv} (rows={len(df_cache):,}, cols={len(df_cache.columns)})")

    # =========== X_use 구성 (훈련 입력) ===========
    if x_mode == "concat":
        X_use = pd.concat([Xp.reset_index(drop=True), S_df.reset_index(drop=True)], axis=1)
        s_appended = True
    else:
        X_use = Xp
        s_appended = False
    print(f"[civil] X_use shape = {X_use.shape}  (x_sensitive='{x_mode}', S appended={s_appended})")

    # ===== Split =====
    X_tr, X_te, y_tr, y_te, S_tr, S_te = train_test_split(
        X_use, y, S_df, test_size=0.2, random_state=seed, stratify=y
    )
    X_tr, X_va, y_tr, y_va, S_tr, S_va = split_train_val(
        X_tr, y_tr, S_tr, val_size=0.2, seed=seed
    )
    print(f"[civil] split: X_tr={X_tr.shape}, X_va={X_va.shape}, X_te={X_te.shape}")
    print(f"[civil] y=1 rate: train={y_tr.mean():.4f}, val={y_va.mean():.4f}, test={y_te.mean():.4f}")
    print(f"[civil] S_train cols = {list(S_tr.columns)}")

    return dict(
        X_train=X_tr, y_train=y_tr, S_train=S_tr,
        X_val=X_va,   y_val=y_va,   S_val=S_va,
        X_test=X_te,  y_test=y_te,  S_test=S_te,
        type="tabular",
        meta=dict(
            target_col="toxicity",
            tox_threshold=tox_thr,
            x_sensitive_mode=x_mode,
            n_rows_cached=len(X_use),
            cache_csv=cache_csv,
        ),
    )
