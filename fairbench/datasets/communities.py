# fairbench/datasets/communities.py
import pandas as pd, numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ..utils.trainval import split_train_val
from .shrink import shrink_smallest_by_global_frac

def _normalize_colname(name: str) -> str:
    # 소문자 + 앞뒤 공백 제거 + 탭 스페이스 정리 + 영문/숫자/밑줄만 남김
    name = name.lower().strip().replace("\t", " ")
    name = re.sub(r'[^a-z0-9_]', '', name)
    return name

def _parse_names_smart(names_path: Path, ncols: int):
    if not names_path.exists():
        return []
    allowed = ("continuous", "integer", "real", "numeric", "binary", "nominal")
    cand = []
    with open(names_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("|") or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            left, right = line.split(":", 1)
            if any(tok in right.lower() for tok in allowed):
                name = _normalize_colname(left)   # ★ 여기서 정규화
                if name:
                    cand.append(name)
    if len(cand) == ncols:
        return cand
    if len(cand) > ncols:
        return cand[-ncols:]
    return []

def _read_communities_with_names(data_dir: str):
    # raw_dir = Path(data_dir) / "raw"
    data_path = Path(data_dir) / "communities.data"
    names_path = Path(data_dir) / "communities.names"

    df = pd.read_csv(
        data_path,
        header=None,
        na_values=["?"],
        dtype=None,
        skipinitialspace=True
    )
    ncols = df.shape[1]
    header = _parse_names_smart(names_path, ncols)
    if header:
        df.columns = header
    else:
        df.columns = [f"col{i}" for i in range(ncols)]
        print(f"[WARN] failed to parse names, fallback to generic colN (n={ncols})")

    # ★ 최종 컬럼에도 정규화 1번 더 (혹시 모를 잔여 문자를 제거)
    df.columns = [_normalize_colname(c) for c in df.columns]

    # 디버깅에 도움: 앞 몇 개 찍기
    print("[communities] first 10 cols:", list(df.columns[:10]))
    return df

# ------------------------------
# (3) 보호(민감) 컬럼 프리셋
#     - 논문/공식 구현과 맞추기: race stats + per-capita incomes + 영어/이민 관련
# ------------------------------
_PRESET_SENS = {
    # 논문에서 “race statistics + 영어 못함 등 몇 개 관련 변수”로 18차원 사용
    # (UCI 컬럼명을 소문자로 통일)
    "paper18": [
        # race percentages (4)
        "racepctwhite", "racepctblack", "racepctasian", "racepcthisp",
        # per-capita income by race (6)
        "whitepercap", "blackpercap", "indianpercap", "asianpercap", "otherpercap", "hisppercap",
        # language/immigration related (8)
        "pctnotspeakenglwell", "pctforeignborn",
        "pctimmigrecent", "pctimmigrec5", "pctimmigrec8", "pctimmigrec10",
        "pctrecentimmig", "pctrecimmig5",
    ],
    # 간단 레이스 기본 세트
    "race_basic": ["racepctwhite", "racepctblack", "racepctasian", "racepcthisp"],
    # 사회경제 예시
    "socio_basic": ["pctpopunderpov", "pcthsdropout", "pctunemployed", "medfaminc"],
    # 논문 문구에 맞춘 기본값: 인종 비율 + 몇 가지 사회지표
    "auto": ["racepctblack", "pctpopunderpov", "pctunemployed"]
}

def _binarize_quantile_strict(series: pd.Series, q: float = 0.5) -> pd.Series:
    """
    결측/동점/상수열까지 고려한 안전 이진화.
    우선 qcut → (x>quantile) → (x>median) → (x>mean) → rank 50% 컷 순으로 시도.
    """
    v = pd.to_numeric(series, errors="coerce")
    finite = v.dropna()
    m = pd.Series(0, index=v.index, dtype=int)

    if finite.empty:
        return m

    try:
        cats = pd.qcut(finite, q=[0, q, 1], labels=[0, 1], duplicates="drop")
        if hasattr(cats, "nunique") and cats.nunique() == 2:
            m.loc[cats.index] = cats.astype(int)
            return m
    except Exception:
        pass

    try:
        thr = finite.quantile(q)
        m2 = (v > thr).astype(int)
        if m2.nunique() == 2:
            return m2
    except Exception:
        pass

    thr = finite.median()
    m3 = (v > thr).astype(int)
    if m3.nunique() == 2:
        return m3

    thr = finite.mean()
    m4 = (v > thr).astype(int)
    if m4.nunique() == 2:
        return m4

    order = finite.rank(method="first")
    kcut = order.median()
    m.loc[order.index] = (order > kcut).astype(int).values
    return m.astype(int)

def _build_sensitive_df_from_df(df: pd.DataFrame, sens_keys, sens_thresh=0.5):
    """
    df: 전체 DF(소문자 컬럼)
    sens_keys: 프리셋 이름(str) 또는 컬럼명 리스트
    sens_thresh: 분위수 컷 (기본 0.5)
    return: (S_df(이진), used_raw_cols(list), dropped(list))
    """
    cols = set(df.columns)
    # 키 파싱
    if sens_keys is None:
        keys = _PRESET_SENS["paper18"]  # 논문 재현 기본: paper18
    elif isinstance(sens_keys, str):
        key = sens_keys.strip().lower()
        if key in _PRESET_SENS:
            keys = _PRESET_SENS[key]
        else:
            keys = [s.strip().lower() for s in key.split(",") if s.strip()]
    else:
        keys = [str(s).lower() for s in sens_keys]

    S = {}
    used, dropped = [], []
    for k in keys:
        if k not in cols:
            dropped.append(k)  # 존재X
            continue
        m = _binarize_quantile_strict(df[k], q=sens_thresh)
        if m.nunique() < 2:
            dropped.append(k)  # 한쪽 클래스만
            continue
        S[f"{k}_high"] = m.astype(int).values
        used.append(k)

    S_df = pd.DataFrame(S).astype(int) if len(S) > 0 else pd.DataFrame()
    if S_df.shape[1] == 0:
        # 후행 코드의 편의상 더미를 넣되, 메타에 dropped 기록
        S_df = pd.DataFrame({"dummy_s": np.zeros(len(df), dtype=int)})

    return S_df, used, dropped

# ------------------------------
# (4) 로더 본체
# ------------------------------
def load_communities(args):
    """
    필요 파일:
      data/raw/communities.names
      data/raw/communities.data

    옵션:
      --x_sensitive {drop,keep,concat}
      --sens_keys {paper18,race_basic,socio_basic,auto 혹은 콤마열}
      --sens_thresh (기본 0.5)
    """
    seed = getattr(args, "seed", 42)
    x_mode = getattr(args, "x_sensitive", "drop")
    sens_keys = getattr(args, "sens_keys", "paper18")      # 기본값을 paper18로!
    sens_thresh = float(getattr(args, "sens_thresh", 0.5))

    # 1) 읽기
    df = _read_communities_with_names(getattr(args, "data_dir", "data"))
    cols = set(df.columns)

    # 2) 타겟: violentcrimesperpop 상위 70% 초과 → 1
    #    (논문/공식 구현과 일치) 70th percentile threshold
    target_candidates = [c for c in df.columns if "violent" in c and "perpop" in c]
    target_col = target_candidates[0] if len(target_candidates) else df.columns[-1]
    y_cont = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)
    thr70 = y_cont.quantile(0.70)  # 상위 30% = 1
    y = (y_cont > thr70).astype(int).values  # strictly greater

    # 3) ID 비예측 컬럼 제거
    id_like = [c for c in ["state","county","community","communityname","fold"] if c in cols]

    # 4) 보호특성 S (이진)
    S_df, used_sens_raw, sens_dropped = _build_sensitive_df_from_df(
        df, sens_keys=sens_keys, sens_thresh=sens_thresh
    )

    # 5) X 원본
    X_raw = df.drop(columns=[target_col] + id_like, errors="ignore")

    # 6) x_sensitive 모드
    if x_mode == "drop":
        X_for_fe = X_raw.drop(columns=used_sens_raw, errors="ignore")
    else:
        X_for_fe = X_raw.copy()

    # 7) 스케일링(숫자만)
    X_for_fe = X_for_fe.apply(pd.to_numeric, errors="ignore")
    X_num = X_for_fe.select_dtypes(include=[np.number]).copy()
    X_num = X_num.fillna(X_num.median(numeric_only=True))
    scaler = StandardScaler()
    Xp = pd.DataFrame(scaler.fit_transform(X_num), columns=X_num.columns)

    if x_mode == "concat" and S_df.shape[1] > 0:
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

    # 8) split
    if S_df.shape[1] == 0:
        S_df = pd.DataFrame({"dummy_s": np.zeros(len(df), dtype=int)})

    X_tr, X_te, y_tr, y_te, S_tr, S_te = train_test_split(
        Xp, y, S_df, test_size=0.2, random_state=seed, stratify=y
    )
    X_tr, X_va, y_tr, y_va, S_tr, S_va = split_train_val(
        X_tr, y_tr, S_tr, val_size=0.2, seed=seed
    )

    # 디버그/로그
    print("[communities] target_col:", target_col, "| thr70:", float(thr70))
    print("[communities] S used:", used_sens_raw)
    print("[communities] S dropped:", sens_dropped)
    print("[communities] S cols(final):", list(S_df.columns))
    print("[communities] S positives:", S_df.sum().to_dict())

    return dict(
        X_train=X_tr, y_train=y_tr, S_train=S_tr,
        X_val=X_va,   y_val=y_va,   S_val=S_va,
        X_test=X_te,  y_test=y_te,  S_test=S_te,
        type="tabular",
        meta=dict(
            target_col=target_col,
            id_cols=id_like,
            x_sensitive_mode=x_mode,
            used_sensitive_raw=used_sens_raw,
            sens_keys=(sens_keys if isinstance(sens_keys, str) else ",".join(sens_keys)),
            sens_thresh=sens_thresh,
            threshold_type="p70_gt",  # 논문 설정
            S_cols=list(S_df.columns),
            S_pos_counts=S_df.sum().to_dict(),
            S_dropped=sens_dropped,
        )
    )
