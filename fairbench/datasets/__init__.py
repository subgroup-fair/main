# fairbench/datasets/__init__.py
from .toy_manyq import load_toy_manyq
from .adult import load_adult
from .communities import load_communities
from .dutch import load_dutch
from .celebA import load_celebA

def _print_sensitive_stats(data):
    """
    Save ONLY intersectional subgroup stats to CSV, aggregated across ALL splits.
    - Tabular: concat S_train/S_val/S_test -> S_all (열 순서는 첫 프레임 기준)
    - Image: train/val/test loader 전체에서 태그를 합쳐 한 번에 집계
    - If k(#columns) <= 16: enumerate ALL 2^k combos (missing -> count=0)
      else: save only PRESENT combos to avoid huge files

    CSV columns: split='all', subgroup, count, proportion, n, columns
    Path: {out_dir or log_dir or save_dir or '.'}/sensitive_stats_{dataset}.csv
    """
    import os, itertools
    import numpy as np
    import pandas as pd
    from collections import Counter, defaultdict

    # ---------- helpers ----------
    def _choose_out_path():
        # root = data.get("out_dir") or data.get("log_dir") or data.get("save_dir") or "."
        dsname = (data.get("dataset") or data.get("name") or "dataset")
        root = 'datasets/'
        os.makedirs(root, exist_ok=True)
        return os.path.join(root, f"sensitive_stats_{dsname}.csv")

    def _prefix_to_drop(cols):
        pref = [c.split("_", 1)[0] if "_" in c else None for c in cols]
        cnt = Counter([p for p in pref if p is not None])
        return {p for p, k in cnt.items() if k >= 2}

    def _clean_label(col, drop_prefixes):
        if "_" in col:
            p, rest = col.split("_", 1)
            if p in drop_prefixes:
                return rest.replace("_", " ")
        return col.replace("_", " ")

    def _to01(series: pd.Series):
        v = pd.to_numeric(series, errors="coerce")
        if v.isna().all():
            return series.astype(bool).astype(int)
        return (v.fillna(0) > 0).astype(int)

    def _name_from_bits(bits, keys, drop_prefixes):
        parts = []
        for b, k in zip(bits, keys):
            lab = _clean_label(k, drop_prefixes)
            parts.append(lab if b == 1 else f"not {lab}")
        return " & ".join(parts)

    def _enumerate_and_count_tabular(S: pd.DataFrame, full_enum: bool = True):
        """Tabular S -> rows for CSV."""
        if not isinstance(S, pd.DataFrame) or S.shape[0] == 0:
            return [], [], 0
        S01 = pd.DataFrame({c: _to01(S[c]) for c in S.columns}, index=S.index)
        keys = list(S01.columns)
        n = int(len(S01))
        drop_prefixes = _prefix_to_drop(keys)

        rows = []
        if full_enum:
            present_counts = S01.groupby(keys, dropna=False).size().to_dict()
            for bits in itertools.product([0, 1], repeat=len(keys)):
                count = int(present_counts.get(tuple(bits), 0))
                name = _name_from_bits(bits, keys, drop_prefixes)
                prop = (count / n) if n > 0 else 0.0
                rows.append((name, count, prop, n, ";".join(keys)))
        else:
            grp = S01.groupby(keys, dropna=False).size().reset_index(name="count")
            for _, r in grp.iterrows():
                bits = tuple(int(r[c]) for c in keys)
                count = int(r["count"])
                name = _name_from_bits(bits, keys, drop_prefixes)
                prop = (count / n) if n > 0 else 0.0
                rows.append((name, count, prop, n, ";".join(keys)))
        return rows, keys, n

    def _concat_S_tabular_all(d):
        frames = []
        base_cols = None
        for S_key in ("S_train", "S_val", "S_test"):
            S = d.get(S_key, None)
            if isinstance(S, pd.DataFrame) and S.shape[0] > 0:
                if base_cols is None:
                    base_cols = list(S.columns)
                # 열 정렬 강제 (없으면 에러)
                S2 = S[base_cols]
                frames.append(S2)
        if frames:
            return pd.concat(frames, axis=0, ignore_index=True)
        return pd.DataFrame()

    def _enumerate_and_count_image_all(d, full_enum: bool = True):
        """Aggregate image tags over all loaders into one count."""
        import math
        # 1) 모든 로더에서 태그 dict를 수집 (positive key set으로 임시 저장)
        pos_sets = []
        keys_set = set()
        n = 0
        for key in ("train_loader", "val_loader", "test_loader"):
            loader = d.get(key, None)
            if loader is None:
                continue
            for _, _, S_list in loader:
                if not isinstance(S_list, list) or len(S_list) == 0:
                    continue
                for s in S_list:
                    # 수집
                    pos = {k for k, v in s.items()
                           if (isinstance(v, (bool, int, np.integer)) and int(v) == 1) or bool(v)}
                    keys_set.update(s.keys())
                    pos_sets.append(pos)
                    n += 1
        if n == 0 or not keys_set:
            return [], [], 0

        keys = sorted(keys_set)  # 안정적 순서
        drop_prefixes = _prefix_to_drop(keys)

        # 2) 패턴 카운트
        cnt = Counter()
        for pos in pos_sets:
            bits = tuple(1 if k in pos else 0 for k in keys)
            cnt[bits] += 1

        rows = []
        if full_enum and len(keys) <= 16:
            for bits in itertools.product([0, 1], repeat=len(keys)):
                count = int(cnt.get(bits, 0))
                name = _name_from_bits(bits, keys, drop_prefixes)
                prop = (count / n) if n > 0 else 0.0
                rows.append((name, count, prop, n, ";".join(keys)))
        else:
            for bits, count in cnt.items():
                name = _name_from_bits(bits, keys, drop_prefixes)
                prop = (count / n) if n > 0 else 0.0
                rows.append((name, int(count), prop, n, ";".join(keys)))
        return rows, keys, n

    # ---------- main ----------
    rows = []
    out_csv = _choose_out_path()

    if data.get("type") == "tabular":
        S_all = _concat_S_tabular_all(data)
        if S_all.shape[0] == 0:
            print("[S|all] tabular: empty S; skip.")
        else:
            k = S_all.shape[1]
            full_enum = (k <= 16)
            subrows, keys, n = _enumerate_and_count_tabular(S_all, full_enum=full_enum)
            for name, count, prop, n, colstr in subrows:
                rows.append({
                    "split": "all",
                    "subgroup": name,
                    "count": count,
                    "proportion": prop,
                    "n": n,
                    "columns": colstr,
                })
            mode_txt = "ALL 2^k" if full_enum else "present-only"
            print(f"[S|all] n={n} cols={keys} (aggregate across splits, {mode_txt}, k={len(keys)})")
    else:
        # image-type
        subrows, keys, n = _enumerate_and_count_image_all(data, full_enum=True)
        if n > 0 and keys:
            for name, count, prop, n, colstr in subrows:
                rows.append({
                    "split": "all",
                    "subgroup": name,
                    "count": count,
                    "proportion": prop,
                    "n": n,
                    "columns": colstr,
                })
            print(f"[S|all] n={n} keys={keys} (aggregate across loaders)")

    # write CSV
    if rows:
        df = pd.DataFrame(rows, columns=["split", "subgroup", "count", "proportion", "n", "columns"])
        try:
            df.to_csv(out_csv, index=False)
            print(f"[S] Saved subgroup INTERSECTION stats to CSV: {out_csv}")
            print(df.head(min(5, len(df))))
        except Exception as e:
            print(f"[WARN] failed to save sensitive stats CSV: {e}")




def load_dataset(args):
    name = args.dataset.lower()
    if name == "toy_manyq":
        data = load_toy_manyq(args)
    elif name == "adult":
        data = load_adult(args)
    elif name == "communities":
        data = load_communities(args)
    elif name == "dutch":
        data = load_dutch(args)
    elif name == "celeba":
        data = load_celebA(args)
    else:
        raise ValueError(name)

    try:
        _print_sensitive_stats(data)
    except Exception as e:
        print(f"[WARN] failed to print sensitive stats: {e}")

    return data
