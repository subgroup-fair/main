import numpy as np

def shrink_smallest_by_global_frac(X_df, y_arr, S_df, frac: float, seed: int = 0):
    n = len(X_df)
    if n == 0 or S_df.shape[1] == 0 or frac == 1.0:
        return X_df, np.asarray(y_arr), S_df, {"shrink_applied": False}
    if frac < 1.0:
        target_keep = max(1, int(np.floor(n * float(frac))))
    else:
        target_keep = int(max(1, float(frac)))
    gb = S_df.groupby(list(S_df.columns), dropna=False).size().sort_values()
    if gb.empty:
        return X_df, np.asarray(y_arr), S_df, {"shrink_applied": False}

    smallest_key = gb.index[0]
    if not isinstance(smallest_key, tuple):
        smallest_key = (smallest_key,)
    mask = np.ones(n, dtype=bool)
    for col, val in zip(S_df.columns, smallest_key):
        mask &= (S_df[col].values == val)
    idx_small = np.where(mask)[0]
    m = len(idx_small)

    if m <= target_keep:
        return X_df, np.asarray(y_arr), S_df, {
            "shrink_applied": False, "smallest_key": tuple(smallest_key),
            "smallest_size_orig": m, "target_keep": int(target_keep)
        }

    rng = np.random.RandomState(seed)
    keep_idx = rng.choice(idx_small, size=target_keep, replace=False)
    keep_mask = np.ones(n, dtype=bool)
    to_drop = np.setdiff1d(idx_small, keep_idx, assume_unique=True)
    keep_mask[to_drop] = False

    X2 = X_df.iloc[keep_mask].reset_index(drop=True) if hasattr(X_df, "iloc") else X_df[keep_mask]
    y2 = np.asarray(y_arr)[keep_mask]
    S2 = S_df.iloc[keep_mask].reset_index(drop=True)

    info = {
        "shrink_applied": True,
        "smallest_key": tuple(smallest_key),
        "smallest_size_orig": m,
        "smallest_size_new": int(target_keep),
        "removed": int(m - target_keep),
        "target_keep": int(target_keep),
        "target_mode": ("global_frac" if frac < 1.0 else "absolute"),
    }
    return X2, y2, S2, info
