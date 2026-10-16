"""Honest permutation null for supervised (MDLP) binning.

Supervised binning fits its cut points on the target, so the plug-in MI of a column against the same target is biased upward even for pure
noise. A permutation null that permutes the target but keeps the cut points fitted on the real target cannot see that bias. This module
builds the null the way it must be built: for every permuted target the cut points are refitted with the same binning routine the fit uses,
and the MI is evaluated against that same permuted target. The result is the maximum over the candidate columns for each permutation.

With a spec the permutations run in a numba prange kernel; otherwise the production categorisation is re-run per permutation.
"""

from __future__ import annotations

import numpy as np

from ._parity_fe import _mi_codes_njit


def _honest_perm_best(honest_codes_fn, factors_data, factors_nbins, y_codes, cols, n, row_perm, eval_rows=None):
    """Maximum corrected plug-in MI over the pool for one row permutation (see :func:`honest_fdr_floor`).

    ``row_perm`` permutes every row (the codes are refitted on the whole permuted target); the MI is evaluated on ``eval_rows`` (the
    screen's subsample) when given, exactly as the observed codes were fitted on the whole target and scored on the subsample.
    """

    codes_p, nb_p = honest_codes_fn(row_perm)
    if eval_rows is not None:
        codes_p = np.asarray(codes_p)[eval_rows]
        factors_data = factors_data[eval_rows]
        n = len(eval_rows)
    yc_p = np.ascontiguousarray(y_codes[row_perm])
    if eval_rows is not None:
        yc_p = np.ascontiguousarray(yc_p[eval_rows])
    ny = int(yc_p.max()) + 1
    ky_eff = float(np.unique(yc_p).shape[0])
    best = 0.0
    for c in cols:
        if c < codes_p.shape[1]:
            # Supervised bins: re-binned on the permuted target above.
            x = np.ascontiguousarray(np.asarray(codes_p[:, c]).astype(np.int64))
            kx = int(nb_p[c])
        else:
            # Columns added after binning (engineered features) are binned without the target, so their observed codes are valid under
            # permutation of the target: the fixed codes are the correct null here.
            x = np.ascontiguousarray(np.asarray(factors_data[:, c]).astype(np.int64))
            kx = int(factors_nbins[c])
        mi = _mi_codes_njit(x, kx, yc_p, ny)
        kx_eff = float(np.unique(x).shape[0])
        corr = mi - (kx_eff - 1.0) * (ky_eff - 1.0) / (2.0 * n)
        if corr > best:
            best = corr
    return best


# Raw columns whose per-permutation null is cached on the spec for the rest of the fit: 999 permutations x 2048 columns x 2 matrices x 8 bytes
# is ~33 MB. The screen floor and the empty-support fallback draw the same permutations (seed 0 by default), so the second call reuses the
# first call's cut searches instead of repeating them.
_NULL_CACHE_MAX_COLS = 2048


def _njit_null_columns(spec: dict, factors_data: np.ndarray, factors_nbins, y_idx: int, cols, n_permutations: int, seed_key: int, eval_rows=None):
    """Per-permutation null MI of ``cols`` (no cache): ``(plain, corrected)`` each ``(n_permutations, len(cols))``, from ``honest_null_columns``."""
    from ._mdlp_oos_honest_njit import honest_validated_null_columns
    from ._analytic_mi_null import analytic_null_min_n, _min_expected_cell

    n = int(factors_data.shape[0])
    m = int(spec["raw"].shape[1])
    P = len(cols)
    is_cont = np.array([c < m and bool(spec["is_cont"][c]) for c in cols], dtype=np.bool_)
    fixed_nb = np.array([int(factors_nbins[c]) for c in cols], dtype=np.int64)
    # Compact layouts: the continuous columns' raw values (the spec's own dtype, the width the binning was fitted on) and the other columns'
    # codes (the narrowest integer that holds their bin counts) each hold only their own columns, mapped by ``col_slot``. Two full-width
    # (n, P) copies, float64 and int64, did not fit next to a 50000-row fit with a pool of ~1500 columns.
    cont_cols = [c for c, f in zip(cols, is_cont) if f]
    fixed_cols = [c for c, f in zip(cols, is_cont) if not f]
    raw_dtype = spec["raw"].dtype
    raw_cont = np.empty((n, max(1, len(cont_cols))), dtype=raw_dtype)
    for s_, c in enumerate(cont_cols):
        raw_cont[:, s_] = spec["raw"][:, c]
    code_dtype = np.min_scalar_type(int(fixed_nb.max(initial=1)))
    fixed = np.zeros((n, max(1, len(fixed_cols))), dtype=code_dtype)
    for s_, c in enumerate(fixed_cols):
        fixed[:, s_] = np.asarray(factors_data[:, c])
    col_slot = np.empty(P, dtype=np.int64)
    n_c = n_f = 0
    for j, f in enumerate(is_cont):
        if f:
            col_slot[j] = n_c
            n_c += 1
        else:
            col_slot[j] = n_f
            n_f += 1
    q_fb_rows, q_rf_rows = [], []
    q_dtype = spec["q_fb"].dtype
    for j, c in enumerate(cols):
        if is_cont[j]:
            q_fb_rows.append(spec["q_fb"][c, : int(spec["q_fb_len"][c])])
            q_rf_rows.append(spec["q_rf"][c, : int(spec["q_rf_len"][c])])
        else:
            q_fb_rows.append(np.empty(0, dtype=q_dtype))
            q_rf_rows.append(np.empty(0, dtype=q_dtype))
    q_fb = np.zeros((P, max(1, max(r.size for r in q_fb_rows))), dtype=q_dtype)
    q_rf = np.zeros((P, max(1, max(r.size for r in q_rf_rows))), dtype=q_dtype)
    q_fb_len = np.array([r.size for r in q_fb_rows], dtype=np.int64)
    q_rf_len = np.array([r.size for r in q_rf_rows], dtype=np.int64)
    for j in range(P):
        q_fb[j, : q_fb_len[j]] = q_fb_rows[j]
        q_rf[j, : q_rf_len[j]] = q_rf_rows[j]
    y_codes = np.ascontiguousarray(np.asarray(factors_data[:, int(y_idx)]).astype(np.int64))
    rng = np.random.default_rng(seed_key)
    ev = np.arange(n, dtype=np.int64) if eval_rows is None else np.ascontiguousarray(np.asarray(eval_rows, dtype=np.int64))
    if spec.get("method") != "oos":
        # Column-parallel kernel: the permutations are applied to the labels one row at a time, so the (permutations, n) int64 permutation matrix is never built.
        from ._mdlp_honest_bycol_njit import honest_null_columns_bycol, permuted_labels

        y_raw = np.asarray(spec["y_enc"])
        perm_rows = (rng.permutation(n) for _ in range(int(n_permutations)))
        y_raw_perm, y_codes_perm = permuted_labels(y_raw, y_codes, perm_rows, int(n_permutations))
        return honest_null_columns_bycol(
            raw_cont,
            is_cont,
            fixed,
            fixed_nb,
            y_raw_perm,
            y_codes_perm,
            int(spec["n_classes_total"]),
            int(spec["min_split_size"]),
            int(spec["max_depth"]),
            float(spec["alpha"]),
            int(spec["n_permutations"]),
            int(spec["seed"]),
            int(analytic_null_min_n()),
            float(_min_expected_cell()),
            q_fb,
            q_fb_len,
            q_rf,
            q_rf_len,
            ev,
            col_slot,
        )
    perms = np.stack([rng.permutation(n) for _ in range(int(n_permutations))]).astype(np.int64)
    if spec.get("method") == "oos":
        return honest_validated_null_columns(
            raw_cont,
            is_cont,
            fixed,
            fixed_nb,
            spec["y_enc"],
            y_codes,
            int(spec["n_classes_total"]),
            perms,
            int(spec["min_split_size"]),
            int(spec["max_depth"]),
            float(spec["oos_tolerance"]),
            int(spec["val_min_split_size"]),
            float(spec["holdout_frac"]),
            int(seed_key),
            q_fb,
            q_fb_len,
            q_rf,
            q_rf_len,
            ev,
            col_slot,
        )


def _njit_null_matrices(spec: dict, factors_data: np.ndarray, factors_nbins, y_idx: int, pool, n_permutations: int, seed, eval_rows=None):
    """``(plain, corrected)`` of shape ``(n_permutations, len(pool))``: the per-column null MI, with raw columns served from the spec cache."""
    n_perm = int(n_permutations)
    seed_key = 0 if seed is None else int(seed)
    pool_idx = [int(c) for c in pool]
    m = int(spec["raw"].shape[1])
    cache = None
    if m <= _NULL_CACHE_MAX_COLS:
        # The evaluation rows are part of the key: the screen scores a subsample, the empty-support fallback the full data.
        eval_key = None if eval_rows is None else (len(eval_rows), hash(np.asarray(eval_rows, dtype=np.int64).tobytes()))
        cache = spec.setdefault("_null_columns", {}).setdefault((n_perm, seed_key, eval_key), {})
    missing = [c for c in pool_idx if cache is None or c >= m or c not in cache]
    computed = {}
    if missing:
        plain_m, corr_m = _njit_null_columns(spec, factors_data, factors_nbins, y_idx, missing, n_perm, seed_key, eval_rows)
        for j, c in enumerate(missing):
            computed[c] = (plain_m[:, j], corr_m[:, j])
            if cache is not None and c < m:
                cache[c] = computed[c]
    # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    plain = np.empty((n_perm, len(pool_idx)), dtype=np.float64)
    corr = np.empty_like(plain)
    for j, c in enumerate(pool_idx):
        if c in computed:
            plain[:, j], corr[:, j] = computed[c]
        else:
            assert cache is not None  # a column missing from ``computed`` is served from the cache
            plain[:, j], corr[:, j] = cache[c]
    return plain, corr


def _njit_fdr_floor(spec: dict, factors_data: np.ndarray, factors_nbins, y_idx: int, pool, n_permutations: int, quantile: float, seed, eval_rows=None) -> float:
    """Numba honest floor: per-permutation maximum of the corrected null MI over the pool (``max(0, max_j)``), then the ``quantile``."""
    _, corr = _njit_null_matrices(spec, factors_data, factors_nbins, y_idx, pool, n_permutations, seed, eval_rows)
    nulls = np.maximum(corr.max(axis=1), 0.0)
    return float(np.quantile(nulls, float(quantile), method="higher"))


def honest_plain_null_maxima(spec: dict, factors_data: np.ndarray, factors_nbins, y_idx: int, pool, n_permutations: int, seed) -> np.ndarray:
    """Per-permutation maximum plain plug-in MI over ``pool`` under the honest null (numba chain), for the empty-support fallback."""
    plain, _ = _njit_null_matrices(spec, factors_data, factors_nbins, y_idx, pool, n_permutations, seed)
    capped: np.ndarray = np.maximum(plain.max(axis=1), 0.0)
    return capped


def honest_fdr_floor(
    honest_codes_fn, factors_data: np.ndarray, factors_nbins, y_idx: int, pool, n_permutations: int, quantile: float, seed, eval_rows=None
) -> float:
    """Screen gain floor from the honest supervised null.

    Plain 'mdlp' binning: the numba chain (prange over permutations). Other strategies: the production categorisation is re-run on each
    permuted target, sequentially. The floor is the ``quantile`` of the per-permutation maxima of the Miller-Madow corrected plug-in MI
    over the pool (the same statistic the gate compares the observed gain against).

    ``factors_data`` holds every row the binning was fitted on; ``eval_rows`` (the screen's row subsample, or None for every row) selects the
    rows the MI is evaluated on. The permutation runs over every row, so the refitted cut points see a whole permuted target, as the observed
    cut points saw the whole target.
    """
    spec = getattr(honest_codes_fn, "spec", None)
    if spec is not None:
        return _njit_fdr_floor(spec, factors_data, factors_nbins, y_idx, pool, n_permutations, quantile, seed, eval_rows)
    y_codes = np.ascontiguousarray(factors_data[:, int(y_idx)]).astype(np.int64)
    n = int(factors_data.shape[0])
    cols = [int(c) for c in pool]
    rng = np.random.default_rng(0 if seed is None else int(seed))
    nulls = np.empty(int(n_permutations), dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    for b in range(int(n_permutations)):
        nulls[b] = _honest_perm_best(honest_codes_fn, factors_data, factors_nbins, y_codes, cols, n, rng.permutation(n), eval_rows)
    return float(np.quantile(nulls, float(quantile), method="higher"))
