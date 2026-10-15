"""Honest permutation null for supervised (MDLP) binning.

Supervised binning fits its cut points on the target, so the plug-in MI of a column against the same target is biased upward even for pure
noise. A permutation null that permutes the target but keeps the cut points fitted on the real target cannot see that bias. This module
builds the null the way it must be built: for every permuted target the cut points are refitted with the same binning routine the fit uses,
and the MI is evaluated against that same permuted target. The result is the maximum over the candidate columns for each permutation.

Permutations are independent, so they are evaluated in parallel threads (the binning routine releases the GIL in its numeric kernels).
"""

from __future__ import annotations

import numpy as np

from ._parity_fe import _mi_codes_njit


def _honest_perm_best(honest_codes_fn, factors_data, factors_nbins, y_codes, cols, n, row_perm):
    """Maximum corrected plug-in MI over the pool for one row permutation (see :func:`honest_fdr_floor`)."""

    codes_p, nb_p = honest_codes_fn(row_perm)
    yc_p = np.ascontiguousarray(y_codes[row_perm])
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


def _njit_null_columns(spec: dict, factors_data: np.ndarray, factors_nbins, y_idx: int, cols, n_permutations: int, seed_key: int):
    """Per-permutation null MI of ``cols`` (no cache): ``(plain, corrected)`` each ``(n_permutations, len(cols))``, from ``honest_null_columns``."""
    from ._mdlp_honest_njit import honest_null_columns
    from ._mdlp_oos_honest_njit import honest_validated_null_columns
    from ._analytic_mi_null import analytic_null_min_n, _min_expected_cell

    n = int(factors_data.shape[0])
    m = int(spec["raw"].shape[1])
    P = len(cols)
    raw_cont = np.zeros((n, P), dtype=np.float64)
    is_cont = np.zeros(P, dtype=np.bool_)
    fixed = np.empty((n, P), dtype=np.int64)
    fixed_nb = np.empty(P, dtype=np.int64)
    q_fb_rows, q_rf_rows = [], []
    for j, c in enumerate(cols):
        fixed[:, j] = np.asarray(factors_data[:, c]).astype(np.int64)
        fixed_nb[j] = int(factors_nbins[c])
        if c < m and spec["is_cont"][c]:
            raw_cont[:, j] = spec["raw"][:, c]
            is_cont[j] = True
            q_fb_rows.append(spec["q_fb"][c, : int(spec["q_fb_len"][c])])
            q_rf_rows.append(spec["q_rf"][c, : int(spec["q_rf_len"][c])])
        else:
            q_fb_rows.append(np.empty(0, dtype=np.float64))
            q_rf_rows.append(np.empty(0, dtype=np.float64))
    q_fb = np.zeros((P, max(1, max(r.size for r in q_fb_rows))), dtype=np.float64)
    q_rf = np.zeros((P, max(1, max(r.size for r in q_rf_rows))), dtype=np.float64)
    q_fb_len = np.array([r.size for r in q_fb_rows], dtype=np.int64)
    q_rf_len = np.array([r.size for r in q_rf_rows], dtype=np.int64)
    for j in range(P):
        q_fb[j, : q_fb_len[j]] = q_fb_rows[j]
        q_rf[j, : q_rf_len[j]] = q_rf_rows[j]
    y_codes = np.ascontiguousarray(np.asarray(factors_data[:, int(y_idx)]).astype(np.int64))
    rng = np.random.default_rng(seed_key)
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
        )
    return honest_null_columns(
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
        float(spec["alpha"]),
        int(spec["n_permutations"]),
        int(spec["seed"]),
        int(analytic_null_min_n()),
        float(_min_expected_cell()),
        q_fb,
        q_fb_len,
        q_rf,
        q_rf_len,
    )


def _njit_null_matrices(spec: dict, factors_data: np.ndarray, factors_nbins, y_idx: int, pool, n_permutations: int, seed):
    """``(plain, corrected)`` of shape ``(n_permutations, len(pool))``: the per-column null MI, with raw columns served from the spec cache."""
    n_perm = int(n_permutations)
    seed_key = 0 if seed is None else int(seed)
    pool_idx = [int(c) for c in pool]
    m = int(spec["raw"].shape[1])
    cache = None
    if m <= _NULL_CACHE_MAX_COLS:
        cache = spec.setdefault("_null_columns", {}).setdefault((n_perm, seed_key), {})
    missing = [c for c in pool_idx if cache is None or c >= m or c not in cache]
    computed = {}
    if missing:
        plain_m, corr_m = _njit_null_columns(spec, factors_data, factors_nbins, y_idx, missing, n_perm, seed_key)
        for j, c in enumerate(missing):
            computed[c] = (plain_m[:, j], corr_m[:, j])
            if cache is not None and c < m:
                cache[c] = computed[c]
    plain = np.empty((n_perm, len(pool_idx)), dtype=np.float64)
    corr = np.empty_like(plain)
    for j, c in enumerate(pool_idx):
        plain[:, j], corr[:, j] = computed[c] if c in computed else cache[c]
    return plain, corr


def _njit_fdr_floor(spec: dict, factors_data: np.ndarray, factors_nbins, y_idx: int, pool, n_permutations: int, quantile: float, seed) -> float:
    """Numba honest floor: per-permutation maximum of the corrected null MI over the pool (``max(0, max_j)``), then the ``quantile``."""
    _, corr = _njit_null_matrices(spec, factors_data, factors_nbins, y_idx, pool, n_permutations, seed)
    nulls = np.maximum(corr.max(axis=1), 0.0)
    return float(np.quantile(nulls, float(quantile), method="higher"))


def honest_plain_null_maxima(spec: dict, factors_data: np.ndarray, factors_nbins, y_idx: int, pool, n_permutations: int, seed) -> np.ndarray:
    """Per-permutation maximum plain plug-in MI over ``pool`` under the honest null (numba chain), for the empty-support fallback."""
    plain, _ = _njit_null_matrices(spec, factors_data, factors_nbins, y_idx, pool, n_permutations, seed)
    return np.maximum(plain.max(axis=1), 0.0)


def honest_fdr_floor(honest_codes_fn, factors_data: np.ndarray, factors_nbins, y_idx: int, pool, n_permutations: int, quantile: float, seed) -> float:
    """Screen gain floor from the honest supervised null.

    Plain 'mdlp' binning: the numba chain (prange over permutations). Other strategies: the production categorisation is re-run on each
    permuted target, sequentially. The floor is the ``quantile`` of the per-permutation maxima of the Miller-Madow corrected plug-in MI
    over the pool (the same statistic the gate compares the observed gain against).
    """
    spec = getattr(honest_codes_fn, "spec", None)
    if spec is not None:
        return _njit_fdr_floor(spec, factors_data, factors_nbins, y_idx, pool, n_permutations, quantile, seed)
    y_codes = np.ascontiguousarray(factors_data[:, int(y_idx)]).astype(np.int64)
    n = int(factors_data.shape[0])
    cols = [int(c) for c in pool]
    rng = np.random.default_rng(0 if seed is None else int(seed))
    nulls = np.empty(int(n_permutations), dtype=np.float64)
    for b in range(int(n_permutations)):
        nulls[b] = _honest_perm_best(honest_codes_fn, factors_data, factors_nbins, y_codes, cols, n, rng.permutation(n))
    return float(np.quantile(nulls, float(quantile), method="higher"))
