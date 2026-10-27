"""Honest permutation null for supervised (MDLP) binning.

Supervised binning fits its cut points on the target, so the plug-in MI of a column against the same target is biased upward even for pure
noise. A permutation null that permutes the target but keeps the cut points fitted on the real target cannot see that bias. This module
builds the null the way it must be built: for every permuted target the cut points are refitted with the same binning routine the fit uses,
and the MI is evaluated against that same permuted target. The result is the maximum over the candidate columns for each permutation.

With a spec the permutations run in a numba prange kernel; otherwise the production categorisation is re-run per permutation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np

from mlframe.utils.log_throttle import log_throttle

from ._parity_fe import _mi_codes_njit

logger = logging.getLogger(__name__)


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


_HONEST_ANALYTIC_MIN_N = 15_000


def _honest_analytic_min_n(get_min_n, lowered) -> int:
    """Row floor of the analytic root test for the honest null: the chi-square asymptotic is validated tight from ~20k rows (see ``_analytic_mi_null``) and measured identical to the permutation-decided root on the honest null from 15k (6% of pairs differ, by up to 5e-4 MI, at 10k), so the 15k-25k band no longer runs the host permutation kernel; an explicit env override still wins."""
    with lowered(_HONEST_ANALYTIC_MIN_N):
        return int(get_min_n())


def _device_honest_null_enabled() -> bool:
    """The device root-split pipeline is used when a GPU is usable; ``MLFRAME_HONEST_NULL_DEVICE=0`` or ``MLFRAME_FE_GPU_STRICT=0`` keeps the host kernel."""
    import os

    if os.environ.get("MLFRAME_HONEST_NULL_DEVICE", "1").strip().lower() not in ("1", "true", "on", "yes"):
        return False
    if os.environ.get("MLFRAME_FE_GPU_STRICT", "").strip().lower() in ("0", "false", "off", "no"):
        return False  # the documented explicit CPU pin of every FE dispatch
    try:
        from ._gpu_policy import gpu_globally_disabled

        if gpu_globally_disabled():
            return False
        import cupy as cp

        has_device: bool = cp.cuda.runtime.getDeviceCount() > 0
        return has_device
    except Exception as e:
        logger.warning("CUDA device probe for the honest MDLP null failed (%s: %s); using the CPU null path", type(e).__name__, e)
        return False


_PERM_LABELS_CACHE_MAX_BYTES = 256 * 1024 * 1024  # per spec: the permuted-label matrices of one (seed, rows, permutations) draw; larger draws are regenerated each call


def _permuted_labels_cached(spec: dict, y_raw: np.ndarray, y_codes: np.ndarray, rng, n: int, n_permutations: int, seed_key: int):
    """``permuted_labels`` of the seeded permutation stream, kept on the spec: the screen asks for the null of one column after another with the same seed, rows and permutation count,
    and every call drew the same ``n_permutations`` shuffles of ``n`` rows again (3483 draws, 1.5 s of a 20000-row fit, for 200 distinct ones)."""
    from ._mdlp_honest_bycol_njit import permuted_labels

    def draw():
        """The permuted label matrices of a fresh draw from ``rng``."""
        return permuted_labels(y_raw, y_codes, (rng.permutation(n) for _ in range(n_permutations)), n_permutations)

    if n_permutations * n * 2 > _PERM_LABELS_CACHE_MAX_BYTES:
        log_throttle(
            logger, "honest_null_perm_labels_cache_too_large", logging.INFO,
            "honest null: the permuted-label matrices (%d permutations x %d rows, %.0f MB) exceed the %.0f MB cache cap; they are redrawn on every call",
            n_permutations, n, n_permutations * n * 2 / 2**20, _PERM_LABELS_CACHE_MAX_BYTES / 2**20,
        )
        return draw()
    # The stream of one seed is a prefix-stable sequence: the first k of a longer draw are the k permutations a shorter draw would make, so one entry serves every count up to its own.
    key = (seed_key, n, y_raw.dtype.str, hash(y_raw.tobytes()), hash(y_codes.tobytes()))
    cache = spec.setdefault("_perm_labels", {})
    hit = cache.get(key)
    if hit is None or hit[0].shape[0] < n_permutations:
        cache.clear()  # one stream at a time: the keys differ only across seeds, never within one fit
        hit = cache[key] = draw()
    return hit[0][:n_permutations], hit[1][:n_permutations]


def _njit_null_columns(spec: dict, factors_data: np.ndarray, factors_nbins, y_idx: int, cols, n_permutations: int, seed_key: int, eval_rows=None):
    """Per-permutation null MI of ``cols`` (no cache): ``(plain, corrected)`` each ``(n_permutations, len(cols))``, from ``honest_null_columns``."""
    from ._mdlp_oos_honest_njit import honest_validated_null_columns
    from ._analytic_mi_null import analytic_null_min_n, analytic_null_min_n_lowered, _min_expected_cell

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
    # Rows that were NaN before the median fill: they keep the column's NaN code instead of their filled value's bin.
    nan_cont = np.zeros((n, max(1, len(cont_cols))), dtype=np.bool_)
    if spec.get("nan_mask") is not None:
        for s_, c in enumerate(cont_cols):
            nan_cont[:, s_] = spec["nan_mask"][:, c]
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
        from ._mdlp_honest_bycol_njit import honest_null_columns_bycol

        y_raw = np.asarray(spec["y_enc"])
        y_raw_perm, y_codes_perm = _permuted_labels_cached(spec, y_raw, y_codes, rng, n, int(n_permutations), seed_key)
        args = (
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
            _honest_analytic_min_n(analytic_null_min_n, analytic_null_min_n_lowered),
            float(_min_expected_cell()),
            q_fb,
            q_fb_len,
            q_rf,
            q_rf_len,
            ev,
            col_slot,
        )
        # The device kernel has no NaN-row handling, so a spec with NaN columns runs the host kernel.
        if not spec.get("has_nan", False) and _device_honest_null_enabled():
            from ._honest_null_device import honest_null_columns_device

            on_device = honest_null_columns_device(*args)
            if on_device is not None:
                return on_device
        return honest_null_columns_bycol(*args, nan_cont)
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


def honest_plain_null_maxima(
    spec: dict, factors_data: np.ndarray, factors_nbins: np.ndarray, y_idx: int, pool: Sequence[int] | np.ndarray, n_permutations: int, seed: int | None
) -> np.ndarray:
    """Per-permutation maximum plain plug-in MI over ``pool`` under the honest null (numba chain), for the empty-support fallback."""
    plain, _ = _njit_null_matrices(spec, factors_data, factors_nbins, y_idx, pool, n_permutations, seed)
    capped: np.ndarray = np.maximum(plain.max(axis=1), 0.0)
    return capped


def honest_fdr_floor(
    honest_codes_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    factors_data: np.ndarray,
    factors_nbins: np.ndarray,
    y_idx: int,
    pool: Sequence[int] | np.ndarray,
    n_permutations: int,
    quantile: float,
    seed: int | None,
    eval_rows: np.ndarray | None = None,
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


def honest_candidate_nfailed(
    honest_codes_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    full_factors: np.ndarray,
    factors_nbins: Sequence[int] | np.ndarray,
    y_idx: int,
    col: int,
    n_permutations: int,
    seed: int | None,
    eval_rows: np.ndarray | None = None,
) -> tuple[int, float] | None:
    """Exceedance count of one raw column's plug-in MI under the honest supervised null.

    The observed MI uses the codes the fit produced (cut points fitted on the whole target). Each permutation draws a whole-target row
    permutation, re-runs the production binning on the permuted target (``honest_codes_fn``), and scores the column against the permuted
    target over the same rows. Returns ``(nfailed, observed_mi)``: the number of permutations whose MI reaches the observed MI, out of
    ``n_permutations``. Returns ``None`` when ``col`` is not a column the honest codes cover (an engineered column appended after binning).
    The row permutations depend only on ``seed``, so the memoised codes are shared by every candidate of one fit.
    """
    n = int(full_factors.shape[0])
    y_full = np.ascontiguousarray(full_factors[:, int(y_idx)]).astype(np.int64)
    x_full = np.ascontiguousarray(full_factors[:, int(col)]).astype(np.int64)
    rows = np.arange(n) if eval_rows is None else np.asarray(eval_rows, dtype=np.int64)
    x_obs, y_obs = x_full[rows], y_full[rows]
    observed = float(_mi_codes_njit(x_obs, int(factors_nbins[int(col)]), y_obs, int(y_obs.max()) + 1))
    spec = getattr(honest_codes_fn, "spec", None)
    if spec is not None:
        # The batched null serves the column with the same permutations (default_rng(seed)) and the same codes: one kernel call per candidate
        # instead of one re-binning of the whole frame per permutation.
        if int(col) >= int(spec["raw"].shape[1]):
            return None
        plain, _corr = _njit_null_columns(spec, full_factors, factors_nbins, y_idx, [int(col)], int(n_permutations), 0 if seed is None else int(seed), eval_rows)
        return int(np.count_nonzero(plain[:, 0] >= observed)), observed
    rng = np.random.default_rng(0 if seed is None else int(seed))
    nfailed = 0
    for _ in range(int(n_permutations)):
        row_perm = rng.permutation(n)
        codes_p, nb_p = honest_codes_fn(row_perm)
        if int(col) >= len(nb_p):
            return None
        x_p = np.ascontiguousarray(np.asarray(codes_p)[rows, int(col)]).astype(np.int64)
        y_p = np.ascontiguousarray(y_full[row_perm][rows]).astype(np.int64)
        if float(_mi_codes_njit(x_p, int(nb_p[int(col)]), y_p, int(y_p.max()) + 1)) >= observed:
            nfailed += 1
    return nfailed, observed
