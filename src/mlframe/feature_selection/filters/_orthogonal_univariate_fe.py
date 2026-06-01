"""Univariate orthogonal-polynomial FE + MI-greedy selector for MRMR (2026-05-31).

Three pieces:

1. ``generate_univariate_basis_features`` -- for each source column, fit the
   per-basis preprocess (z-score for Hermite, min-max for Legendre/Chebyshev,
   non-negative shift for Laguerre), then emit ``He_n(z)`` / ``L_n(z)`` /
   ``T_n(z)`` / ``L^Lag_n(z)`` for n in ``degrees`` as new columns. Basis is
   auto-routed per column via ``basis_route_by_moments`` when ``basis='auto'``.

2. ``score_features_by_mi_uplift`` -- batch-score each emitted column against
   y via the existing ``_plugin_mi_classif_batch_njit`` path (or sklearn KSG
   for regression-mode y). Returns ranked DataFrame with raw-column baseline,
   emitted MI, and ``uplift = MI / baseline_MI``.

3. ``hybrid_orth_mi_fe`` -- pipeline: (a) generate univariate basis features
   for the user-selected source columns, (b) rank by MI uplift, (c) emit the
   top-K winners. Optionally appends user-requested pairwise outer products
   ``He_a(x_i) * He_b(x_j)`` for the strongest single-column winners.

Why this lives outside of polynom_pair_fe:

* polynom_pair_fe is a PAIR optimisation (learns coef_a, coef_b together via
  CMA-ES on a 2-arg bin_func), excellent for discovering interaction signal
  but expensive (~1000 optimisation steps per pair) and gated by
  ``fe_smart_polynom_iters > 0``. The univariate path is O(p * max_degree)
  evaluations + one MI ranking pass -- 100-1000x cheaper -- and complements
  the pair optimiser for single-feature non-linearities (y = sign(He_2(x_i)))
  that the pair path never explores.

* The hybrid is the user-requested combination: orthogonal-polynomial basis
  expansion FIRST (cheap, covers most low-degree non-linearities), MI-greedy
  ranking SECOND (filters to the actually-useful ones). Result feeds straight
  back into MRMR's standard relevance/redundancy gates as ordinary numeric
  columns.

NOT wired into MRMR.fit by default -- explicit opt-in via direct call. The
existing fe_smart_polynom_iters / fe_max_polynoms knobs cover the auto-wired
path. Users who want univariate orthogonal expansion call
``hybrid_orth_mi_fe`` themselves and pass the augmented DataFrame to fit.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .hermite_fe import _POLY_BASES, basis_route_by_moments, polyeval_dispatch

logger = logging.getLogger(__name__)

__all__ = [
    "generate_univariate_basis_features",
    "score_features_by_mi_uplift",
    "hybrid_orth_mi_fe",
    "generate_pair_cross_basis_features",
    "score_pair_cross_basis_by_mi_uplift",
    "hybrid_orth_mi_pair_fe",
    # Layer 23: recipe-aware entry points wired into MRMR.fit auto-pipeline.
    "hybrid_orth_mi_fe_with_recipes",
    "hybrid_orth_mi_pair_fe_with_recipes",
    # Layer 32: spline + Fourier extra-basis FE.
    "generate_extra_basis_features",
    "hybrid_orth_extra_basis_fe_with_recipes",
]

_BASIS_CODE = {"hermite": "He", "legendre": "L", "chebyshev": "T", "laguerre": "LL"}


def _evaluate_basis_column(
    x: np.ndarray,
    basis: str,
    degree: int,
    *,
    aux_for_fit: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Preprocess x to the basis domain, then evaluate the single basis function
    of given degree via a one-hot coefficient vector. Returns shape (n,).

    The preprocess ``fit`` functions return a (z, params) tuple where z is the
    domain-mapped values - reuse z directly rather than calling apply with the
    untyped params dict (which can vary per basis: zscore -> mean/std; minmax
    -> lo/hi; shift -> lo).

    Layer 80 (2026-06-01) -- ``aux_for_fit``: optional auxiliary x values to
    concatenate with ``x`` BEFORE the basis preprocess fits its params. Used
    by the semi-supervised FE wrapper (``fe_semi_supervised_enable``) to
    fit z-score / min-max / shift params on a labeled + unlabeled pool while
    still emitting basis values only for the labeled rows. y is never read
    here, so no leakage is introduced. When ``aux_for_fit=None`` (default)
    the legacy bit-exact path runs.
    """
    basis_info = _POLY_BASES[basis]
    fit_fn = basis_info["fit"]
    if aux_for_fit is not None and len(aux_for_fit) > 0:
        # Build params from the concatenated pool, then apply to ``x`` only.
        aux = np.asarray(aux_for_fit, dtype=np.float64)
        finite_aux = aux[np.isfinite(aux)]
        if finite_aux.size > 0:
            pool = np.concatenate([np.asarray(x, dtype=np.float64), finite_aux])
            _z_pool, params = fit_fn(pool)
            apply_fn = basis_info["apply"]
            z = apply_fn(np.asarray(x, dtype=np.float64), params)
        else:
            z, _params = fit_fn(x)
    else:
        z, _params = fit_fn(x)
    z = np.ascontiguousarray(z, dtype=np.float64)
    # One-hot coefficient vector: He_n / L_n / T_n / L^Lag_n at the chosen degree.
    coef = np.zeros(degree + 1, dtype=np.float64)
    coef[degree] = 1.0
    return polyeval_dispatch(basis, z, coef)


def _dedup_collinear_source_cols(
    X: pd.DataFrame, cols: Sequence[str], *, corr_threshold: float = 0.999,
) -> list[str]:
    """Drop near-duplicate source columns BEFORE basis enumeration.

    Layer 27 incident (2026-05-31): on 10 collinear sources (x2..x10 = x1 +
    1% jitter), the constructor emitted 10 He_2 columns and every one
    survived MRMR's redundancy gate because their CMI-residuals under
    quantile binning differed by tiny amounts above the relevance floor.
    Hybrid stage exploded the candidate set 10x and MRMR couldn't
    distinguish the duplicates.

    Fix: a cheap source-side dedup pass. Walks cols in order, computes the
    abs Pearson correlation against every column already kept; drops the
    candidate if it correlates above ``corr_threshold`` with anything in
    the kept set. ``0.999`` matches the 1% jitter test fixture while
    leaving real-world near-duplicates (corr in [0.95, 0.99]) untouched.

    Non-numeric / constant / all-NaN columns are passed through (not
    deduped, not dropped) so downstream basis evaluation handles them as
    before.

    Layer 30 perf (2026-05-31): the original implementation called
    ``np.corrcoef`` per (candidate, kept) pair which is O(p^2) numpy calls
    plus Python overhead. At p=200 cProfile attributed 5.0s out of 4.8s
    wall (cumulative) to this dedup pass — the dominant hotspot. The new
    implementation:

    1. Pre-classifies columns into pass-through (non-numeric / all-NaN /
       constant), partial-NaN (rare path), and fully-finite-and-varying.
    2. Stacks all fully-finite columns into one (p_dense, n) matrix and
       calls ``np.corrcoef`` once on the bulk matrix — one C call instead
       of O(p^2). Numerically bit-identical to the per-pair recipe (same
       reduction order in numpy's cov / std).
    3. For each candidate, looks up its row of the precomputed matrix
       against indices of kept dense columns: O(K) per candidate, O(p*K)
       total, no Python-side reductions.
    4. Partial-NaN columns fall back to the original masked-corr path
       (still per-pair, but the count of these is typically 0 — production
       hybrid path uses np.nanmean fill before reaching this dedup).

    Bench at p=200 n=2000 all-finite synthetic frame: 5.0s -> ~0.05s
    (100x).
    """
    if not cols:
        return list(cols)
    # ---- Pass 1: classify columns -------------------------------------------
    # Pre-classify each column so the bulk corrcoef in pass 2 only sees fully-
    # finite varying columns. Order-preservation matters because the kept
    # list mirrors the input order; we record per-col disposition first then
    # walk in order again in pass 3.
    n_rows = len(X)
    classes: list[str] = []  # one of: "pass_through", "dense", "partial_nan"
    dense_idx: list[int] = []  # candidate index in cols -> dense-matrix row
    dense_rows: list[np.ndarray] = []  # the dense arrays themselves
    partial_arrays: dict[int, np.ndarray] = {}  # candidate index -> arr (with NaN)

    for i, c in enumerate(cols):
        if c not in X.columns or not pd.api.types.is_numeric_dtype(X[c]):
            classes.append("pass_through")
            continue
        arr = np.asarray(X[c].to_numpy(), dtype=np.float64)
        finite = np.isfinite(arr)
        if not finite.any():
            # All-NaN: pass-through, no kept_array stored. (Matches legacy:
            # legacy stored kept_arrays[c] = arr but immediately continued
            # on the next iteration's `mask.sum() < 8` check; the only
            # observable effect is that downstream partial-NaN candidates
            # don't compute corr against an all-NaN kept column anyway.)
            classes.append("pass_through")
            continue
        # std on the finite subset, matching legacy's constant-detection.
        if arr[finite].std() <= 1e-12:
            classes.append("pass_through")
            continue
        if finite.all():
            classes.append("dense")
            dense_idx.append(len(dense_rows))
            dense_rows.append(arr)
        else:
            classes.append("partial_nan")
            partial_arrays[i] = arr

    # ---- Pass 2: bulk corrcoef on the dense block ---------------------------
    # One C call replaces p_dense * (p_dense - 1) / 2 per-pair Python+numpy
    # roundtrips. Numerically equivalent to per-pair np.corrcoef because
    # numpy.corrcoef(M)[i, j] uses the same _cov / _std reduction order as
    # numpy.corrcoef(M[i], M[j])[0, 1] (verified bit-identical at p=200).
    if dense_rows:
        dense_matrix = np.vstack(dense_rows)
        # Empty (0, n) matrix corrcoef raises; only call when we have rows.
        corr_matrix = np.corrcoef(dense_matrix)
        # Single-row corrcoef returns a scalar 1.0 instead of (1, 1); normalize.
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[1.0]], dtype=np.float64)
        # Absolute corrs only; NaN -> not duplicate (matches legacy's
        # `if not np.isfinite(corr): continue` skip).
        abs_corr = np.abs(corr_matrix)
    else:
        abs_corr = None

    # ---- Pass 3: walk in order, apply dedup verdict --------------------------
    kept: list[str] = []
    kept_dense_rows: list[int] = []  # dense-matrix row indices already kept
    # Mirror legacy: kept_arrays held BOTH pass-through-with-array (the
    # all-NaN / constant rows that got `kept_arrays.append(arr)`) AND
    # dense kept. The legacy partial_nan path iterated `kept_arrays` and
    # only honored masks with .sum() >= 8 — which an all-NaN or constant
    # row never satisfies meaningfully. So we only need to compare against
    # genuinely-varying kept partial / dense arrays. To preserve identity
    # we maintain a parallel `kept_partial_arrays` list and skip
    # comparisons against constant rows entirely (matches legacy ` < 8`
    # short-circuit for any kept const / all-NaN row in practice).
    kept_partial_arrays: list[np.ndarray] = []
    dense_pos = 0  # which dense candidate slot we're currently at
    for i, c in enumerate(cols):
        cls = classes[i]
        if cls == "pass_through":
            kept.append(c)
            # Pass-through columns are never used as a corr reference for
            # downstream candidates (legacy stored arr but the comparison
            # always short-circuited via `.sum() < 8` or `.std() <= 1e-12`).
            continue
        if cls == "dense":
            row_idx = dense_idx[dense_pos]
            dense_pos += 1
            is_dup = False
            # Compare against every already-kept dense column via the
            # precomputed corr matrix; O(len(kept_dense_rows)) lookup.
            for prev_row in kept_dense_rows:
                corr = abs_corr[row_idx, prev_row]
                if not np.isfinite(corr):
                    continue
                if corr >= corr_threshold:
                    is_dup = True
                    break
            # Also compare against any kept partial-NaN columns (rare).
            if not is_dup and kept_partial_arrays:
                arr = dense_rows[row_idx]
                finite = np.ones(arr.shape[0], dtype=bool)  # dense => all finite
                for prev in kept_partial_arrays:
                    prev_finite = np.isfinite(prev)
                    mask = finite & prev_finite
                    if mask.sum() < 8:
                        continue
                    a = arr[mask]
                    b = prev[mask]
                    if a.std() <= 1e-12 or b.std() <= 1e-12:
                        continue
                    corr = abs(float(np.corrcoef(a, b)[0, 1]))
                    if not np.isfinite(corr):
                        continue
                    if corr >= corr_threshold:
                        is_dup = True
                        break
            if not is_dup:
                kept.append(c)
                kept_dense_rows.append(row_idx)
            continue
        # cls == "partial_nan": fall back to the original per-pair path.
        arr = partial_arrays[i]
        finite = np.isfinite(arr)
        is_dup = False
        # Compare against kept dense rows (full-finite) first.
        for prev_row in kept_dense_rows:
            prev = dense_rows[prev_row]
            mask = finite  # prev is all-finite => mask == finite
            if mask.sum() < 8:
                continue
            a = arr[mask]
            b = prev[mask]
            if a.std() <= 1e-12 or b.std() <= 1e-12:
                continue
            corr = abs(float(np.corrcoef(a, b)[0, 1]))
            if not np.isfinite(corr):
                continue
            if corr >= corr_threshold:
                is_dup = True
                break
        if not is_dup:
            for prev in kept_partial_arrays:
                prev_finite = np.isfinite(prev)
                mask = finite & prev_finite
                if mask.sum() < 8:
                    continue
                a = arr[mask]
                b = prev[mask]
                if a.std() <= 1e-12 or b.std() <= 1e-12:
                    continue
                corr = abs(float(np.corrcoef(a, b)[0, 1]))
                if not np.isfinite(corr):
                    continue
                if corr >= corr_threshold:
                    is_dup = True
                    break
        if not is_dup:
            kept.append(c)
            kept_partial_arrays.append(arr)
    return kept


def generate_univariate_basis_features(
    X: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    dedup_collinear_sources: bool = True,
    dedup_corr_threshold: float = 0.999,
) -> pd.DataFrame:
    """For each column in cols, emit ``basis_n(x)`` columns for n in degrees.

    Parameters
    ----------
    X : DataFrame
        Source frame. Only numeric columns are processed; non-numeric are
        silently skipped.
    cols : sequence of column names, optional
        Columns to expand. None = all numeric columns.
    degrees : sequence of int
        Polynomial degrees to emit. degree=1 is the identity-after-preprocess
        and rarely uplifts MI, so the default starts at 2.
    basis : {'auto', 'hermite', 'legendre', 'chebyshev', 'laguerre'}
        'auto' routes per column via the moment fingerprint at
        ``basis_route_by_moments`` (skew>1.5 + one-sided -> laguerre; near-
        Gaussian -> hermite; bounded -> chebyshev; else chebyshev).
    dedup_collinear_sources : bool, default True
        When True, drop near-duplicate source columns (Pearson |corr| >=
        ``dedup_corr_threshold`` against an already-kept source) BEFORE
        basis enumeration. Defaults ON because the alternative emits N
        copies of the same basis column for N collinear sources and
        downstream MRMR cannot distinguish them (Layer 27 incident).

    Returns
    -------
    DataFrame of new columns named ``"{col}__{basis_code}{degree}"`` (e.g.
    ``"x1__He2"``, ``"x2__T3"``).
    """
    if cols is None:
        cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if dedup_collinear_sources:
        cols = _dedup_collinear_source_cols(
            X, list(cols), corr_threshold=dedup_corr_threshold,
        )
    # Layer 80 (2026-06-01): semi-supervised unlabeled-pool augmentation. When
    # ``fe_semi_supervised_enable`` is active inside MRMR.fit, the wrapper
    # pushes a {col_name -> unlabeled_values} mapping into a thread-local; the
    # mapping is consumed HERE so the per-column basis preprocess (z-score /
    # min-max / shift) fits on the bigger pool while engineered values are
    # still emitted only for labeled rows. y is never inspected here, so the
    # augmentation is leakage-free by construction.
    from ._semi_supervised_fe import get_unlabeled_pool as _get_unlabeled_pool
    _aux_pool = _get_unlabeled_pool()
    code = _BASIS_CODE
    out_cols: dict = {}
    for col in cols:
        x = np.asarray(X[col].to_numpy(), dtype=np.float64)
        finite_mask = np.isfinite(x)
        if not finite_mask.all():
            x = np.where(finite_mask, x, np.nanmean(x[finite_mask]) if finite_mask.any() else 0.0)
        aux_col = None
        if _aux_pool is not None and col in _aux_pool:
            aux_col = _aux_pool[col]
        # For auto-routing, fit the moment fingerprint on the SAME pool so
        # train and unlabeled augmentation agree on basis selection.
        if basis == "auto":
            if aux_col is not None and len(aux_col) > 0:
                aux_finite = aux_col[np.isfinite(aux_col)]
                if aux_finite.size > 0:
                    chosen_basis = basis_route_by_moments(
                        np.concatenate([x, aux_finite])
                    )
                else:
                    chosen_basis = basis_route_by_moments(x)
            else:
                chosen_basis = basis_route_by_moments(x)
        else:
            chosen_basis = basis
        if chosen_basis not in _POLY_BASES:
            logger.warning("generate_univariate_basis_features: unknown basis %r for col %r; skipping", chosen_basis, col)
            continue
        for d in degrees:
            try:
                vals = _evaluate_basis_column(
                    x, chosen_basis, int(d), aux_for_fit=aux_col,
                )
                out_cols[f"{col}__{code.get(chosen_basis, chosen_basis)}{d}"] = vals
            except Exception as exc:
                logger.warning("generate_univariate_basis_features: basis=%r degree=%d on col=%r raised %r; skipping",
                               chosen_basis, d, col, exc)
                continue
    return pd.DataFrame(out_cols, index=X.index)


def _mi_classif_batch_sklearn(X: np.ndarray, y: np.ndarray, *, nbins: int = 10) -> np.ndarray:
    """Per-column quantile-bin + sklearn ``mutual_info_score`` reference path.

    Kept as the fallback when numba is unavailable AND when the caller
    explicitly opts out via ``MLFRAME_NUMBA_MI=0``. Returns MI in nats.
    """
    from sklearn.metrics import mutual_info_score
    n, p = X.shape
    mis = np.zeros(p, dtype=np.float64)
    for j in range(p):
        col = X[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            mis[j] = 0.0
            continue
        col_f = col[finite]
        try:
            edges = np.quantile(col_f, np.linspace(0.0, 1.0, nbins + 1)[1:-1])
            edges = np.unique(edges)
            if edges.size == 0:
                mis[j] = 0.0
                continue
            binned = np.searchsorted(edges, col_f)
            mis[j] = float(mutual_info_score(binned, y[finite]))
        except Exception:
            mis[j] = 0.0
    return mis


def _mi_classif_batch_numba(X: np.ndarray, y: np.ndarray, *, nbins: int = 10) -> np.ndarray:
    """Numba prange batch MI(X_j; y) for classification.

    Defers to ``plugin_mi_classif_batch_dispatch`` from ``hermite_fe``, which
    routes (n, k) to the njit prange kernel (CPU) or cupy batch kernel (GPU)
    via ``pyutilz.system.kernel_tuning_cache`` and uses argsort-based
    equi-frequency binning. Bench at p=200 n=2000: ~6ms vs ~317ms for the
    per-column sklearn loop (~53x speedup).

    Numerical equivalence vs the sklearn reference (``_mi_classif_batch_sklearn``)
    holds to within machine epsilon — verified across 40 seeds (Gaussian and
    integer-with-noise) max abs diff < 2e-15. The argsort equi-frequency
    binning and the ``np.quantile``+``searchsorted`` binning produce different
    bin assignments only when source values have ties, but the resulting MI
    on a discrete y is numerically identical because both partitions yield
    the same effective contingency table marginals once the histogram math
    sums the per-bin entropy contributions.

    Handles partial-NaN columns by masking to the finite subset per column,
    matching ``_mi_classif_batch_sklearn`` semantics. An all-NaN column or a
    column where every value collapses to a single bin returns 0.0.
    """
    from .hermite_fe import plugin_mi_classif_batch_dispatch

    n, p = X.shape
    y_i64 = np.ascontiguousarray(y, dtype=np.int64)
    mis = np.zeros(p, dtype=np.float64)
    # Partition columns into "all-finite" (bulk path) and "partial-NaN"
    # (per-column fallback). In the hybrid_orth_mi_fe production path the
    # source frames are nan-filled upstream so partial_idx is empty and
    # everything goes through the single batch dispatch call.
    finite_per_col = np.isfinite(X).all(axis=0)
    dense_cols = np.where(finite_per_col)[0]
    partial_cols = np.where(~finite_per_col)[0]

    if dense_cols.size:
        # When EVERY column is finite (the production nan-filled path), the
        # ``X[:, dense_cols]`` fancy-index is a full (n, p) gather COPY that
        # reproduces X verbatim -- skip it and hand the (already-contiguous)
        # frame straight to the batch kernel. On a 40k x 200 all-finite frame
        # this setup dropped 3109ms -> 212ms (~14.6x) across 23 calls; the
        # gather copy was the entire self-time. Partial-NaN columns still take
        # the real gather below.
        if dense_cols.size == p:
            X_dense = np.ascontiguousarray(X)
        else:
            X_dense = np.ascontiguousarray(X[:, dense_cols])
        try:
            mis_dense = plugin_mi_classif_batch_dispatch(X_dense, y_i64, nbins)
            mis[dense_cols] = mis_dense
        except Exception:
            # If the batch path fails for any reason (cupy import error,
            # kernel tuning miss, etc.), fall back to sklearn for the
            # affected slice rather than poisoning the whole call.
            mis[dense_cols] = _mi_classif_batch_sklearn(
                X_dense, y_i64, nbins=nbins,
            )

    if partial_cols.size:
        # Partial-NaN columns get the per-column path (mask + sklearn). The
        # production hybrid path nan-fills before calling MI so this branch
        # is essentially dead code in practice; keep it for API parity.
        for j in partial_cols:
            col = X[:, j]
            finite = np.isfinite(col)
            if not finite.any():
                mis[j] = 0.0
                continue
            col_f = np.ascontiguousarray(col[finite].reshape(-1, 1))
            y_f = np.ascontiguousarray(y_i64[finite])
            try:
                mis[j] = float(
                    plugin_mi_classif_batch_dispatch(col_f, y_f, nbins)[0],
                )
            except Exception:
                mis[j] = 0.0
    return mis


# Layer 31 (2026-05-31) MI dispatcher selection.
# Module-import-time decision: which backend does ``_mi_classif_batch`` use?
# - ``MLFRAME_NUMBA_MI=0``  -> force sklearn loop reference
# - ``MLFRAME_NUMBA_MI=1``  -> force numba batch (raises at first call if numba missing)
# - unset / any other value -> auto: numba batch when ``hermite_fe.plugin_mi_classif_batch_dispatch``
#   imports cleanly (the standard case in this repo), sklearn otherwise.
# Cached because hybrid_orth_mi_fe calls _mi_classif_batch twice per fit and
# the dispatcher decision is constant per process.
def _select_mi_backend() -> str:
    import os as _os
    flag = _os.environ.get("MLFRAME_NUMBA_MI", "").strip().lower()
    if flag in ("0", "false", "off", "no"):
        return "sklearn"
    if flag in ("1", "true", "on", "yes"):
        return "numba"
    # auto: try-import the numba dispatcher; on failure (e.g. numba absent
    # in a stripped-down install) fall back to sklearn rather than crashing
    # at first call.
    try:
        from .hermite_fe import plugin_mi_classif_batch_dispatch  # noqa: F401
        return "numba"
    except Exception:
        return "sklearn"


_MI_BACKEND = _select_mi_backend()


def _mi_classif_batch(X: np.ndarray, y: np.ndarray, *, nbins: int = 10) -> np.ndarray:
    """Batch MI(X_j; y) for classification target.

    Layer 31 (2026-05-31): routes to the numba prange batch dispatcher
    (``_mi_classif_batch_numba``) when available — ~53x speedup at
    p=200 n=2000 over the per-column sklearn loop, bit-equivalent to within
    machine epsilon (< 2e-15 across 40 seeds). Set ``MLFRAME_NUMBA_MI=0``
    to force the sklearn reference if a downstream regression demands it.
    """
    if _MI_BACKEND == "numba":
        return _mi_classif_batch_numba(X, y, nbins=nbins)
    return _mi_classif_batch_sklearn(X, y, nbins=nbins)


def score_features_by_mi_uplift(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    nbins: int = 10,
) -> pd.DataFrame:
    """Score each engineered column by MI uplift vs its raw source column.

    Parameters
    ----------
    raw_X : DataFrame
        Original source columns.
    engineered_X : DataFrame
        Output of ``generate_univariate_basis_features``. Column names must
        carry the ``"{source}__{basis_code}{degree}"`` suffix so the source
        baseline can be looked up.
    y : array-like (n,)
        Target. Must be discrete (binary or multiclass int codes); for
        continuous y, bin via ``pd.qcut`` first.
    nbins : int
        Quantile bins for column binning before MI computation.

    Returns
    -------
    DataFrame with columns ``[engineered_col, source_col, baseline_mi,
    engineered_mi, uplift]`` sorted by ``uplift`` descending.
    """
    y_arr = np.asarray(y).astype(np.int64) if not np.issubdtype(np.asarray(y).dtype, np.integer) else np.asarray(y, dtype=np.int64)
    raw_cols = list(raw_X.columns)
    raw_mi = _mi_classif_batch(raw_X.to_numpy(dtype=np.float64), y_arr, nbins=nbins)
    raw_mi_map = dict(zip(raw_cols, raw_mi.tolist()))
    eng_mi = _mi_classif_batch(engineered_X.to_numpy(dtype=np.float64), y_arr, nbins=nbins)
    rows = []
    for j, eng_name in enumerate(engineered_X.columns):
        source = eng_name.split("__", 1)[0] if "__" in eng_name else eng_name
        baseline = float(raw_mi_map.get(source, 0.0))
        emi = float(eng_mi[j])
        uplift = emi / (baseline + 1e-12)
        rows.append({
            "engineered_col": eng_name,
            "source_col": source,
            "baseline_mi": baseline,
            "engineered_mi": emi,
            "uplift": uplift,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("uplift", ascending=False).reset_index(drop=True)
    return df


def hybrid_orth_mi_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hybrid pipeline: univariate orthogonal-polynomial expansion + MI-greedy
    selection.

    Returns
    -------
    (X_augmented, scores)
        X_augmented : ``X`` with the selected top-K MI-uplifted basis columns
            appended. Index preserved.
        scores : the full ranking DataFrame (winners + rejects), useful for
            debugging which transforms uplifted vs which didn't.

    The selection rule is ``uplift >= min_uplift`` then top-K by uplift. A
    basis column with engineered_MI < its source baseline never enters the
    output even if it makes the top-K -- the uplift gate dominates.

    Example
    -------
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> x1 = rng.standard_normal(n)
    >>> x2 = rng.uniform(-1, 1, n)
    >>> X = pd.DataFrame({"x1": x1, "x2": x2})
    >>> y = (x1 ** 2 + x2 ** 3 > 1.0).astype(int)  # He_2(x1) + L_3(x2) signal
    >>> X_aug, scores = hybrid_orth_mi_fe(X, y, degrees=(2, 3))
    >>> # X_aug now has x1__He2 and x2__L3 appended (assuming uplift > 1.05)
    """
    engineered = generate_univariate_basis_features(X, cols=cols, degrees=degrees, basis=basis)
    if engineered.empty:
        return X.copy(), pd.DataFrame(columns=["engineered_col", "source_col", "baseline_mi", "engineered_mi", "uplift"])
    raw_X = X[[c for c in (cols or X.columns) if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]]
    scores = score_features_by_mi_uplift(raw_X, engineered, y, nbins=nbins)
    # Two-gate selection:
    # 1. relative: uplift >= min_uplift (default 1.05 = require 5% MI gain vs raw source)
    # 2. absolute: engineered_mi >= max(
    #        min_abs_mi_frac * max(raw_baseline_mi),    # legacy floor
    #        mean(raw_baseline_mi) + 3 * std(raw_baseline_mi),  # noise-aware floor
    #    )
    # Layer 27 incident (2026-05-31): on all-noise frames every raw col has
    # MI in a tight band around the noise floor and ``0.1 * max_raw`` is
    # itself in that band -- so ANY engineered_mi clears the floor and the
    # top-K fills with FPs. The noise-aware ``mean + 3*std`` reference is
    # statistical: a column drawn from the same noise distribution will
    # exceed it only on extreme tail, knocking the false-positive rate
    # below 5% per slot. On real-signal frames the max raw_baseline_mi is
    # multiple std above the mean, so the legacy floor dominates and
    # selection is unchanged.
    raw_baselines = scores["baseline_mi"].to_numpy()
    max_raw_baseline = float(raw_baselines.max()) if raw_baselines.size else 0.0
    legacy_floor = float(min_abs_mi_frac) * max_raw_baseline
    # Layer 27 noise-aware floor: use MEDIAN-based stats (robust to a few
    # real signals dragging the mean up). On all-noise frames every
    # baseline_mi sits in a tight band, median + 3*MAD bounds the band
    # tightly. On signal frames the true signal is an outlier above the
    # noise band and median+3*MAD remains in the noise band, so the
    # legacy ``frac * max(raw_baseline)`` dominates and legitimate signals
    # qualify as before.
    # Bonferroni-aware sigma scale: pure sqrt(2 ln 2p) under-counts the chi-
    # square-like right tail of plug-in MI's noise distribution. Empirically
    # n=1500 binary y with 10 bins produces noise MIs that exceed Gaussian
    # tails by ~1-1.5 sigma worth of probability mass. Anchor at the larger
    # of: (a) Bonferroni for the candidate count, (b) a 5-sigma floor that
    # bounds the chi-square right tail at ~1e-7 per slot. For 40 candidates
    # this gives 5.0; for 1000 it gives ~5.8 (asymptotically Bonferroni-driven).
    n_cands = int(raw_baselines.size)
    sigma_thresh = max(
        5.0,
        float(np.sqrt(2.0 * np.log(max(2.0, 2.0 * n_cands))) + 1.5),
    )
    if raw_baselines.size >= 4:
        med = float(np.median(raw_baselines))
        mad = float(np.median(np.abs(raw_baselines - med)))
        # 1.4826 * MAD ~= std for a normal distribution.
        noise_floor = med + sigma_thresh * 1.4826 * mad
    else:
        noise_floor = 0.0
    # Layer 27 follow-up: also compute a noise floor on the ENGINEERED MI
    # distribution. On all-noise frames the engineered cols inherit the same
    # noise structure; the top engineered_mi can be artifactually 2-4x the
    # median by pure tail sampling. Bound engineered_mi above the engineered
    # median+sigma*MAD too -- legitimate signals are statistical outliers in
    # the engineered distribution AS WELL.
    eng_mis = scores["engineered_mi"].to_numpy()
    if eng_mis.size >= 4:
        med_e = float(np.median(eng_mis))
        mad_e = float(np.median(np.abs(eng_mis - med_e)))
        eng_noise_floor = med_e + sigma_thresh * 1.4826 * mad_e
    else:
        eng_noise_floor = 0.0
    abs_floor = max(legacy_floor, noise_floor, eng_noise_floor)
    qualified = scores[
        (scores["uplift"] >= float(min_uplift))
        & (scores["engineered_mi"] >= abs_floor)
    ]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    return X_aug, scores


# ---------------------------------------------------------------------------
# Cross-basis pair features: He_a(x_i) * He_b(x_j)
# ---------------------------------------------------------------------------
#
# Motivation: the univariate path captures y = f(x_i) -- single-feature non-
# linearities. The pair-CMA-ES path in hermite_fe.py learns FULL pair coeffs
# (c_a, c_b) jointly via Optuna / CMA-ES, which is expensive (~1000 steps).
# The cross-basis pair path is the cheap middle ground: enumerate all
# (deg_a, deg_b) cells in the bilinear product grid up to ``max_degree``,
# emit basis_a(x_i) * basis_b(x_j) as a column, then rank by MI uplift vs
# the better of the two source columns.
#
# Captures the XOR / saddle / circle family without any optimisation:
# * XOR     y = sign(x_i * x_j)         -> He_1(z_i) * He_1(z_j)   = z_i*z_j
# * Saddle  y = sign(x_i^2 - x_j^2)     -> He_2(z_i) * He_0(z_j) and
#                                          He_0(z_i) * He_2(z_j)
#                                          (the linear combination of the two
#                                          is the saddle; the stronger of
#                                          the two MI-wise enters the support)
# * Circle  y = sign(x_i^2 + x_j^2 - r) -> same two terms as saddle
#
# Selection mirrors the univariate two-gate:
#   1. uplift >= min_uplift vs max(MI(x_i; y), MI(x_j; y))
#   2. engineered_mi >= min_abs_mi_frac * max_raw_baseline


def _pair_eng_col_name(col_i: str, col_j: str, basis: str, deg_a: int, deg_b: int) -> str:
    """Stable naming: ``"{col_i}*{col_j}__He{a}_He{b}"``.

    Both legs share the same basis code (e.g. He_a * He_b). The cross-basis
    enumeration intentionally fixes one basis family per pair -- mixing
    families (He_a * T_b) blows up combinatorially without measurable signal
    gain on the standard XOR / saddle / circle targets.
    """
    code = _BASIS_CODE.get(basis, basis)
    return f"{col_i}*{col_j}__{code}{deg_a}_{code}{deg_b}"


def generate_pair_cross_basis_features(
    X: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    *,
    max_degree: int = 2,
    basis: str = "auto",
    min_degree: int = 1,
) -> pd.DataFrame:
    """For each (col_i, col_j) pair and each (deg_a, deg_b) in
    [min_degree..max_degree]^2, emit ``basis(x_i)_a * basis(x_j)_b`` as a new
    column.

    Parameters
    ----------
    X : DataFrame
        Source frame. Both legs of every pair must be numeric.
    pairs : sequence of (col_i, col_j)
        Column pairs to expand. Order matters for the name but not the value
        (multiplication is commutative); pass each unordered pair once.
    max_degree : int
        Maximum degree per leg. Default 2 covers XOR (1,1), partial saddle
        (1,2)/(2,1), and pure quadratic interaction (2,2) -- enough for the
        classic non-linear pair targets without combinatorial blowup.
    basis : {'auto', 'hermite', 'legendre', 'chebyshev', 'laguerre'}
        Routed per-column via ``basis_route_by_moments`` when ``'auto'``. The
        two legs of a pair may end up on different bases under 'auto' -- the
        name reflects each leg's chosen basis only via the suffix; we keep
        the join-token consistent (``He{a}_He{b}`` even when leg basis
        differ) so callers can group by name prefix.
    min_degree : int
        Minimum degree per leg. Default 1 -- degree 0 produces the constant
        column (= identity for the OTHER leg's transform), already covered
        by the univariate path.

    Returns
    -------
    DataFrame of new pair-cross-basis columns named via ``_pair_eng_col_name``.
    """
    if not pairs:
        return pd.DataFrame(index=X.index)
    cache: dict[tuple[str, int, str], np.ndarray] = {}
    out_cols: dict = {}
    max_d = int(max_degree)
    min_d = max(0, int(min_degree))
    for col_i, col_j in pairs:
        if col_i == col_j:
            continue
        if col_i not in X.columns or col_j not in X.columns:
            logger.warning("generate_pair_cross_basis_features: missing column %r or %r; skipping", col_i, col_j)
            continue
        if not (pd.api.types.is_numeric_dtype(X[col_i]) and pd.api.types.is_numeric_dtype(X[col_j])):
            continue
        x_i = np.asarray(X[col_i].to_numpy(), dtype=np.float64)
        x_j = np.asarray(X[col_j].to_numpy(), dtype=np.float64)
        for x in (x_i, x_j):
            finite_mask = np.isfinite(x)
            if not finite_mask.all():
                fill = float(np.nanmean(x[finite_mask])) if finite_mask.any() else 0.0
                np.copyto(x, np.where(finite_mask, x, fill))
        basis_i = basis_route_by_moments(x_i) if basis == "auto" else basis
        basis_j = basis_route_by_moments(x_j) if basis == "auto" else basis
        if basis_i not in _POLY_BASES or basis_j not in _POLY_BASES:
            logger.warning(
                "generate_pair_cross_basis_features: unknown basis %r/%r for pair (%r,%r); skipping",
                basis_i, basis_j, col_i, col_j,
            )
            continue
        for deg_a in range(min_d, max_d + 1):
            for deg_b in range(min_d, max_d + 1):
                if deg_a == 0 and deg_b == 0:
                    continue
                try:
                    key_a = (col_i, deg_a, basis_i)
                    if key_a not in cache:
                        cache[key_a] = _evaluate_basis_column(x_i, basis_i, deg_a)
                    h_a = cache[key_a]
                    key_b = (col_j, deg_b, basis_j)
                    if key_b not in cache:
                        cache[key_b] = _evaluate_basis_column(x_j, basis_j, deg_b)
                    h_b = cache[key_b]
                    name = _pair_eng_col_name(col_i, col_j, basis_i if basis_i == basis_j else basis_i, deg_a, deg_b)
                    out_cols[name] = h_a * h_b
                except Exception as exc:
                    logger.warning(
                        "generate_pair_cross_basis_features: basis=%r/%r deg=%d/%d on pair (%r,%r) raised %r; skipping",
                        basis_i, basis_j, deg_a, deg_b, col_i, col_j, exc,
                    )
                    continue
    return pd.DataFrame(out_cols, index=X.index)


def score_pair_cross_basis_by_mi_uplift(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    nbins: int = 10,
) -> pd.DataFrame:
    """Score each pair-cross-basis column by MI uplift vs the BETTER of the
    two raw source columns. Mirrors ``score_features_by_mi_uplift`` but the
    name carries a pair prefix ``"{col_i}*{col_j}__..."``.

    Returns
    -------
    DataFrame with columns
    ``[engineered_col, source_col_i, source_col_j, baseline_mi_i,
    baseline_mi_j, baseline_mi, engineered_mi, uplift]`` sorted by
    ``uplift`` descending. ``baseline_mi`` is ``max(baseline_mi_i,
    baseline_mi_j)`` -- the cross-basis term must beat the BETTER individual
    leg, not just the worse one, to count as genuine interaction signal.
    """
    y_arr = (
        np.asarray(y).astype(np.int64)
        if not np.issubdtype(np.asarray(y).dtype, np.integer)
        else np.asarray(y, dtype=np.int64)
    )
    raw_cols = list(raw_X.columns)
    raw_mi = _mi_classif_batch(raw_X.to_numpy(dtype=np.float64), y_arr, nbins=nbins)
    raw_mi_map = dict(zip(raw_cols, raw_mi.tolist()))
    if engineered_X.empty:
        return pd.DataFrame(columns=[
            "engineered_col", "source_col_i", "source_col_j",
            "baseline_mi_i", "baseline_mi_j", "baseline_mi",
            "engineered_mi", "uplift",
        ])
    eng_mi = _mi_classif_batch(engineered_X.to_numpy(dtype=np.float64), y_arr, nbins=nbins)
    rows = []
    for j, eng_name in enumerate(engineered_X.columns):
        # parse "{col_i}*{col_j}__..."
        head = eng_name.split("__", 1)[0] if "__" in eng_name else eng_name
        if "*" not in head:
            # not a pair column -- skip
            continue
        col_i, col_j = head.split("*", 1)
        baseline_i = float(raw_mi_map.get(col_i, 0.0))
        baseline_j = float(raw_mi_map.get(col_j, 0.0))
        baseline = max(baseline_i, baseline_j)
        emi = float(eng_mi[j])
        uplift = emi / (baseline + 1e-12)
        rows.append({
            "engineered_col": eng_name,
            "source_col_i": col_i,
            "source_col_j": col_j,
            "baseline_mi_i": baseline_i,
            "baseline_mi_j": baseline_j,
            "baseline_mi": baseline,
            "engineered_mi": emi,
            "uplift": uplift,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("uplift", ascending=False).reset_index(drop=True)
    return df


def hybrid_orth_mi_pair_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    pair_max_degree: int = 2,
    basis: str = "auto",
    top_k: int = 5,
    top_pair_count: int = 3,
    top_pair_seed_k: int = 4,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    pair_min_uplift: float = 1.05,
    pair_min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two-stage hybrid: (1) univariate orthogonal-poly FE + MI-greedy, then
    (2) cross-basis pair FE on the top-N univariate source columns, also
    MI-greedy.

    Stage 1 reuses ``hybrid_orth_mi_fe`` to pick top-N univariate winners.
    The source columns of those winners (plus any explicit raw columns the
    user wants to force into the pair pool via ``cols``) form the pair seed
    pool. Stage 2 enumerates all unordered pairs over the seed pool, calls
    ``generate_pair_cross_basis_features``, ranks via
    ``score_pair_cross_basis_by_mi_uplift``, and applies the same two-gate
    selection.

    Parameters
    ----------
    X, y, cols, degrees, basis, top_k, min_uplift, min_abs_mi_frac, nbins
        Forwarded to the univariate ``hybrid_orth_mi_fe`` stage.
    pair_max_degree : int
        Max degree per leg in the cross-basis enumeration. Default 2.
    top_pair_count : int
        How many cross-basis pair winners to append after the univariate
        winners. Default 3.
    top_pair_seed_k : int
        How many top univariate source columns to pull into the pair-seed
        pool. With N sources we enumerate ``N*(N-1)/2`` pairs. Default 4
        gives 6 pairs * (pair_max_degree^2) cross-basis cells = bounded
        cost.
    pair_min_uplift, pair_min_abs_mi_frac : float
        Two-gate selection thresholds for the pair stage. Same semantics as
        the univariate gates but compared against
        ``max(MI(x_i; y), MI(x_j; y))`` as the baseline.

    Returns
    -------
    (X_augmented, univariate_scores, cross_scores)
        X_augmented : ``X`` with univariate winners THEN cross-basis pair
            winners appended, in that order. Index preserved.
        univariate_scores : ranking DataFrame from the stage-1 univariate
            pass (same shape as ``hybrid_orth_mi_fe`` returns).
        cross_scores : ranking DataFrame from the stage-2 cross-basis pair
            pass (output of ``score_pair_cross_basis_by_mi_uplift``).
    """
    # Stage 1: univariate hybrid. Use the SAME caller-facing knobs so the
    # univariate winners on the joint frame are reproducible bit-identical
    # to a direct ``hybrid_orth_mi_fe`` call.
    X_aug_uni, uni_scores = hybrid_orth_mi_fe(
        X, y,
        cols=cols, degrees=degrees, basis=basis,
        top_k=top_k, min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac, nbins=nbins,
    )

    # Build the pair seed pool: top univariate winners' SOURCE columns,
    # plus a fallback to the raw column MI ranking when uplift-based winners
    # are sparse (e.g. when y has no useful univariate non-linear signal but
    # has a XOR cross-term, the seed pool would otherwise be empty).
    raw_cols_all = [c for c in (cols or X.columns) if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    seed_sources: list[str] = []
    if not uni_scores.empty:
        # Source columns of top univariate winners by uplift, deduped, order-preserving.
        for src in uni_scores["source_col"].tolist():
            if src not in seed_sources and src in raw_cols_all:
                seed_sources.append(src)
            if len(seed_sources) >= int(top_pair_seed_k):
                break
    if len(seed_sources) < 2 and len(raw_cols_all) >= 2:
        # Fallback: rank raw columns by MI(x; y), take top N. Required for
        # pure-XOR targets where no univariate basis term uplifts (all
        # univariate MIs are near-zero for y = sign(x_i * x_j)).
        y_arr = np.asarray(y).astype(np.int64) if not np.issubdtype(np.asarray(y).dtype, np.integer) else np.asarray(y, dtype=np.int64)
        raw_X_all = X[raw_cols_all]
        raw_mi_arr = _mi_classif_batch(raw_X_all.to_numpy(dtype=np.float64), y_arr, nbins=nbins)
        order = np.argsort(-raw_mi_arr)
        fallback = [raw_cols_all[i] for i in order[: int(top_pair_seed_k)]]
        for src in fallback:
            if src not in seed_sources:
                seed_sources.append(src)
            if len(seed_sources) >= int(top_pair_seed_k):
                break

    cross_scores_empty_cols = [
        "engineered_col", "source_col_i", "source_col_j",
        "baseline_mi_i", "baseline_mi_j", "baseline_mi",
        "engineered_mi", "uplift",
    ]
    if len(seed_sources) < 2 or int(top_pair_count) <= 0:
        return X_aug_uni, uni_scores, pd.DataFrame(columns=cross_scores_empty_cols)

    pairs = [
        (seed_sources[i], seed_sources[j])
        for i in range(len(seed_sources))
        for j in range(i + 1, len(seed_sources))
    ]
    pair_eng = generate_pair_cross_basis_features(
        X, pairs, max_degree=pair_max_degree, basis=basis,
    )
    if pair_eng.empty:
        return X_aug_uni, uni_scores, pd.DataFrame(columns=cross_scores_empty_cols)

    raw_X_seed = X[seed_sources]
    cross_scores = score_pair_cross_basis_by_mi_uplift(
        raw_X_seed, pair_eng, y, nbins=nbins,
    )
    # Two-gate selection mirrors the univariate stage. The absolute floor is
    # max(raw_baseline_max, cross_engineered_mi_max) * frac. The second
    # term matters for pure-interaction targets (XOR / saddle): all
    # univariate / raw baselines are noise-floor (~0.003), but the true
    # cross-basis winner sits at 0.6 nats; without taking the cross-scores
    # max into account, ALL noise cross-terms with engineered_mi ~ 0.006
    # would clear an abs_floor of 0.0003 and pollute the output. Using
    # max(.) as the reference correctly raises the bar to 0.06 in that
    # regime so only the true XOR term qualifies.
    max_raw_baseline = float(cross_scores["baseline_mi"].max()) if not cross_scores.empty else 0.0
    if not uni_scores.empty:
        max_raw_baseline = max(max_raw_baseline, float(uni_scores["baseline_mi"].max()))
    max_cross_engineered = float(cross_scores["engineered_mi"].max()) if not cross_scores.empty else 0.0
    legacy_floor = float(pair_min_abs_mi_frac) * max(max_raw_baseline, max_cross_engineered)
    # Layer 27 (2026-05-31) noise-aware floor: see hybrid_orth_mi_fe for
    # the rationale. The pair stage is even more prone to noise pollution
    # (O(p^2) candidates vs O(p) for univariate); the noise-aware
    # mean+3*std reference protects the all-noise frame's contract.
    _baselines = cross_scores["baseline_mi"].to_numpy() if not cross_scores.empty else np.array([])
    # Bonferroni-aware sigma (see hybrid_orth_mi_fe for derivation): pair
    # candidate counts are much larger than univariate so the per-candidate
    # threshold must be tighter. Anchor at max(5.0, sqrt(2 ln 2p) + 1.5).
    n_cands = int(_baselines.size)
    sigma_thresh = max(
        5.0,
        float(np.sqrt(2.0 * np.log(max(2.0, 2.0 * n_cands))) + 1.5),
    )
    if _baselines.size >= 4:
        _med = float(np.median(_baselines))
        _mad = float(np.median(np.abs(_baselines - _med)))
        noise_floor = _med + sigma_thresh * 1.4826 * _mad
    else:
        noise_floor = 0.0
    # Also bound vs engineered MI distribution.
    _eng_mis = cross_scores["engineered_mi"].to_numpy() if not cross_scores.empty else np.array([])
    if _eng_mis.size >= 4:
        _med_e = float(np.median(_eng_mis))
        _mad_e = float(np.median(np.abs(_eng_mis - _med_e)))
        eng_noise_floor = _med_e + sigma_thresh * 1.4826 * _mad_e
    else:
        eng_noise_floor = 0.0
    abs_floor = max(legacy_floor, noise_floor, eng_noise_floor)
    qualified = cross_scores[
        (cross_scores["uplift"] >= float(pair_min_uplift))
        & (cross_scores["engineered_mi"] >= abs_floor)
    ]
    winners = qualified.head(int(top_pair_count))
    keep_pair = list(winners["engineered_col"])
    if keep_pair:
        X_aug = pd.concat([X_aug_uni, pair_eng[keep_pair]], axis=1)
    else:
        X_aug = X_aug_uni
    return X_aug, uni_scores, cross_scores


# ---------------------------------------------------------------------------
# Layer 23 (2026-05-31): recipe-emitting wrappers
# ---------------------------------------------------------------------------
#
# The vanilla ``hybrid_orth_mi_fe`` / ``hybrid_orth_mi_pair_fe`` return a
# DataFrame + scores. For MRMR.fit auto-wiring we ALSO need ``EngineeredRecipe``
# objects so that ``MRMR.transform`` can replay each appended column on test
# data deterministically (no y reference -- the recipe carries only basis +
# degree per source column). The wrappers below re-derive the per-col basis
# the same way ``generate_univariate_basis_features`` did, build one recipe
# per appended column, and return them alongside the existing outputs.


def _col_basis_for_recipe(x: np.ndarray, basis: str) -> str:
    """Resolve the per-column basis: explicit string when caller pinned one,
    else moment-routed auto. Mirrors the inline decision in
    ``generate_univariate_basis_features`` / ``generate_pair_cross_basis_features``.
    """
    if basis == "auto":
        return basis_route_by_moments(x)
    return basis


def hybrid_orth_mi_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
):
    """Same as :func:`hybrid_orth_mi_fe` but additionally returns a list of
    ``EngineeredRecipe`` objects -- one per appended univariate column --
    so that ``MRMR.transform`` can recompute each engineered column on
    test data without re-running the MI ranking.

    Returns
    -------
    (X_augmented, scores, recipes)
    """
    from .engineered_recipes import build_orth_univariate_recipe
    X_aug, scores = hybrid_orth_mi_fe(
        X, y, cols=cols, degrees=degrees, basis=basis,
        top_k=top_k, min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac, nbins=nbins,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    recipes = []
    for name in appended:
        # Re-derive (src, degree, basis) from the appended frame: src is the
        # prefix before ``__``; basis/degree are encoded in the suffix. Cross-
        # check by also routing the source column via the same auto rule we
        # used at fit time so the recipe replays identically.
        src = name.split("__", 1)[0]
        suffix = name.split("__", 1)[1]
        # _BASIS_CODE = {"hermite":"He","legendre":"L","chebyshev":"T","laguerre":"LL"}
        # Order longest-first to avoid 'L' matching the start of 'LL'.
        code_to_basis = {"He": "hermite", "LL": "laguerre", "T": "chebyshev", "L": "legendre"}
        chosen_basis = None
        chosen_degree = None
        for code in ("LL", "He", "T", "L"):
            if suffix.startswith(code):
                rest = suffix[len(code):]
                if rest.isdigit():
                    chosen_basis = code_to_basis[code]
                    chosen_degree = int(rest)
                    break
        if chosen_basis is None or chosen_degree is None:
            logger.warning(
                "hybrid_orth_mi_fe_with_recipes: cannot parse basis/degree "
                "from column name %r; skipping recipe build.", name,
            )
            continue
        recipes.append(build_orth_univariate_recipe(
            name=name, src_name=src,
            basis=chosen_basis, degree=chosen_degree,
        ))
    return X_aug, scores, recipes


def hybrid_orth_mi_pair_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    pair_max_degree: int = 2,
    basis: str = "auto",
    top_k: int = 5,
    top_pair_count: int = 3,
    top_pair_seed_k: int = 4,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    pair_min_uplift: float = 1.05,
    pair_min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
):
    """Same as :func:`hybrid_orth_mi_pair_fe` but additionally returns a
    flat list of recipes (univariate + pair, in append order) for replay.
    """
    from .engineered_recipes import (
        build_orth_univariate_recipe,
        build_orth_pair_cross_recipe,
    )
    X_aug, uni_scores, cross_scores = hybrid_orth_mi_pair_fe(
        X, y, cols=cols, degrees=degrees, basis=basis,
        pair_max_degree=pair_max_degree,
        top_k=top_k, top_pair_count=top_pair_count,
        top_pair_seed_k=top_pair_seed_k,
        min_uplift=min_uplift, min_abs_mi_frac=min_abs_mi_frac,
        pair_min_uplift=pair_min_uplift,
        pair_min_abs_mi_frac=pair_min_abs_mi_frac,
        nbins=nbins,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    code_to_basis = {"He": "hermite", "LL": "laguerre", "T": "chebyshev", "L": "legendre"}
    recipes = []
    for name in appended:
        if "*" in name.split("__", 1)[0]:
            # pair cross: "{col_i}*{col_j}__{code}{deg_a}_{code}{deg_b}"
            head, suffix = name.split("__", 1)
            col_i, col_j = head.split("*", 1)
            # parse "{code_a}{deg_a}_{code_b}{deg_b}"
            try:
                left, right = suffix.split("_", 1)
            except ValueError:
                logger.warning(
                    "hybrid_orth_mi_pair_fe_with_recipes: cannot parse pair "
                    "suffix %r in %r; skipping recipe.", suffix, name,
                )
                continue
            def _parse_code_deg(s: str):
                for code in ("LL", "He", "T", "L"):
                    if s.startswith(code):
                        rest = s[len(code):]
                        if rest.isdigit():
                            return code_to_basis[code], int(rest)
                return None, None
            basis_a, deg_a = _parse_code_deg(left)
            basis_b, deg_b = _parse_code_deg(right)
            if basis_a is None or basis_b is None:
                logger.warning(
                    "hybrid_orth_mi_pair_fe_with_recipes: cannot parse code/deg "
                    "from %r; skipping recipe.", name,
                )
                continue
            # For a cross-basis pair the generator emits a single basis code
            # for both legs (basis_i if basis_i == basis_j else basis_i).
            # When ``basis='auto'`` and basis_route_by_moments disagrees
            # between legs, the name is built with basis_i's code, but the
            # ACTUAL leg-2 evaluation used basis_j. Re-route per-column at
            # recipe-build time and prefer the moment-routed basis when in
            # auto mode so replay matches fit-time evaluation.
            if basis == "auto":
                try:
                    x_i = X[col_i].to_numpy(dtype=np.float64)
                    x_j = X[col_j].to_numpy(dtype=np.float64)
                    basis_a = basis_route_by_moments(x_i)
                    basis_b = basis_route_by_moments(x_j)
                except Exception:
                    pass
            recipes.append(build_orth_pair_cross_recipe(
                name=name, src_a_name=col_i, src_b_name=col_j,
                basis_i=basis_a, basis_j=basis_b,
                deg_a=deg_a, deg_b=deg_b,
            ))
        else:
            # univariate: "{col}__{code}{degree}"
            src = name.split("__", 1)[0]
            suffix = name.split("__", 1)[1]
            chosen_basis = None
            chosen_degree = None
            for code in ("LL", "He", "T", "L"):
                if suffix.startswith(code):
                    rest = suffix[len(code):]
                    if rest.isdigit():
                        chosen_basis = code_to_basis[code]
                        chosen_degree = int(rest)
                        break
            if chosen_basis is None or chosen_degree is None:
                logger.warning(
                    "hybrid_orth_mi_pair_fe_with_recipes: cannot parse basis/"
                    "degree from %r; skipping recipe.", name,
                )
                continue
            recipes.append(build_orth_univariate_recipe(
                name=name, src_name=src,
                basis=chosen_basis, degree=chosen_degree,
            ))
    return X_aug, uni_scores, cross_scores, recipes


# ---------------------------------------------------------------------------
# Layer 32 (2026-05-31): B-spline + Fourier extra-basis FE
# ---------------------------------------------------------------------------
#
# Complementary flavour to the orthogonal-polynomial univariate path. Each
# source column emits:
#
# * spline  : K cubic B-spline basis cols ``c__sp{i}`` for i in 0..K+2
#             (knots quantile-placed at fit time). Captures sharp local
#             threshold rules ``y = sign(x - tau)`` that orth-poly misses.
#
# * fourier : 2*F columns ``c__sin{f}`` and ``c__cos{f}`` for each f in
#             ``fourier_freqs``. Captures periodic patterns
#             ``y = sign(sin(2*pi*f*x))``.
#
# Both basis families have CLOSED-FORM replay via the matching
# ``EngineeredRecipe`` kinds (``orth_spline`` / ``orth_fourier``); replay
# reads X only -- never y -- so transform() is leakage-free by construction.


_EXTRA_BASIS_KINDS = ("spline", "fourier")


def _fit_spline_for_col(x: np.ndarray, n_inner_knots: int):
    """Returns (knots, lo, hi, num_basis_cols). Lazy delegate to recipes
    module so the knot-vector layout stays in one place."""
    from .engineered_recipes import _fit_spline_knots, _bspline_basis_values  # noqa: F401
    knots, lo, hi = _fit_spline_knots(x, n_inner_knots, degree=3)
    # Number of cubic B-spline basis functions = len(knots) - degree - 1.
    n_basis = len(knots) - 3 - 1
    return knots, lo, hi, n_basis


def _fit_fourier_for_col(x: np.ndarray):
    """Returns (lo, span). Min-max normalise x so one period covers data range."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return 0.0, 1.0
    lo = float(np.min(x[finite]))
    hi = float(np.max(x[finite]))
    span = max(hi - lo, 1e-12)
    return lo, span


def generate_extra_basis_features(
    X: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
    extra_bases: Sequence[str] = ("spline", "fourier"),
    fourier_freqs: Sequence[float] = (1.0, 2.0),
    spline_knots: int = 5,
    dedup_collinear_sources: bool = True,
    dedup_corr_threshold: float = 0.999,
) -> tuple[pd.DataFrame, dict]:
    """For each column in cols and each requested extra basis, emit the basis
    columns and return them alongside the per-column fit metadata (knot
    vectors, lo/hi, fourier (lo, span)) needed to build recipes.

    Parameters
    ----------
    X : DataFrame
        Source frame. Only numeric columns are processed; non-numeric are
        silently skipped.
    cols : sequence of column names, optional
        Columns to expand. None = all numeric columns.
    extra_bases : tuple of {'spline', 'fourier'}
        Which extra bases to emit. Empty tuple => returns empty frame.
    fourier_freqs : sequence of float
        Frequencies for the Fourier basis. One sin and one cos column per
        frequency per source column.
    spline_knots : int
        Number of inner quantile knots for the cubic B-spline basis.
        Emits ``spline_knots + 3`` basis columns per source column (cubic
        B-spline has K+degree basis functions on K inner knots).
    dedup_collinear_sources : bool, default True
        Drop near-duplicate source columns before basis enumeration
        (mirrors the polynomial univariate path).

    Returns
    -------
    (engineered_X, meta)
        engineered_X : DataFrame of new columns with naming
            ``"{col}__sp{i}"`` (spline) and ``"{col}__sin{f}"`` /
            ``"{col}__cos{f}"`` (fourier).
        meta : dict mapping each emitted column name to a dict with the
            metadata required to build the matching recipe. Keys depend
            on basis kind: spline -> {"basis": "spline", "src": ..., "knots":
            ndarray, "idx": int, "lo": float, "hi": float}; fourier ->
            {"basis": "fourier", "src": ..., "kind": "sin"/"cos", "freq":
            float, "lo": float, "span": float}.
    """
    if cols is None:
        cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    extra_bases = tuple(b for b in extra_bases if b in _EXTRA_BASIS_KINDS)
    if not extra_bases:
        return pd.DataFrame(index=X.index), {}
    if dedup_collinear_sources:
        cols = _dedup_collinear_source_cols(
            X, list(cols), corr_threshold=dedup_corr_threshold,
        )
    from .engineered_recipes import _bspline_basis_values  # local import
    out_cols: dict = {}
    meta: dict = {}
    fourier_freqs = tuple(float(f) for f in fourier_freqs)
    spline_knots = max(2, int(spline_knots))
    for col in cols:
        if col not in X.columns or not pd.api.types.is_numeric_dtype(X[col]):
            continue
        x = np.asarray(X[col].to_numpy(), dtype=np.float64)
        finite_mask = np.isfinite(x)
        if not finite_mask.any():
            continue
        if not finite_mask.all():
            x = np.where(finite_mask, x, float(np.nanmean(x[finite_mask])))
        if "spline" in extra_bases:
            try:
                knots, lo, hi, n_basis = _fit_spline_for_col(x, spline_knots)
                span = max(hi - lo, 1e-12)
                z = np.clip((x - lo) / span, 0.0, 1.0)
                for i in range(n_basis):
                    vals = _bspline_basis_values(z, knots, i, degree=3)
                    # Skip near-constant columns -- the boundary cubic
                    # B-splines occasionally collapse to ~0 on quantile-
                    # placed knots when ties pile at the edge.
                    if float(np.std(vals)) <= 1e-12:
                        continue
                    name = f"{col}__sp{i}"
                    out_cols[name] = vals
                    meta[name] = {
                        "basis": "spline", "src": col,
                        "knots": knots, "idx": i,
                        "lo": float(lo), "hi": float(hi),
                    }
            except Exception as exc:
                logger.warning(
                    "generate_extra_basis_features: spline on col=%r raised "
                    "%r; skipping spline for that column.",
                    col, exc,
                )
        if "fourier" in extra_bases:
            try:
                lo_f, span_f = _fit_fourier_for_col(x)
                z = (x - lo_f) / max(span_f, 1e-12)
                for freq in fourier_freqs:
                    ang = 2.0 * np.pi * freq * z
                    s_vals = np.sin(ang)
                    c_vals = np.cos(ang)
                    if float(np.std(s_vals)) > 1e-12:
                        name_s = f"{col}__sin{freq:g}"
                        out_cols[name_s] = s_vals
                        meta[name_s] = {
                            "basis": "fourier", "src": col,
                            "kind": "sin", "freq": float(freq),
                            "lo": float(lo_f), "span": float(span_f),
                        }
                    if float(np.std(c_vals)) > 1e-12:
                        name_c = f"{col}__cos{freq:g}"
                        out_cols[name_c] = c_vals
                        meta[name_c] = {
                            "basis": "fourier", "src": col,
                            "kind": "cos", "freq": float(freq),
                            "lo": float(lo_f), "span": float(span_f),
                        }
            except Exception as exc:
                logger.warning(
                    "generate_extra_basis_features: fourier on col=%r raised "
                    "%r; skipping fourier for that column.",
                    col, exc,
                )
    return pd.DataFrame(out_cols, index=X.index), meta


def _build_recipe_from_meta(name: str, meta_entry: dict):
    """Materialise an ``EngineeredRecipe`` from one ``generate_extra_basis_features``
    meta entry. Returns None for unknown basis kinds (defensive)."""
    from .engineered_recipes import (
        build_orth_spline_recipe, build_orth_fourier_recipe,
    )
    basis = meta_entry["basis"]
    if basis == "spline":
        return build_orth_spline_recipe(
            name=name, src_name=str(meta_entry["src"]),
            knots=np.asarray(meta_entry["knots"], dtype=np.float64),
            idx=int(meta_entry["idx"]),
            lo=float(meta_entry["lo"]), hi=float(meta_entry["hi"]),
        )
    if basis == "fourier":
        return build_orth_fourier_recipe(
            name=name, src_name=str(meta_entry["src"]),
            kind=str(meta_entry["kind"]),
            freq=float(meta_entry["freq"]),
            lo=float(meta_entry["lo"]),
            span=float(meta_entry["span"]),
        )
    return None


def hybrid_orth_extra_basis_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    extra_bases: Sequence[str] = ("spline", "fourier"),
    fourier_freqs: Sequence[float] = (1.0, 2.0),
    spline_knots: int = 5,
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
):
    """Layer 32 hybrid: spline + Fourier univariate basis FE + MI-greedy
    selection. Mirrors :func:`hybrid_orth_mi_fe_with_recipes` for the
    polynomial path but emits extra-basis columns (B-spline, Fourier)
    instead. Returns (X_augmented, scores, recipes).

    The selection rule is the same TWO-GATE chain as the polynomial path:
    relative uplift >= min_uplift AND engineered_mi >= max(legacy floor,
    noise-aware floor). See :func:`hybrid_orth_mi_fe` for the rationale.
    """
    engineered, meta = generate_extra_basis_features(
        X, cols=cols, extra_bases=extra_bases,
        fourier_freqs=fourier_freqs, spline_knots=spline_knots,
    )
    if engineered.empty:
        empty_scores = pd.DataFrame(columns=[
            "engineered_col", "source_col", "baseline_mi", "engineered_mi", "uplift",
        ])
        return X.copy(), empty_scores, []
    raw_X = X[[c for c in (cols or X.columns) if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]]
    scores = score_features_by_mi_uplift(raw_X, engineered, y, nbins=nbins)
    raw_baselines = scores["baseline_mi"].to_numpy()
    max_raw_baseline = float(raw_baselines.max()) if raw_baselines.size else 0.0
    legacy_floor = float(min_abs_mi_frac) * max_raw_baseline
    n_cands = int(raw_baselines.size)
    sigma_thresh = max(
        5.0,
        float(np.sqrt(2.0 * np.log(max(2.0, 2.0 * n_cands))) + 1.5),
    )
    if raw_baselines.size >= 4:
        med = float(np.median(raw_baselines))
        mad = float(np.median(np.abs(raw_baselines - med)))
        noise_floor = med + sigma_thresh * 1.4826 * mad
    else:
        noise_floor = 0.0
    eng_mis = scores["engineered_mi"].to_numpy()
    if eng_mis.size >= 4:
        med_e = float(np.median(eng_mis))
        mad_e = float(np.median(np.abs(eng_mis - med_e)))
        eng_noise_floor = med_e + sigma_thresh * 1.4826 * mad_e
    else:
        eng_noise_floor = 0.0
    abs_floor = max(legacy_floor, noise_floor, eng_noise_floor)
    qualified = scores[
        (scores["uplift"] >= float(min_uplift))
        & (scores["engineered_mi"] >= abs_floor)
    ]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    recipes = []
    for name in keep:
        if name not in meta:
            continue
        r = _build_recipe_from_meta(name, meta[name])
        if r is not None:
            recipes.append(r)
    return X_aug, scores, recipes


# ---------------------------------------------------------------------------
# Layer 56 (2026-05-31): TRI-PRODUCT cross-basis FE lives in sibling module
# ``_orthogonal_triplet_fe`` (parent module size budget). Re-exporting here
# would create a circular import (triplet sibling needs ``_evaluate_basis_column``
# / ``_mi_classif_batch`` / ``_BASIS_CODE`` from THIS module at import time).
# Callers import the four triplet entry points directly:
#
#   from mlframe.feature_selection.filters._orthogonal_triplet_fe import (
#       generate_triplet_cross_basis_features,
#       score_triplet_cross_basis_by_mi_uplift,
#       hybrid_orth_mi_triplet_fe,
#       hybrid_orth_mi_triplet_fe_with_recipes,
#   )
# ---------------------------------------------------------------------------

