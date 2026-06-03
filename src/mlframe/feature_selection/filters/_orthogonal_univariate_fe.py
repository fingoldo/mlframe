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
    "basis_route_by_signal",
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
    preprocess_params: Optional[dict] = None,
    return_params: bool = False,
):
    """Preprocess x to the basis domain, then evaluate the single basis function
    of given degree via a one-hot coefficient vector. Returns shape (n,) -- or
    ``(values, params)`` when ``return_params=True``.

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

    2026-06-03 (audit cluster-aggregate-6) -- ``preprocess_params`` / ``return_-
    params``: persist-and-replay the fit-time preprocess. With ``preprocess_-
    params`` set, the basis preprocess is APPLIED with the stored params instead
    of refit on ``x`` -- so a recipe replayed on drifted test data maps a given
    row to the same engineered value as at fit. With ``return_params=True`` the
    fit path also returns the params it computed so the caller can persist them.
    """
    basis_info = _POLY_BASES[basis]
    fit_fn = basis_info["fit"]
    if preprocess_params is not None:
        # REPLAY: apply the stored fit-time params; never refit on test x.
        apply_fn = basis_info["apply"]
        z = apply_fn(np.asarray(x, dtype=np.float64), preprocess_params)
        params = preprocess_params
    elif aux_for_fit is not None and len(aux_for_fit) > 0:
        # Build params from the concatenated pool, then apply to ``x`` only.
        aux = np.asarray(aux_for_fit, dtype=np.float64)
        finite_aux = aux[np.isfinite(aux)]
        if finite_aux.size > 0:
            pool = np.concatenate([np.asarray(x, dtype=np.float64), finite_aux])
            _z_pool, params = fit_fn(pool)
            apply_fn = basis_info["apply"]
            z = apply_fn(np.asarray(x, dtype=np.float64), params)
        else:
            z, params = fit_fn(x)
    else:
        z, params = fit_fn(x)
    z = np.ascontiguousarray(z, dtype=np.float64)
    # One-hot coefficient vector: He_n / L_n / T_n / L^Lag_n at the chosen degree.
    coef = np.zeros(degree + 1, dtype=np.float64)
    coef[degree] = 1.0
    out = polyeval_dispatch(basis, z, coef)
    return (out, params) if return_params else out


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


def basis_route_by_signal(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degrees: Sequence[int] = (2, 3),
    candidate_bases: Sequence[str] = _POLY_BASES,
    aux_for_fit: Optional[np.ndarray] = None,
) -> str:
    """Signal-adaptive orthogonal-polynomial basis routing (2026-06-03).

    Choose the basis whose best low-degree expansion is most LINEARLY usable for
    ``y`` (max ``|Pearson corr|`` over ``degrees``). The legacy
    ``basis_route_by_moments`` picks the basis from the marginal distribution of
    ``x`` ALONE (skew / kurtosis / spread), which mis-routes whenever the target's
    best linearising basis is not x's distributional "home" basis: a heavy-tailed
    or skewed x whose target is a clean polynomial is far better linearised by the
    z-scored Hermite expansion than by the moment-preferred Chebyshev / Laguerre.

    Bench (benchmarks/bench_basis_routing equivalent, 2026-06-03, 30 cases x 3
    seeds): moment-routing picked the signal-best basis in only 19/30 (63%); mean
    |corr| gap +0.128, max +0.80. Catastrophic moment mis-routes fixed -- heavy-
    tailed cubic 0.17->0.93, gamma cubic 0.43->0.92, lognormal-square 0.68->0.92.

    Routing by linear usability (|corr|) rather than raw MI is deliberate: MI is
    monotone-invariant and would pick a basis whose feature is informative but NOT
    linearly usable by the shallow downstream the FE feeds (the project's
    MI-vs-linear-usability principle). The chosen basis is fixed into the
    EngineeredRecipe at fit time and replayed deterministically at transform time
    (no y needed at transform -> leakage-free). Falls back to
    ``basis_route_by_moments`` when y is degenerate / size-mismatched / too small.
    """
    x = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64).ravel()
    if (x.size < 30 or yv.size != x.size
            or not np.isfinite(yv).all() or float(np.std(yv)) < 1e-12):
        return basis_route_by_moments(x)
    best_basis = None
    best_corr = -1.0
    for basis in candidate_bases:
        bcorr = 0.0
        for d in degrees:
            try:
                v = _evaluate_basis_column(x, basis, int(d), aux_for_fit=aux_for_fit)
            except Exception:
                continue
            v = np.asarray(v, dtype=np.float64)
            if v.size != yv.size or not np.all(np.isfinite(v)) or float(np.std(v)) < 1e-12:
                continue
            c = abs(float(np.corrcoef(v, yv)[0, 1]))
            if np.isfinite(c) and c > bcorr:
                bcorr = c
        if bcorr > best_corr:
            best_corr = bcorr
            best_basis = basis
    if best_basis is None:
        return basis_route_by_moments(x)
    return best_basis


def generate_univariate_basis_features(
    X: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    dedup_collinear_sources: bool = True,
    dedup_corr_threshold: float = 0.999,
    y: Optional[np.ndarray] = None,
    basis_routing: str = "signal",
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
            # 2026-06-03: signal-adaptive routing (route by which basis best
            # LINEARISES y) beats moment-routing on both linear and tree OOS
            # recovery (bench: corr-routing linear R^2 0.919 vs MI 0.769, tree
            # 0.852 vs 0.829; moment-routing mis-routed 11/30 cases, catastrophic
            # on heavy-tailed / skewed x). Requires y; falls back to moment
            # routing when y is unavailable (standalone callers) or degenerate.
            if basis_routing == "signal" and y is not None:
                chosen_basis = basis_route_by_signal(
                    x, np.asarray(y), degrees=degrees, aux_for_fit=aux_col,
                )
            elif aux_col is not None and len(aux_col) > 0:
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
    engineered = generate_univariate_basis_features(X, cols=cols, degrees=degrees, basis=basis, y=y)
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

    Notes
    -----
    bench-rejected (2026-06-03): "product-signal JOINT routing" -- choosing the
    (basis_a, deg_a, basis_b, deg_b) cell that maximises ``|corr(basis_a(x_i)*
    basis_b(x_j), y)|`` instead of moment-routing each leg -- was benchmarked and
    REJECTED. Premise (from a poly-synergy probe) was that per-leg routing never
    materialises the Hermite leg of a pure-synergy product like ``He2(a)*b``. False
    for THIS path: moment-routing sends a Gaussian leg to Hermite / a bounded leg to
    Chebyshev regardless of marginal corr, then the (deg_a,deg_b) sweep + MI-uplift
    scorer already keeps the synergy cell (``He2(a)*b`` recovered |corr|=0.998,
    mixed ``He2(a)*T2(b)`` 0.999). The product search gave ZERO lift on the synergy
    targets, REGRESSED the plain ``a*b`` control (classif |corr| 1.000->0.877, a
    best-of-144-cells selection-bias swap), and HIJACKED the univariate ``He3(a)``
    control (manufactured spurious weaker pairs 0.60-0.96, leak-free over-search).
    Don't re-add joint product routing here. (D:/Temp/item5_product_routing_findings.md)
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
    module so the knot-vector layout stays in one place.

    Knots are placed at QUANTILES of x (unsupervised). bench-rejected
    (2026-06-03): a TARGET-SUPERVISED knot strategy (knots at a shallow x->y
    tree's splits / conditional-mean curvature) was benchmarked and REJECTED.
    (1) In the real MRMR pipeline NO spline column -- quantile OR supervised --
    survives the MI-uplift gate, so the support is byte-identical with either
    strategy (the gate, not knot placement, is the binding constraint; and
    supervised knots score LOWER at the gate -- narrower, individually-lower-MI
    basis columns). (2) Even at the raw block level the win reverses by shape:
    supervised wins a narrow bump (|corr| 0.884 vs 0.614) but LOSES a sharp step
    (0.793 vs 0.931) and kink (0.719 vs 0.933). (3) The one shape it helps is
    already recovered by the default-on Fourier multitone. Leak-safety would have
    held (knots baked into the recipe, replay reads only knots/lo/hi), but moot.
    Don't add fe_spline_knot_strategy="supervised". (D:/Temp/item7_supervised_knots_findings.md)
    """
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


def _fit_chirp_warp_for_col(x: np.ndarray):
    """Fit the QUADRATIC-ARGUMENT ("chirp") warp params on ``x`` (2026-06-03).

    The chirp axis is ``u = sign(z) * z**2`` where ``z = (x - mean) / std``.
    Squaring the STANDARDISED z (signed, so the map stays monotone and one-to-one
    across the whole real line rather than folding ``x`` and ``-x`` together)
    turns an oscillation whose frequency GROWS with the argument
    (``y ~ sin(2*pi*f*z**2)``) into a STATIONARY-frequency sinusoid in ``u`` --
    which the existing periodogram peak-search then locks onto. A Fourier on the
    LINEAR argument cannot represent a frequency that grows with z (Phase-0 bench:
    linear multitone R^2 0.07-0.53 vs chirp warp 0.88 on a fast chirp f=2.5).

    Returns ``(mean, std, lo, span)``:
    * ``mean`` / ``std`` -- standardisation of x into z (fit on the finite subset).
    * ``lo`` / ``span``  -- min / range of ``u`` so ``(u - lo) / span`` lands the
      warped axis in [0, 1], matching the linear emitter's z normalisation so the
      same coarse frequency grid applies.

    All four are baked into the recipe at fit time and replayed verbatim at
    transform time (no y, so leakage-free)."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return 0.0, 1.0, 0.0, 1.0
    xf = x[finite]
    mean = float(np.mean(xf))
    std = float(np.std(xf))
    std = std if std > 1e-12 else 1.0
    z = (xf - mean) / std
    u = np.sign(z) * (z * z)
    lo = float(np.min(u))
    hi = float(np.max(u))
    span = max(hi - lo, 1e-12)
    return mean, std, lo, span


def _chirp_axis(x: np.ndarray, mean: float, std: float, lo: float, span: float) -> np.ndarray:
    """Apply the stored chirp warp: x -> z=(x-mean)/std -> u=sign(z)*z**2 ->
    (u-lo)/span. Pure function of the fit-time params -- the single source of
    truth shared by fit-time detection (``generate_extra_basis_features``) and
    transform-time replay (``_apply_orth_fourier``) so both produce a
    bit-identical axis."""
    x = np.asarray(x, dtype=np.float64)
    z = (x - float(mean)) / max(float(std), 1e-12)
    u = np.sign(z) * (z * z)
    return (u - float(lo)) / max(float(span), 1e-12)


def _corr_sq_centered(v: np.ndarray, y_centered: np.ndarray, y_ss: float) -> float:
    """Squared Pearson correlation of ``v`` with a pre-centered ``y`` whose
    sum-of-squares is ``y_ss``. Avoids ``np.corrcoef`` (2x2-matrix build + two
    std passes) -- a direct centered dot product. Returns 0.0 on a degenerate
    ``v``."""
    vc = v - v.mean()
    v_ss = float(vc @ vc)
    if v_ss < 1e-24 or y_ss < 1e-24:
        return 0.0
    num = float(vc @ y_centered)
    return (num * num) / (v_ss * y_ss)


def _periodogram_power(z01: np.ndarray, y: np.ndarray, freq: float) -> float:
    """Phase-invariant periodogram power of ``y`` at z-space frequency ``freq``.

    ``corr(sin(2*pi*freq*z), y)^2 + corr(cos(2*pi*freq*z), y)^2`` -- the sum of
    the squared linear correlations of the sin and cos projections. Phase-
    invariant because a pure ``sin(2*pi*freq*z + phi)`` decomposes into a
    sin + cos mix whose combined power is independent of phi. Returns 0.0 when
    either projection degenerates (constant), so a frequency whose sin/cos
    collapse over the slice never wins.

    Convenience wrapper that centers ``y`` once; the hot per-column loops call
    :func:`_corr_sq_centered` directly with a pre-centered ``y`` to skip the
    redundant centering on every frequency.
    """
    yc = y - y.mean()
    y_ss = float(yc @ yc)
    if y_ss < 1e-24:
        return 0.0
    ang = 2.0 * np.pi * float(freq) * z01
    return (
        _corr_sq_centered(np.sin(ang), yc, y_ss)
        + _corr_sq_centered(np.cos(ang), yc, y_ss)
    )


def _power_centered(z: np.ndarray, yc: np.ndarray, y_ss: float, freq: float) -> float:
    """Periodogram power at ``freq`` against a pre-centered ``y`` (``yc``,
    sum-of-squares ``y_ss``). Hot-loop variant that skips re-centering y."""
    ang = 2.0 * np.pi * float(freq) * z
    return (
        _corr_sq_centered(np.sin(ang), yc, y_ss)
        + _corr_sq_centered(np.cos(ang), yc, y_ss)
    )


def _refine_peak_freq(
    z_tr: np.ndarray, yc: np.ndarray, y_ss: float, coarse_f: float,
) -> float:
    """Two-stage local-refine of ``coarse_f`` on the TRAIN rows (pre-centered
    ``yc`` / ``y_ss``), maximising periodogram power.

    Stage 1 scans +-0.25 at 0.05 step (the coarse-grid spacing); stage 2 then
    scans +-0.05 at 0.0125 step around the stage-1 winner. The finer second
    pass tightens secondary-peak localisation after deflation -- which widens
    the downstream Ridge recovery margin on multitone signals (a 0.05-only
    refine left secondary tones mis-located by up to ~0.3, costing R^2)."""
    def _scan(center: float, half_width: float, step: float) -> tuple[float, float]:
        lo_r = max(0.05, center - half_width)
        hi_r = center + half_width
        n_steps = int(round((hi_r - lo_r) / step)) + 1
        best_f = center
        best_p = _power_centered(z_tr, yc, y_ss, center)
        for k in range(n_steps):
            f = lo_r + step * k
            p = _power_centered(z_tr, yc, y_ss, f)
            if p > best_p:
                best_p = p
                best_f = f
        return best_f, best_p
    f1, _ = _scan(coarse_f, 0.25, 0.05)
    f2, _ = _scan(f1, 0.05, 0.0125)
    return float(f2)


def _deflate_sincos(z: np.ndarray, y: np.ndarray, freq: float) -> np.ndarray:
    """Residual of ``y`` after least-squares projection onto
    ``[1, sin(2*pi*freq*z), cos(2*pi*freq*z)]``. Removes the contribution of
    one detected frequency so the next peak-pick sees the remaining tones."""
    ang = 2.0 * np.pi * float(freq) * z
    A = np.column_stack([np.ones_like(z), np.sin(ang), np.cos(ang)])
    try:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ coef
    except Exception:
        return y


def _detect_fourier_freqs_for_col(
    z01: np.ndarray,
    y: np.ndarray,
    *,
    f_grid: Sequence[float],
    min_val_corr: float = 0.15,
    min_rows: int = 800,
    max_freqs: int = 4,
) -> list[float]:
    """MULTI-FREQUENCY adaptive detector (2026-06-03).

    Returns the list of held-out-validated dominant z-space frequencies of
    ``y`` as a function of ``z01`` -- the multitone generalisation of
    :func:`_detect_fourier_freq_for_col`. Real signals routinely superpose
    several arbitrary-period oscillations (``sin(3.7x) + sin(5.3x) +
    sin(6.8x)``); detecting only the single dominant frequency leaves a Ridge
    on the support unable to recover the sum.

    Before any frequency search the target is POLYNOMIAL-DETRENDED: y is
    regressed on ``[1, z, z^2, z^3]`` (cubic coefficients fit on TRAIN, applied
    to VAL) and detection runs on the RESIDUAL. A monotone / smooth trend (the
    linear-additive ``y = sign(x1 + 0.7*x2)``) has high LOW-frequency periodogram
    power because a sub-1-cycle sinusoid mimics a ramp; the cubic absorbs it so
    its z-frequency power collapses to ~0, while a genuine oscillation (which a
    cubic cannot express) is left intact. This is the discriminator that
    separates "arbitrary-period oscillation" from "trend the poly basis covers".

    Each iteration then:

    * picks the coarse peak by periodogram power on TRAIN, local-refines +-0.25
      at 0.05 step,
    * confirms ``sqrt(val-slice power) >= max(min_val_corr, 0.30)`` on the
      held-out stride slice -- the 0.30 robust floor rejects finite-sample
      chance peaks (40-seed linear-fixture max spurious 0.232 vs genuine >= 0.96
      at n=800); ``min_val_corr`` is the user-raisable lower bound,
    * DEFLATES both the train and val targets by least-squares-projecting out
      that frequency's ``[1, sin, cos]`` so the next iteration sees the
      remaining tones,

    stopping at ``max_freqs`` or the first frequency that fails the held-out
    gate. N-gated at ``n >= min_rows`` (default 800) so a small-n chance
    frequency never fires. Frequencies already in the running list (within a
    coarse-grid spacing) are skipped to avoid re-locking the same peak.
    """
    z01 = np.asarray(z01, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = z01.size
    if n != y.size or n < int(min_rows):
        return []
    if not np.all(np.isfinite(z01)) or not np.all(np.isfinite(y)):
        return []
    if float(np.std(z01)) < 1e-12 or float(np.std(y)) < 1e-12:
        return []
    grid = [float(f) for f in f_grid if float(f) > 0.0]
    if not grid:
        return []
    idx = np.arange(n)
    val_mask = (idx % 3) == 0
    train_mask = ~val_mask
    z_tr, z_va = z01[train_mask], z01[val_mask]
    y_tr = y[train_mask].copy()
    y_va = y[val_mask].copy()
    if z_tr.size < 16 or z_va.size < 8:
        return []
    if float(np.std(y_tr)) < 1e-12 or float(np.std(y_va)) < 1e-12:
        return []
    # POLYNOMIAL DETREND (2026-06-03): regress y on [1, z, z^2, z^3] and run
    # detection on the RESIDUAL. A monotone / smooth trend (the linear-additive
    # target ``y = sign(x1 + 0.7*x2)``) has HIGH periodogram power at LOW
    # frequencies because a sub-1-cycle sinusoid mimics a monotone ramp -- so
    # the raw periodogram would FALSE-POSITIVE a "frequency" on a non-periodic
    # target (the linear fixture emitted a spurious ``x1__sin0.75``). A low-
    # degree polynomial in z ABSORBS any such trend, so its z-frequency power
    # collapses to ~0 after detrending (measured 0.689 -> 0.002), while a
    # genuine oscillation -- which a cubic CANNOT express -- is left intact
    # (sin(5.3x) power 0.94 retained). The cubic coefficients are fit on TRAIN
    # and APPLIED to VAL (no val leakage); this is the discriminator that
    # separates "arbitrary-period oscillation" from "smooth trend the poly
    # basis already covers".
    _V_tr = np.vander(z_tr, 4)  # [z^3, z^2, z, 1]
    try:
        _poly_coef, *_ = np.linalg.lstsq(_V_tr, y_tr, rcond=None)
        y_tr = y_tr - _V_tr @ _poly_coef
        y_va = y_va - np.vander(z_va, 4) @ _poly_coef
    except Exception:
        pass
    if float(np.std(y_tr)) < 1e-9 or float(np.std(y_va)) < 1e-9:
        return []
    # Effective held-out floor. Even after the polynomial detrend, a FINITE-
    # SAMPLE chance frequency can clear a lenient floor: across 40 linear-
    # additive fixtures (``y = sign(x1 + 0.7*x2)``, n=1200) the max spurious
    # held-out sqrt-power was 0.232, while a genuine oscillation sits at >= 0.96
    # even at n=800 -- a wide gap. A robust 0.30 floor rejects the chance peaks
    # without touching genuine recovery (gate-A multitone tones clear 0.6+).
    # ``min_val_corr`` is honoured as a LOWER bound a caller can RAISE; the
    # built-in 0.30 is the anti-false-positive guard the small-n regime needs.
    _eff_min_val_corr = max(float(min_val_corr), 0.30)
    # Precompute the coarse-grid sin/cos bases on TRAIN once: they depend only
    # on z, not y, so deflation iterations reuse them (cProfile: the per-freq
    # np.sin/np.cos + np.corrcoef was the dominant cost at p=200; this drops
    # the coarse sweep to a centered dot product per cached basis).
    _coarse_basis = []  # (sin_centered, sin_ss, cos_centered, cos_ss) per grid freq
    for f in grid:
        ang = 2.0 * np.pi * f * z_tr
        s = np.sin(ang); c = np.cos(ang)
        sc = s - s.mean(); cc = c - c.mean()
        _coarse_basis.append((sc, float(sc @ sc), cc, float(cc @ cc)))
    out: list[float] = []
    for _ in range(max(1, int(max_freqs))):
        if float(np.std(y_tr)) < 1e-9 or float(np.std(y_va)) < 1e-9:
            break
        yc = y_tr - y_tr.mean()
        y_ss = float(yc @ yc)
        if y_ss < 1e-24:
            break
        best_f = None
        best_power = -1.0
        for gi, f in enumerate(grid):
            sc, s_ss, cc, c_ss = _coarse_basis[gi]
            num_s = float(sc @ yc)
            num_c = float(cc @ yc)
            p = 0.0
            if s_ss >= 1e-24:
                p += (num_s * num_s) / (s_ss * y_ss)
            if c_ss >= 1e-24:
                p += (num_c * num_c) / (c_ss * y_ss)
            if p > best_power:
                best_power = p
                best_f = f
        if best_f is None:
            break
        refined_f = _refine_peak_freq(z_tr, yc, y_ss, best_f)
        # Skip a frequency we've already locked (within half a coarse step).
        if any(abs(refined_f - g) < 0.25 for g in out):
            # Deflate at the coarse peak anyway so the loop can advance, then
            # continue searching the remaining spectrum.
            y_tr = _deflate_sincos(z_tr, y_tr, refined_f)
            y_va = _deflate_sincos(z_va, y_va, refined_f)
            continue
        val_power = _periodogram_power(z_va, y_va, refined_f)
        if val_power <= 0.0 or np.sqrt(val_power) < _eff_min_val_corr:
            break
        out.append(float(refined_f))
        # Deflate both slices so the next peak-pick sees the residual tones.
        y_tr = _deflate_sincos(z_tr, y_tr, refined_f)
        y_va = _deflate_sincos(z_va, y_va, refined_f)
    return out


def _detect_fourier_freq_for_col(
    z01: np.ndarray,
    y: np.ndarray,
    *,
    f_grid: Sequence[float],
    min_val_corr: float = 0.15,
    min_rows: int = 800,
) -> Optional[float]:
    """ADAPTIVE-FREQUENCY Fourier detector (2026-06-03).

    The fixed Fourier univariate grid only covers z-space frequencies {1, 2}.
    An ARBITRARY-period oscillation (e.g. ``y = sin(3.7*x)``, ``sin(5.3*x)``)
    lands at a non-integer z-space frequency and is missed by the fixed grid
    (recovered at |corr| 0.02-0.23). This detector sweeps a coarse z-space
    frequency grid, locally refines around the peak, and returns the dominant
    frequency ONLY when a held-out validation slice confirms it -- otherwise
    None (no adaptive column emitted).

    Method
    ------
    * Deterministic stride train/val split: ``val = arange(n) % 3 == 0`` (a
      third held out, no RNG so the recipe replays identically). The frequency
      is RANKED on train rows and CONFIRMED on the held-out val rows -- a
      chance frequency that fits a train slice but not the held-out slice is
      rejected. This is the n-gated false-positive guard: a naive default-on
      version regressed 9 tests because at small n a chance frequency clears
      the gate. We require ``n >= min_rows`` (default 800) AND val-slice
      confirmation.
    * Rank ``f_grid`` by PERIODOGRAM POWER ``corr(sin)^2 + corr(cos)^2`` on the
      TRAIN rows (phase-invariant: a single sin or cos alone has low |corr| for
      a phase-shifted signal, so we must score the sin+cos pair jointly).
    * Local-refine ``+-0.25`` at ``0.05`` step around the coarse peak (still on
      train).
    * KEEP the refined freq only if ``sqrt(val-slice periodogram power) >=
      max(min_val_corr, 0.30)`` (the held-out effective |corr| of the sin+cos
      support clears the floor). Otherwise return None.

    Before the search, y is POLYNOMIAL-DETRENDED (cubic in z, train-fit /
    val-applied) so a monotone / smooth trend cannot masquerade as a low
    frequency; the 0.30 robust floor then rejects finite-sample chance peaks.
    See :func:`_detect_fourier_freqs_for_col` for the full rationale.

    ``z01`` is the SAME ``z = (x - lo) / span`` in [0, 1] that the Fourier
    emitter uses, so the detected frequency drops straight into the emitter's
    ``fourier_freqs`` for that column. ``y`` may be discrete or continuous;
    Pearson on y is fine because we only need a phase-invariant linear-usability
    score, not MI.

    Returns the SINGLE dominant validated frequency (or None). The multitone
    superposition case is handled by :func:`_detect_fourier_freqs_for_col`,
    which this delegates to (taking the first detected peak) -- so the coarse-
    sweep + local-refine + held-out-gate contract is shared verbatim.
    """
    freqs = _detect_fourier_freqs_for_col(
        z01, y, f_grid=f_grid, min_val_corr=min_val_corr,
        min_rows=min_rows, max_freqs=1,
    )
    return float(freqs[0]) if freqs else None


def generate_extra_basis_features(
    X: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
    extra_bases: Sequence[str] = ("spline", "fourier"),
    fourier_freqs: Sequence[float] = (1.0, 2.0),
    fourier_powers: Sequence[int] = (1, 2),
    spline_knots: int = 5,
    dedup_collinear_sources: bool = True,
    dedup_corr_threshold: float = 0.999,
    y: Optional[np.ndarray] = None,
    fourier_adaptive: bool = False,
    fourier_adaptive_min_val_corr: float = 0.15,
    fourier_chirp: bool = False,
    fourier_chirp_min_val_corr: float = 0.15,
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
    y : array-like, optional
        Target. Only consulted when ``fourier_adaptive`` is True (and only by
        the ADAPTIVE-FREQUENCY detector). Never read for the fixed-grid
        emission, so the legacy path stays leakage-free / y-independent.
    fourier_adaptive : bool, default False
        When True and ``y`` is given, run :func:`_detect_fourier_freqs_for_col`
        on each source column's z (power==1 only) and -- for each held-out-
        validated dominant frequency found (multitone: several peaks via
        residual deflation) -- ADD it to that column's Fourier frequency set.
        The emitted sin/cos meta entries for adaptive frequencies are tagged
        ``"adaptive": True`` so MRMR can protect them past screening. Covers
        arbitrary-period oscillations and their superpositions
        (``sin(3.7*x) + sin(5.3*x) + sin(6.8*x)``) the fixed grid {1, 2} misses.
    fourier_adaptive_min_val_corr : float, default 0.15
        Held-out validation effective-|corr| floor for the adaptive detector.
    fourier_chirp : bool, default False
        ADAPTIVE-CHIRP path (2026-06-03). When True and ``y`` is given, run the
        SAME held-out-validated detector on the QUADRATIC-ARGUMENT warp
        ``u = sign(z) * z**2`` (z standardised on the column) for each source
        column. A chirp ``y ~ sin(2*pi*f*z**2)`` -- whose frequency GROWS with z
        -- is STATIONARY in u, so the detector locks its frequency and the
        emitted ``sin(2*pi*f*u)`` / ``cos(2*pi*f*u)`` reconstruct it; a Fourier on
        the LINEAR argument cannot express a frequency that grows with z. The
        emitted sin/cos meta entries carry ``"arg": "quadratic"`` (the warp the
        recipe replays) AND ``"adaptive": True`` (so MRMR protects them past the
        screen, identical to the linear adaptive legs). This is an ADDITIVE
        second path alongside the linear adaptive one -- both fire; on a plain
        linear target the chirp legs are harmless (Ridge regularises them to ~0,
        Phase-0 bench: combined R^2 == linear-only on linear targets, +0.3-0.5
        R^2 on fast chirps). N-gated at >= 800 MI rows like the linear path.
    fourier_chirp_min_val_corr : float, default 0.15
        Held-out validation effective-|corr| floor for the chirp detector.

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
            float, "lo": float, "span": float, "power": int[, "adaptive": True]}.

    Notes
    -----
    bench-rejected (2026-06-03): a per-column "poly-vs-Fourier COMPETITION gate"
    -- emit only the better of {orth-poly basis, this Fourier path} per column to
    cut the redundant cross-family features that co-occur in the support -- was
    benchmarked and REJECTED. The co-occurrence is genuine COMPLEMENTARITY, not
    redundancy: on kink/step/bump targets (e.g. y=|x|) the Fourier legs carry
    independent residual R^2 0.16-0.60 that a degree<=4 poly under-fits, so a
    winner-takes-all gate HURT the |x| target OOS by -0.06; no OOS win on a tree
    downstream (deltas +/-0.008, inconsistent sign); on a mixed frame the gate
    declines to fire (different columns have different winners). The existing
    Fleuret redundancy + Spearman cross-stage dedup already remove the only real
    redundancy. Don't add a competition gate. (D:/Temp/item3_poly_fourier_findings.md)
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
    # Adaptive-frequency detection runs only when requested AND y is supplied.
    # The y array is coerced to float once here (Pearson on y is all the
    # phase-invariant periodogram needs); detection is gated per-column below.
    _y_adapt = None
    if (fourier_adaptive or fourier_chirp) and y is not None:
        _y_adapt = np.asarray(y, dtype=np.float64).ravel()
        if _y_adapt.size != len(X) or not np.all(np.isfinite(_y_adapt)):
            _y_adapt = None
    # Coarse z-space frequency sweep grid for the adaptive detector. Covers a
    # wide period range (0.5 .. 8.0) at 0.5 stride; local refinement then
    # snaps to the true non-integer frequency. The set of frequencies the
    # detector may ADD is disjoint from the fixed grid by construction (a
    # fixed freq that already recovers the signal needs no adaptive twin).
    _adaptive_f_grid = tuple(0.5 * k for k in range(1, 17))  # 0.5 .. 8.0
    # The CHIRP warp (u = sign(z)*z**2) concentrates a growing-frequency signal
    # at a HIGHER z-space frequency than the linear axis, so the chirp detector
    # sweeps a WIDER grid (0.5 .. 24.0). Phase-0: fast chirps land at u-space
    # peaks up to ~12 and the multitone deflation needs headroom above them.
    _chirp_f_grid = tuple(0.5 * k for k in range(1, 49))  # 0.5 .. 24.0
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
                # POWER-ARGUMENT Fourier (2026-06-03): build the Fourier on x**p for
                # p in fourier_powers, as a SELF-CONTAINED replayable recipe (raw x ->
                # x**p -> Fourier; 1-deep, no nesting). p=2 captures even-argument
                # CHIRPS like ``sin(a**2)`` (freq~1 on the a**2 argument reproduces it
                # exactly) that a Fourier on the linear argument cannot. p=1 keeps the
                # original ``{col}__sin{freq}`` name (back-compat with prior recipes).
                for pwr in fourier_powers:
                    _p = int(pwr)
                    _xp = x if _p == 1 else np.power(x, _p)
                    if not np.all(np.isfinite(_xp)) or float(np.std(_xp)) <= 1e-12:
                        continue
                    lo_f, span_f = _fit_fourier_for_col(_xp)
                    z = (_xp - lo_f) / max(span_f, 1e-12)
                    _pfx = "" if _p == 1 else f"p{_p}"
                    # ADAPTIVE-FREQUENCY (2026-06-03): for the linear argument
                    # (power==1) detect the column's dominant z-space frequency
                    # from a coarse sweep + local refine, held-out validated.
                    # The detected freq is ADDED to this column's freq set and
                    # its sin/cos meta is tagged adaptive=True so MRMR protects
                    # it past screening. Disjoint-by-detection from the fixed
                    # grid: a fixed freq that already recovers the signal makes
                    # the periodogram peak land near it, so the detector's
                    # held-out gate is satisfied by the fixed twin too -- but
                    # we still tag/add the refined freq because the fixed grid
                    # cannot express a non-integer period.
                    _adaptive_freqs: list[float] = []
                    if _p == 1 and _y_adapt is not None:
                        # max_freqs=6: a multitone superposition (3-4 genuine
                        # tones) needs enough sin/cos pairs to SPAN the signal
                        # subspace after the per-iteration deflation leaves a
                        # residual -- 4 pairs recovered the 3-tone gate-A signal
                        # at OOS R^2 ~0.95 but 6 pairs lift it to ~0.985, a far
                        # safer margin above the 0.9 bar. Each extra freq still
                        # passes the held-out 0.30 floor, so noise never inflates
                        # the count (a pure-noise column stops at the first peak).
                        _adaptive_freqs = _detect_fourier_freqs_for_col(
                            z, _y_adapt,
                            f_grid=_adaptive_f_grid,
                            min_val_corr=float(fourier_adaptive_min_val_corr),
                            min_rows=800,
                            max_freqs=6,
                        )
                    _freqs_for_col = list(fourier_freqs)
                    _adaptive_set: set[float] = set()
                    for _af in _adaptive_freqs:
                        if not any(abs(_af - f) < 1e-9 for f in _freqs_for_col):
                            _freqs_for_col.append(_af)
                            _adaptive_set.add(_af)
                    for freq in _freqs_for_col:
                        _is_adaptive = freq in _adaptive_set
                        ang = 2.0 * np.pi * freq * z
                        s_vals = np.sin(ang)
                        c_vals = np.cos(ang)
                        if float(np.std(s_vals)) > 1e-12:
                            name_s = f"{col}__{_pfx}sin{freq:g}"
                            out_cols[name_s] = s_vals
                            meta[name_s] = {
                                "basis": "fourier", "src": col,
                                "kind": "sin", "freq": float(freq),
                                "lo": float(lo_f), "span": float(span_f),
                                "power": _p, "adaptive": _is_adaptive,
                            }
                        if float(np.std(c_vals)) > 1e-12:
                            name_c = f"{col}__{_pfx}cos{freq:g}"
                            out_cols[name_c] = c_vals
                            meta[name_c] = {
                                "basis": "fourier", "src": col,
                                "kind": "cos", "freq": float(freq),
                                "lo": float(lo_f), "span": float(span_f),
                                "power": _p, "adaptive": _is_adaptive,
                            }
                # ADAPTIVE-CHIRP (2026-06-03): a SECOND argument-warp alongside
                # the linear-adaptive path above. The chirp axis u = sign(z)*z**2
                # (z standardised on the column) makes a growing-frequency
                # oscillation ``y ~ sin(2*pi*f*z**2)`` STATIONARY in u, so the
                # SAME held-out-validated multitone detector locks its frequency
                # and the emitted sin/cos on u reconstruct it -- which a Fourier
                # on the linear argument cannot (Phase-0: linear R^2 0.07-0.53 vs
                # chirp 0.88 on a fast chirp). Emitted legs carry arg="quadratic"
                # (the warp the recipe replays) + adaptive=True (so MRMR protects
                # them past the screen, exactly like the linear adaptive legs).
                # Disjoint by name (``__qsin``/``__qcos``) from the linear legs;
                # additive (on a plain linear target the chirp legs are harmless,
                # Ridge regularises them to ~0). N-gated identically (>= 800 rows
                # inside the detector); a pure-noise column admits none.
                if fourier_chirp and _y_adapt is not None:
                    _c_mean, _c_std, _c_lo, _c_span = _fit_chirp_warp_for_col(x)
                    if _c_span > 1e-12 and _c_std > 1e-12:
                        u_axis = _chirp_axis(x, _c_mean, _c_std, _c_lo, _c_span)
                        if np.all(np.isfinite(u_axis)) and float(np.std(u_axis)) > 1e-12:
                            _chirp_freqs = _detect_fourier_freqs_for_col(
                                u_axis, _y_adapt,
                                f_grid=_chirp_f_grid,
                                min_val_corr=float(fourier_chirp_min_val_corr),
                                min_rows=800,
                                max_freqs=6,
                            )
                            for _cf in _chirp_freqs:
                                ang_c = 2.0 * np.pi * _cf * u_axis
                                sc_vals = np.sin(ang_c)
                                cc_vals = np.cos(ang_c)
                                if float(np.std(sc_vals)) > 1e-12:
                                    name_qs = f"{col}__qsin{_cf:g}"
                                    out_cols[name_qs] = sc_vals
                                    meta[name_qs] = {
                                        "basis": "fourier", "src": col,
                                        "kind": "sin", "freq": float(_cf),
                                        "arg": "quadratic",
                                        "mean": float(_c_mean), "std": float(_c_std),
                                        "lo": float(_c_lo), "span": float(_c_span),
                                        "power": 1, "adaptive": True,
                                    }
                                if float(np.std(cc_vals)) > 1e-12:
                                    name_qc = f"{col}__qcos{_cf:g}"
                                    out_cols[name_qc] = cc_vals
                                    meta[name_qc] = {
                                        "basis": "fourier", "src": col,
                                        "kind": "cos", "freq": float(_cf),
                                        "arg": "quadratic",
                                        "mean": float(_c_mean), "std": float(_c_std),
                                        "lo": float(_c_lo), "span": float(_c_span),
                                        "power": 1, "adaptive": True,
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
            power=int(meta_entry.get("power", 1)),
            adaptive=bool(meta_entry.get("adaptive", False)),
            arg=str(meta_entry.get("arg", "linear")),
            mean=(None if meta_entry.get("mean") is None else float(meta_entry["mean"])),
            std=(None if meta_entry.get("std") is None else float(meta_entry["std"])),
        )
    return None


def hybrid_orth_extra_basis_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    extra_bases: Sequence[str] = ("spline", "fourier"),
    fourier_freqs: Sequence[float] = (1.0, 2.0),
    fourier_powers: Sequence[int] = (1, 2),
    spline_knots: int = 5,
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
    fourier_adaptive: bool = False,
    fourier_adaptive_min_val_corr: float = 0.15,
    fourier_chirp: bool = False,
    fourier_chirp_min_val_corr: float = 0.15,
):
    """Layer 32 hybrid: spline + Fourier univariate basis FE + MI-greedy
    selection. Mirrors :func:`hybrid_orth_mi_fe_with_recipes` for the
    polynomial path but emits extra-basis columns (B-spline, Fourier)
    instead. Returns (X_augmented, scores, recipes).

    The selection rule is the same TWO-GATE chain as the polynomial path:
    relative uplift >= min_uplift AND engineered_mi >= max(legacy floor,
    noise-aware floor). See :func:`hybrid_orth_mi_fe` for the rationale.

    ``fourier_adaptive`` (default False) forwards to
    :func:`generate_extra_basis_features` -- when True, each source column's
    dominant z-space frequency is detected (held-out validated) and added to
    its Fourier set, with the emitted sin/cos recipes tagged ``adaptive=True``.

    ``fourier_chirp`` (default False) likewise forwards the ADAPTIVE-CHIRP path:
    the same held-out detector run on the quadratic-argument warp
    ``u = sign(z)*z**2``, emitting ``__qsin``/``__qcos`` legs tagged
    ``arg="quadratic"`` + ``adaptive=True`` (force-admitted past the uplift gate
    and MRMR-protected identically to the linear adaptive legs).
    """
    engineered, meta = generate_extra_basis_features(
        X, cols=cols, extra_bases=extra_bases,
        fourier_freqs=fourier_freqs, fourier_powers=fourier_powers,
        spline_knots=spline_knots,
        y=y, fourier_adaptive=fourier_adaptive,
        fourier_adaptive_min_val_corr=fourier_adaptive_min_val_corr,
        fourier_chirp=fourier_chirp,
        fourier_chirp_min_val_corr=fourier_chirp_min_val_corr,
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
    # FORCE-ADMIT adaptive Fourier columns: a single adaptive sin OR cos has a
    # LOW marginal |corr| / MI for a phase-shifted oscillation (the phase is
    # split between the two), so the per-column MI-uplift gate would drop them
    # even though the sin+cos PAIR recovers the signal. The adaptive detector
    # already validated the frequency on a held-out slice, so both legs are
    # admitted unconditionally here; the downstream MRMR adaptive-protection
    # block then keeps them past screening. Append in deterministic name order.
    _adaptive_names = [
        nm for nm, m in meta.items()
        if m.get("basis") == "fourier" and m.get("adaptive", False)
    ]
    _keep_set = set(keep)
    for nm in _adaptive_names:
        if nm not in _keep_set and nm in engineered.columns:
            keep.append(nm)
            _keep_set.add(nm)
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

