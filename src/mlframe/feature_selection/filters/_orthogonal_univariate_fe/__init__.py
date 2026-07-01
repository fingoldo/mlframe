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

from ..hermite_fe import (
    _POLY_BASES,
    basis_route_by_moments,
    polyeval_dispatch,
)
from ._orth_mi_backends import (  # noqa: F401
    _MI_BACKEND,
    _mi_classif_batch,
    mi_classif_batch_chunked,
    _mi_classif_batch_numba,
    _mi_classif_batch_sklearn,
    _select_mi_backend,
)

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


def _source_from_engineered_name(name: str, raw_names) -> str:
    """Recover the TRUE source column of an engineered name ``"{src}__{suffix}"``.

    The engineered-name grammar is ``"{src}__{basis_code}{degree}"`` -- but blindly
    splitting on the FIRST ``"__"`` (``name.split("__", 1)[0]``) MISPARSES any raw
    input whose own name contains ``"__"`` (one-hot / dummy names like ``"city__NY"``):
    it stems ``"city__NY__He2"`` to ``"city"`` instead of ``"city__NY"``, so the
    per-source raw-MI baseline lookup misses, the uplift denominator collapses to the
    ``1e-12`` floor, and EVERY such engineered column clears the uplift gate spuriously.

    FIX (D1, 2026-06-22): the true source is always a member of the known raw-column
    set ``raw_names``. Recover it by LONGEST matching ``"{raw}__"`` prefix (longest so
    a raw ``"city"`` cannot shadow a raw ``"city__NY"``). Fall back to the legacy
    first-``"__"`` split only when no raw name prefixes the engineered name (defensive;
    keeps standalone callers that never pass a raw set working).
    """
    best = None
    for raw in raw_names:
        if name == raw or name.startswith(raw + "__"):
            if best is None or len(raw) > len(best):
                best = raw
    if best is not None:
        return best
    return name.split("__", 1)[0] if "__" in name else name


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


# Carved out to _orth_dedup.py (2026-06-22 monolith-split, LOC budget); re-exported here so the
# package-facade name `_dedup_collinear_source_cols` and all internal callers keep working.
from ._orth_dedup import _dedup_collinear_source_cols  # noqa: E402


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
    _route_no_aux = aux_for_fit is None or len(np.asarray(aux_for_fit)) == 0
    # MEMO SCOPE (2026-06-21): all 4 candidate bases' preprocess fits run _detect_heavy_tail on the SAME x
    # (the verdict depends on x's values, not the basis), so memoise it for this routing sweep -> 4 sort-based
    # median+MAD probes collapse to 1 (byte-identical; the memo caches a deterministic boolean, id-keyed +
    # identity-verified, cleared at scope exit). Covers both the host and the GPU-resident orth-FE routing.
    from ..hermite_fe._hermite_robust import heavy_tail_memo_scope
    with heavy_tail_memo_scope():
        for basis in candidate_bases:
            bcorr = 0.0
            # FIT-ONCE per basis (2026-06-21): z depends on (x, basis) not degree, so fit the
            # preprocess (incl the robust heavy-tail np.median axis) ONCE and evaluate both degrees
            # on the cached z -- byte-identical to the per-degree _evaluate_basis_column, halving the
            # routing-probe medians (4 bases x 2 degrees = 8 fits -> 4). Aux-pool path stays per-degree.
            _zc = None
            if _route_no_aux:
                try:
                    _zf, _ = _POLY_BASES[basis]["fit"](x)
                    _zc = np.ascontiguousarray(_zf, dtype=np.float64)
                except Exception:
                    _zc = None
            for d in degrees:
                try:
                    if _zc is not None:
                        _coef = np.zeros(int(d) + 1, dtype=np.float64)
                        _coef[int(d)] = 1.0
                        v = polyeval_dispatch(basis, _zc, _coef)
                    else:
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
    _cols_auto = cols is None
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
    from .._semi_supervised_fe import get_unlabeled_pool as _get_unlabeled_pool
    _aux_pool = _get_unlabeled_pool()
    code = _BASIS_CODE
    out_cols: dict = {}
    from .._fe_deadline import fe_deadline_passed
    for col in cols:
        # Optional-enrichment wall-clock budget: stop the per-column basis scan once MRMR.fit's deadline passes and return
        # whatever was engineered so far (the core selection still produces a usable partial). No-op when no budget is set.
        if fe_deadline_passed():
            break
        x = np.asarray(X[col].to_numpy(), dtype=np.float64)
        # Skip orthogonal-polynomial expansion on integer-valued low-cardinality categorical group keys: T_n / He_n of an
        # arbitrary label code (region 0..9) is spurious -- it fits the label->target mapping, floods the candidate pool, and
        # displaces the genuinely useful grouped aggregates of that key. Continuous / high-card columns keep the expansion. The
        # skip applies ONLY when ``cols`` was auto-detected (the default-on MRMR scan over every column): an EXPLICIT caller-
        # provided ``cols`` is trusted as-is, because a 3-level ORDINAL integer (a binned threshold) is a legitimate basis-FE
        # axis the caller asked for -- only the auto scan over arbitrary group-key labels needs protecting.
        if _cols_auto and _is_int_as_cat_axis(x):
            continue
        finite_mask = np.isfinite(x)
        if not finite_mask.all():
            # An orthogonal-polynomial basis over a NaN-containing column is unsound: the recipe replay path does NOT impute (NaN in -> NaN out, so
            # transform() emits an all-NaN engineered column), and fit-time the nanmean-imputed basis becomes a MISSINGNESS PROXY whose binned MI ties /
            # beats the genuine missingness-FE columns (is_missing__/missingness_pattern), displacing them from MRMR selection. Skip the column; the
            # missingness signal belongs to the dedicated missingness-FE family, not to a non-replayable mean-imputed polynomial.
            continue
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
        # FIT-ONCE-PER-COLUMN (2026-06-21): the basis preprocess ``z`` + params depend ONLY on
        # (x, basis), NOT on degree -- the degree only swaps the one-hot coefficient. Re-fitting it
        # inside the degrees loop recomputed the robust heavy-tail axis (np.median/MAD) once per
        # degree, the dominant np.median caller in the post-subsample CPU tail. Fit ``z`` ONCE here
        # and evaluate each degree via polyeval_dispatch on the cached ``z`` -- BYTE-IDENTICAL to the
        # per-degree _evaluate_basis_column (same fit_fn(x) -> same z -> same polyeval). The rare
        # aux-pool path keeps the per-degree call (its fit concatenates the pool; left untouched).
        _z_cached = None
        if aux_col is None or len(np.asarray(aux_col)) == 0:
            try:
                _z_fit, _ = _POLY_BASES[chosen_basis]["fit"](x)
                _z_cached = np.ascontiguousarray(_z_fit, dtype=np.float64)
            except Exception:
                _z_cached = None
        for d in degrees:
            try:
                if _z_cached is not None:
                    _coef = np.zeros(int(d) + 1, dtype=np.float64)
                    _coef[int(d)] = 1.0
                    vals = polyeval_dispatch(chosen_basis, _z_cached, _coef)
                else:
                    vals = _evaluate_basis_column(
                        x, chosen_basis, int(d), aux_for_fit=aux_col,
                    )
                out_cols[f"{col}__{code.get(chosen_basis, chosen_basis)}{d}"] = vals
            except Exception as exc:
                logger.warning("generate_univariate_basis_features: basis=%r degree=%d on col=%r raised %r; skipping",
                               chosen_basis, d, col, exc)
                continue
    return pd.DataFrame(out_cols, index=X.index)


def _gpu_build_and_score_univariate(X, cols, degrees, basis, y, nbins):
    """MATRIX-NATIVE (Piece 3, gated): build the univariate orth-basis candidate matrix ON the device
    (_gpu_evaluate_basis_column) and score its plug-in MI RESIDENT (_plugin_mi_classif_batch_cuda_resident,
    no H2D) -- mirroring generate_univariate_basis_features + score_features_by_mi_uplift. Routing/dedup/
    skip rules match the host builder exactly (only the per-(col,basis,degree) eval + the MI move to the
    GPU). Returns ``(eng_matrix_cupy, names, scores_df)`` or ``(None, [], empty_scores)`` when no candidate.
    Raises on GPU failure so the caller falls back to the host path (never a correctness regression)."""
    import cupy as cp
    from ..hermite_fe import _plugin_mi_classif_batch_cuda_resident
    from ..hermite_fe._hermite_robust import _robust_axis_enabled
    from .._fe_deadline import fe_deadline_passed
    _cols_auto = cols is None
    if cols is None:
        cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cols = _dedup_collinear_source_cols(X, list(cols), corr_threshold=0.999)
    _empty = pd.DataFrame(columns=["engineered_col", "source_col", "baseline_mi", "engineered_mi", "uplift"])
    ra = _robust_axis_enabled()
    _ya = np.asarray(y)
    y_arr = np.asarray(_ya, dtype=np.int64) if np.issubdtype(_ya.dtype, np.integer) else _ya.astype(np.int64)
    # y is a FIT-CONSTANT re-uploaded on every orth-family call (univariate-decide / pair-cross / triplet /
    # quadruplet / meta-scorer / adaptive-arity each re-run this builder over the SAME X/y). Route through the
    # resident operand cache so it is uploaded ONCE per fit (selection-equivalent: same int64 labels).
    from .._fe_resident_operands import resident_operand, assemble_resident_matrix
    y_gpu = resident_operand(y_arr, "orth_uni_y", dtype=np.int64)
    # y min/max is a fit-constant -> compute ONCE and reuse for both the raw-MI and the eng-MI resident
    # calls below, instead of each recomputing it (cp.min/max + scalar D2H). Bit-identical (y is invariant).
    _ymm = cp.asnumpy(cp.stack((cp.min(y_gpu), cp.max(y_gpu))))
    _ymin = int(_ymm[0]); _ncls = int(_ymm[1]) - _ymin + 1
    # Baseline RAW-column MI (resident), for the uplift denominator. Built here but SCORED BELOW in ONE
    # resident MI call stacked with eng_mat -- per-column MI is independent, so the values are identical to
    # two separate calls while issuing one launch set instead of two (raw_mi feeds raw_mi_map at the rows
    # loop, which runs after eng_mat exists, so deferring the score is safe).
    raw_cols = [c for c in cols if pd.api.types.is_numeric_dtype(X[c])]
    raw_mi_map: dict = {}
    raw_mat = None
    if raw_cols:
        # raw_mat is the base raw feature-column matrix, a FIT-CONSTANT re-built+re-uploaded per orth-family
        # call over the SAME X. DEVICE-ASSEMBLE it from its per-column resident operands: each raw column is
        # already uploaded once by the basis builders, so stacking the resident columns content-hits the cache
        # and the whole (n, k) matrix never crosses H2D (vs the prior single whole-matrix upload, a distinct
        # blob that never deduped). Column j is X[raw_cols[j]] verbatim -> same bytes -> selection-equivalent.
        raw_mat = assemble_resident_matrix(
            X[raw_cols].to_numpy(dtype=np.float64), raw_cols, ("orth_raw_mat", tuple(raw_cols)), dtype=np.float64,
        )
    code = _BASIS_CODE
    # Routing + skips run on the HOST (cheap njit / moment fingerprint), mirroring the host builder;
    # only the heavy per-(col,basis,degree) eval + the MI move to the GPU -- and the eval is BATCHED.
    # First pass: apply the cheap host skip rules to pick candidate columns + their operand arrays.
    cand_cols: list = []
    cand_x: list = []
    for col in cols:
        if fe_deadline_passed():
            break
        x = np.asarray(X[col].to_numpy(), dtype=np.float64)
        if _cols_auto and _is_int_as_cat_axis(x):
            continue
        if not np.isfinite(x).all():
            continue
        cand_cols.append(col)
        cand_x.append(np.ascontiguousarray(x))
    if not cand_x:
        return None, [], _empty
    from .._gpu_resident_fe import (
        _gpu_evaluate_basis_matrix, fe_gpu_routing_enabled, _gpu_route_bases_batched,
    )
    # GPU ROUTING (opt-in, default OFF): decide every candidate column's basis on the device at once,
    # mirroring the per-column host basis_route_by_signal. Falls back to the host router per column on any
    # GPU failure or where the device router returned None (degenerate). The top-level guards (y usable,
    # n>=30) match basis_route_by_signal's host fallback conditions, applied once since y/n are shared.
    _gpu_routed = None
    _Mr = None  # resident (n, n_cand) operand matrix uploaded ONCE for routing, reused for the basis-MI build
    if basis == "auto" and y is not None and fe_gpu_routing_enabled():
        _yc = np.asarray(_ya, dtype=np.float64).ravel()
        if (_yc.size == cand_x[0].size and cand_x[0].size >= 30
                and np.isfinite(_yc).all() and float(np.std(_yc)) >= 1e-12):
            try:
                # _Mr is the candidate-column matrix -- each column is a RAW X base column (cand_x[j] =
                # X[cand_cols[j]] verbatim). DEVICE-ASSEMBLE it from the per-column resident operands so it
                # never crosses H2D as a whole (n, n_cand) blob: each raw column is already resident under the
                # shared ("xbasis_op", col) role, so stacking the resident columns content-hits the cache.
                # Column j is the raw column verbatim -> same bytes -> selection-equivalent; a name/shape
                # mismatch or cupy fault falls back to the whole-matrix upload.
                _Mr = assemble_resident_matrix(
                    np.column_stack(cand_x), cand_cols, ("orth_Mr", tuple(cand_cols)), dtype=np.float64,
                )
                # _yc is the FIT-CONSTANT routing target, re-uploaded per orth-family call (a fresh cp.asarray
                # each time). Route through the content-keyed resident cache so it uploads ONCE per fit and
                # every later orth call reuses the resident copy. Read-only f64 -> selection-equivalent.
                _yc_gpu = resident_operand(_yc, "orth_route_y", dtype=np.float64)
                _gpu_routed = _gpu_route_bases_batched(
                    cp, _Mr, _yc_gpu, list(_POLY_BASES), tuple(degrees), robust_axis=ra,
                )
            except Exception:
                _gpu_routed = None
                _Mr = None
    used_x: list = []
    used_bases: list = []
    used_src: list = []
    used_idx: list = []   # index into cand_x of each survivor, so a resident _Mr can be reused by slice
    for _i, col in enumerate(cand_cols):
        x = cand_x[_i]
        if basis == "auto":
            if _gpu_routed is not None and _gpu_routed[_i] is not None:
                chosen = _gpu_routed[_i]
            else:
                chosen = basis_route_by_signal(x, _ya, degrees=degrees) if y is not None else basis_route_by_moments(x)
        else:
            chosen = basis
        if chosen not in _POLY_BASES:
            continue
        used_x.append(x)
        used_bases.append(chosen)
        used_src.append(col)
        used_idx.append(_i)
    if not used_x:
        return None, [], _empty
    # ONE H2D of the (n, n_used) operand matrix, then ONE vectorised preprocess+Clenshaw per (basis, robust)
    # group/degree. When GPU routing already uploaded the candidate matrix, REUSE it (device slice, no second
    # H2D) -- in residency mode the operands are already on the GPU; re-uploading would be a redundant copy.
    if _Mr is not None:
        M = _Mr if used_idx == list(range(_Mr.shape[1])) else _Mr[:, used_idx]
    else:
        M = cp.asarray(np.ascontiguousarray(np.column_stack(used_x), dtype=np.float64))
    eng_mat, meta = _gpu_evaluate_basis_matrix(cp, M, used_bases, list(degrees), robust_axis=ra)
    if eng_mat is None:
        return None, [], _empty
    names = [f"{used_src[_ci]}__{code.get(_b, _b)}{_d}" for (_ci, _b, _d) in meta]
    # D1 (2026-06-22): the TRUE source per emitted name is ``used_src[_ci]`` -- carry it
    # directly rather than re-parsing the name via ``split("__", 1)[0]`` (which mis-stems a
    # one-hot source ``"city__NY"`` and collapses the uplift denominator to the 1e-12 floor).
    name_src = [used_src[_ci] for (_ci, _b, _d) in meta]
    # ONE resident MI over [raw_mat | eng_mat] stacked -- per-column independent, so raw_mi/eng_mi are
    # bit-identical to two separate calls, with one launch set instead of two. Both are already device-
    # resident, so this stays H2D-free (do NOT route through the host-input batcher, which would round-trip).
    if raw_mat is not None:
        _stacked = cp.concatenate((raw_mat, eng_mat.astype(cp.float64, copy=False)), axis=1)
        _all_mi = _plugin_mi_classif_batch_cuda_resident(_stacked, y_gpu, nbins, y_min=_ymin, n_classes=_ncls)
        _rk = int(raw_mat.shape[1])
        raw_mi_map = dict(zip(raw_cols, [float(v) for v in _all_mi[:_rk]]))
        eng_mi = _all_mi[_rk:]
    else:
        eng_mi = _plugin_mi_classif_batch_cuda_resident(eng_mat, y_gpu, nbins, y_min=_ymin, n_classes=_ncls)
    rows = []
    for j, nm in enumerate(names):
        src = name_src[j]
        base = float(raw_mi_map.get(src, 0.0))
        emi = float(eng_mi[j])
        rows.append({
            "engineered_col": nm, "source_col": src,
            "baseline_mi": base, "engineered_mi": emi, "uplift": emi / (base + 1e-12),
        })
    scores = pd.DataFrame(rows).sort_values("uplift", ascending=False).reset_index(drop=True)
    return eng_mat, names, scores


def score_features_by_mi_uplift(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    nbins: int = 10,
    meta: dict = None,
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
    raw_np = raw_X.to_numpy(dtype=np.float64)
    # SF1a class-B :311 collapse (2026-06-30): the RAW baseline matrix is the fit-constant raw columns verbatim;
    # under STRICT it already routes through the resident plug-in but re-uploads fresh at _orth_mi_backends:311.
    # Ride the resident-operand cache so it uploads ONCE. Same percentile-edge resident estimator the host STRICT
    # path uses -> byte-identical per-column raw MI -> byte-identical uplift baseline. None on cupy failure /
    # non-strict -> the EXACT host scorer (byte-identical default path untouched).
    from .._resident_raw_mi import resident_raw_baseline_mi

    raw_mi = resident_raw_baseline_mi(raw_np, y_arr, ("uplift_raw_baseline", tuple(raw_cols)), nbins=nbins)
    if raw_mi is None:
        raw_mi = _mi_classif_batch(raw_np, y_arr, nbins=nbins)
    raw_mi = np.asarray(raw_mi, dtype=np.float64)
    raw_mi_map = dict(zip(raw_cols, raw_mi.tolist()))
    # SF1b/1c class-A/C :311 collapse (2026-06-30): when EVERY engineered column is a poly leg (He/T/L/LL), the
    # engineered matrix is rebuilt DEVICE-BORN from the resident raw operands and scored through the SAME resident
    # plug-in MI (no host materialise/upload) -- the uplift RATIO stays internally consistent (numerator + baseline
    # on the SAME estimator). EXTRA-BASIS columns (spline/Fourier/chirp/wavelet) are not GPU-ported -> the helper
    # returns None and the engineered matrix stays on the host chunked scorer (SF1c irreducible born-fresh transient).
    # ENGINEERED-matrix MI, device-born when possible so the host matrix never uploads at _orth_mi_backends:311.
    # ``meta`` (supplied by the EXTRA-BASIS caller) carries per-column fit params (freq/knots/lo/span/mean/std),
    # so the extra-basis families (spline/Fourier/chirp/wavelet) rebuild ON device from the resident raw operands
    # (SF1c all-device). Without meta (the poly-univariate caller) the poly-leg twin rebuilds from the names.
    # Either device path returns None (STRICT off / non-poly / unrecognised basis / cupy fault) -> the exact host
    # chunked scorer (bit-identical, bounds peak RAM). BOTH engineered + raw baseline use the SAME resident
    # estimator so the uplift RATIO stays internally consistent.
    if meta is not None:
        from ._extra_basis_resident import extra_basis_eng_mi_resident

        eng_mi = extra_basis_eng_mi_resident(raw_X, engineered_X, y_arr, meta, nbins=nbins)
    else:
        from ._uplift_univariate_resident import uplift_univariate_eng_mi_resident

        eng_mi = uplift_univariate_eng_mi_resident(raw_X, engineered_X, y_arr, nbins=nbins)
    if eng_mi is None:
        # Column-chunked MI scoring -> bit-identical, bounds peak RAM at scale (see mi_classif_batch_chunked).
        eng_mi = mi_classif_batch_chunked(engineered_X, y_arr, nbins=nbins)
    eng_mi = np.asarray(eng_mi, dtype=np.float64)
    rows = []
    for j, eng_name in enumerate(engineered_X.columns):
        # D1 (2026-06-22): recover the true source via longest raw-prefix match, not a blind
        # first-``"__"`` split (which mis-stems one-hot sources like ``"city__NY"``).
        source = _source_from_engineered_name(eng_name, raw_cols)
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
    # MATRIX-NATIVE resident path (Piece 3, gated default-off): build the candidate matrix ON the device +
    # score plug-in MI resident (no H2D). Falls back to the host build/scoring on any failure or when off.
    # STRICT (2026-06-28): under MLFRAME_FE_GPU_STRICT this device-born build is FORCED even when the
    # MLFRAME_FE_GPU_RESIDENT_BASIS_MI opt-out turned it off -- the STRICT mandate is "every selection-
    # equivalent GPU twin runs on the device", and the orth-univariate basis build IS one (already default-on,
    # parity-validated). Without this, STRICT + that opt-out would host-materialise the (n, cols*bases*degrees)
    # expansion and upload it at _orth_mi_backends._mi_classif_batch:311; with it the operands upload once and
    # the basis is evaluated on-device (line 311 is never reached for THIS family). Scope note: the orth-uni
    # hybrid is NOT the dominant :311 H2D site -- a full-fit byte-audit (2026-06-28, F2 300k strict) attributes
    # ~78% of the :311 upload to the conditional-gate _gate_grid_mi (host-built tau-grid (n, k<=527) matrices,
    # which have NO device operand/basis handoff and whose residency was separately bench-rejected, see
    # _conditional_gate_fe.cheap_row_argmax_scan). This edit closes only the narrow STRICT+opt-out orth-uni
    # residual; it is selection-equivalent and wall-neutral. Any GPU failure resets _gpu_eng -> host fallback.
    _gpu_eng = None  # (eng_matrix_cupy, names) when the GPU path produced candidates
    try:
        from .._gpu_resident_fe import fe_gpu_resident_basis_mi_enabled, _cuda_present
        from .._fe_gpu_strict import fe_gpu_strict_enabled
        if (fe_gpu_resident_basis_mi_enabled() or fe_gpu_strict_enabled()) and _cuda_present():
            _g_mat, _g_names, scores = _gpu_build_and_score_univariate(X, cols, degrees, basis, y, nbins)
            if _g_mat is None:
                return X.copy(), scores
            _gpu_eng = (_g_mat, _g_names)
    except Exception:
        logger.debug("hybrid_orth_mi_fe: GPU resident basis-MI path failed; host fallback", exc_info=True)
        _gpu_eng = None
    if _gpu_eng is None:
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
    if _gpu_eng is None:
        X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    else:
        # GPU resident path: D2H ONLY the winning columns from the device candidate matrix.
        import cupy as cp
        _g_mat, _g_names = _gpu_eng
        _idx = {nm: i for i, nm in enumerate(_g_names)}
        _cols_host = {k: cp.asnumpy(_g_mat[:, _idx[k]]) for k in keep if k in _idx}
        X_aug = (pd.concat([X, pd.DataFrame(_cols_host, index=X.index)], axis=1)
                 if _cols_host else X.copy())
    return X_aug, scores




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
    subsample_n: int = 0,
    subsample_seed: int = 42,
):
    """Same as :func:`hybrid_orth_mi_fe` but additionally returns a list of
    ``EngineeredRecipe`` objects -- one per appended univariate column --
    so that ``MRMR.transform`` can recompute each engineered column on
    test data without re-running the MI ranking.

    SUBSAMPLED DECISION (2026-06-21). When ``subsample_n`` > 0 and the frame is
    larger, the basis-selection DECISION (MI ranking / uplift gate) is made on a
    seeded row SUBSAMPLE -- the SAME pattern the pair-search FE
    (``check_prospective_fe_pairs``) uses -- and the chosen engineered columns are
    then REBUILT at FULL n via the basis evaluator (the recipe loop already
    recomputes each winner's full-column values to freeze its preprocess params, so
    the OUTPUT is byte-identical to a full-data fit GIVEN the same winners). This
    removes the ~n/subsample_n redundant rows from the (CPU-heavy) orth-FE MI sweep
    and aligns it with the pair-search so both families decide on the same data.
    Default 0 = legacy full-data decision (byte-for-byte unchanged).

    Returns
    -------
    (X_augmented, scores, recipes)
    """
    from ..engineered_recipes import build_orth_univariate_recipe
    _full_n = len(X)
    _do_sub = isinstance(subsample_n, int) and 0 < subsample_n < _full_n
    if _do_sub:
        _sub_idx = np.sort(
            np.random.default_rng(int(subsample_seed)).choice(
                _full_n, size=int(subsample_n), replace=False,
            )
        )
        _X_fit = X.iloc[_sub_idx].reset_index(drop=True)
        _y_fit = np.asarray(y)[_sub_idx]
    else:
        _X_fit, _y_fit = X, y
    # DECISION on the (possibly subsampled) fit frame.
    X_aug_fit, scores = hybrid_orth_mi_fe(
        _X_fit, _y_fit, cols=cols, degrees=degrees, basis=basis,
        top_k=top_k, min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac, nbins=nbins,
    )
    appended = [c for c in X_aug_fit.columns if c not in _X_fit.columns]
    # D1 (2026-06-22): the authoritative raw-source set for un-stemming engineered names.
    _raw_src_cols = [c for c in _X_fit.columns]
    recipes = []
    # When subsampling, accumulate the FULL-n engineered columns (rebuilt from full
    # X[src] in the recipe loop below) so X_aug carries full-n output, not the
    # subsample-sized X_aug_fit.
    _full_eng_cols: dict = {}
    for name in appended:
        # Re-derive (src, degree, basis) from the appended frame: src is the
        # prefix before ``__``; basis/degree are encoded in the suffix. Cross-
        # check by also routing the source column via the same auto rule we
        # used at fit time so the recipe replays identically.
        src = _source_from_engineered_name(name, _raw_src_cols)
        # suffix = everything after the recovered ``"{src}__"`` (NOT the first ``"__"``, which
        # would mis-split a one-hot source ``"city__NY__He2"`` -> src="city", suffix="NY__He2").
        suffix = name[len(src) + 2:] if name.startswith(src + "__") else name.split("__", 1)[1]
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
        # BUG2 FIX (2026-06-12): freeze the fit-time basis-preprocess params into
        # the recipe so transform() replays byte-exactly on any row-slice. ``X`` is
        # the FULL fit frame, and the fit path preprocessed via ``fit_fn(full_col)``,
        # so recomputing the params from the same full column here reproduces the
        # exact train-time axis (mean/std / lo-hi / shift). Without this, replay
        # refits the axis from the transform slice and the z-score drifts by ~1e-3,
        # which the downstream quantiser turns into a |delta|=1 bin drift on a nested
        # ``a__He2`` sub-operand (the BUG2 replay-determinism regression).
        _pp = None
        try:
            _col_full = np.asarray(X[src].values, dtype=np.float64)
            _vals_full, _pp = _evaluate_basis_column(
                _col_full, chosen_basis, int(chosen_degree), return_params=True,
            )
            # The full-column evaluation IS the full-n engineered output for this
            # winner -- reuse it (the subsampled DECISION only chose basis/degree;
            # given that, the values equal a full-data fit, so the OUTPUT is exact).
            if _do_sub:
                _full_eng_cols[name] = np.asarray(_vals_full)
        except Exception:
            _pp = None  # best-effort: fall back to legacy refit-at-replay path
        recipes.append(build_orth_univariate_recipe(
            name=name, src_name=src,
            basis=chosen_basis, degree=chosen_degree,
            preprocess_params=_pp,
        ))
    # Build the augmented frame at FULL n. Without subsampling X_aug_fit is already
    # full-n; with subsampling, append the full-n engineered columns rebuilt above
    # (any winner whose full-column rebuild raised is dropped here AND has no recipe).
    if _do_sub:
        if _full_eng_cols:
            X_aug = pd.concat(
                [X, pd.DataFrame(_full_eng_cols, index=X.index)], axis=1,
            )
        else:
            X_aug = X.copy()
    else:
        X_aug = X_aug_fit
    return X_aug, scores, recipes


from ._orth_pair_cross_fe import (  # noqa: E402,F401
    _pair_eng_col_name,
    generate_pair_cross_basis_features,
    hybrid_orth_mi_pair_fe,
    hybrid_orth_mi_pair_fe_with_recipes,
    score_pair_cross_basis_by_mi_uplift,
)
from ._orth_extra_basis_fe import (  # noqa: E402,F401
    _ADAPTIVE_FE_RAW_USABILITY_CAP,
    _EXTRA_BASIS_KINDS,
    _FOURIER_INT_AS_CAT_MAX_CARD,
    _build_recipe_from_meta,
    _chirp_axis,
    _detect_fourier_freq_for_col,
    _detect_fourier_freqs_for_col,
    _fit_chirp_warp_for_col,
    _fit_fourier_for_col,
    _fit_spline_for_col,
    _heldout_smooth_r2,
    _is_int_as_cat_axis,
    generate_extra_basis_features,
    hybrid_orth_extra_basis_fe_with_recipes,
)



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

