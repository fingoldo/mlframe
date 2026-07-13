"""Layer 77 (2026-06-01): QUADRUPLET cross-basis FE.

Extends the Layer 22 pair-cross-basis path and the Layer 56 triplet path
to FOUR legs: ``basis_a(x_i) * basis_b(x_j) * basis_c(x_k) * basis_d(x_l)``.

Why quadruplets matter
----------------------

The pair path (Layer 22) captures bilinear interactions. The triplet
path (Layer 56) captures ``y = sign(x_i * x_j * x_k)`` (3-way XOR /
volume targets). Layer 77 reaches the next tier:

* 4-way XOR ``y = sign(x_1 * x_2 * x_3 * x_4)`` -- every triplet
  marginal MI is zero by symmetry (the 4th leg randomises balanced),
  so the Layer 56 triplet stage cannot find it. Only the cell
  ``He_1*He_1*He_1*He_1`` carries signal.

* Real-world ``revenue = price * qty * count * discount`` style
  decompositions where each leg is centred and the multiplicative
  structure surfaces only at the 4-way order.

Cost guard
----------

Quadruplet enumeration is O(p^4 * deg^4). Two safety knobs:

1. ``seed_count`` (caller-level): rank source columns by univariate MI,
   keep top-N, enumerate quadruplets only from that subset. At
   seed_count=4 we get C(4,4)=1 quadruplet * deg^4 cells. At
   seed_count=5 we get C(5,4)=5 * deg^4 cells. Bounded regardless of
   input width.

2. Default ``max_degree=1`` in the MRMR ctor wiring. ``He_1^4`` IS the
   dominant 4-way interaction signal for every multiplicative 4-way
   target the literature pins; max_degree>=2 multiplies cell count by
   16 for marginal MI lift on synthetic 4-way XOR / volume targets.

Recipe parity
-------------

Each appended quadruplet column is backed by an ``orth_quadruplet_cross``
``EngineeredRecipe`` so ``MRMR.transform`` replays it deterministically
from the source columns alone (no y reference).
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .hermite_fe import basis_route_by_moments, _POLY_BASES
from ._orthogonal_univariate_fe import (
    _evaluate_basis_column,
    _mi_classif_batch, mi_classif_batch_chunked,
    _BASIS_CODE,
    hybrid_orth_mi_fe,
    cached_raw_mi_baseline,
)

logger = logging.getLogger(__name__)

__all__ = [
    "generate_quadruplet_cross_basis_features",
    "score_quadruplet_cross_basis_by_mi_uplift",
    "hybrid_orth_mi_quadruplet_fe",
    "hybrid_orth_mi_quadruplet_fe_with_recipes",
]


def _quadruplet_eng_col_name(
    col_i: str, col_j: str, col_k: str, col_l: str,
    basis: str, deg_a: int, deg_b: int, deg_c: int, deg_d: int,
) -> str:
    """Stable naming: ``"{col_i}*{col_j}*{col_k}*{col_l}__He{a}_He{b}_He{c}_He{d}"``.

    Mirrors the Layer 22 pair / Layer 56 triplet naming convention with
    a fourth leg. Per-leg basis round-trips via the recipe's ``extra``
    dict, not the name (the name uses leg-1's basis code for stability).
    """
    code = _BASIS_CODE.get(basis, basis)
    return f"{col_i}*{col_j}*{col_k}*{col_l}__" f"{code}{deg_a}_{code}{deg_b}_{code}{deg_c}_{code}{deg_d}"


def generate_quadruplet_cross_basis_features(
    X: pd.DataFrame,
    quadruplets: Sequence[tuple[str, str, str, str]],
    *,
    max_degree: int = 1,
    basis: str = "auto",
    min_degree: int = 1,
) -> pd.DataFrame:
    """For each (col_i, col_j, col_k, col_l) quadruplet and each
    (deg_a, deg_b, deg_c, deg_d) in [min_degree..max_degree]^4, emit
    ``basis_a(x_i) * basis_b(x_j) * basis_c(x_k) * basis_d(x_l)`` as a
    new column.

    Parameters
    ----------
    X : DataFrame
        Source frame. All four legs of every quadruplet must be numeric.
    quadruplets : sequence of (col_i, col_j, col_k, col_l)
        Column quadruplets to expand. Multiplication is commutative;
        pass each unordered quadruplet once. Self-aliased quadruplets
        (any two legs equal) are skipped.
    max_degree : int
        Maximum degree per leg. Default 1: ``He_1^4`` captures every
        multiplicative 4-way target (4-way XOR, volume). max_degree=2
        emits 2^4=16 cells per quadruplet; rarely worth the combinatorial
        blowup on synthetic targets, may help on curved 4-way manifolds.
    basis : {'auto', 'hermite', 'legendre', 'chebyshev', 'laguerre'}
        Routed per-column via ``basis_route_by_moments`` when ``'auto'``.
    min_degree : int
        Minimum degree per leg. Default 1 -- degree 0 produces a leg
        equal to the constant, collapsing the quadruplet to a triplet
        which is already covered by Layer 56.

    Returns
    -------
    DataFrame of quadruplet-cross-basis columns.
    """
    if not quadruplets:
        return pd.DataFrame(index=X.index)
    cache: dict[tuple[str, int, str], np.ndarray] = {}
    out_cols: dict = {}
    max_d = int(max_degree)
    min_d = max(0, int(min_degree))
    for quad in quadruplets:
        if len(quad) != 4:
            continue
        col_i, col_j, col_k, col_l = quad
        legs_set = {col_i, col_j, col_k, col_l}
        # Skip degenerate quadruplets where any two legs alias the same column.
        if len(legs_set) != 4:
            continue
        if col_i not in X.columns or col_j not in X.columns or col_k not in X.columns or col_l not in X.columns:
            logger.warning(
                "generate_quadruplet_cross_basis_features: missing column in "
                "(%r,%r,%r,%r); skipping",
                col_i, col_j, col_k, col_l,
            )
            continue
        if not (
            pd.api.types.is_numeric_dtype(X[col_i])
            and pd.api.types.is_numeric_dtype(X[col_j])
            and pd.api.types.is_numeric_dtype(X[col_k])
            and pd.api.types.is_numeric_dtype(X[col_l])
        ):
            continue
        from ._fe_usability_signal import _crit_np_dtype
        _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); matches the GPU device
        # builder's operand dtype (_gpu_resident_cross_basis.py) so host and device run the polynomial
        # recurrence at the SAME precision -- see build_leg_product_matrix_gpu's docstring.
        x_i = np.asarray(X[col_i].to_numpy(), dtype=_dt)
        x_j = np.asarray(X[col_j].to_numpy(), dtype=_dt)
        x_k = np.asarray(X[col_k].to_numpy(), dtype=_dt)
        x_l = np.asarray(X[col_l].to_numpy(), dtype=_dt)
        for x in (x_i, x_j, x_k, x_l):
            finite_mask = np.isfinite(x)
            if not finite_mask.all():
                fill = float(np.nanmean(x[finite_mask])) if finite_mask.any() else 0.0
                np.copyto(x, np.where(finite_mask, x, fill))
        basis_i = basis_route_by_moments(x_i) if basis == "auto" else basis
        basis_j = basis_route_by_moments(x_j) if basis == "auto" else basis
        basis_k = basis_route_by_moments(x_k) if basis == "auto" else basis
        basis_l = basis_route_by_moments(x_l) if basis == "auto" else basis
        if basis_i not in _POLY_BASES or basis_j not in _POLY_BASES or basis_k not in _POLY_BASES or basis_l not in _POLY_BASES:
            logger.warning(
                "generate_quadruplet_cross_basis_features: unknown basis "
                "%r/%r/%r/%r for quadruplet (%r,%r,%r,%r); skipping",
                basis_i, basis_j, basis_k, basis_l,
                col_i, col_j, col_k, col_l,
            )
            continue
        # A 4-way basis product is even more overflow-prone than the pair-cross 2-way product; suppress
        # the resulting numpy RuntimeWarnings and scrub the product below rather than leaving a
        # non-finite value to surface as a silent NaN downstream (mirrors generate_pair_cross_basis_features).
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for deg_a in range(min_d, max_d + 1):
                for deg_b in range(min_d, max_d + 1):
                    for deg_c in range(min_d, max_d + 1):
                        for deg_d in range(min_d, max_d + 1):
                            if deg_a == 0 and deg_b == 0 and deg_c == 0 and deg_d == 0:
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
                                key_c = (col_k, deg_c, basis_k)
                                if key_c not in cache:
                                    cache[key_c] = _evaluate_basis_column(x_k, basis_k, deg_c)
                                h_c = cache[key_c]
                                key_d = (col_l, deg_d, basis_l)
                                if key_d not in cache:
                                    cache[key_d] = _evaluate_basis_column(x_l, basis_l, deg_d)
                                h_d = cache[key_d]
                                name = _quadruplet_eng_col_name(
                                    col_i, col_j, col_k, col_l, basis_i,
                                    deg_a, deg_b, deg_c, deg_d,
                                )
                                out_cols[name] = np.nan_to_num(h_a * h_b * h_c * h_d, nan=0.0, posinf=0.0, neginf=0.0)
                            except Exception as exc:
                                logger.warning(
                                    "generate_quadruplet_cross_basis_features: "
                                    "basis=%r/%r/%r/%r deg=%d/%d/%d/%d on quadruplet "
                                    "(%r,%r,%r,%r) raised %r; skipping",
                                    basis_i, basis_j, basis_k, basis_l,
                                    deg_a, deg_b, deg_c, deg_d,
                                    col_i, col_j, col_k, col_l, exc,
                                )
                                continue
    return pd.DataFrame(out_cols, index=X.index)


_QUADRUPLET_SCORE_EMPTY_COLS = [
    "engineered_col",
    "source_col_i", "source_col_j", "source_col_k", "source_col_l",
    "baseline_mi_i", "baseline_mi_j", "baseline_mi_k", "baseline_mi_l",
    "baseline_mi",
    "engineered_mi", "uplift",
]


def _quadruplet_device_col_specs(eng_columns, raw_cols):
    """Per-column device leg specs aligned 1:1 with ``eng_columns`` for the quadruplet family (4 legs),
    recovering ``(col, degree)`` per leg from each ``"{i}*{j}*{k}*{l}__{a}_{b}_{c}_{d}"`` name. ``None`` if ANY
    column does not resolve to exactly four legs."""
    from ._orthogonal_univariate_fe._gpu_resident_cross_basis import _parse_code_deg

    specs = []
    raw_set = set(raw_cols)
    for name in eng_columns:
        head = name.split("__", 1)[0] if "__" in name else name
        legs = head.split("*")
        if len(legs) != 4 or any(leg not in raw_set for leg in legs):
            return None
        try:
            suffix = name.split("__", 1)[1]
            parts = suffix.split("_")
        except (ValueError, IndexError):
            return None
        if len(parts) != 4:
            return None
        degs = [_parse_code_deg(p) for p in parts]
        if any(d is None for d in degs):
            return None
        specs.append({"legs": [(legs[i], degs[i]) for i in range(4)]})
    return specs


def score_quadruplet_cross_basis_by_mi_uplift(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    nbins: int = 10,
    basis: str = "auto",
) -> pd.DataFrame:
    """Score each quadruplet-cross-basis column by MI uplift vs the BEST
    of the four raw source columns.

    Mirrors ``score_triplet_cross_basis_by_mi_uplift`` with a fourth leg.
    Baseline is ``max(MI(x_i;y), MI(x_j;y), MI(x_k;y), MI(x_l;y))`` -- a
    real 4-way interaction must beat the BEST individual leg, not just
    the worst, to count as genuine 4-way signal.

    ``basis`` mirrors the ``generate_quadruplet_cross_basis_features`` call that produced ``engineered_X`` so the
    DEVICE-BORN STRICT-resident scorer re-routes each leg to the SAME basis the host generator used. Unused on
    the host default path.
    """
    from ._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); hoisted so _dt is bound on every branch
    y_arr = np.asarray(y).astype(np.int64) if not np.issubdtype(np.asarray(y).dtype, np.integer) else np.asarray(y, dtype=np.int64)
    raw_cols = list(raw_X.columns)
    if engineered_X.empty:
        return pd.DataFrame(columns=_QUADRUPLET_SCORE_EMPTY_COLS)
    # DEVICE-BORN (STRICT-resident): rebuild the quadruplet product matrix on the GPU + score both it and the raw
    # baseline through the SAME resident plug-in MI -- collapsing the host product-matrix upload at :311.
    raw_mi_map = eng_mi = None
    _specs = _quadruplet_device_col_specs(engineered_X.columns, raw_cols)
    if _specs is not None:
        from ._orthogonal_univariate_fe._orth_pair_cross_fe import _crossbasis_device_born_on
        if _crossbasis_device_born_on():
            from ._orthogonal_univariate_fe._gpu_resident_cross_basis import raw_and_product_mi_resident
            _res = raw_and_product_mi_resident(raw_X, engineered_X, y_arr, _specs, nbins=nbins, basis=basis)
            if _res is not None:
                raw_mi_map, eng_mi = _res
    if eng_mi is None:
        from ._fe_usability_signal import _crit_np_dtype
        _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust
        raw_mi = _mi_classif_batch(raw_X.to_numpy(dtype=_dt), y_arr, nbins=nbins)
        raw_mi_map = dict(zip(raw_cols, raw_mi.tolist()))
        eng_mi = mi_classif_batch_chunked(engineered_X, y_arr, nbins=nbins)
    assert raw_mi_map is not None  # set by either the device-born path or the host fallback above
    rows = []
    for j, eng_name in enumerate(engineered_X.columns):
        # parse "{col_i}*{col_j}*{col_k}*{col_l}__..."
        head = eng_name.split("__", 1)[0] if "__" in eng_name else eng_name
        legs = head.split("*")
        if len(legs) != 4:
            # Not a quadruplet column -- skip (triplet/pair/univariate output
            # mistakenly routed through this scorer would land here).
            continue
        col_i, col_j, col_k, col_l = legs
        baseline_i = float(raw_mi_map.get(col_i, 0.0))
        baseline_j = float(raw_mi_map.get(col_j, 0.0))
        baseline_k = float(raw_mi_map.get(col_k, 0.0))
        baseline_l = float(raw_mi_map.get(col_l, 0.0))
        baseline = max(baseline_i, baseline_j, baseline_k, baseline_l)
        emi = float(eng_mi[j])
        uplift = emi / (baseline + 1e-12)
        rows.append({
            "engineered_col": eng_name,
            "source_col_i": col_i,
            "source_col_j": col_j,
            "source_col_k": col_k,
            "source_col_l": col_l,
            "baseline_mi_i": baseline_i,
            "baseline_mi_j": baseline_j,
            "baseline_mi_k": baseline_k,
            "baseline_mi_l": baseline_l,
            "baseline_mi": baseline,
            "engineered_mi": emi,
            "uplift": uplift,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("uplift", ascending=False).reset_index(drop=True)
    return df


def _noise_aware_floor(values: np.ndarray, sigma_thresh: float) -> float:
    """Median + sigma * 1.4826 * MAD noise floor used by Layer 22 / 56.
    Returns 0 when too few values to estimate robustly.
    """
    if values.size < 4:
        return 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med + sigma_thresh * 1.4826 * mad


def hybrid_orth_mi_quadruplet_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    quadruplet_max_degree: int = 1,
    basis: str = "auto",
    top_k: int = 5,
    top_quadruplet_count: int = 2,
    top_quadruplet_seed_k: int = 4,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    quadruplet_min_uplift: float = 1.05,
    quadruplet_min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two-stage hybrid: (1) univariate orthogonal-poly FE + MI-greedy,
    then (2) cross-basis QUADRUPLET FE on the top-N raw source columns,
    also MI-greedy.

    Stage 1 reuses ``hybrid_orth_mi_fe`` (Layer 21). The quadruplet stage
    enumerates ALL ordered-by-name C(N, 4) quadruplets over the seed pool.

    Cost: enumerating quadruplets over full input would be O(p^4); the
    seed_k cap turns it into O(seed_k^4) regardless of input width.
    Default ``top_quadruplet_seed_k=4`` -> 1 quadruplet * 1 cell (deg 1)
    = 1 candidate -- bounded.

    Parameters
    ----------
    X, y, cols, degrees, basis, top_k, min_uplift, min_abs_mi_frac, nbins
        Forwarded to the univariate ``hybrid_orth_mi_fe`` stage.
    quadruplet_max_degree : int
        Max degree per leg in the quadruplet stage. Default 1 emits
        exactly one cell per quadruplet (``He_1^4``).
    top_quadruplet_count : int
        How many quadruplet winners to append after the univariate winners.
    top_quadruplet_seed_k : int
        How many top raw source columns to pull into the quadruplet seed
        pool. With N=4 we enumerate C(4,4)=1; with N=5, C(5,4)=5.
    quadruplet_min_uplift, quadruplet_min_abs_mi_frac : float
        Two-gate thresholds for the quadruplet stage. Same semantics as
        the univariate gates, compared against the BEST individual leg
        MI as the baseline.

    Returns
    -------
    (X_augmented, univariate_scores, quadruplet_scores)
    """
    # Stage 1: univariate hybrid (Layer 21).
    from ._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); hoisted so _dt is bound on every branch
    X_aug_uni, uni_scores = hybrid_orth_mi_fe(
        X, y,
        cols=cols, degrees=degrees, basis=basis,
        top_k=top_k, min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac, nbins=nbins,
    )

    # Build the quadruplet seed pool. Same caveat as Layer 56: on pure
    # 4-way XOR every leg has zero univariate MI to y, so a raw-MI
    # top-k cut would drop signal legs. When the caller explicitly
    # passes ``cols``, respect that order; only fall back to raw-MI
    # ranking when ``cols=None`` AND input width > seed_k.
    raw_cols_all = [c for c in (cols or X.columns) if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    seed_sources: list[str] = []
    if len(raw_cols_all) >= 4:
        if cols is not None:
            seed_sources = list(raw_cols_all[: int(top_quadruplet_seed_k)])
        else:
            # Stage 1 (hybrid_orth_mi_fe, just above) already ran a full raw-column MI batch
            # internally and surfaced it per-source in uni_scores["baseline_mi"] (same y coercion,
            # same raw-column universe when cols=None) -- reuse it instead of a second full
            # _mi_classif_batch pass; only recompute for a raw column uni_scores doesn't cover
            # (skipped source: all-NaN / int-as-cat / dedup'd), so the ranking stays exactly
            # selection-equivalent to the old always-recompute path.
            _baseline_map: dict = {}
            if not uni_scores.empty:
                _baseline_map = uni_scores.groupby("source_col")["baseline_mi"].first().to_dict()
            _missing = [c for c in raw_cols_all if c not in _baseline_map]
            if _missing:
                y_arr = np.asarray(y).astype(np.int64) if not np.issubdtype(np.asarray(y).dtype, np.integer) else np.asarray(y, dtype=np.int64)
                from ._fe_usability_signal import _crit_np_dtype
                _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust
                # Fit-scoped memo: no-op passthrough outside an active orth_scoring_memo_scope(); inside a
                # scope, shares this residual MI batch with sibling opt-in layers.
                _baseline_map.update(cached_raw_mi_baseline(_missing, X[_missing].to_numpy(dtype=_dt), y_arr, nbins=nbins))
            raw_mi_arr = np.array([float(_baseline_map.get(c, 0.0)) for c in raw_cols_all])
            order = np.argsort(-raw_mi_arr)
            seed_sources = [raw_cols_all[i] for i in order[: int(top_quadruplet_seed_k)]]

    if len(seed_sources) < 4 or int(top_quadruplet_count) <= 0:
        return X_aug_uni, uni_scores, pd.DataFrame(columns=_QUADRUPLET_SCORE_EMPTY_COLS)

    quadruplets = [
        (seed_sources[i], seed_sources[j], seed_sources[k], seed_sources[m])
        for i in range(len(seed_sources))
        for j in range(i + 1, len(seed_sources))
        for k in range(j + 1, len(seed_sources))
        for m in range(k + 1, len(seed_sources))
    ]
    quad_eng = generate_quadruplet_cross_basis_features(
        X, quadruplets, max_degree=quadruplet_max_degree, basis=basis,
    )
    if quad_eng.empty:
        return X_aug_uni, uni_scores, pd.DataFrame(columns=_QUADRUPLET_SCORE_EMPTY_COLS)

    raw_X_seed = X[seed_sources]
    quad_scores = score_quadruplet_cross_basis_by_mi_uplift(
        raw_X_seed, quad_eng, y, nbins=nbins, basis=basis,
    )

    # Two-gate selection mirrors Layer 22 / 56. Quadruplet candidate
    # counts are larger (O(seed_k^4 * deg^4)) so the Bonferroni-scaled
    # sigma threshold is correspondingly stricter.
    max_raw_baseline = float(quad_scores["baseline_mi"].max()) if not quad_scores.empty else 0.0
    if not uni_scores.empty:
        max_raw_baseline = max(max_raw_baseline, float(uni_scores["baseline_mi"].max()))
    max_quad_engineered = float(quad_scores["engineered_mi"].max()) if not quad_scores.empty else 0.0
    legacy_floor = float(quadruplet_min_abs_mi_frac) * max(
        max_raw_baseline,
        max_quad_engineered,
    )
    _baselines = quad_scores["baseline_mi"].to_numpy() if not quad_scores.empty else np.array([])
    n_cands = int(_baselines.size)
    sigma_thresh = max(
        5.0,
        float(np.sqrt(2.0 * np.log(max(2.0, 2.0 * n_cands))) + 1.5),
    )
    noise_floor = _noise_aware_floor(_baselines, sigma_thresh)
    _eng_mis = quad_scores["engineered_mi"].to_numpy() if not quad_scores.empty else np.array([])
    eng_noise_floor = _noise_aware_floor(_eng_mis, sigma_thresh)
    abs_floor = max(legacy_floor, noise_floor, eng_noise_floor)
    qualified = quad_scores[(quad_scores["uplift"] >= float(quadruplet_min_uplift)) & (quad_scores["engineered_mi"] >= abs_floor)]
    winners = qualified.head(int(top_quadruplet_count))
    keep_quad = list(winners["engineered_col"])
    if keep_quad:
        X_aug = pd.concat([X_aug_uni, quad_eng[keep_quad]], axis=1)
    else:
        X_aug = X_aug_uni
    return X_aug, uni_scores, quad_scores


from ._fe_family_timing import fe_timed


@fe_timed("quadruplet")
def hybrid_orth_mi_quadruplet_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    quadruplet_max_degree: int = 1,
    basis: str = "auto",
    top_k: int = 5,
    top_quadruplet_count: int = 2,
    top_quadruplet_seed_k: int = 4,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    quadruplet_min_uplift: float = 1.05,
    quadruplet_min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
):
    """Same as :func:`hybrid_orth_mi_quadruplet_fe` but additionally returns
    a flat list of recipes (univariate + quadruplet, in append order) so
    that ``MRMR.transform`` can replay each engineered column from X
    alone (no y).
    """
    from .engineered_recipes import build_orth_univariate_recipe
    from ._orthogonal_quadruplet_fe_recipes import build_orth_quadruplet_cross_recipe

    X_aug, uni_scores, quad_scores = hybrid_orth_mi_quadruplet_fe(
        X, y, cols=cols, degrees=degrees, basis=basis,
        quadruplet_max_degree=quadruplet_max_degree,
        top_k=top_k,
        top_quadruplet_count=top_quadruplet_count,
        top_quadruplet_seed_k=top_quadruplet_seed_k,
        min_uplift=min_uplift, min_abs_mi_frac=min_abs_mi_frac,
        quadruplet_min_uplift=quadruplet_min_uplift,
        quadruplet_min_abs_mi_frac=quadruplet_min_abs_mi_frac,
        nbins=nbins,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    code_to_basis = {"He": "hermite", "LL": "laguerre", "T": "chebyshev", "L": "legendre"}

    def _parse_code_deg(s: str):
        """Parse a basis-code-plus-degree token (e.g. ``"He3"``) into its ``(basis_name, degree)`` pair."""
        for code in ("LL", "He", "T", "L"):
            if s.startswith(code):
                rest = s[len(code) :]
                if rest.isdigit():
                    return code_to_basis[code], int(rest)
        return None, None

    recipes = []
    for name in appended:
        head = name.split("__", 1)[0]
        legs = head.split("*")
        if len(legs) == 4:
            # quadruplet cross: "{i}*{j}*{k}*{l}__{code_a}{deg_a}_{code_b}{deg_b}_{code_c}{deg_c}_{code_d}{deg_d}"
            try:
                suffix = name.split("__", 1)[1]
                parts = suffix.split("_")
            except (ValueError, IndexError):
                logger.warning(
                    "hybrid_orth_mi_quadruplet_fe_with_recipes: cannot parse " "suffix in %r; skipping recipe.",
                    name,
                )
                continue
            if len(parts) != 4:
                logger.warning(
                    "hybrid_orth_mi_quadruplet_fe_with_recipes: expected 4 deg " "parts in %r; skipping recipe.",
                    name,
                )
                continue
            basis_a, deg_a = _parse_code_deg(parts[0])
            basis_b, deg_b = _parse_code_deg(parts[1])
            basis_c, deg_c = _parse_code_deg(parts[2])
            basis_d, deg_d = _parse_code_deg(parts[3])
            if basis_a is None or basis_b is None or basis_c is None or basis_d is None:
                logger.warning(
                    "hybrid_orth_mi_quadruplet_fe_with_recipes: cannot parse " "code/deg from %r; skipping recipe.",
                    name,
                )
                continue
            col_i, col_j, col_k, col_l = legs
            # Value-construction (recovers the basis preprocess params for recipe replay): always
            # float64, matching the fresh-path's own dtype (see generate_quadruplet_cross_basis_features).
            x_i = X[col_i].to_numpy(dtype=np.float64)
            x_j = X[col_j].to_numpy(dtype=np.float64)
            x_k = X[col_k].to_numpy(dtype=np.float64)
            x_l = X[col_l].to_numpy(dtype=np.float64)
            if basis == "auto":
                try:
                    basis_a = basis_route_by_moments(x_i)
                    basis_b = basis_route_by_moments(x_j)
                    basis_c = basis_route_by_moments(x_k)
                    basis_d = basis_route_by_moments(x_l)
                except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                    logger.debug("suppressed in _orthogonal_quadruplet_fe.py:573: %s", e)
                    pass
            # REPLAY-FIDELITY FIX (2026-06-13): freeze each leg's fit-time basis-preprocess params so
            # transform() replays the basis axis byte-exactly (no slice-vs-full refit drift). Guarded.
            _pp_a = _pp_b = _pp_c = _pp_d = None
            try:
                _, _pp_a = _evaluate_basis_column(x_i, basis_a, deg_a, return_params=True)
                _, _pp_b = _evaluate_basis_column(x_j, basis_b, deg_b, return_params=True)
                _, _pp_c = _evaluate_basis_column(x_k, basis_c, deg_c, return_params=True)
                _, _pp_d = _evaluate_basis_column(x_l, basis_d, deg_d, return_params=True)
            except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed in _orthogonal_quadruplet_fe.py:583: %s", e)
                pass
            recipes.append(build_orth_quadruplet_cross_recipe(
                name=name,
                src_a_name=col_i, src_b_name=col_j,
                src_c_name=col_k, src_d_name=col_l,
                basis_i=basis_a, basis_j=basis_b,
                basis_k=basis_c, basis_l=basis_d,
                deg_a=deg_a, deg_b=deg_b, deg_c=deg_c, deg_d=deg_d,
                preprocess_params_i=_pp_a, preprocess_params_j=_pp_b,
                preprocess_params_k=_pp_c, preprocess_params_l=_pp_d,
            ))
        elif len(legs) == 1:
            # univariate: "{col}__{code}{degree}"
            src = legs[0]
            suffix = name.split("__", 1)[1] if "__" in name else ""
            chosen_basis, chosen_degree = _parse_code_deg(suffix)
            if chosen_basis is None or chosen_degree is None:
                logger.warning(
                    "hybrid_orth_mi_quadruplet_fe_with_recipes: cannot parse " "basis/degree from %r; skipping recipe.",
                    name,
                )
                continue
            # REPLAY-FIDELITY FIX (2026-06-13): freeze the fit-time basis-preprocess params.
            _pp_u = None
            try:
                _x_u = X[src].to_numpy(dtype=np.float64)  # value-construction: always float64, see above
                _, _pp_u = _evaluate_basis_column(_x_u, chosen_basis, chosen_degree, return_params=True)
            except Exception:  # nosec B110 - optional dependency import guard
                pass
            recipes.append(build_orth_univariate_recipe(
                name=name, src_name=src,
                basis=chosen_basis, degree=chosen_degree,
                preprocess_params=_pp_u,
            ))
        else:
            logger.warning(
                "hybrid_orth_mi_quadruplet_fe_with_recipes: unexpected leg "
                "count in %r (legs=%r); skipping recipe.", name, legs,
            )
            continue
    return X_aug, uni_scores, quad_scores, recipes
