"""Layer 56 (2026-05-31): TRI-PRODUCT cross-basis FE.

Extends the Layer 22 pair-cross-basis path to triplets:
``basis_a(x_i) * basis_b(x_j) * basis_c(x_k)``.

Why triplets matter
-------------------

The pair path (``He_a(z_i) * He_b(z_j)``) captures XOR / saddle / circle and
every other bilinear interaction. But many real-world signals are
genuinely 3-way: ``volume = price * quantity * count``,
``revenue_anomaly = sign(margin * units * region_flag)``, the 3-way XOR
``y = sign(x_1 * x_2 * x_3)`` that no pairwise term can resolve.

For 3-way XOR the marginal MI of each pair-cross-basis term is exactly
zero (the third variable acts as a balanced randomiser); only the
``He_1 * He_1 * He_1`` triplet carries signal. The pair path misses it
completely; this module emits exactly that cell.

Cost guard
----------

Triplet enumeration is O(p^3 * deg^3). Two safety knobs:

1. ``seed_count`` (caller-level): rank source columns by univariate MI
   first, keep the top-N, enumerate triplets only from that subset.
   At seed_count=4 we get C(4,3)=4 triplets * deg^3 cells, which is
   bounded regardless of input width.

2. Default ``max_degree=1`` in the MRMR ctor wiring. ``He_1 * He_1 * He_1``
   IS the dominant 3-way interaction signal for nearly every triplet
   target the literature pins; max_degree>=2 multiplies the cell count
   by 8 for marginal MI lift on synthetic 3-way XOR / volume targets.

Recipe parity
-------------

Each appended triplet column is backed by an ``orth_triplet_cross``
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
    "generate_triplet_cross_basis_features",
    "score_triplet_cross_basis_by_mi_uplift",
    "hybrid_orth_mi_triplet_fe",
    "hybrid_orth_mi_triplet_fe_with_recipes",
]


def _coerce_y_int64(y) -> np.ndarray:
    """Dense int64 class labels. Non-integer y is densified via
    ``np.unique(return_inverse=...)`` rather than truncated with
    ``.astype(int64)`` -- plain truncation merges distinct labels and destroys
    continuous-y signal (everything in [0, 1) collapses to class 0)."""
    arr = np.asarray(y).ravel()
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False)
    _, inv = np.unique(arr, return_inverse=True)
    return inv.astype(np.int64, copy=False)


def _triplet_eng_col_name(
    col_i: str, col_j: str, col_k: str,
    basis: str, deg_a: int, deg_b: int, deg_c: int,
) -> str:
    """Stable naming: ``"{col_i}*{col_j}*{col_k}__He{a}_He{b}_He{c}"``.

    Mirrors the Layer 22 pair-naming convention with a third leg. The
    basis code is the leg-1 basis even when the auto router picks
    different bases for the other legs; downstream parsing recovers
    the per-leg basis via the recipe's ``extra`` dict, not via the name.
    """
    code = _BASIS_CODE.get(basis, basis)
    return f"{col_i}*{col_j}*{col_k}__{code}{deg_a}_{code}{deg_b}_{code}{deg_c}"


def generate_triplet_cross_basis_features(
    X: pd.DataFrame,
    triplets: Sequence[tuple[str, str, str]],
    *,
    max_degree: int = 1,
    basis: str = "auto",
    min_degree: int = 1,
) -> pd.DataFrame:
    """For each (col_i, col_j, col_k) triplet and each
    (deg_a, deg_b, deg_c) in [min_degree..max_degree]^3, emit
    ``basis_a(x_i) * basis_b(x_j) * basis_c(x_k)`` as a new column.

    Parameters
    ----------
    X : DataFrame
        Source frame. All three legs of every triplet must be numeric.
    triplets : sequence of (col_i, col_j, col_k)
        Column triplets to expand. Multiplication is commutative;
        pass each unordered triplet once. Self-triplets (any two legs
        equal) are skipped.
    max_degree : int
        Maximum degree per leg. Default 1: ``He_1 * He_1 * He_1``
        captures 3-way XOR / volume / 3-way product signals which is
        the dominant family of 3-way interactions. max_degree=2 emits
        2^3=8 cells per triplet; rarely worth the combinatorial blowup
        on synthetic targets, may help on curved 3-way manifolds.
    basis : {'auto', 'hermite', 'legendre', 'chebyshev', 'laguerre'}
        Routed per-column via ``basis_route_by_moments`` when ``'auto'``.
    min_degree : int
        Minimum degree per leg. Default 1 -- degree 0 produces a leg
        equal to the constant, collapsing the triplet to a pair which
        is already covered by Layer 22.

    Returns
    -------
    DataFrame of triplet-cross-basis columns.
    """
    if not triplets:
        return pd.DataFrame(index=X.index)
    cache: dict[tuple[str, int, str], np.ndarray] = {}
    out_cols: dict = {}
    max_d = int(max_degree)
    min_d = max(0, int(min_degree))
    for triplet in triplets:
        if len(triplet) != 3:
            continue
        col_i, col_j, col_k = triplet
        # Skip degenerate triplets where any two legs alias the same column.
        if col_i == col_j or col_i == col_k or col_j == col_k:
            continue
        if col_i not in X.columns or col_j not in X.columns or col_k not in X.columns:
            logger.warning(
                "generate_triplet_cross_basis_features: missing column in (%r,%r,%r); skipping",
                col_i, col_j, col_k,
            )
            continue
        if not (pd.api.types.is_numeric_dtype(X[col_i]) and pd.api.types.is_numeric_dtype(X[col_j]) and pd.api.types.is_numeric_dtype(X[col_k])):
            continue
        from ._fe_usability_signal import _crit_np_dtype
        _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); matches the GPU device
        # builder's operand dtype (_gpu_resident_cross_basis.py) so host and device run the polynomial
        # recurrence at the SAME precision -- see build_leg_product_matrix_gpu's docstring.
        # np.array (copy=True): X[col].to_numpy() can alias the DataFrame's backing block -- e.g. a
        # zero-copy Arrow-backed view or a frozen MRMR-fit-cache array reused via fe_append_columns from
        # an earlier stage -- and the np.copyto NaN-fill below would then either mutate the CALLER's X or
        # raise "assignment destination is read-only" on a genuinely read-only block. A fresh copy (matching
        # the sibling pair-cross / GPU-resident generators' established pattern) keeps the fill local and safe.
        x_i = np.array(X[col_i].to_numpy(), dtype=_dt)
        x_j = np.array(X[col_j].to_numpy(), dtype=_dt)
        x_k = np.array(X[col_k].to_numpy(), dtype=_dt)
        for x in (x_i, x_j, x_k):
            finite_mask = np.isfinite(x)
            if not finite_mask.all():
                fill = float(np.nanmean(x[finite_mask])) if finite_mask.any() else 0.0
                np.copyto(x, np.where(finite_mask, x, fill))
        basis_i = basis_route_by_moments(x_i) if basis == "auto" else basis
        basis_j = basis_route_by_moments(x_j) if basis == "auto" else basis
        basis_k = basis_route_by_moments(x_k) if basis == "auto" else basis
        if basis_i not in _POLY_BASES or basis_j not in _POLY_BASES or basis_k not in _POLY_BASES:
            logger.warning(
                "generate_triplet_cross_basis_features: unknown basis %r/%r/%r "
                "for triplet (%r,%r,%r); skipping",
                basis_i, basis_j, basis_k, col_i, col_j, col_k,
            )
            continue
        # A 3-way basis product is strictly more overflow-prone than the pair-cross 2-way product;
        # suppress the resulting numpy RuntimeWarnings and scrub the product below rather than leaving
        # a non-finite value to surface as a silent NaN downstream (mirrors generate_pair_cross_basis_features).
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for deg_a in range(min_d, max_d + 1):
                for deg_b in range(min_d, max_d + 1):
                    for deg_c in range(min_d, max_d + 1):
                        if deg_a == 0 and deg_b == 0 and deg_c == 0:
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
                            # Use leg-1 basis code for the name; per-leg basis
                            # round-trips via the recipe (Layer 22 same trick).
                            name = _triplet_eng_col_name(
                                col_i, col_j, col_k, basis_i,
                                deg_a, deg_b, deg_c,
                            )
                            out_cols[name] = np.nan_to_num(h_a * h_b * h_c, nan=0.0, posinf=0.0, neginf=0.0)
                        except Exception as exc:
                            logger.warning(
                                "generate_triplet_cross_basis_features: basis=%r/%r/%r "
                                "deg=%d/%d/%d on triplet (%r,%r,%r) raised %r; skipping",
                                basis_i, basis_j, basis_k,
                                deg_a, deg_b, deg_c,
                                col_i, col_j, col_k, exc,
                            )
                            continue
    return pd.DataFrame(out_cols, index=X.index)


_TRIPLET_SCORE_EMPTY_COLS = [
    "engineered_col",
    "source_col_i", "source_col_j", "source_col_k",
    "baseline_mi_i", "baseline_mi_j", "baseline_mi_k",
    "baseline_mi",
    "engineered_mi", "uplift",
]


def _triplet_device_col_specs(eng_columns, raw_cols):
    """Build per-column device leg specs aligned 1:1 with ``eng_columns`` for the triplet family (3 legs),
    recovering ``(col, degree)`` per leg from each ``"{i}*{j}*{k}__{a}_{b}_{c}"`` name. Returns ``None`` if ANY
    column does not resolve to exactly three legs (the device matrix must align 1:1 with the host columns)."""
    from ._orthogonal_univariate_fe._gpu_resident_cross_basis import _parse_code_deg

    specs = []
    raw_set = set(raw_cols)
    for name in eng_columns:
        head = name.split("__", 1)[0] if "__" in name else name
        legs = head.split("*")
        if len(legs) != 3 or any(leg not in raw_set for leg in legs):
            return None
        try:
            suffix = name.split("__", 1)[1]
            parts = suffix.split("_")
        except (ValueError, IndexError):
            return None
        if len(parts) != 3:
            return None
        degs = [_parse_code_deg(p) for p in parts]
        if any(d is None for d in degs):
            return None
        specs.append({"legs": [(legs[i], degs[i]) for i in range(3)]})
    return specs


def score_triplet_cross_basis_by_mi_uplift(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    nbins: int = 10,
    basis: str = "auto",
) -> pd.DataFrame:
    """Score each triplet-cross-basis column by MI uplift vs the BEST of the
    three raw source columns.

    Mirrors ``score_pair_cross_basis_by_mi_uplift`` with a third leg. The
    baseline is ``max(MI(x_i; y), MI(x_j; y), MI(x_k; y))`` -- the triplet
    must beat the BEST individual source, not just the worst, to count as
    genuine 3-way interaction signal (a triplet that only beats the
    weakest leg is more likely picking up the strongest leg's marginal).

    ``basis`` mirrors the ``generate_triplet_cross_basis_features`` call that produced ``engineered_X`` so the
    DEVICE-BORN STRICT-resident scorer re-routes each leg to the SAME basis the host generator used. Unused on
    the host default path.
    """
    from ._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); hoisted so _dt is bound on every branch
    y_arr = _coerce_y_int64(y)
    raw_cols = list(raw_X.columns)
    if engineered_X.empty:
        return pd.DataFrame(columns=_TRIPLET_SCORE_EMPTY_COLS)
    # DEVICE-BORN (STRICT-resident): rebuild the triplet product matrix on the GPU + score both it and the raw
    # baseline through the SAME resident plug-in MI -- collapsing the host product-matrix upload at
    # _orth_mi_backends.py:311. None (-> exact host path) on no-cupy / non-strict / cupy failure / unsupported.
    raw_mi_map = eng_mi = None
    _specs = _triplet_device_col_specs(engineered_X.columns, raw_cols)
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
        # parse "{col_i}*{col_j}*{col_k}__..."
        head = eng_name.split("__", 1)[0] if "__" in eng_name else eng_name
        legs = head.split("*")
        if len(legs) != 3:
            # Not a triplet column -- skip (pair/univariate output mistakenly
            # routed through this scorer would land here).
            continue
        col_i, col_j, col_k = legs
        baseline_i = float(raw_mi_map.get(col_i, 0.0))
        baseline_j = float(raw_mi_map.get(col_j, 0.0))
        baseline_k = float(raw_mi_map.get(col_k, 0.0))
        baseline = max(baseline_i, baseline_j, baseline_k)
        emi = float(eng_mi[j])
        uplift = emi / (baseline + 1e-12)
        rows.append({
            "engineered_col": eng_name,
            "source_col_i": col_i,
            "source_col_j": col_j,
            "source_col_k": col_k,
            "baseline_mi_i": baseline_i,
            "baseline_mi_j": baseline_j,
            "baseline_mi_k": baseline_k,
            "baseline_mi": baseline,
            "engineered_mi": emi,
            "uplift": uplift,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("uplift", ascending=False).reset_index(drop=True)
    return df


def _noise_aware_floor(values: np.ndarray, sigma_thresh: float) -> float:
    """Median + sigma * 1.4826 * MAD noise floor used by Layer 22's pair
    pipeline. Returns 0 when too few values to estimate robustly.
    """
    if values.size < 4:
        return 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med + sigma_thresh * 1.4826 * mad


def hybrid_orth_mi_triplet_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    triplet_max_degree: int = 1,
    basis: str = "auto",
    top_k: int = 5,
    top_triplet_count: int = 2,
    top_triplet_seed_k: int = 4,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    triplet_min_uplift: float = 1.05,
    triplet_min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
    explicit_triplets: Optional[Sequence[tuple]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two-stage hybrid: (1) univariate orthogonal-poly FE + MI-greedy,
    then (2) cross-basis TRIPLET FE on the top-N raw source columns by
    univariate MI, also MI-greedy.

    Stage 1 reuses ``hybrid_orth_mi_fe`` (Layer 21) to pick top-N
    univariate winners. The seed pool for the triplet stage is the
    top-N raw source columns ranked by ``MI(x_c; y)``; we enumerate ALL
    ordered-by-name C(N, 3) triplets over that pool.

    Cost: enumerating triplets over the full input would be O(p^3); the
    seed_k cap turns it into O(seed_k^3) regardless of input width.
    Default ``top_triplet_seed_k=4`` -> 4 triplets * 1 cell (deg 1) = 4
    candidates -- bounded.

    Parameters
    ----------
    X, y, cols, degrees, basis, top_k, min_uplift, min_abs_mi_frac, nbins
        Forwarded to the univariate ``hybrid_orth_mi_fe`` stage.
    triplet_max_degree : int
        Max degree per leg in the triplet stage. Default 1 emits exactly
        one cell per triplet (``He_1*He_1*He_1``) which is the dominant
        3-way signal for nearly every real triplet target.
    top_triplet_count : int
        How many triplet winners to append after the univariate winners.
    top_triplet_seed_k : int
        How many top raw source columns (by MI to y) to pull into the
        triplet seed pool. With N=4 we enumerate C(4,3)=4 triplets.
    triplet_min_uplift, triplet_min_abs_mi_frac : float
        Two-gate thresholds for the triplet stage. Same semantics as
        the univariate gates, compared against the BEST individual leg
        MI as the baseline.

    Returns
    -------
    (X_augmented, univariate_scores, triplet_scores)
        X_augmented : ``X`` with univariate winners THEN triplet winners
            appended, in that order.
        univariate_scores : ranking DataFrame from stage 1.
        triplet_scores : ranking DataFrame from stage 2.
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

    # Build the triplet seed pool. Critical bug class on 3-way XOR:
    # MI(x_i; y) is statistically indistinguishable from MI(noise; y)
    # when y = sign(x_1 * x_2 * x_3) -- every signal leg looks marginally
    # balanced, so a raw-MI top-k cut would silently drop one of the
    # three signal legs and the triplet path never gets to enumerate
    # the right cell.
    #
    # Resolution: when the caller explicitly enumerates ``cols``, respect
    # that as the seed pool intent (truncate only to the first
    # ``top_triplet_seed_k`` of them, preserving caller order). Only fall
    # back to raw-MI ranking when ``cols=None`` AND the input width is
    # larger than ``top_triplet_seed_k`` (the algo has to pick somehow).
    raw_cols_all = [c for c in (cols or X.columns) if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    seed_sources: list[str] = []
    if len(raw_cols_all) >= 3:
        if cols is not None:
            # Caller-specified pool: respect caller order, truncate to the
            # budget. The caller already encoded which columns are
            # candidates; ranking-pruning a hand-picked list would silently
            # drop signal legs on 3-way XOR (every signal leg has
            # marginally zero MI to y because the other two factors
            # randomise balanced). The Layer 22 pair path got away with
            # MI-ranking because 2-way XOR's marginal MI is also zero but
            # the seed pool falls back to the union with univariate
            # uplift; in 3-way XOR even univariate uplift is zero, so
            # ranking must be off when cols are explicit.
            seed_sources = list(raw_cols_all[: int(top_triplet_seed_k)])
        else:
            # cols=None on wide X: rank by MI(x; y) and keep the top-k.
            # This is best-effort; on a 3-way XOR with 100 columns and
            # seed_k=4 the caller should pass cols=[<candidates>].
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
                y_arr = _coerce_y_int64(y)
                from ._fe_usability_signal import _crit_np_dtype
                _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust
                # Fit-scoped memo: no-op passthrough outside an active orth_scoring_memo_scope(); inside a
                # scope, shares this residual MI batch with sibling opt-in layers.
                _baseline_map.update(cached_raw_mi_baseline(_missing, X[_missing].to_numpy(dtype=_dt), y_arr, nbins=nbins))
            raw_mi_arr = np.array([float(_baseline_map.get(c, 0.0)) for c in raw_cols_all])
            order = np.argsort(-raw_mi_arr)
            seed_sources = [raw_cols_all[i] for i in order[: int(top_triplet_seed_k)]]

    # EXPLICIT TRIPLETS (GBM seeder): when the caller passes an order-3
    # proposer's surviving triples (each already order-3-maxT-floored), enumerate EXACTLY
    # those triples instead of the C(seed_k, 3) over the univariate-MI seed pool -- the whole
    # point of the surrogate seeder is to reach the zero-marginal 3-way needle whose operands
    # the univariate seed_k never ranks. The proposer GENERATES; the per-triplet uplift / abs-MI
    # gates below still GATE, so a spurious explicit triple cannot survive. Only triples whose
    # legs all map to valid numeric columns in X are kept.
    if explicit_triplets:
        _xset = set(X.columns)
        triplets = []
        _seen = set()
        for _tr in explicit_triplets:
            if len(_tr) != 3:
                continue
            if not all((c in _xset and pd.api.types.is_numeric_dtype(X[c])) for c in _tr):
                continue
            _key = tuple(sorted(_tr))
            if _key in _seen:
                continue
            _seen.add(_key)
            triplets.append(tuple(_tr))
        if not triplets:
            return X_aug_uni, uni_scores, pd.DataFrame(columns=_TRIPLET_SCORE_EMPTY_COLS)
        # The MI-uplift baseline / seed-X frame need the union of explicit-triple legs.
        seed_sources = sorted({c for _tr in triplets for c in _tr})
    else:
        if len(seed_sources) < 3 or int(top_triplet_count) <= 0:
            return X_aug_uni, uni_scores, pd.DataFrame(columns=_TRIPLET_SCORE_EMPTY_COLS)

        triplets = [
            (seed_sources[i], seed_sources[j], seed_sources[k])
            for i in range(len(seed_sources))
            for j in range(i + 1, len(seed_sources))
            for k in range(j + 1, len(seed_sources))
        ]
    triplet_eng = generate_triplet_cross_basis_features(
        X, triplets, max_degree=triplet_max_degree, basis=basis,
    )
    if triplet_eng.empty:
        return X_aug_uni, uni_scores, pd.DataFrame(columns=_TRIPLET_SCORE_EMPTY_COLS)

    raw_X_seed = X[seed_sources]
    triplet_scores = score_triplet_cross_basis_by_mi_uplift(
        raw_X_seed, triplet_eng, y, nbins=nbins, basis=basis,
    )

    # Two-gate selection mirrors Layer 22's pair stage. Triplet candidate
    # counts are larger (O(seed_k^3 * deg^3)) so the Bonferroni-scaled
    # sigma threshold is correspondingly stricter.
    max_raw_baseline = float(triplet_scores["baseline_mi"].max()) if not triplet_scores.empty else 0.0
    if not uni_scores.empty:
        max_raw_baseline = max(max_raw_baseline, float(uni_scores["baseline_mi"].max()))
    max_triplet_engineered = float(triplet_scores["engineered_mi"].max()) if not triplet_scores.empty else 0.0
    legacy_floor = float(triplet_min_abs_mi_frac) * max(
        max_raw_baseline,
        max_triplet_engineered,
    )
    _baselines = triplet_scores["baseline_mi"].to_numpy() if not triplet_scores.empty else np.array([])
    n_cands = int(_baselines.size)
    sigma_thresh = max(
        5.0,
        float(np.sqrt(2.0 * np.log(max(2.0, 2.0 * n_cands))) + 1.5),
    )
    noise_floor = _noise_aware_floor(_baselines, sigma_thresh)
    _eng_mis = triplet_scores["engineered_mi"].to_numpy() if not triplet_scores.empty else np.array([])
    eng_noise_floor = _noise_aware_floor(_eng_mis, sigma_thresh)
    abs_floor = max(legacy_floor, noise_floor, eng_noise_floor)
    qualified = triplet_scores[(triplet_scores["uplift"] >= float(triplet_min_uplift)) & (triplet_scores["engineered_mi"] >= abs_floor)]
    winners = qualified.head(int(top_triplet_count))
    keep_triplet = list(winners["engineered_col"])
    if keep_triplet:
        X_aug = pd.concat([X_aug_uni, triplet_eng[keep_triplet]], axis=1)
    else:
        X_aug = X_aug_uni
    return X_aug, uni_scores, triplet_scores


from ._fe_family_timing import fe_timed


@fe_timed("triplet")
def hybrid_orth_mi_triplet_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    triplet_max_degree: int = 1,
    basis: str = "auto",
    top_k: int = 5,
    top_triplet_count: int = 2,
    top_triplet_seed_k: int = 4,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    triplet_min_uplift: float = 1.05,
    triplet_min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
    explicit_triplets: Optional[Sequence[tuple]] = None,
):
    """Same as :func:`hybrid_orth_mi_triplet_fe` but additionally returns
    a flat list of recipes (univariate + triplet, in append order) so
    that ``MRMR.transform`` can replay each engineered column from X
    alone (no y). ``explicit_triplets`` (GBM seeder) forces the
    triplet stage to enumerate EXACTLY the given column-name triples (the
    order-3-floored proposer survivors) instead of the C(seed_k, 3) over the
    univariate-MI seed pool.
    """
    from .engineered_recipes import (
        build_orth_univariate_recipe,
        build_orth_triplet_cross_recipe,
    )
    X_aug, uni_scores, triplet_scores = hybrid_orth_mi_triplet_fe(
        X, y, cols=cols, degrees=degrees, basis=basis,
        triplet_max_degree=triplet_max_degree,
        top_k=top_k,
        top_triplet_count=top_triplet_count,
        top_triplet_seed_k=top_triplet_seed_k,
        min_uplift=min_uplift, min_abs_mi_frac=min_abs_mi_frac,
        triplet_min_uplift=triplet_min_uplift,
        triplet_min_abs_mi_frac=triplet_min_abs_mi_frac,
        nbins=nbins,
        explicit_triplets=explicit_triplets,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    code_to_basis = {"He": "hermite", "LL": "laguerre", "T": "chebyshev", "L": "legendre"}

    def _parse_code_deg(s: str):
        """Parse a leg-code token like ``He3``/``LL2``/``T1``/``L4`` into ``(basis name, degree)``, checking two-letter codes before single-letter ones so ``LL`` isn't mis-parsed as ``L``; returns ``(None, None)`` when ``s`` doesn't match any known code prefix."""
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
        if len(legs) == 3:
            # triplet cross: "{col_i}*{col_j}*{col_k}__{code_a}{deg_a}_{code_b}{deg_b}_{code_c}{deg_c}"
            try:
                suffix = name.split("__", 1)[1]
                parts = suffix.split("_")
            except (ValueError, IndexError):
                logger.warning(
                    "hybrid_orth_mi_triplet_fe_with_recipes: cannot parse " "suffix in %r; skipping recipe.",
                    name,
                )
                continue
            if len(parts) != 3:
                logger.warning(
                    "hybrid_orth_mi_triplet_fe_with_recipes: expected 3 deg " "parts in %r; skipping recipe.",
                    name,
                )
                continue
            basis_a, deg_a = _parse_code_deg(parts[0])
            basis_b, deg_b = _parse_code_deg(parts[1])
            basis_c, deg_c = _parse_code_deg(parts[2])
            if basis_a is None or basis_b is None or basis_c is None:
                logger.warning(
                    "hybrid_orth_mi_triplet_fe_with_recipes: cannot parse " "code/deg from %r; skipping recipe.",
                    name,
                )
                continue
            # Same auto-routing fixup as Layer 22 pair recipe builder.
            col_i, col_j, col_k = legs
            # Value-construction (recovers the basis preprocess params for recipe replay): always
            # float64, matching the fresh-path's own dtype (see generate_triplet_cross_basis_features).
            x_i = X[col_i].to_numpy(dtype=np.float64)
            x_j = X[col_j].to_numpy(dtype=np.float64)
            x_k = X[col_k].to_numpy(dtype=np.float64)
            if basis == "auto":
                try:
                    basis_a = basis_route_by_moments(x_i)
                    basis_b = basis_route_by_moments(x_j)
                    basis_c = basis_route_by_moments(x_k)
                except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                    logger.debug("suppressed in _orthogonal_triplet_fe.py:613: %s", e)
                    pass
            # REPLAY-FIDELITY FIX (2026-06-13): freeze each leg's fit-time basis-preprocess params
            # (z-score mean/std, min-max lo/hi, ...) so ``transform()`` replays the basis axis
            # byte-exactly instead of refitting it from the apply-time rows (slice-replay corruption).
            # Guarded -> on any failure params stay None (legacy refit path), never crashing the emit.
            _pp_a = _pp_b = _pp_c = None
            try:
                _, _pp_a = _evaluate_basis_column(x_i, basis_a, deg_a, return_params=True)
                _, _pp_b = _evaluate_basis_column(x_j, basis_b, deg_b, return_params=True)
                _, _pp_c = _evaluate_basis_column(x_k, basis_c, deg_c, return_params=True)
            except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed in _orthogonal_triplet_fe.py:624: %s", e)
                pass
            recipes.append(build_orth_triplet_cross_recipe(
                name=name,
                src_a_name=col_i, src_b_name=col_j, src_c_name=col_k,
                basis_i=basis_a, basis_j=basis_b, basis_k=basis_c,
                deg_a=deg_a, deg_b=deg_b, deg_c=deg_c,
                preprocess_params_i=_pp_a, preprocess_params_j=_pp_b, preprocess_params_k=_pp_c,
            ))
        elif len(legs) == 1:
            # univariate: "{col}__{code}{degree}"
            src = legs[0]
            suffix = name.split("__", 1)[1] if "__" in name else ""
            chosen_basis, chosen_degree = _parse_code_deg(suffix)
            if chosen_basis is None or chosen_degree is None:
                logger.warning(
                    "hybrid_orth_mi_triplet_fe_with_recipes: cannot parse basis/" "degree from %r; skipping recipe.",
                    name,
                )
                continue
            # REPLAY-FIDELITY FIX (2026-06-13): freeze the fit-time basis-preprocess params (mirrors
            # the orth_univariate BUG2 fix); without them replay refits the axis from apply-time rows.
            _pp_u = None
            try:
                _x_u = X[src].to_numpy(dtype=np.float64)  # value-construction: always float64, see above
                _, _pp_u = _evaluate_basis_column(_x_u, chosen_basis, chosen_degree, return_params=True)
            except Exception as e:  # nosec B110 - optional dependency import guard
                logger.debug("Could not freeze fit-time basis-preprocess params for %r (%s: %s); recipe replays without them", src, type(e).__name__, e)
            recipes.append(build_orth_univariate_recipe(
                name=name, src_name=src,
                basis=chosen_basis, degree=chosen_degree,
                preprocess_params=_pp_u,
            ))
        else:
            logger.warning(
                "hybrid_orth_mi_triplet_fe_with_recipes: unexpected leg count "
                "in %r (legs=%r); skipping recipe.", name, legs,
            )
            continue
    return X_aug, uni_scores, triplet_scores, recipes
