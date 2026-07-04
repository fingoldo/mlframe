"""Pair cross-basis orthogonal FE: ``basis(x_i)_a * basis(x_j)_b`` interaction terms.

Generates and MI-greedy-ranks degree-(a,b) cross products of two source
columns' orthogonal-basis transforms, then the two-stage
``hybrid_orth_mi_pair_fe`` (univariate winners + cross-basis pair winners)
plus its recipe-emitting variant. The shared univariate scaffolding
(`_evaluate_basis_column`, `generate_univariate_basis_features`,
`hybrid_orth_mi_fe`, `_BASIS_CODE`) lives in the parent module
``_orthogonal_univariate_fe`` and is lazy-imported in-body to avoid a cycle.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..hermite_fe import _POLY_BASES, basis_route_by_moments
from ._orth_mi_backends import _mi_classif_batch, mi_classif_batch_chunked

logger = logging.getLogger(__name__)

__all__ = [
    "generate_pair_cross_basis_features",
    "score_pair_cross_basis_by_mi_uplift",
    "hybrid_orth_mi_pair_fe",
    "hybrid_orth_mi_pair_fe_with_recipes",
]


def _pair_sources_from_engineered_name(name: str, raw_names):
    """Recover ``(col_i, col_j)`` from a pair-cross name ``"{col_i}*{col_j}__{suffix}"``.

    D1 (2026-06-22): the legacy ``name.split("__", 1)[0]`` then ``head.split("*", 1)``
    MISPARSES whenever a one-hot raw source contains ``"__"`` (e.g.
    ``"city__NY*age__He2_He3"`` -> head ``"city"`` -> not a pair). Recover the legs by
    matching against the known raw-column set: split on each ``"*"``, take the FIRST split
    where both halves (the second half un-stemmed of its ``"__{suffix}"``) are raw columns.

    Returns ``(col_i, col_j)`` or ``(None, None)`` when no pair structure resolves.
    """
    raw_set = set(raw_names)
    # The right leg is ``"{col_j}__{suffix}"``; recover col_j by longest raw prefix.
    star_positions = [i for i, ch in enumerate(name) if ch == "*"]
    for sp in star_positions:
        left = name[:sp]
        right = name[sp + 1:]
        if left not in raw_set:
            continue
        # right = "{col_j}__{suffix}"; longest raw prefix is col_j.
        best = None
        for raw in raw_names:
            if right == raw or right.startswith(raw + "__"):
                if best is None or len(raw) > len(best):
                    best = raw
        if best is not None:
            return left, best
    # Legacy fallback (no raw set match): first-"__" then first-"*".
    head = name.split("__", 1)[0] if "__" in name else name
    if "*" in head:
        ci, cj = head.split("*", 1)
        return ci, cj
    return None, None


def _pair_eng_col_name(col_i: str, col_j: str, basis: str, deg_a: int, deg_b: int) -> str:
    """Stable naming: ``"{col_i}*{col_j}__He{a}_He{b}"``.

    Both legs share the same basis code (e.g. He_a * He_b). The cross-basis
    enumeration intentionally fixes one basis family per pair -- mixing
    families (He_a * T_b) blows up combinatorially without measurable signal
    gain on the standard XOR / saddle / circle targets.
    """
    from . import _BASIS_CODE

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
    from .._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); hoisted so _dt is bound on every branch
    from . import _evaluate_basis_column

    if not pairs:
        return pd.DataFrame(index=X.index)
    cache: dict[tuple[str, int, str], np.ndarray] = {}
    out_cols: dict = {}
    max_d = int(max_degree)
    min_d = max(0, int(min_degree))
    from .._fe_deadline import fe_deadline_passed
    for col_i, col_j in pairs:
        # Optional-enrichment wall-clock budget: stop materialising pair-cross basis candidates once MRMR.fit's deadline
        # passes and return the partial set (downstream scoring then runs on fewer candidates). No-op without a budget.
        if fe_deadline_passed():
            break
        if col_i == col_j:
            continue
        if col_i not in X.columns or col_j not in X.columns:
            logger.warning("generate_pair_cross_basis_features: missing column %r or %r; skipping", col_i, col_j)
            continue
        if not (pd.api.types.is_numeric_dtype(X[col_i]) and pd.api.types.is_numeric_dtype(X[col_j])):
            continue
        # np.array (copy=True): X[col].to_numpy() can alias the DataFrame's backing block for a
        # contiguous float64 column, and the np.copyto NaN-fill below would then mutate the CALLER's X
        # (corrupting downstream missingness-FE). A fresh copy keeps the fill local to this function.
        from .._fe_usability_signal import _crit_np_dtype
        _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust
        x_i = np.array(X[col_i].to_numpy(), dtype=_dt)
        x_j = np.array(X[col_j].to_numpy(), dtype=_dt)
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


def _crossbasis_device_born_on() -> bool:
    """Device-born cross-basis scoring engages ONLY under STRICT-residency (+ the per-family opt-out). Read live
    (no frozen cache) so it tracks the env per call. Any import failure -> host path (never a correctness loss)."""
    try:
        from .._gpu_strict_fe import fe_gpu_device_born_crossbasis_enabled
        return bool(fe_gpu_device_born_crossbasis_enabled())
    except Exception:
        return False


def _pair_device_col_specs(eng_columns, raw_cols, *, nbins: int):
    """Build per-column device leg specs aligned 1:1 with ``eng_columns`` for the pair-cross family, recovering
    ``(col_i, deg_a)`` / ``(col_j, deg_b)`` from each engineered name via the SAME name-parsing the recipe
    builder uses. Returns ``None`` if ANY column does not resolve to exactly two legs (the device matrix must
    align 1:1 with the host columns; a single unresolved column -> host fallback for the whole batch)."""
    from ._gpu_resident_cross_basis import _parse_code_deg

    specs = []
    for name in eng_columns:
        col_i, col_j = _pair_sources_from_engineered_name(name, raw_cols)
        if col_i is None or col_j is None:
            return None
        _head = col_i + "*" + col_j
        if name.startswith(_head + "__"):
            suffix = name[len(_head) + 2:]
        elif "__" in name:
            suffix = name.split("__", 1)[1]
        else:
            return None
        try:
            left, right = suffix.split("_", 1)
        except ValueError:
            return None
        deg_a = _parse_code_deg(left)
        deg_b = _parse_code_deg(right)
        if deg_a is None or deg_b is None:
            return None
        specs.append({"legs": [(col_i, deg_a), (col_j, deg_b)]})
    return specs


def _resident_pair_cross_mi(raw_X, engineered_X, y_arr, col_specs, *, nbins: int, basis: str = "auto"):
    """Thin wrapper over the shared device-born cross-basis MI twin for the pair family."""
    from ._gpu_resident_cross_basis import raw_and_product_mi_resident

    return raw_and_product_mi_resident(raw_X, engineered_X, y_arr, col_specs, nbins=nbins, basis=basis)


def score_pair_cross_basis_by_mi_uplift(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    nbins: int = 10,
    basis: str = "auto",
) -> pd.DataFrame:
    """Score each pair-cross-basis column by MI uplift vs the BETTER of the
    two raw source columns. Mirrors ``score_features_by_mi_uplift`` but the
    name carries a pair prefix ``"{col_i}*{col_j}__..."``.

    ``basis`` mirrors the ``generate_pair_cross_basis_features`` call that produced ``engineered_X`` so the
    DEVICE-BORN STRICT-resident scorer re-routes each leg to the SAME basis the host generator used (it routes
    legs ITSELF rather than reading the name's leg-1 code). It is unused on the host default path. The MRMR
    pipeline + the hybrid entry points thread the generation basis through; a direct caller that built
    ``engineered_X`` under an explicit basis must pass the same value when the device path is engaged.

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
    if engineered_X.empty:
        return pd.DataFrame(columns=[
            "engineered_col", "source_col_i", "source_col_j",
            "baseline_mi_i", "baseline_mi_j", "baseline_mi",
            "engineered_mi", "uplift",
        ])
    # DEVICE-BORN (STRICT-resident): rebuild the pair-cross product matrix on the GPU from the small raw operand
    # columns and score BOTH it and the raw baseline through the SAME resident plug-in MI -- collapsing the host
    # product-matrix upload at _orth_mi_backends.py:311. Returns None (-> exact host path below) on no-cupy /
    # non-strict / any cupy failure / unsupported basis. Selection-equivalent (device Clenshaw vs host forward
    # recurrence ~1e-12, same estimator for numerator + baseline so the uplift ratio cannot flip).
    raw_mi_map = eng_mi = None
    _specs = _pair_device_col_specs(engineered_X.columns, raw_cols, nbins=nbins)
    if _specs is not None and _crossbasis_device_born_on():
        _res = _resident_pair_cross_mi(raw_X, engineered_X, y_arr, _specs, nbins=nbins, basis=basis)
        if _res is not None:
            raw_mi_map, eng_mi = _res
    if eng_mi is None:
        from .._fe_usability_signal import _crit_np_dtype
        raw_mi = _mi_classif_batch(raw_X.to_numpy(dtype=_crit_np_dtype()), y_arr, nbins=nbins)
        raw_mi_map = dict(zip(raw_cols, raw_mi.tolist()))
        # Column-chunked MI scoring -> bit-identical, bounds peak RAM on the wide pair-cross-basis matrix.
        eng_mi = mi_classif_batch_chunked(engineered_X, y_arr, nbins=nbins)
    rows = []
    for j, eng_name in enumerate(engineered_X.columns):
        # D1 (2026-06-22): recover legs against the raw-column set, not a blind first-"__"
        # split + first-"*" (which mis-parses one-hot sources like "city__NY*age__He2_He3").
        col_i, col_j = _pair_sources_from_engineered_name(eng_name, raw_cols)
        if col_i is None or col_j is None:
            # not a pair column -- skip
            continue
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
    from . import hybrid_orth_mi_fe

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
        from .._fe_usability_signal import _crit_np_dtype
        raw_mi_arr = _mi_classif_batch(raw_X_all.to_numpy(dtype=_crit_np_dtype()), y_arr, nbins=nbins)
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
        raw_X_seed, pair_eng, y, nbins=nbins, basis=basis,
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


from .._fe_family_timing import fe_timed


@fe_timed("orth_pair")
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
    from ..engineered_recipes import (
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
    # D1 (2026-06-22): authoritative raw-source set for un-stemming engineered names.
    _raw_src_cols = [c for c in X.columns]
    recipes = []
    from .._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); recipe-replay axis matches the fit-time f32 operands
    for name in appended:
        _pair_ci, _pair_cj = _pair_sources_from_engineered_name(name, _raw_src_cols)
        if _pair_ci is not None and _pair_cj is not None:
            # pair cross: "{col_i}*{col_j}__{code}{deg_a}_{code}{deg_b}"
            col_i, col_j = _pair_ci, _pair_cj
            # suffix = everything after "{col_i}*{col_j}__" (NOT first "__", which mis-splits
            # one-hot sources like "city__NY*age__He2_He3").
            _head = col_i + "*" + col_j
            suffix = name[len(_head) + 2:] if name.startswith(_head + "__") else name.split("__", 1)[1]
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
                    x_i = X[col_i].to_numpy(dtype=_dt)
                    x_j = X[col_j].to_numpy(dtype=_dt)
                    basis_a = basis_route_by_moments(x_i)
                    basis_b = basis_route_by_moments(x_j)
                except Exception:
                    pass
            # BUG2 FIX (2026-06-12): freeze each operand's fit-time preprocess
            # params from the FULL fit column so replay is byte-exact on a slice.
            from . import _evaluate_basis_column as _ebc
            _ppi = _ppj = None
            try:
                _xi = X[col_i].to_numpy(dtype=_dt)
                _, _ppi = _ebc(_xi, basis_a, int(deg_a), return_params=True)
            except Exception:
                _ppi = None
            try:
                _xj = X[col_j].to_numpy(dtype=_dt)
                _, _ppj = _ebc(_xj, basis_b, int(deg_b), return_params=True)
            except Exception:
                _ppj = None
            recipes.append(build_orth_pair_cross_recipe(
                name=name, src_a_name=col_i, src_b_name=col_j,
                basis_i=basis_a, basis_j=basis_b,
                deg_a=deg_a, deg_b=deg_b,
                preprocess_params_i=_ppi, preprocess_params_j=_ppj,
            ))
        else:
            # univariate: "{col}__{code}{degree}"
            from . import _source_from_engineered_name as _src_of
            src = _src_of(name, _raw_src_cols)
            suffix = name[len(src) + 2:] if name.startswith(src + "__") else name.split("__", 1)[1]
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
            # BUG2 FIX (2026-06-12): freeze the fit-time preprocess params.
            from . import _evaluate_basis_column as _ebc
            _pp = None
            try:
                _xc = X[src].to_numpy(dtype=_dt)
                _, _pp = _ebc(_xc, chosen_basis, int(chosen_degree), return_params=True)
            except Exception:
                _pp = None
            recipes.append(build_orth_univariate_recipe(
                name=name, src_name=src,
                basis=chosen_basis, degree=chosen_degree,
                preprocess_params=_pp,
            ))
    return X_aug, uni_scores, cross_scores, recipes
