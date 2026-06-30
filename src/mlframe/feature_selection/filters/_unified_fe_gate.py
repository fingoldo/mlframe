"""Two-tier information-theoretic gates for the recipe-emitting FE mechanisms
(Layer 91, 2026-06-01).

Four FE mechanisms emit engineered columns from a per-source / per-pair /
per-group enumeration with NO relevance gate of their own:

* Layer 33 K-fold target encoding (``_target_encoding_fe``)
* Layer 34 count / frequency / cat-num residual (``_count_freq_interaction_fe``)
* Layer 37 missingness indicator / count / pattern (``_missingness_fe``)
* Layer 38 ratio / grouped-delta / lagged-diff (``_ratio_delta_fe``)

Their pools grow combinatorially: 50 categorical columns -> 50 count-encoded
columns, p numeric columns -> p*(p-1) ratio columns, and so on. Every emitted
column then flows into MRMR's relevance / redundancy screen, inflating its
work and (on small n) admitting noise survivors. This module adds two
independent gates the wrappers / MRMR.fit opt into.

Tier 1 -- LOCAL MI FLOOR (cheap, per-mechanism)
-----------------------------------------------
``local_mi_gate`` scores every freshly-emitted engineered column by plug-in
``MI(candidate; y)`` (reusing ``_orthogonal_univariate_fe._mi_classif_batch``)
and drops any whose MI falls below a noise floor. The floor is anchored on the
RAW baseline MI distribution (median + ``mad_mult`` * MAD of the raw columns'
MI), NOT on the engineered columns -- the Layer 90 lesson: an engineered-pool-
relative floor moves with the pool and lets a pool of uniformly-weak columns
set its own low bar. After flooring, the top-``top_k`` survivors by MI are
kept (bounding the pool even when many columns clear the floor).

Tier 2 -- UNIFIED SECOND-PASS CMI GATE (cross-mechanism)
--------------------------------------------------------
``unified_second_pass_gate`` runs a greedy conditional-MI selection over ALL
engineered columns (regardless of which mechanism emitted them), conditioning
on a running support that starts from the raw signal. An engineered column is
dropped when ``CMI(col; y | already_selected) < threshold`` -- i.e. it adds no
new information beyond what raw columns + earlier-selected engineered columns
already carry. This catches CROSS-mechanism redundancy that no single
mechanism's local gate can see: ``count(cat_a)`` and ``freq(cat_a)`` are an
affine transform of each other (identical bin pattern), so once one is seated
the other's CMI collapses to ~0 and it is dropped.

Both gates are pure functions of (engineered X, y, raw column names). They
return the SUBSET of engineered column names to KEEP; the caller drops the
rest. They never reference y at transform time (they run only at fit).
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "raw_mi_noise_floor",
    "local_mi_gate",
    "unified_second_pass_gate",
]


# ---------------------------------------------------------------------------
# Tier 1: local MI floor
# ---------------------------------------------------------------------------


def _coerce_y_classes(y) -> np.ndarray:
    """Promote y to a dense int64 class array (binary / multiclass). Continuous
    y is quantile-binned into 10 bins so the plug-in classification MI estimator
    still applies (mirrors the Layer 60 ``score_candidates_by_cmi`` contract)."""
    y_arr = np.asarray(y)
    if np.issubdtype(y_arr.dtype, np.integer) or y_arr.dtype == bool:
        _, y_bin = np.unique(y_arr.astype(np.int64), return_inverse=True)
        return y_bin.astype(np.int64)
    if np.issubdtype(y_arr.dtype, np.floating):
        finite = y_arr[np.isfinite(y_arr)]
        n_unique = int(np.unique(finite).size) if finite.size else 0
        if n_unique <= 20:
            _, y_bin = np.unique(y_arr, return_inverse=True)
            return y_bin.astype(np.int64)
        # Continuous: 10-bin quantile discretisation.
        from ._mi_greedy_cmi_fe import _quantile_bin
        return _quantile_bin(y_arr, nbins=10)
    # bench-attempt-rejected (2026-06-11, wave-2 W8 "extreme-imbalance / tiny-n target-binning
    # guard"): hypothesis was that equi-frequency TARGET binning silently yields a single-class
    # degenerate bin under rare<<1% / n<200 y, corrupting MI -> unstable rare-feature ranking, and
    # that a merge-degenerate-bins + coarser-nbins fallback would stabilise it. FALSIFIED:
    #   * Classification rare-1% (n=150, ~1.5 positives): y is integer -> the factorize branch above,
    #     never _quantile_bin, so a degenerate target bin CANNOT form on the classification path.
    #   * Regression continuous y: _quantile_bin already drops duplicate quantile edges via
    #     np.unique(edges) (see _mi_greedy_cmi_fe._quantile_bin docstring + ``edges.size <= 2``
    #     branch) so collapse to <nbins bins is EXPLICIT and documented, not silent; tied-value bins
    #     produce finite, sane, stable plug-in MI (verified). Existing collapse guards already live at
    #     _adaptive_nbins.py:534 / _discretization_edges.py:259 / info_theory/_entropy_kernels.py:67.
    #   * The proposed cap-to-nbins / merge fix did NOT stabilise the rare-feature rank across 12
    #     seeds at n=150/1% (mean rank 5.25->5.33, std 1.96->2.36 -- no gain, marginally worse). The
    #     rank instability is intrinsic small-n statistics (rare signal vs noise-MI chance at ~1.5
    #     positives), NOT a binning artefact; it is the cross-project "rare imbalance needs large-n"
    #     fact and no target-binning trick rescues it. Distinct from (and confirms) rejected idea #18
    #     (imbalance-MI rescore). Large-n balanced is byte-identical to the cap variant either way.
    # Do not re-attempt a degenerate-target-bin merge / coarser-nbins guard here.
    _, y_bin = np.unique(y_arr, return_inverse=True)
    return y_bin.astype(np.int64)


# bench-attempt-rejected (2026-06-11, backlog #2 / frontier-idea-3 "leave-candidate-out
# noise floor"): the hypothesis was that ``median(pool) + 3.5*1.4826*MAD(pool)`` self-gates
# a LONE strong signal (signal pooled into med/MAD lifts the floor above itself), fixable by
# computing med/MAD leave-candidate-out or upper-trimmed. FALSIFIED by 300k-draw Monte Carlo:
#   * median/MAD is already robust to a single outlier, so the self-gating bug fires in only
#     ~10% of signal pools and there it is driven by WIDE NOISE spread, not signal self-pooling.
#   * upper-trim (cap 10%) fixed 50.2% of bug cases; true per-candidate LOO only 39.6%.
#   * BOTH catastrophically REGRESS the all-noise case: removing the top / held-out value lowers
#     the median while barely moving MAD, so the floor drops below the remaining noise band ->
#     all-noise leak rose classic 13575 -> trim 36741 / LOO 29619 (~2-3x more noise admitted).
# Net: attacks the wrong term and admits noise. Keep the pooled median+MAD floor as-is. Do not
# re-attempt LOO/trim on the noise floor; a real fix for the rare wide-noise miss would target
# the SIGMA threshold or an absolute-MI anchor, not the pooling of the candidate.


def raw_mi_noise_floor(
    raw_X: pd.DataFrame,
    y,
    *,
    nbins: int = 10,
    mad_mult: float = 3.5,
) -> float:
    """Noise floor = ``median(raw_mi) + mad_mult * MAD(raw_mi)`` over the
    numeric raw columns' marginal ``MI(col; y)``.

    Anchoring on the RAW distribution (not the engineered pool) is the Layer 90
    lesson: an engineered-pool floor drifts with the pool, so a pool of
    uniformly weak engineered columns sets its own low bar and lets noise
    through. The raw distribution is a stable reference for "what an
    uninformative-to-weak column's MI looks like on this y at this n".

    Returns 0.0 when there are no usable numeric raw columns (degenerate: the
    caller then keeps everything that clears 0, i.e. any column with MI > 0).
    """
    from ._orthogonal_univariate_fe import _mi_classif_batch

    if not isinstance(raw_X, pd.DataFrame) or raw_X.shape[1] == 0:
        return 0.0
    num_cols = [c for c in raw_X.columns if pd.api.types.is_numeric_dtype(raw_X[c])]
    if not num_cols:
        return 0.0
    y_bin = _coerce_y_classes(y)
    arr = raw_X[num_cols].to_numpy(dtype=np.float64)
    # Class-B :311 collapse (2026-06-30): under STRICT-residency ``_mi_classif_batch(arr)`` already routes
    # through the resident plug-in but re-uploads this FIT-CONSTANT raw matrix fresh at _orth_mi_backends:311.
    # The matrix is the raw numeric columns verbatim (a pure baseline, re-scored across the fit), so route it
    # through the resident-operand cache -> uploaded ONCE. Same percentile-edge resident estimator the host
    # STRICT path uses -> byte-identical per-column MI -> byte-identical median+MAD floor. None on any cupy
    # failure / non-strict -> the EXACT host scorer below (byte-identical default path untouched).
    from ._resident_raw_mi import resident_raw_baseline_mi

    raw_mi = resident_raw_baseline_mi(arr, y_bin, ("raw_noise_floor", tuple(num_cols)), nbins=nbins)
    if raw_mi is None:
        raw_mi = _mi_classif_batch(arr, y_bin, nbins=nbins)
    raw_mi = np.asarray(raw_mi, dtype=np.float64)
    raw_mi = raw_mi[np.isfinite(raw_mi)]
    if raw_mi.size == 0:
        return 0.0
    med = float(np.median(raw_mi))
    mad = float(np.median(np.abs(raw_mi - med)))
    # MAD scaled to a std-equivalent (1.4826) so ``mad_mult`` reads like a
    # sigma multiplier on a roughly-normal raw-MI distribution.
    return med + float(mad_mult) * 1.4826 * mad


def local_mi_gate(
    enc_df: pd.DataFrame,
    y,
    raw_X: Optional[pd.DataFrame] = None,
    *,
    top_k: Optional[int] = None,
    nbins: int = 10,
    mad_mult: float = 3.5,
    floor: Optional[float] = None,
    reject_sink: Optional[Callable[..., None]] = None,
) -> list[str]:
    """Tier-1 local MI floor. Returns the subset of ``enc_df`` columns to keep.

    A column is kept when ``MI(col; y) >= floor`` where ``floor`` is the
    ``raw_mi_noise_floor`` of ``raw_X`` (or the explicit ``floor`` argument).
    Survivors are then ranked by MI and the top-``top_k`` retained (no cap when
    ``top_k`` is None / <= 0). Column order in the returned list follows
    descending MI.

    Non-numeric engineered columns (should not occur -- every emitter produces
    numeric output) are dropped defensively.
    """
    from ._orthogonal_univariate_fe import _mi_classif_batch

    if not isinstance(enc_df, pd.DataFrame) or enc_df.shape[1] == 0:
        return []
    cand_cols = [c for c in enc_df.columns if pd.api.types.is_numeric_dtype(enc_df[c])]
    if not cand_cols:
        return []
    if floor is None:
        floor = raw_mi_noise_floor(raw_X, y, nbins=nbins, mad_mult=mad_mult) if raw_X is not None else 0.0
    y_bin = _coerce_y_classes(y)
    arr = enc_df[cand_cols].to_numpy(dtype=np.float64)
    # NOTE (device-born gate, 2026-06-29): unlike the CONDITIONAL gate (whose tau-grid candidates are derived
    # from a few RESIDENT operand columns and so can be built device-born, collapsing the host matrix upload),
    # ``local_mi_gate``'s candidates ARE the engineered ``enc_df`` block -- arbitrary per-call columns with no
    # cacheable operand basis. Its single ``cp.asarray`` upload under the STRICT resident ``_mi_classif_batch``
    # is therefore irreducible (the matrix must reach the device once). Routing it through a separate resident
    # wrapper only RELOCATES the same one-shot upload without collapsing it, so this path stays on the exact
    # host ``_mi_classif_batch`` (byte-identical; the STRICT branch inside already uploads once + caches y).
    cand_mi = np.asarray(_mi_classif_batch(arr, y_bin, nbins=nbins), dtype=np.float64)
    scored = [
        (col, float(cand_mi[j]))
        for j, col in enumerate(cand_cols)
        if np.isfinite(cand_mi[j]) and cand_mi[j] >= floor
    ]
    # W6 (class-closure): record every candidate the shared abs-MAD noise floor
    # kills (raw_mi_noise_floor = med+k*MAD). Pure-record; the kept set is
    # computed independently above so selection is byte-identical.
    if reject_sink is not None:
        for j, col in enumerate(cand_cols):
            _mi = cand_mi[j]
            if np.isfinite(_mi) and _mi < floor:
                try:
                    reject_sink(
                        gate="marginal_uplift_floor",
                        candidate=str(col),
                        operands=None,
                        operator="unified_local_mi_gate",
                        observed=float(_mi),
                        threshold=float(floor),
                        reason="unified local-MI abs-MAD floor: MI below med+k*MAD raw noise floor",
                    )
                except Exception:
                    pass
    scored.sort(key=lambda t: t[1], reverse=True)
    if top_k is not None and int(top_k) > 0:
        scored = scored[: int(top_k)]
    return [col for col, _mi in scored]


# ---------------------------------------------------------------------------
# Tier 2: unified second-pass CMI gate (cross-mechanism)
# ---------------------------------------------------------------------------


def unified_second_pass_gate(
    X_with_all_engineered: pd.DataFrame,
    y,
    raw_cols: Sequence[str],
    engineered_cols: Sequence[str],
    *,
    max_keep: Optional[int] = None,
    min_cmi_gain: float = 0.005,
    nbins: int = 10,
    seed_raw_cols_count: int = 4,
) -> list[str]:
    """Tier-2 unified second-pass CMI gate over ALL engineered columns.

    Greedily seats engineered columns by ``CMI(col; y | running_support)``.
    The running support is seeded with the top-``seed_raw_cols_count`` RAW
    numeric columns by marginal MI (the "raw signal" the engineered pool must
    beat), then grows by each seated engineered winner. A column is dropped
    when its best CMI over the current support is ``< min_cmi_gain`` -- it adds
    no new information beyond raw + already-seated engineered columns.

    This catches CROSS-mechanism redundancy: ``count(cat_a)`` and ``freq(cat_a)``
    have identical equi-frequency bin patterns, so once one is seated the
    other's CMI collapses to ~0 and it is dropped, even though each mechanism's
    own local gate would keep both.

    Bin-pattern dedup: a candidate whose quantile-bin fingerprint matches an
    already-seated winner is skipped outright (its CMI is identically zero given
    the seated column, but the explicit skip avoids float ties admitting it).

    Parameters
    ----------
    X_with_all_engineered : DataFrame
        Frame containing both the raw and the engineered columns.
    y : array-like
    raw_cols : sequence of str
        Raw (non-engineered) column names; the conditioning seed is drawn from
        their numeric subset.
    engineered_cols : sequence of str
        Engineered column names to gate. Only these are eligible to be kept.
    max_keep : int or None
        Hard cap on the number of engineered columns kept. None = no cap
        (greedy still stops when no candidate clears ``min_cmi_gain``).
    min_cmi_gain : float
        Minimum CMI a candidate must add over the current support to be seated.
    nbins : int
        Equi-frequency bins per column.
    seed_raw_cols_count : int
        Number of top-marginal-MI raw numeric columns folded into the initial
        conditioning support. 0 = start with empty support (pure marginal MI
        ranking at the first step).

    Returns
    -------
    list[str]
        Engineered column names to KEEP, in selection order.
    """
    from ._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin, _renumber_joint
    from ._orthogonal_univariate_fe import _mi_classif_batch

    if not isinstance(X_with_all_engineered, pd.DataFrame):
        raise TypeError(
            "unified_second_pass_gate: X must be a pandas DataFrame; got "
            f"{type(X_with_all_engineered).__name__}"
        )
    eng = [
        c for c in engineered_cols
        if c in X_with_all_engineered.columns
        and pd.api.types.is_numeric_dtype(X_with_all_engineered[c])
    ]
    if not eng:
        return []
    y_bin = _coerce_y_classes(y)

    # Seed the conditioning support with the top-N raw numeric columns by
    # marginal MI -- the "raw signal" the engineered pool must add to.
    raw_num = [
        c for c in raw_cols
        if c in X_with_all_engineered.columns
        and pd.api.types.is_numeric_dtype(X_with_all_engineered[c])
    ]
    z_joint: Optional[np.ndarray] = None
    if raw_num and int(seed_raw_cols_count) > 0:
        raw_arr = X_with_all_engineered[raw_num].to_numpy(dtype=np.float64)
        raw_mi = np.asarray(_mi_classif_batch(raw_arr, y_bin, nbins=nbins), dtype=np.float64)
        order = np.argsort(-raw_mi)
        seed_cols = [raw_num[i] for i in order[: int(seed_raw_cols_count)]]
        seed_bins = [
            _quantile_bin(X_with_all_engineered[c].to_numpy(), nbins=nbins)
            for c in seed_cols
        ]
        if seed_bins:
            z_joint, _ = _renumber_joint(*seed_bins)

    # Pre-bin every engineered candidate + fingerprint for monotone-equivalence
    # dedup against seated winners.
    cand_bins: dict[str, np.ndarray] = {
        c: _quantile_bin(X_with_all_engineered[c].to_numpy(), nbins=nbins)
        for c in eng
    }
    cand_fp: dict[str, bytes] = {c: cand_bins[c].tobytes() for c in eng}

    n_samples = int(y_bin.size)
    frag_cap = max(2, n_samples // 5)
    winners: list[str] = []
    winner_fps: set[bytes] = set()
    remaining = set(eng)
    cap = int(max_keep) if (max_keep is not None and int(max_keep) > 0) else len(eng)

    while remaining and len(winners) < cap:
        best_name = None
        best_cmi = -1.0
        for name in remaining:
            if cand_fp[name] in winner_fps:
                continue
            cmi = _cmi_from_binned(cand_bins[name], y_bin, z_joint)
            if cmi > best_cmi:
                best_cmi = cmi
                best_name = name
        if best_name is None or best_cmi < float(min_cmi_gain):
            break
        winners.append(best_name)
        winner_fps.add(cand_fp[best_name])
        remaining.discard(best_name)
        # Fold the winner into Z (under the fragmentation cap) so the next CMI
        # measures the gain ON TOP OF this column -- this is what drops the
        # cross-mechanism redundant siblings.
        new_bin = cand_bins[best_name]
        if z_joint is None or z_joint.size == 0:
            z_joint = new_bin.copy()
        else:
            candidate_joint, _ = _renumber_joint(z_joint, new_bin)
            if int(np.unique(candidate_joint).size) <= frag_cap:
                z_joint = candidate_joint
            # else: freeze Z so later candidates stay measurable.

    return winners
