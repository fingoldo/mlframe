"""Layer 73 (2026-06-01): Total Correlation (Watanabe 1960) multivariate-
redundancy ranking for hybrid orth-poly FE.

Why this layer
--------------

Layers 21 / 65 / 66 / 67 / 71 rank each engineered column by its MARGINAL
dependence with the target; Layer 72 (JMIM) ranks by the WORST-CASE pairwise
joint MI with the already-selected support. Both criteria are still
PAIRWISE in their redundancy book-keeping: JMIM evaluates a separate joint
MI against each selected column and takes the min. Higher-order
redundancy -- the case where three or more columns are PAIRWISE
near-independent but JOINTLY redundant (the canonical example is the XOR
parity ``x_3 = x_1 XOR x_2``) -- is invisible to a pairwise scorer.

Total Correlation closes that hole. For a column set ``Z = (Z_1, ..., Z_d)``
TC is

    TC(Z) = sum_i H(Z_i) - H(Z_1, ..., Z_d)

(Watanabe, S. (1960). "Information theoretical analysis of multivariate
correlation." *IBM Journal of Research and Development* 4(1):66-82). TC
generalises mutual information from 2 variables to arbitrarily many: it is
the total amount of information SHARED among the variables in ``Z``, and
collapses to ``I(X; Y)`` when ``d = 2``. Critically, TC of an XOR triple
``(x_1, x_2, x_3=x_1 XOR x_2)`` is POSITIVE because the joint entropy
``H(Z_1, Z_2, Z_3)`` is one bit less than ``sum H(Z_i)`` even though
``I(Z_i; Z_j) = 0`` for every pair -- the higher-order redundancy that
pairwise MI misses entirely.

Score for FE ranking
--------------------

For each engineered candidate ``c`` we compute the TC UPLIFT when ``c`` is
added to the current support union ``y``:

    score(c) = TC([support, c, y]) - TC([support, y])

Higher = ``c`` adds GENUINE new information (the joint entropy grows more
than the per-column entropy sum implies redundancy). Lower / negative =
``c`` is jointly redundant with the support given ``y``. A column that is
marginally informative with ``y`` but JOINTLY redundant with the support
posts a low / negative delta_tc -- the property pairwise MI cannot see.

TC vs JMIM (Layer 72)
---------------------

* JMIM ``min_j I((X_k, X_j); Y)`` is a TWO-variable joint MI (candidate
  paired with ONE support member at a time). It catches pairwise
  redundancy but not higher-order: on XOR ``(x_1, x_2, x_3=x_1 XOR x_2)``
  every PAIRWISE joint MI is zero, so JMIM scores ``x_3`` the same as a
  pure-noise column even though jointly ``x_3`` is fully determined.
* TC delta is a ``d+2``-variable joint MI (candidate + all of support +
  y). It catches all-order redundancy at the cost of an exponentially-
  sized joint histogram -- mitigated by the dense-renumber trick
  ``_renumber_joint`` borrowed from Layer 60: the cartesian space
  ``prod_j K_j`` never materialises; only the ``<= n`` occupied joint
  cells are tracked.

Cost
----

TC of a ``d``-variable set with equi-frequency binning and the dense-
renumber trick is ``O(n * d)`` time and ``O(n)`` memory for the joint
histogram (one ``np.unique`` per fold). Each ranking pass over ``p``
engineered candidates is ``O(p * n * |support|)``. Calibration: ``n =
2000``, ``p = 50``, ``|support| = 5`` -- about 500k joint-renumber ops,
sub-second on a modern laptop.

Recipe replay
-------------

Each emitted column is backed by an ``orth_univariate`` recipe -- the
SAME kind Layer 21 uses -- because the engineered VALUES are bit-equal
to Layer 21; only the SCORING (and therefore the selection) changes.

NOT wired into ``MRMR.fit`` by default -- opt-in via
``fe_hybrid_orth_tc_enable=True``.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._mi_greedy_cmi_fe import _quantile_bin, _renumber_joint
from ._orthogonal_univariate_fe import (
    generate_univariate_basis_features,
    cached_raw_mi_baseline,
)

logger = logging.getLogger(__name__)

_INT64_MAX = np.iinfo(np.int64).max

__all__ = [
    "total_correlation",
    "score_features_by_tc_uplift",
    "hybrid_orth_mi_tc_fe",
    "hybrid_orth_mi_tc_fe_with_recipes",
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


def _entropy_from_classes(classes: np.ndarray) -> float:
    """Plug-in Shannon entropy (natural log) of a dense integer class array.

    Returns 0.0 for an empty / single-class array.
    """
    if classes.size == 0:
        return 0.0
    counts = np.bincount(classes)
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    p = counts.astype(np.float64) / float(classes.size)
    return float(-(p * np.log(p)).sum())


def _factorize_pack(*cols: np.ndarray) -> np.ndarray:
    """Layer 86: Horner-packed int64 dedup via ``pd.factorize``.

    Mirrors the Layer 84 CMIM optimization (see
    ``_orthogonal_cmim_fe._factorize_pack``): packs each int64 column
    into a single int64 key via running multiplication by per-column
    ``max+1``, then ``pd.factorize(sort=False)`` (a hash-based dedup,
    ~3x faster than ``np.unique``-chained ``_renumber_joint`` at
    n=1000+ on multi-fold joins).

    Resulting class ids may be PERMUTED relative to ``_renumber_joint``
    but the count multiset is identical, and ``_entropy_from_classes``
    (via ``np.bincount``) is invariant under class-id permutation, so
    any plug-in entropy / TC built from these codes is bit-equal (up
    to float summation order) to the canonical path.
    """
    if not cols:
        return np.zeros(0, dtype=np.int64)
    n = cols[0].size
    key = np.zeros(n, dtype=np.int64)
    radix = 1  # running product of per-column (max+1)
    for c in cols:
        c64 = np.asarray(c, dtype=np.int64)
        cmax = int(c64.max()) + 1 if c64.size else 1
        # Horner pack. Normally safe at our bin counts, but a high-cardinality
        # column (cmax ~= n) can push ``radix * cmax`` past int64 max, silently
        # wrapping the key and corrupting the joint count multiset. Detect and
        # fall back to a sort-based row renumber that cannot overflow.
        if radix > _INT64_MAX // max(cmax, 1):
            stacked = np.column_stack([np.asarray(cc, dtype=np.int64) for cc in cols])
            _, inv = np.unique(stacked, axis=0, return_inverse=True)
            return inv.astype(np.int64, copy=False).ravel()
        radix *= cmax
        key = key * cmax + c64
    codes, _ = pd.factorize(key, sort=False)
    return np.asarray(codes.astype(np.int64, copy=False))


def _quantile_bin_batched(arr: np.ndarray, nbins: int) -> np.ndarray:
    """Vectorised equi-frequency bin of a 2-D (n, k) all-finite float array.

    Mirrors the Layer 86 JMIM ``_quantile_bin_batched``. Computes
    ``np.quantile(arr, qs, axis=0)`` ONCE for the whole batch so the
    underlying partition-based selector amortises across columns much
    better than ``k`` separate ``np.quantile`` calls; then a per-column
    dedup + ``np.searchsorted`` produces dense int64 bin codes matching
    the contract of :func:`_quantile_bin` on the all-finite path.

    Bit-equivalent to ``_quantile_bin`` on all-finite numeric input;
    the per-column path is the fallback for mixed-NaN / Inf data.
    """
    n, k = arr.shape
    out = np.zeros((n, k), dtype=np.int64)
    if n == 0 or k == 0:
        return out
    qs = np.linspace(0.0, 1.0, int(nbins) + 1)
    edges_all = np.quantile(arr, qs, axis=0)  # shape (nbins+1, k)
    for j in range(k):
        col_edges = np.unique(edges_all[:, j])
        if col_edges.size <= 2:
            if col_edges.size == 2:
                out[:, j] = (arr[:, j] >= col_edges[1]).astype(np.int64)
            continue
        inner = col_edges[1:-1]
        out[:, j] = np.searchsorted(inner, arr[:, j], side="right").astype(np.int64)
    return out


def _bin_dataframe_batched(
    df: pd.DataFrame, nbins: int,
) -> list[np.ndarray]:
    """Quantile-bin every column of ``df`` via the batched fast path
    when possible; fall back to per-column ``_quantile_bin`` for mixed
    finite / NaN data.

    Returns a list of dense int64 bin arrays (one per column).
    """
    cols = list(df.columns)
    if not cols:
        return []
    arr = df.to_numpy()
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    try:
        arr64 = np.ascontiguousarray(arr, dtype=np.float64)
    except (TypeError, ValueError):
        arr64 = None
    if arr64 is not None and np.isfinite(arr64).all():
        bins_arr = _quantile_bin_batched(arr64, nbins=int(nbins))
        return [np.ascontiguousarray(bins_arr[:, j]) for j in range(bins_arr.shape[1])]
    from ._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust
    return [
        _quantile_bin(
            np.ascontiguousarray(df[c].to_numpy(), dtype=_dt),
            nbins=int(nbins),
        )
        for c in cols
    ]


def total_correlation(
    cols_2d_arr: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Total Correlation (Watanabe 1960) of the columns of a 2-D array.

    Equi-frequency-binned plug-in estimator:

        TC = sum_i H(Z_i) - H(Z_1, ..., Z_d)

    Parameters
    ----------
    cols_2d_arr : np.ndarray, shape (n, d)
        Real-valued columns. Each column is binned independently via
        ``_quantile_bin`` (same equi-frequency binning Layers 21 / 65 / 66
        / 67 / 71 / 72 use) so TC numbers are directly comparable across
        the scorer family. Integer-class arrays are accepted as-is via
        the ``np.searchsorted`` path of ``_quantile_bin`` (which is a no-
        op on already-dense integer codes after the quantile pass).
    n_bins : int
        Bins per column. Default 10 matches the rest of the orth-poly
        scorer family.

    Returns
    -------
    float
        Plug-in TC in nats. Always non-negative for d >= 2 in the
        asymptotic limit; the finite-sample plug-in estimate can drift
        slightly negative on near-independent low-n data -- callers
        comparing TC values across candidate sets should treat
        differences below the per-sample noise floor (~ 0.01 nat at
        n=2000, d=4) as ties.
    """
    arr = np.asarray(cols_2d_arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"total_correlation: expected 2-D array (n, d), got shape " f"{arr.shape}")
    n, d = arr.shape
    if d == 0 or n == 0:
        return 0.0
    bins: list[np.ndarray] = []
    sum_marginal_H = 0.0
    for j in range(d):
        col = arr[:, j]
        if np.issubdtype(col.dtype, np.integer):
            b = col.astype(np.int64)
            # Densify the integer codes so the bincount path doesn't
            # allocate a sparse counts array.
            _, inverse = np.unique(b, return_inverse=True)
            b = inverse.astype(np.int64)
        else:
            b = _quantile_bin(np.ascontiguousarray(col, dtype=np.float64), nbins=int(n_bins))
        bins.append(b)
        sum_marginal_H += _entropy_from_classes(b)
    if d == 1:
        # TC degenerates to 0 for a single column (sum H - joint H = 0).
        return 0.0
    joint, _ = _renumber_joint(*bins)
    joint_H = _entropy_from_classes(joint)
    return float(sum_marginal_H - joint_H)


def _build_support_bins(
    raw_X: pd.DataFrame,
    current_support: Optional[pd.DataFrame],
    n_bins: int,
) -> tuple[list[np.ndarray], float]:
    """Quantile-bin the current support DataFrame; return (bins, sum_marginal_H).

    Falls back to an empty support when ``current_support`` is None /
    empty / misaligned with ``raw_X`` (the natural "first-round"
    behaviour -- no extra TC contribution from the empty support, so the
    score reduces to the candidate's own TC contribution with y).
    """
    if current_support is None or not isinstance(current_support, pd.DataFrame) or current_support.shape[1] == 0:
        return [], 0.0
    if len(current_support) != len(raw_X):
        logger.warning(
            "score_features_by_tc_uplift: current_support length %d does " "not match raw_X length %d; falling back to empty support.",
            len(current_support),
            len(raw_X),
        )
        return [], 0.0
    bins: list[np.ndarray] = []
    sum_H = 0.0
    for c in current_support.columns:
        col = current_support[c].to_numpy()
        b = _quantile_bin(np.ascontiguousarray(col, dtype=np.float64), nbins=int(n_bins))
        bins.append(b)
        sum_H += _entropy_from_classes(b)
    return bins, sum_H


def score_features_by_tc_uplift(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    current_support: Optional[pd.DataFrame] = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Rank engineered columns by their CONDITIONAL new-info contribution
    derived from the TC increment around the candidate's insertion::

        tc_before    = TC([current_support, y])
        tc_after     = TC([current_support, c, y])
        delta_tc     = tc_after - tc_before                  # = I(c; (S, y))
        score(c)     = delta_tc - I(c ; joint_S)             # = I(c; y | S)

    The raw TC increment ``delta_tc`` is the mutual information of the
    candidate with the joint ``(support, y)`` -- it decomposes exactly
    as ``I(c; S) + I(c; y | S)``. Ranking by ``delta_tc`` itself would
    REWARD candidates that overlap with the support (the redundancy
    term), which is the opposite of what we want. We strip the ``I(c;
    joint_S)`` redundancy term and rank by the conditional ``I(c; y |
    S)`` -- the genuinely-new information the candidate carries about
    ``y`` GIVEN the support.

    The ``joint_S`` conditioning is what makes this score TC-flavoured
    (and distinct from Layer 60's bin-renumbered CMI): the support
    contribution to TC is the FULL JOINT entropy of S, not a sum of
    pairwise terms, so the higher-order redundancy of an XOR-style
    support triple is captured. Higher score = more new info about y
    that the support does not already carry.

    Parameters
    ----------
    raw_X : DataFrame
        Source columns. Used to look up the per-source baseline marginal
        MI ``MI(source; y)`` so the returned ``baseline_mi`` column is
        comparable to the rest of the Layer-21 family.
    engineered_X : DataFrame
        Output of :func:`generate_univariate_basis_features`. Column
        names must carry the ``"{source}__{basis_code}{degree}"`` suffix
        so the per-source baseline can be looked up.
    y : array-like (n,)
        Discrete target (binary or multiclass int codes). Continuous y
        must be binned upstream.
    current_support : Optional[DataFrame]
        Reference set ``S`` whose joint with ``c`` and ``y`` defines
        ``tc_after``. When None / empty, ``S`` is empty and the score
        becomes ``TC([c, y]) = I(c; y)`` -- equivalent to marginal MI
        for the first-round case (so callers without an explicit
        support get a sensible ordering on cold start).
    n_bins : int
        Equi-frequency bins per column. Default 10.

    Returns
    -------
    DataFrame with columns ``[engineered_col, source_col, baseline_mi,
    engineered_mi, uplift]`` sorted by ``engineered_mi`` (the raw
    ``delta_tc``) descending. ``baseline_mi`` is the per-source marginal
    MI with y (consistency with Layers 21 / 65 / 66 / 67 / 71 / 72);
    ``engineered_mi`` is the ``delta_tc`` itself; ``uplift = delta_tc /
    (baseline_mi + eps)`` is kept for callers that gate on the relative
    floor.
    """
    if len(raw_X) != len(engineered_X):
        raise ValueError(
            f"score_features_by_tc_uplift: raw_X has {len(raw_X)} rows " f"but engineered_X has {len(engineered_X)}; positional row " f"alignment required."
        )
    if len(raw_X) != len(np.asarray(y)):
        raise ValueError(f"score_features_by_tc_uplift: raw_X has {len(raw_X)} rows " f"but y has {len(np.asarray(y))}; positional row alignment " f"required.")
    empty_cols = [
        "engineered_col", "source_col",
        "baseline_mi", "engineered_mi", "uplift",
    ]
    if engineered_X.empty:
        return pd.DataFrame(columns=empty_cols)

    y_int = _coerce_y_int64(y)
    from ._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust
    # Fit-scoped memo (cached_raw_mi_baseline): a no-op passthrough outside an active
    # orth_scoring_memo_scope(), so this stays byte-for-byte the same _mi_classif_batch call by default;
    # inside a scope it lets sibling opt-in layers (routing / adaptive-degree / cluster-basis / diff-basis
    # / adaptive-arity) share this raw MI(x; y) batch instead of each recomputing it.
    raw_mi_map = cached_raw_mi_baseline(list(raw_X.columns), raw_X.to_numpy(dtype=_dt), y_int, nbins=int(n_bins))

    # y bin support: equi-frequency-bin discrete y by dense renumbering.
    _, y_bin = np.unique(y_int, return_inverse=True)
    y_bin = y_bin.astype(np.int64)
    y_H = _entropy_from_classes(y_bin)

    sup_bins, sup_sum_H = _build_support_bins(
        raw_X, current_support, n_bins=int(n_bins),
    )

    # Layer 86 optimization: pre-compute the support-side joints ONCE.
    # ``_renumber_joint(*sup_bins)`` and ``_renumber_joint(*sup_bins,
    # y_bin)`` are INVARIANT across the p_eng candidates -- chaining
    # them per-candidate (the pre-opt path) re-did ``len(sup_bins) + 1``
    # ``np.unique`` calls each loop iter. We hoist them outside the
    # loop AND switch the per-candidate fold from sort-based
    # ``_renumber_joint`` to hash-based ``_factorize_pack``: each
    # candidate's joint_after = factorize_pack(joint_Sy, cand) and
    # joint_cS = factorize_pack(joint_S, cand) -- two hash-dedup ops
    # vs. (len(sup_bins) + 2) chained np.unique folds. Bit-equivalent
    # (entropy invariant under class-id permutation).
    if sup_bins:
        joint_s_invariant, _ = _renumber_joint(*sup_bins)
        H_S = _entropy_from_classes(joint_s_invariant)
        joint_sy_invariant, _ = _renumber_joint(*sup_bins, y_bin)
        H_sy = _entropy_from_classes(joint_sy_invariant)
        tc_before = sup_sum_H + y_H - H_sy
    else:
        joint_s_invariant = None
        joint_sy_invariant = None
        H_S = 0.0
        H_sy = y_H
        tc_before = 0.0

    # Pre-bin all candidate columns via the batched quantile path.
    eng_bins = _bin_dataframe_batched(engineered_X, nbins=int(n_bins))
    eng_names_list = list(engineered_X.columns)

    rows: list[dict] = []
    for j, eng_name in enumerate(eng_names_list):
        source = eng_name.split("__", 1)[0] if "__" in eng_name else eng_name
        baseline = float(raw_mi_map.get(source, 0.0))
        cand_bin = eng_bins[j]
        cand_H = _entropy_from_classes(cand_bin)
        # Build joint(S, c, y) via hash-dedup over the pre-computed
        # invariant (S, y) joint -- one factorize_pack call instead of
        # the (len(sup_bins)+1)-fold _renumber_joint chain.
        if joint_sy_invariant is not None:
            joint_after = _factorize_pack(joint_sy_invariant, cand_bin)
        else:
            # No support: H(c, y) only.
            joint_after = _factorize_pack(cand_bin, y_bin)
        H_after = _entropy_from_classes(joint_after)
        tc_after = sup_sum_H + cand_H + y_H - H_after
        # Raw TC increment (Watanabe). delta_tc decomposes EXACTLY as
        #     delta_tc = I(c ; (S, y))
        #             = I(c ; S) + I(c ; y | S)
        # so the "TC grew" includes BOTH the new-info-with-y term AND
        # the candidate-support-redundancy term. A near-copy of an
        # already-selected support col scores high purely from the
        # ``I(c; S)`` redundancy contribution -- the OPPOSITE of "new
        # info". We rank by the conditional ``I(c; y | S)`` term,
        # obtained by subtracting ``I(c; joint_S)`` from delta_tc.
        if joint_s_invariant is not None:
            joint_cs = _factorize_pack(joint_s_invariant, cand_bin)
            H_cs = _entropy_from_classes(joint_cs)
            # I(c; joint_S) = H(c) + H(S) - H(c, S)
            mi_c_S = cand_H + H_S - H_cs
        else:
            mi_c_S = 0.0
        delta_tc_raw = tc_after - tc_before
        new_info_term = delta_tc_raw - mi_c_S  # = I(c; y | S)
        uplift = new_info_term / (baseline + 1e-12)
        rows.append({
            "engineered_col": eng_name,
            "source_col": source,
            "baseline_mi": baseline,
            "engineered_mi": float(new_info_term),
            "uplift": float(uplift),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        # Rank by raw delta_tc -- the per-source baseline uplift would
        # explode on near-zero-MI sources, the same pathology Layer 72
        # documents for JMIM. Bennasar-style: trust the joint-info score
        # itself.
        df = df.sort_values(
            "engineered_mi", ascending=False,
        ).reset_index(drop=True)
    return df


def hybrid_orth_mi_tc_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    current_support: Optional[pd.DataFrame] = None,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 0.95,
    min_abs_mi_frac: float = 0.05,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Total-Correlation variant of :func:`hybrid_orth_mi_fe`.

    Replaces the plug-in marginal MI estimator with the Watanabe 1960
    total-correlation uplift criterion: each engineered candidate's
    contribution to the joint shared information of ``[support, c, y]``
    is measured against the support's TC alone, and the top-K winners
    by ``delta_tc`` are appended.

    Parameters
    ----------
    current_support : Optional[DataFrame]
        Reference set ``S`` for the TC computation. When None, the
        score collapses to the marginal MI of each candidate with y
        (the cold-start case).
    cols / degrees / basis / top_k / min_uplift / min_abs_mi_frac :
        see :func:`hybrid_orth_mi_fe`. The ``min_uplift`` ratio gate is
        preserved at the input for cross-layer kwargs parity but is not
        the primary admission criterion under TC semantics (a candidate
        with near-zero baseline_mi would post an exploding ratio); the
        absolute floor ``min_abs_mi_frac * max(delta_tc)`` dominates.
    n_bins : int
        Equi-frequency bins per column (default 10).

    Returns
    -------
    (X_augmented, scores)
        X_augmented : ``X`` with the TC-ranked top-K winners appended.
        scores : full ranking DataFrame (winners + rejects).
    """
    engineered = generate_univariate_basis_features(
        X, cols=cols, degrees=degrees, basis=basis,
    )
    empty_cols = [
        "engineered_col", "source_col",
        "baseline_mi", "engineered_mi", "uplift",
    ]
    if engineered.empty:
        return X.copy(), pd.DataFrame(columns=empty_cols)

    raw_X = X[[c for c in (cols or X.columns) if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]]
    scores = score_features_by_tc_uplift(
        raw_X, engineered, y,
        current_support=current_support,
        n_bins=int(n_bins),
    )
    if scores.empty:
        return X.copy(), scores
    eng_mis = scores["engineered_mi"].to_numpy()
    max_tc = float(eng_mis.max()) if eng_mis.size else 0.0
    abs_floor = float(min_abs_mi_frac) * max(0.0, max_tc)
    qualified = scores[scores["engineered_mi"] >= abs_floor]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    return X_aug, scores


def hybrid_orth_mi_tc_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    current_support: Optional[pd.DataFrame] = None,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 0.95,
    min_abs_mi_frac: float = 0.05,
    n_bins: int = 10,
):
    """Same as :func:`hybrid_orth_mi_tc_fe` plus a list of
    ``orth_univariate`` recipes -- one per appended column -- so
    ``MRMR.transform`` can recompute each engineered column on test
    data without re-running the TC ranking.

    Recipes are byte-identical to Layer 21 because the engineered
    VALUES are byte-identical -- only the SCORING (and therefore the
    selection) differs.
    """
    from .engineered_recipes import build_orth_univariate_recipe
    from ._orthogonal_univariate_fe import _evaluate_basis_column

    X_aug, scores = hybrid_orth_mi_tc_fe(
        X, y,
        current_support=current_support,
        cols=cols, degrees=degrees, basis=basis,
        top_k=top_k, min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac,
        n_bins=int(n_bins),
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    recipes = []
    code_to_basis = {
        "He": "hermite", "LL": "laguerre", "T": "chebyshev", "L": "legendre",
    }
    for name in appended:
        if "__" not in name:
            continue
        src, suffix = name.split("__", 1)
        chosen_basis = None
        chosen_degree = None
        for code in ("LL", "He", "T", "L"):
            if suffix.startswith(code):
                rest = suffix[len(code) :]
                if rest.isdigit():
                    chosen_basis = code_to_basis[code]
                    chosen_degree = int(rest)
                    break
        if chosen_basis is None or chosen_degree is None:
            logger.warning(
                "hybrid_orth_mi_tc_fe_with_recipes: cannot parse " "basis/degree from column name %r; skipping recipe build.",
                name,
            )
            continue
        _pp = None
        try:
            _col_full = np.asarray(X[src].to_numpy(), dtype=np.float64)
            _, _pp = _evaluate_basis_column(_col_full, chosen_basis, int(chosen_degree), return_params=True)
        except Exception:
            _pp = None
        recipes.append(build_orth_univariate_recipe(
            name=name, src_name=src,
            basis=chosen_basis, degree=chosen_degree,
            preprocess_params=_pp,
        ))
    return X_aug, scores, recipes
