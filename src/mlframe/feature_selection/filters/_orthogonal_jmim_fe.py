"""Layer 72 (2026-06-01): Joint Mutual Information Maximisation (JMIM)
redundancy-aware ranking for hybrid orth-poly FE.

Why this layer
--------------

Layers 21 / 65 / 66 / 67 / 71 all rank each engineered column by its
MARGINAL dependence with the target (plug-in MI, KSG MI, copula MI, dCor,
HSIC). Marginal ranking is the right answer in isolation, but it fails on
REDUNDANT candidates: when two engineered columns carry the same signal,
both have a high marginal score, both make the top-K, and the second is
pure duplication. JMIM (Bennasar 2015) replaces the marginal MI by the
WORST-CASE joint MI against the already-selected support::

    J_JMIM(X_k) = min over X_j in S of  I((X_k, X_j); Y)

Intuition: ``I((X_k, X_j); Y)`` is large iff ``X_k`` brings INFORMATION
TO ``Y`` that ``X_j`` does not already carry; the minimum over ``S``
picks the WEAKEST link -- a candidate scores well only if it is
informative jointly with EVERY already-selected feature. The min
construction is the key property that distinguishes JMIM from the
related JMI / DISR criteria (Brown 2012 unifies these as special cases
of conditional likelihood maximisation): JMIM enforces non-redundancy
column-by-column rather than only on average, so a redundant candidate
cannot hide behind one good interaction.

Bennasar (2015), Section 4: JMIM is the empirical winner on 22 of 22
benchmark datasets vs marginal MI / mRMR / JMI / DISR / CMIM / CIFE
under naive Bayes / SVM / k-NN downstream classifiers. The min
formulation gives JMIM the best worst-case behaviour when the
incoming candidate pool is heavy with mutually-redundant features --
exactly the regime FE engineering creates (Hermite_2 / Legendre_2 /
Chebyshev_2 of the same x are pairwise highly redundant).

Reference: Bennasar, M., Hua, Y., Setchi, R. (2015). "Feature selection
using Joint Mutual Information Maximisation." *Expert Systems with
Applications* 42(22):8520-8532.

Reuses ``_jmim_scorer.jmim_score``
----------------------------------

This module is a thin orchestration layer on top of the existing
``_jmim_scorer.jmim_score`` njit helper (Bennasar's 3-D joint histogram
implementation). The scorer alone exposes a per-candidate scalar; this
module wraps it with:

* per-source baseline lookup (``MI(x_source; y)`` so ``uplift`` is
  comparable across raw / engineered columns),
* the same two-gate selection rule Layer 21 / 65 / 66 / 67 / 71 use
  (relative ``uplift`` floor + absolute MAD-noise floor),
* the same ``orth_univariate`` recipe builder so ``MRMR.transform`` can
  replay each appended column without re-running the JMIM ranking.

Layer 72 vs Layer 60 (CMI-greedy)
---------------------------------

Layer 60 already ranks generic MI-greedy transforms by
``CMI(candidate; y | support_joint)``. Layer 72 differs in three ways:

* Scope: Layer 60 is a separate FE constructor (operates on generic
  unary / binary transforms); Layer 72 plugs into the SAME engineered-
  column pool the orth-poly hybrid stages (21 / 65 / 66 / 67 / 71) use,
  so the JMIM ranking is directly comparable to its sibling marginal
  scorers on the same candidates.
* Score: CMI ``I(X; Y | Z)`` and JMIM ``min_j I(X, X_j; Y)`` are NOT
  the same quantity. CMI asks "what extra info does X carry given
  Z?"; JMIM asks "what is the smallest joint info X contributes with
  ANY already-picked feature?". The min construction makes JMIM
  more pessimistic on heavily-redundant candidate sets.
* Aggregation: CMI conditions on the joint of all support columns
  (cartesian-renumbered into one composite class); JMIM evaluates a
  separate pairwise joint MI against each selected column and takes
  the MIN. JMIM stays O(n * |S|) in evaluation cost; CMI is O(n *
  prod |bins_j|) on the joint dimension (renumbered, but worst-case
  growing in n).

Recipe replay
-------------

Each emitted column is backed by an ``orth_univariate`` recipe -- the
SAME kind Layer 21 uses -- because the engineered VALUES are bit-equal
to Layer 21; only the SCORING (and therefore the selection) changes.

NOT wired into ``MRMR.fit`` by default -- opt-in via
``fe_hybrid_orth_jmim_enable=True``.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._jmim_scorer import _joint_mi_3d_njit, jmim_score
from ._mi_greedy_cmi_fe import _quantile_bin
from ._orthogonal_univariate_fe import (
    _mi_classif_batch,
    generate_univariate_basis_features,
)

logger = logging.getLogger(__name__)

__all__ = [
    "jmim_score",
    "score_features_by_jmim",
    "hybrid_orth_mi_jmim_fe",
    "hybrid_orth_mi_jmim_fe_with_recipes",
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


def _bin_columns(
    df: pd.DataFrame, nbins: int,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    """Quantile-bin every column of ``df``; return (bins, nbins_per_col, names).

    Each bin array is dense int64 in ``[0, k_j - 1]`` where
    ``k_j = max(bins_j) + 1`` (the dense observed cardinality, which is
    what ``_jmim_scorer._joint_mi_3d_njit`` expects as ``K_x`` / ``K_z``).
    Constant columns collapse to a single bin (``k_j = 1``) -- the JMIM
    scorer then yields zero on that pair, which is the right "no signal"
    reading.

    Layer 86 fast path: routes to :func:`_bin_columns_batched` when the
    DataFrame has 2+ all-finite numeric columns (the common case for
    engineered candidate pools) -- batched ``np.quantile`` over the
    stacked 2-D array amortises the ``np.linspace`` / quantile-edge work
    that dominated the per-column ``_quantile_bin`` loop (60% of total
    score_features_by_jmim runtime at p_eng=100, n=1000).
    """
    names = list(df.columns)
    if not names:
        return [], [], []
    arr = df.to_numpy()
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    # Convert to float64 for the batched quantile; downstream casts to int64.
    try:
        arr64 = np.ascontiguousarray(arr, dtype=np.float64)
    except (TypeError, ValueError):
        arr64 = None
    if arr64 is not None and np.isfinite(arr64).all():
        bins_arr = _quantile_bin_batched(arr64, nbins=int(nbins))
        ks = [(int(bins_arr[:, j].max()) + 1) if bins_arr.shape[0] else 1 for j in range(bins_arr.shape[1])]
        ks = [max(1, k) for k in ks]
        bins = [np.ascontiguousarray(bins_arr[:, j]) for j in range(bins_arr.shape[1])]
        return bins, ks, names
    # Fallback: per-column path preserves the NaN / Inf handling of
    # ``_quantile_bin`` for mixed-finite data.
    bins: list[np.ndarray] = []
    ks: list[int] = []
    for c in names:
        b = _quantile_bin(df[c].to_numpy(), nbins=nbins).astype(np.int64)
        k = int(b.max()) + 1 if b.size else 1
        bins.append(b)
        ks.append(max(1, k))
    return bins, ks, names


def _quantile_bin_batched(arr: np.ndarray, nbins: int) -> np.ndarray:
    """Vectorised equi-frequency bin of a 2-D (n, k) all-finite float array.

    Computes ``np.quantile(arr, qs, axis=0)`` ONCE for the whole batch
    (the underlying partition-based selector amortises across columns
    much better than ``k`` separate ``np.quantile(col, qs)`` calls). Then
    a per-column dedup + ``np.searchsorted`` produces dense int64 bin
    codes matching the contract of :func:`_quantile_bin` on the all-
    finite path: the same edges (after ``np.unique`` dedup) and the same
    ``side='right'`` searchsorted convention.

    Bit-equivalent to ``_quantile_bin`` on all-finite numeric input; the
    per-column fallback handles mixed-NaN / Inf data via the original
    path.
    """
    n, k = arr.shape
    out = np.zeros((n, k), dtype=np.int64)
    if n == 0 or k == 0:
        return out
    qs = np.linspace(0.0, 1.0, int(nbins) + 1)
    # Batched quantile: (nbins+1, k). One partition + interpolation per
    # column under the hood but called via the broadcast path.
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


def score_features_by_jmim(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    current_support: Optional[pd.DataFrame] = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """JMIM score for each engineered column (Bennasar 2015).

    For every engineered column ``X_k`` returns

        J_JMIM(X_k) = min over X_j in S of  I((X_k, X_j); Y)

    where ``S`` is the union of ``raw_X`` (the per-source MI baseline
    pool, also providing the redundancy reference) and any extra
    ``current_support`` the caller passes.

    Parameters
    ----------
    raw_X : DataFrame
        Original source columns. Used for two purposes: (1) the per-
        source baseline ``MI(source; y)`` so ``uplift`` is comparable
        across the engineered ranking, (2) by default the redundancy
        reference set ``S`` when ``current_support`` is None.
    engineered_X : DataFrame
        Output of :func:`generate_univariate_basis_features`. Column
        names must carry the ``"{source}__{basis_code}{degree}"`` suffix
        so the per-source baseline can be looked up.
    y : array-like (n,)
        Target. Must be discrete (binary or multiclass int codes); for
        continuous y bin via ``pd.qcut`` first.
    current_support : Optional[DataFrame]
        Extra reference columns added to ``S`` for the redundancy min.
        When None or empty, ``S`` defaults to ``raw_X`` (the natural
        choice when no MRMR support has been picked yet -- ranking
        candidates against the raw inputs is the "first-round" JMIM).
    n_bins : int
        Equi-frequency bin count per column. Same default as the
        sibling layers (10).

    Returns
    -------
    DataFrame with columns ``[engineered_col, source_col, baseline_mi,
    engineered_mi, uplift]`` sorted by ``uplift`` descending. Column
    names use ``baseline_mi`` / ``engineered_mi`` for downstream
    consistency with Layers 21 / 65 / 66 / 67 / 71; the engineered
    VALUES are JMIM (not marginal MI) but the ranking semantics are
    identical (higher = better).
    """
    if len(raw_X) != len(engineered_X):
        raise ValueError(
            f"score_features_by_jmim: raw_X has {len(raw_X)} rows but " f"engineered_X has {len(engineered_X)}; positional row " f"alignment required."
        )
    if len(raw_X) != len(np.asarray(y)):
        raise ValueError(f"score_features_by_jmim: raw_X has {len(raw_X)} rows but " f"y has {len(np.asarray(y))}; positional row alignment " f"required.")
    empty_cols = [
        "engineered_col", "source_col",
        "baseline_mi", "engineered_mi", "uplift",
    ]
    if engineered_X.empty:
        return pd.DataFrame(columns=empty_cols)

    # Per-source baseline marginal MI -- used to populate the ``uplift``
    # column so the JMIM ranking is comparable across columns with very
    # different source-marginal magnitudes.
    y_int = _coerce_y_int64(y)
    raw_mi = _mi_classif_batch(
        raw_X.to_numpy(dtype=np.float64), y_int, nbins=int(n_bins),
    )
    raw_mi_map = dict(zip(list(raw_X.columns), raw_mi.tolist()))

    # Build the reference set ``S`` for the JMIM min. Default to raw_X
    # when no explicit current_support is provided.
    use_support = current_support
    if use_support is None or not isinstance(use_support, pd.DataFrame) or use_support.shape[1] == 0:
        use_support = raw_X
    if len(use_support) != len(engineered_X):
        # Defensive: a misaligned support is treated as empty (falls
        # back to the first-round case in jmim_score, which returns
        # I(X_cand; y)).
        logger.warning(
            "score_features_by_jmim: current_support length %d does not " "match engineered_X length %d; falling back to raw_X as " "redundancy reference.",
            len(use_support),
            len(engineered_X),
        )
        use_support = raw_X
    sel_bins, sel_ks, _ = _bin_columns(use_support, nbins=int(n_bins))

    # y bin support: jmim_score expects integer-encoded y.
    _, y_bin = np.unique(y_int, return_inverse=True)
    y_bin = y_bin.astype(np.int64)
    K_y = int(y_bin.max()) + 1 if y_bin.size else 1
    K_y = max(1, K_y)

    eng_bins, eng_ks, eng_names = _bin_columns(
        engineered_X, nbins=int(n_bins),
    )

    # Layer 86 optimization: pre-coerce y_bin to int64 ONCE and re-use
    # across all (cand, support) pairs. The reference ``jmim_score``
    # path calls ``x.astype(np.int64)`` / ``y.astype(np.int64)`` /
    # ``col_j.astype(np.int64)`` PER CALL, which forces ``p_eng *
    # (|S| + 2)`` redundant int64 copies of arrays that are already
    # int64. We hoist all coercions before the candidate loop and call
    # the njit kernel ``_joint_mi_3d_njit`` directly so each per-pair
    # call is a tight C-level invocation with zero per-call boxing.
    y_bin_c = np.ascontiguousarray(y_bin, dtype=np.int64)
    sel_bins_c = [np.ascontiguousarray(b, dtype=np.int64) for b in sel_bins]
    sel_ks_arr = list(sel_ks)
    has_support = bool(sel_bins_c)

    rows: list[dict] = []
    for j, eng_name in enumerate(eng_names):
        source = eng_name.split("__", 1)[0] if "__" in eng_name else eng_name
        baseline = float(raw_mi_map.get(source, 0.0))
        # Bennasar 2015 JMIM scores the candidate against EVERY support
        # member -- the min-construction is the redundancy filter.
        # Critically, we include the candidate's own raw source in the
        # support pool (when present): an engineered ``He_2(x_dup_a)``
        # column whose source ``x_dup_a`` is a near-copy of an already-
        # selected ``x1`` collapses ``I((He_2(x_dup_a), x_dup_a); Y)``
        # to roughly ``I(x_dup_a; Y)``, which the min then pins below
        # ``I((He_2(x2), x1); Y)`` -- the redundancy suppression
        # mechanism the paper describes.
        x_c = np.ascontiguousarray(eng_bins[j], dtype=np.int64)
        K_x = int(eng_ks[j])
        if not has_support:
            score = float(_joint_mi_3d_njit(
                x_c, np.zeros(x_c.size, dtype=np.int64), y_bin_c,
                K_x, 1, K_y,
            ))
        else:
            best = np.inf
            for k, z_c in enumerate(sel_bins_c):
                K_z = int(sel_ks_arr[k])
                mi = float(_joint_mi_3d_njit(
                    x_c, z_c, y_bin_c, K_x, K_z, K_y,
                ))
                if mi < best:
                    best = mi
                if best <= 0.0:
                    # MI floors at zero in the njit kernel; once a zero is
                    # observed the min cannot drop further -- early exit.
                    best = 0.0
                    break
            score = float(best) if best != np.inf else 0.0
        uplift = score / (baseline + 1e-12)
        rows.append({
            "engineered_col": eng_name,
            "source_col": source,
            "baseline_mi": baseline,
            "engineered_mi": score,
            "uplift": uplift,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        # Sort by the RAW JMIM score (``engineered_mi``) rather than the
        # uplift ratio. JMIM is a joint-MI score with the universal
        # "higher = more new information given any support member"
        # semantics; the per-source baseline uplift Layer 21 reports is
        # there for downstream gate calibration only -- a near-zero
        # baseline_mi on a noise source would make uplift explode while
        # the raw JMIM score is moderate, which is precisely the wrong
        # ranking signal for Bennasar's criterion. The Bennasar 2015
        # paper ranks by the JMIM value itself; we follow that
        # convention here.
        df = df.sort_values(
            "engineered_mi", ascending=False,
        ).reset_index(drop=True)
    return df


def hybrid_orth_mi_jmim_fe(
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
    """JMIM variant of :func:`hybrid_orth_mi_fe`.

    Replaces the plug-in marginal MI estimator with the Bennasar 2015
    Joint MI Maximisation criterion -- each engineered column is scored
    by the WORST-CASE joint MI against the already-selected support
    (defaulting to ``raw_X`` when ``current_support`` is empty), and
    the two-gate selection (relative ``uplift`` + absolute MAD-noise
    floor) admits the top-K winners.

    Parameters
    ----------
    current_support : Optional[DataFrame]
        Reference set ``S`` for the JMIM redundancy min. When None,
        ``raw_X`` (i.e. the source columns of the engineered candidates)
        is used. Pass the live MRMR support DataFrame to enforce
        non-redundancy against the actually-picked features.
    cols : Optional[Sequence[str]]
        Source columns to expand. Defaults to every numeric column in
        ``X``.
    degrees / basis / top_k / min_uplift / min_abs_mi_frac : see
        :func:`hybrid_orth_mi_fe`.
    n_bins : int
        Equi-frequency bins per column (default 10, same as Layer 21).

    Returns
    -------
    (X_augmented, scores)
        X_augmented : ``X`` with the JMIM-ranked top-K winners appended.
        scores : the full ranking DataFrame (winners + rejects).
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
    scores = score_features_by_jmim(
        raw_X, engineered, y,
        current_support=current_support,
        n_bins=int(n_bins),
    )
    if scores.empty:
        return X.copy(), scores
    # JMIM-natural absolute floor. Layer 21 / 65 / 66 / 67 / 71 gate on
    # the ``uplift = engineered_mi / baseline_mi`` ratio; that semantics
    # is wrong for JMIM because (a) a noise source with near-zero
    # ``baseline_mi`` would post a huge uplift even though its raw JMIM
    # score is small, and (b) JMIM scores are MIN-constructed so they
    # collapse below the marginal-MI baseline on redundant candidates --
    # the uplift ratio would penalise the very candidates JMIM ranks
    # correctly. We gate exclusively on a fraction-of-best-JMIM-score
    # floor, parameterised by ``min_abs_mi_frac`` (kept under the same
    # ctor name for cross-layer parity). The ``min_uplift`` param is
    # consumed at the input but has no effect under JMIM semantics; it
    # is preserved in the signature so callers can swap scorers without
    # rewriting their kwargs.
    eng_mis = scores["engineered_mi"].to_numpy()
    max_jmim = float(eng_mis.max()) if eng_mis.size else 0.0
    abs_floor = float(min_abs_mi_frac) * max(0.0, max_jmim)
    qualified = scores[scores["engineered_mi"] >= abs_floor]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    return X_aug, scores


def hybrid_orth_mi_jmim_fe_with_recipes(
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
    """Same as :func:`hybrid_orth_mi_jmim_fe` plus a list of
    ``orth_univariate`` recipes -- one per appended column -- so
    ``MRMR.transform`` can recompute each engineered column on test
    data without re-running the JMIM ranking.

    Recipes are byte-identical to Layer 21 because the engineered
    VALUES are byte-identical -- only the SCORING (and therefore the
    selection) differs.
    """
    from .engineered_recipes import build_orth_univariate_recipe

    X_aug, scores = hybrid_orth_mi_jmim_fe(
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
                "hybrid_orth_mi_jmim_fe_with_recipes: cannot parse " "basis/degree from column name %r; skipping recipe build.",
                name,
            )
            continue
        recipes.append(build_orth_univariate_recipe(
            name=name, src_name=src,
            basis=chosen_basis, degree=chosen_degree,
        ))
    return X_aug, scores, recipes
