"""Layer 74 (2026-06-01): CMIM (Conditional Mutual Information Maximisation,
Fleuret 2004) redundancy-aware ranking for hybrid orth-poly FE.

Why this layer
--------------

Layers 21 / 65 / 66 / 67 / 71 rank each engineered column by its MARGINAL
dependence with the target. Layer 72 (JMIM, Bennasar 2015) scores the
candidate's WORST-CASE PAIRWISE JOINT MI ``min_j I((X_k, X_j); Y)`` against
the already-selected support. Layer 73 (Total Correlation, Watanabe 1960)
scores the FULL-ORDER joint shared-information delta of the candidate with
the support union and ``y``.

CMIM (Fleuret 2004) sits between marginal MI and JMIM: it scores the
candidate by the WORST-CASE CONDITIONAL MI against each already-selected
support member INDIVIDUALLY::

    J_CMIM(X_k) = min over X_j in S of  CMI(X_k ; Y | X_j)

Intuition: ``CMI(X_k; Y | X_j)`` measures how much NEW information about
``Y`` the candidate ``X_k`` carries GIVEN that ``X_j`` is already in the
model. The min over ``S`` enforces robustness: a candidate scores well
only if it adds NEW information given EACH already-selected feature
individually. A candidate that is redundant with even ONE support member
gets a low score because the corresponding CMI collapses near zero.

Reference: Fleuret, F. (2004). "Fast Binary Feature Selection with
Conditional Mutual Information." *Journal of Machine Learning Research*
5:1531-1555. CMIM is the original member of the Brown 2012
"conditional likelihood maximisation" family (Brown, G., Pocock, A.,
Zhao, M.-J., Lujan, M. (2012). "Conditional Likelihood Maximisation: A
Unifying Framework for Information Theoretic Feature Selection."
*Journal of Machine Learning Research* 13:27-66) and predates the JMIM
refinement of Bennasar 2015.

CMIM vs JMIM (Layer 72)
-----------------------

* JMIM scores ``min_j I((X_k, X_j); Y)`` -- a pairwise JOINT MI taken
  together with each support member. The joint construction means the
  candidate gets credit for INTERACTION with the support: a candidate
  that is marginally uninformative but jointly informative with ``X_j``
  posts a HIGH JMIM contribution at that ``j``.
* CMIM scores ``min_j I(X_k; Y | X_j)`` -- a pairwise CONDITIONAL MI
  given each support member. The conditional construction REMOVES
  ``X_j``'s contribution: a candidate that is redundant with ``X_j``
  (i.e. carries the SAME info ``X_j`` already carries about ``Y``) gets
  CMI near zero at that ``j``.

The practical difference: JMIM rewards complementarity, CMIM penalises
redundancy. On a fixture where ``X_k`` and ``X_j`` are near-copies but
both correlated with ``Y``, JMIM scores ``I((X_k, X_j); Y) ~ I(X_j; Y)``
(no penalty -- the joint is as informative as the existing support),
whereas CMIM scores ``I(X_k; Y | X_j) ~ 0`` (full redundancy penalty).
Giving users a choice between the two is the empirical recommendation
of Brown 2012 (Sec. 6): JMIM wins on heavily-INTERACTING feature
families (e.g. signal split across multiple bases), CMIM wins on
heavily-DUPLICATING feature families (e.g. near-copies of one strong
predictor). Layer 74 ships CMIM so the user can match the scorer to
the candidate-pool topology.

CMIM vs Layer 60 (CMI-greedy FE constructor)
--------------------------------------------

Layer 60 already builds GENERIC transform candidates ranked by ``CMI(X;
Y | joint_S)`` where the JOINT renumbered support is the conditioning
set. CMIM differs from Layer 60 in two ways:

* Scope: Layer 60 is a separate FE constructor (operates on
  log_abs / sqrt_abs / etc. transforms); Layer 74 plugs into the SAME
  engineered-column pool the orth-poly hybrid stages (21 / 65 / 66 / 67
  / 71 / 72 / 73) use, so the CMIM ranking is directly comparable to its
  sibling scorers on the same candidates.
* Aggregation: Layer 60 conditions on the JOINT renumbered support
  (one composite class id); CMIM conditions on EACH support member
  INDIVIDUALLY and takes the MIN. The two are NOT equivalent: a candidate
  redundant with the joint support but not with any single member would
  score zero under Layer 60 yet high under CMIM (no individual ``X_j``
  alone fully accounts for the candidate). The Fleuret 2004 min-form is
  more pessimistic about EACH pairwise redundancy and less sensitive to
  emergent joint redundancies the support cooperatively encodes.

Cost
----

For ``p`` candidates and ``|S|`` support columns, CMIM runs ``p * |S|``
``CMI(X; Y | X_j)`` evaluations. Each CMI is ``O(n)`` after
``_renumber_joint`` (dense-renumber trick from Layer 60 keeps memory at
``O(n)`` regardless of the cartesian space). Calibration: ``n = 2000``,
``p = 50``, ``|S| = 5`` -- ~ 250 CMI evaluations, sub-second on a modern
laptop.

Recipe replay
-------------

Each emitted column is backed by an ``orth_univariate`` recipe -- the
SAME kind Layer 21 uses -- because the engineered VALUES are bit-equal
to Layer 21; only the SCORING (and therefore the selection) changes.

NOT wired into ``MRMR.fit`` by default -- opt-in via
``fe_hybrid_orth_cmim_enable=True``.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._mi_greedy_cmi_fe import (
    _cmi_from_binned,
    _entropy_from_classes,
    _quantile_bin,
    _renumber_joint,
)
from ._orthogonal_univariate_fe import (
    _mi_classif_batch,
    generate_univariate_basis_features,
)

logger = logging.getLogger(__name__)

_INT64_MAX = np.iinfo(np.int64).max

__all__ = [
    "cmim_score",
    "score_features_by_cmim",
    "hybrid_orth_mi_cmim_fe",
    "hybrid_orth_mi_cmim_fe_with_recipes",
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


def _factorize_pack(*cols: np.ndarray) -> tuple[np.ndarray, int]:
    """Faster equivalent of :func:`_renumber_joint` for entropy/CMI use.

    Packs each int64 column into a single int64 key via running
    multiplication by per-column ``max+1``, then ``pd.factorize`` (a
    sort-free hash dedup, ~3x faster than ``np.unique``-based chaining
    at n=2500).

    The output class ids may differ from :func:`_renumber_joint`'s
    SORTED ids, but the resulting class-count MULTISET is identical --
    and the downstream consumer (:func:`_entropy_from_classes` via
    ``np.bincount``) is invariant under class-id permutation, so the
    final CMI value is bit-equal.
    """
    if not cols:
        return np.zeros(0, dtype=np.int64), 1
    n = cols[0].size
    key = np.zeros(n, dtype=np.int64)
    radix = 1  # running product of per-column (max+1)
    overflow = False
    for c in cols:
        c64 = np.asarray(c, dtype=np.int64)
        cmax = int(c64.max()) + 1 if c64.size else 1
        # ``key * cmax + c64`` is the standard Horner pack for mixed-radix
        # integer compositions. It is normally safe at our bin counts, but a
        # high-cardinality column (cmax ~= n, e.g. a continuous source binned
        # to n levels) can push ``radix * cmax`` past int64 max, silently
        # wrapping the key and corrupting the joint count multiset. Detect that
        # and fall back to a sort-based per-column renumber (no Horner radix).
        if radix > _INT64_MAX // max(cmax, 1):
            overflow = True
            break
        radix *= cmax
        key = key * cmax + c64
    if overflow:
        return _renumber_joint_safe(*cols)
    codes, uniques = pd.factorize(key, sort=False)
    return codes.astype(np.int64, copy=False), int(len(uniques))


def _renumber_joint_safe(*cols: np.ndarray) -> tuple[np.ndarray, int]:
    """Overflow-proof joint renumber: stack columns and factorize the rows
    directly (no Horner radix product), so cardinality cannot overflow int64."""
    stacked = np.column_stack([np.asarray(c, dtype=np.int64) for c in cols])
    _, inv = np.unique(stacked, axis=0, return_inverse=True)
    inv = inv.astype(np.int64, copy=False).ravel()
    k = int(inv.max()) + 1 if inv.size else 1
    return inv, k


def _build_cmi_yz_cache(
    y_bin: np.ndarray,
    selected_bins: Sequence[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray, float, float, int, int]]:
    """Pre-compute the ``(yz, z, h_z, h_yz, k_z, k_yz)`` tuple for every
    support member -- these depend ONLY on ``y_bin`` and the support
    member's bin codes and so are invariant across candidates. Reusing
    them across the ``p`` candidate columns turns the inner CMI computation
    from 3 joint-renumbers + 4 entropies per (cand, support) pair to
    2 joint-renumbers + 2 entropies per pair -- matching the analytic
    structure of CMIM where (yz, z) terms are shared.

    Returns a list of ``(yz_joint, z_int64, h_z, h_yz, k_z, k_yz)`` tuples
    aligned positionally with ``selected_bins``.
    """
    cache: list[tuple[np.ndarray, np.ndarray, float, float, int, int]] = []
    y_i = np.ascontiguousarray(y_bin, dtype=np.int64)
    for s_bin in selected_bins:
        z_i = np.ascontiguousarray(s_bin, dtype=np.int64)
        yz, _ = _factorize_pack(y_i, z_i)
        h_z, k_z = _entropy_from_classes(z_i)
        h_yz, k_yz = _entropy_from_classes(yz)
        cache.append((yz, z_i, h_z, h_yz, k_z, k_yz))
    return cache


def _cmi_from_binned_with_cached_z(
    x_bin: np.ndarray,
    z_bin: np.ndarray,
    yz_joint: np.ndarray,
    h_z: float,
    h_yz: float,
    k_z: int,
    k_yz: int,
    n: float,
) -> float:
    """Bit-exact equivalent of ``_cmi_from_binned(x, y, z)`` that re-uses
    pre-computed ``(yz, h_z, h_yz, k_z, k_yz)`` from
    :func:`_build_cmi_yz_cache`. Saves one ``np.unique`` call (the yz
    renumber) and two entropy passes per (candidate, support) pair.

    The arithmetic is identical to :func:`_cmi_from_binned`:
    ``CMI = H_xz + H_yz - H_z - H_xyz`` with the same Miller-Madow
    correction ``(k_xyz + k_z - k_xz - k_yz) / (2n)``.
    """
    x_i = np.ascontiguousarray(x_bin, dtype=np.int64)
    # xyz is built from x and the pre-renumbered yz -- numerically
    # equivalent to renumber(x, y, z) because the packed factorisation
    # is associative on the COUNT MULTISET (entropy is invariant under
    # class-id permutation).
    xz, _ = _factorize_pack(x_i, z_bin)
    xyz, _ = _factorize_pack(x_i, yz_joint)
    h_xz, k_xz = _entropy_from_classes(xz)
    h_xyz, k_xyz = _entropy_from_classes(xyz)
    cmi_plugin = h_xz + h_yz - h_z - h_xyz
    cmi_bias = (k_xyz + k_z - k_xz - k_yz) / (2.0 * n)
    return max(0.0, cmi_plugin - cmi_bias)


def _cmim_score_cached(
    candidate_bin: np.ndarray,
    y_bin: np.ndarray,
    cache: list,
    n: float,
) -> float:
    """Cached-z fast path for :func:`cmim_score` -- bit-equivalent."""
    if not cache:
        # Empty support -> marginal MI fallback via the public helper so
        # the bias correction path remains a single source of truth.
        return float(_cmi_from_binned(candidate_bin, y_bin, None))
    best = np.inf
    for yz, z_bin, h_z, h_yz, k_z, k_yz in cache:
        cmi = _cmi_from_binned_with_cached_z(
            candidate_bin, z_bin, yz, h_z, h_yz, k_z, k_yz, n,
        )
        if cmi < best:
            best = cmi
        if best <= 0.0:
            # Same early-exit as cmim_score: a zero is the floor; the
            # min cannot drop below.
            return 0.0
    return float(best)


def cmim_score(
    candidate_bin: np.ndarray,
    y_bin: np.ndarray,
    selected_bins: Sequence[np.ndarray],
) -> float:
    """Fleuret 2004 CMIM score for a binned candidate column.

        J_CMIM(X_k) = min over X_j in S of CMI(X_k ; Y | X_j)

    When ``selected_bins`` is empty, the score reduces to the marginal
    MI ``I(X_k; Y)`` (the natural first-round fallback when no support
    is available).

    Parameters
    ----------
    candidate_bin : np.ndarray, shape (n,)
        Dense int64 bin codes for the candidate column.
    y_bin : np.ndarray, shape (n,)
        Dense int64 bin codes for the target.
    selected_bins : Sequence[np.ndarray]
        One bin array per already-selected support column.

    Returns
    -------
    float
        The CMIM score in nats. Always non-negative (Miller-Madow-
        corrected CMI floors at zero in :func:`_cmi_from_binned`).
    """
    if not selected_bins:
        # Empty support -> marginal MI fallback (CMI with z=None).
        return float(_cmi_from_binned(candidate_bin, y_bin, None))
    best = np.inf
    for s_bin in selected_bins:
        cmi = float(_cmi_from_binned(candidate_bin, y_bin, s_bin))
        if cmi < best:
            best = cmi
        if best <= 0.0:
            # Early-exit: CMI cannot drop below zero (Miller-Madow floor),
            # so once we see a zero we know the min is zero. Saves the
            # remaining |S|-j-1 evaluations on heavily-redundant candidates.
            return 0.0
    return float(best)


def score_features_by_cmim(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    current_support: Optional[pd.DataFrame] = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """CMIM score for each engineered column (Fleuret 2004).

    For every engineered column ``X_k`` returns

        J_CMIM(X_k) = min over X_j in S of  CMI(X_k ; Y | X_j)

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
        candidates against the raw inputs is the "first-round" CMIM).
    n_bins : int
        Equi-frequency bin count per column. Same default as the
        sibling layers (10).

    Returns
    -------
    DataFrame with columns ``[engineered_col, source_col, baseline_mi,
    engineered_mi, uplift]`` sorted by ``engineered_mi`` descending.
    Column names use ``baseline_mi`` / ``engineered_mi`` for downstream
    consistency with Layers 21 / 65 / 66 / 67 / 71 / 72 / 73; the
    engineered VALUES are CMIM scores (not marginal MI) but the ranking
    semantics are identical (higher = better).
    """
    if len(raw_X) != len(engineered_X):
        raise ValueError(
            f"score_features_by_cmim: raw_X has {len(raw_X)} rows but "
            f"engineered_X has {len(engineered_X)}; positional row "
            f"alignment required."
        )
    if len(raw_X) != len(np.asarray(y)):
        raise ValueError(
            f"score_features_by_cmim: raw_X has {len(raw_X)} rows but "
            f"y has {len(np.asarray(y))}; positional row alignment "
            f"required."
        )
    empty_cols = [
        "engineered_col", "source_col",
        "baseline_mi", "engineered_mi", "uplift",
    ]
    if engineered_X.empty:
        return pd.DataFrame(columns=empty_cols)

    # Per-source baseline marginal MI -- used to populate the ``uplift``
    # column so the CMIM ranking is comparable across columns with very
    # different source-marginal magnitudes.
    y_int = _coerce_y_int64(y)
    raw_mi = _mi_classif_batch(
        raw_X.to_numpy(dtype=np.float64), y_int, nbins=int(n_bins),
    )
    raw_mi_map = dict(zip(list(raw_X.columns), raw_mi.tolist()))

    # Build the reference set ``S`` for the CMIM min. Default to raw_X
    # when no explicit current_support is provided (mirrors Layer 72).
    use_support = current_support
    if (
        use_support is None
        or not isinstance(use_support, pd.DataFrame)
        or use_support.shape[1] == 0
    ):
        use_support = raw_X
    if len(use_support) != len(engineered_X):
        # Defensive: a misaligned support is treated as raw_X (fall back
        # to the first-round case).
        logger.warning(
            "score_features_by_cmim: current_support length %d does not "
            "match engineered_X length %d; falling back to raw_X as "
            "redundancy reference.",
            len(use_support), len(engineered_X),
        )
        use_support = raw_X
    sel_bins: list[np.ndarray] = []
    for c in use_support.columns:
        col = use_support[c].to_numpy()
        sel_bins.append(_quantile_bin(
            np.ascontiguousarray(col, dtype=np.float64),
            nbins=int(n_bins),
        ))

    # y bin support: cmim_score expects integer-encoded y.
    _, y_bin = np.unique(y_int, return_inverse=True)
    y_bin = y_bin.astype(np.int64)

    # Layer 84 optimization: pre-compute the (yz, z, h_z, h_yz, k_z, k_yz)
    # tuple for every support member once. These depend only on
    # ``y_bin`` + the support member's bin codes and are invariant across
    # the ``p_eng`` candidate columns -- caching saves ``p_eng-1``
    # ``np.unique`` calls per support member (~6x speedup on a typical
    # n=2500, p=20 fixture). Bit-equivalent to the per-call path.
    support_col_names = list(use_support.columns)
    n_rows = float(max(1, y_bin.size))
    full_cache = _build_cmi_yz_cache(y_bin, sel_bins)

    rows: list[dict] = []
    for eng_name in engineered_X.columns:
        source = eng_name.split("__", 1)[0] if "__" in eng_name else eng_name
        baseline = float(raw_mi_map.get(source, 0.0))
        cand_col = engineered_X[eng_name].to_numpy()
        cand_bin = _quantile_bin(
            np.ascontiguousarray(cand_col, dtype=np.float64),
            nbins=int(n_bins),
        )
        # Fleuret 2004 CMIM: min over support of CMI(cand; y | X_j).
        # CRITICAL: skip the candidate's own raw source when scoring --
        # CMI(He_2(x_dup_a); y | x_dup_a) collapses to ~0 trivially
        # (the engineered column is a deterministic function of its
        # source). We want the redundancy filter to penalise OVERLAP
        # with OTHER support members; the self-source overlap is
        # accounted for by ranking on engineered_mi vs baseline_mi.
        cache_filtered = [
            entry for entry, sname in zip(full_cache, support_col_names)
            if sname != source
        ]
        if not cache_filtered:
            score = float(_cmi_from_binned(cand_bin, y_bin, None))
        else:
            score = _cmim_score_cached(cand_bin, y_bin, cache_filtered, n_rows)
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
        # Sort by the RAW CMIM score (``engineered_mi``). The per-source
        # baseline uplift ratio would explode on near-zero-MI sources --
        # the same pathology Layer 72 / 73 document for JMIM / TC. The
        # Fleuret 2004 paper ranks by the CMIM value itself; we follow.
        df = df.sort_values(
            "engineered_mi", ascending=False,
        ).reset_index(drop=True)
    return df


def hybrid_orth_mi_cmim_fe(
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
    """CMIM (Fleuret 2004) variant of :func:`hybrid_orth_mi_fe`.

    Replaces the plug-in marginal MI estimator with the Fleuret 2004
    Conditional MI Maximisation criterion -- each engineered column is
    scored by the WORST-CASE conditional MI against each already-
    selected support member individually (defaulting to ``raw_X`` when
    ``current_support`` is empty), and the absolute floor selection
    admits the top-K winners.

    Parameters
    ----------
    current_support : Optional[DataFrame]
        Reference set ``S`` for the CMIM redundancy min. When None,
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
        X_augmented : ``X`` with the CMIM-ranked top-K winners appended.
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

    raw_X = X[[
        c for c in (cols or X.columns)
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
    ]]
    scores = score_features_by_cmim(
        raw_X, engineered, y,
        current_support=current_support,
        n_bins=int(n_bins),
    )
    if scores.empty:
        return X.copy(), scores
    # CMIM-natural absolute floor (same semantics as Layer 72 / 73): a
    # noise-source uplift ratio would explode while the raw CMIM is
    # tiny. ``min_uplift`` is consumed for cross-layer kwargs parity but
    # has no effect under CMIM semantics.
    eng_mis = scores["engineered_mi"].to_numpy()
    max_cmim = float(eng_mis.max()) if eng_mis.size else 0.0
    abs_floor = float(min_abs_mi_frac) * max(0.0, max_cmim)
    qualified = scores[scores["engineered_mi"] >= abs_floor]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    return X_aug, scores


def hybrid_orth_mi_cmim_fe_with_recipes(
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
    """Same as :func:`hybrid_orth_mi_cmim_fe` plus a list of
    ``orth_univariate`` recipes -- one per appended column -- so
    ``MRMR.transform`` can recompute each engineered column on test
    data without re-running the CMIM ranking.

    Recipes are byte-identical to Layer 21 because the engineered
    VALUES are byte-identical -- only the SCORING (and therefore the
    selection) differs.
    """
    from .engineered_recipes import build_orth_univariate_recipe

    X_aug, scores = hybrid_orth_mi_cmim_fe(
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
                rest = suffix[len(code):]
                if rest.isdigit():
                    chosen_basis = code_to_basis[code]
                    chosen_degree = int(rest)
                    break
        if chosen_basis is None or chosen_degree is None:
            logger.warning(
                "hybrid_orth_mi_cmim_fe_with_recipes: cannot parse "
                "basis/degree from column name %r; skipping recipe build.",
                name,
            )
            continue
        recipes.append(build_orth_univariate_recipe(
            name=name, src_name=src,
            basis=chosen_basis, degree=chosen_degree,
        ))
    return X_aug, scores, recipes
