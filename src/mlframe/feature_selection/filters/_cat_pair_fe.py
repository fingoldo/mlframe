"""Cat x cat synergy cross with interaction-information pre-filter (Layer 89, 2026-06-01).

NVIDIA cuDF Kaggle-Grandmaster blog technique #3: combine two categorical
columns into a new high-cardinality categorical ``hash(cat_i || cat_j)`` then
target-encode it. The raw cross captures conditional structure that neither
parent column carries alone -- the textbook example is the categorical XOR
``y = cat_a XOR cat_b`` where each parent is marginally uninformative but the
pair is fully predictive.

The IT enhancement (THE KEY)
----------------------------

Materialising every (cat_i, cat_j) cross explodes the column count and most
crosses carry no synergy. We pre-filter pairs by INTERACTION INFORMATION:

    II(cat_i, cat_j; y) = I(cat_i, cat_j; y) - I(cat_i; y) - I(cat_j; y)

* ``II > 0`` -- synergy: the joint tells more than the sum of the marginals
  (XOR-like). These are the crosses worth materialising.
* ``II < 0`` -- redundancy: the parents overlap (e.g. one is a copy of the
  other), so the cross adds nothing over either parent.
* ``II ~ 0`` -- the parents are conditionally independent given y; the cross
  is no better than concatenating two independent encodings.

Only pairs with ``II > threshold`` are materialised. The discrete MI / joint
MI primitives reuse the Layer 60 / Layer 19 plug-in estimator family via the
self-contained count-based ``_plug_in_mi`` (``_adaptive_nbins``), matching the
Layer 88 scoring discipline.

Cardinality control (Layer 29 pre-screen)
-----------------------------------------

A cat x cat cross can have up to ``card_i * card_j`` cells. Following the
Layer 29 cardinality pre-screen, a cross whose distinct-cell count exceeds
``0.5 * n_samples`` is refused as a raw one-hot/factorize feature (each cell
would be near-unique -> memorisation, not signal). When that happens the cross
is routed through K-fold OOF target encoding (Layer 33) instead, which is
leak-safe and collapses the high-cardinality cross to a single dense numeric
column. Low-cardinality crosses are emitted as the integer cell code directly.

Recipe-based replay
-------------------

Recipe kind ``cat_pair_cross`` stores the (cat_i, cat_j) pair plus the
fit-time ``value-pair -> integer code`` mapping (and, when target-encoded, the
per-code mean-of-y lookup + global mean). Replay reads ONLY X: each test row's
(cat_i_val, cat_j_val) tuple is looked up in the stored mapping; unseen tuples
map to a sentinel code (raw mode) or the global mean (TE mode). No ``y``
reference is captured, so ``transform()`` is leakage-free by construction.
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._target_encoding_fe import _column_to_str, _smooth

logger = logging.getLogger(__name__)

__all__ = [
    "engineered_name_cat_pair_cross",
    "generate_cat_pair_crosses",
    "score_cat_pairs_by_interaction_information",
    "hybrid_cat_pair_fe",
    "auto_detect_cat_pair_cols",
]


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def engineered_name_cat_pair_cross(cat_i: str, cat_j: str) -> str:
    """Stable engineered column name for the (cat_i, cat_j) cross."""
    return f"cross_{cat_i}_{cat_j}"


# ---------------------------------------------------------------------------
# Cardinality control threshold (Layer 29 pre-screen)
# ---------------------------------------------------------------------------


def _cross_too_high_card(n_distinct_cells: int, n_samples: int) -> bool:
    """Layer 29 pre-screen: a cross with more distinct cells than ``0.5 * n``
    is refused as a raw factorize feature (cells near-unique -> memorisation).
    Such crosses are routed through target encoding instead."""
    return n_distinct_cells > 0.5 * float(max(n_samples, 1))


# ---------------------------------------------------------------------------
# Cross materialisation -- pure X-only kernel (no y reference)
# ---------------------------------------------------------------------------


def _encode_pair(
    cats_i: np.ndarray, cats_j: np.ndarray,
) -> tuple[np.ndarray, dict[tuple, int]]:
    """Factorise the (cat_i_val, cat_j_val) tuple stream into a dense integer
    code per row. Returns ``(codes, mapping)`` where ``mapping`` is the
    fit-time ``(str_i, str_j) -> int_code`` table (replay-only payload).

    Codes are assigned in first-seen order for determinism. NaN values were
    already mapped to the ``"__nan__"`` sentinel by ``_column_to_str`` upstream,
    so they form their own implicit category at fit AND replay.
    """
    n = len(cats_i)
    if n == 0:
        return np.empty(0, dtype=np.int64), {}
    # Vectorised factorisation: encode each parent's string stream to ints,
    # fold the pair into a single combined key, then np.unique(return_inverse)
    # assigns dense codes in one C pass (the per-row Python loop was the
    # materialisation hotspot at scale -- cProfile 2026-06-01). Codes are
    # assigned in SORTED-pair order (np.unique sorts) rather than first-seen;
    # the absolute code values are arbitrary anyway (consumed as a categorical
    # / target-encoded), and replay maps via the stored value-pair lookup, so
    # the ordering convention does not affect correctness.
    uniq_i, inv_i = np.unique(cats_i, return_inverse=True)
    uniq_j, inv_j = np.unique(cats_j, return_inverse=True)
    combined = inv_i.astype(np.int64) * (len(uniq_j)) + inv_j.astype(np.int64)
    uniq_pairs, codes = np.unique(combined, return_inverse=True)
    codes = codes.astype(np.int64, copy=False)
    # Reconstruct the (str_i, str_j) -> code mapping from the surviving unique
    # combined keys only (one entry per cell, not per row).
    mapping: dict[tuple, int] = {}
    for code, key in enumerate(uniq_pairs):
        ii = int(key) // len(uniq_j)
        jj = int(key) % len(uniq_j)
        mapping[(uniq_i[ii], uniq_j[jj])] = code
    return codes, mapping


def generate_cat_pair_crosses(
    X: pd.DataFrame,
    cat_cols: Sequence[str],
    pairs: Optional[Sequence[tuple[str, str]]] = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Materialise an integer-coded ``cross_{cat_i}_{cat_j}`` column for each
    (cat_i, cat_j) pair.

    Parameters
    ----------
    X : pd.DataFrame
        Input frame containing every column referenced by ``cat_cols`` /
        ``pairs``.
    cat_cols : sequence of str
        Categorical columns. When ``pairs`` is None, ALL unordered pairs of
        ``cat_cols`` are generated.
    pairs : sequence of (str, str), optional
        Explicit pairs to cross. Overrides the all-pairs default.

    Returns
    -------
    enc_df : pd.DataFrame
        Shape (n, n_pairs), columns named ``cross_{cat_i}_{cat_j}``, integer
        cell codes.
    raw_recipes : dict
        ``{name: {"cat_i": str, "cat_j": str, "mapping": dict[(str,str), int],
                  "n_cells": int}}``. Replay-only payload (no y reference).
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"generate_cat_pair_crosses: X must be a pandas DataFrame; got " f"{type(X).__name__}")
    if len(X) == 0:
        raise ValueError("generate_cat_pair_crosses: X is empty")
    cat_cols = [c for c in cat_cols if c in X.columns]
    if pairs is None:
        pairs = list(combinations(cat_cols, 2))
    else:
        pairs = [(a, b) for (a, b) in pairs if a in X.columns and b in X.columns and a != b]
    encoded: dict[str, np.ndarray] = {}
    raw_recipes: dict[str, dict] = {}
    str_cache: dict[str, np.ndarray] = {}
    for cat_i, cat_j in pairs:
        if cat_i not in str_cache:
            str_cache[cat_i] = _column_to_str(X[cat_i])
        if cat_j not in str_cache:
            str_cache[cat_j] = _column_to_str(X[cat_j])
        codes, mapping = _encode_pair(str_cache[cat_i], str_cache[cat_j])
        name = engineered_name_cat_pair_cross(cat_i, cat_j)
        encoded[name] = codes
        raw_recipes[name] = {
            "cat_i": str(cat_i),
            "cat_j": str(cat_j),
            "mapping": mapping,
            "n_cells": len(mapping),
        }
    enc_df = pd.DataFrame(encoded, index=X.index)
    return enc_df, raw_recipes


# ---------------------------------------------------------------------------
# Interaction-information pre-filter (THE KEY)
# ---------------------------------------------------------------------------


def _bin_target(y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Coerce y to integer class codes (regression -> n_bins quantile bins)."""
    y_arr = np.asarray(y)
    if y_arr.dtype.kind in "iub":
        _, codes = np.unique(y_arr, return_inverse=True)
        return codes.astype(np.int64)
    if y_arr.dtype.kind in "fc" and int(np.unique(y_arr).size) > n_bins:
        q = np.quantile(y_arr.astype(np.float64), np.linspace(0, 1, n_bins + 1))
        q = np.unique(q)
        if q.size < 2:
            return np.zeros(len(y_arr), dtype=np.int64)
        return np.searchsorted(q[1:-1], y_arr.astype(np.float64), side="right").astype(np.int64)
    _, codes = np.unique(y_arr, return_inverse=True)
    return codes.astype(np.int64)


def score_cat_pairs_by_interaction_information(
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: Sequence[str],
    *,
    n_bins: int = 10,
    pairs: Optional[Sequence[tuple[str, str]]] = None,
) -> pd.DataFrame:
    """Score every candidate (cat_i, cat_j) pair by interaction information.

    ``II(cat_i, cat_j; y) = I(cat_i, cat_j; y) - I(cat_i; y) - I(cat_j; y)``.

    Positive II = synergy (materialise); negative II = redundancy (skip).
    Marginal ``I(cat; y)`` is cached per column. The joint
    ``I(cat_i, cat_j; y)`` treats the (cat_i, cat_j) cell code as a single
    high-cardinality discrete variable. The plug-in MI estimator
    (``_adaptive_nbins._plug_in_mi``) is reused so the estimator family matches
    the rest of the FE stack (Layer 19 / Layer 88).

    Returns a frame sorted by ``ii`` descending with columns
    ``[cat_i, cat_j, engineered_col, mi_joint, mi_i, mi_j, ii]``.
    """
    from ._adaptive_nbins import _plug_in_mi

    cat_cols = [c for c in cat_cols if c in X.columns]
    if pairs is None:
        pairs = list(combinations(cat_cols, 2))
    else:
        pairs = [(a, b) for (a, b) in pairs if a in X.columns and b in X.columns and a != b]
    if not pairs:
        return pd.DataFrame(columns=["cat_i", "cat_j", "engineered_col", "mi_joint", "mi_i", "mi_j", "ii"])

    y_bin = _bin_target(y, n_bins=n_bins)

    # O(p^2) -> O(p) + O(p^2)-cheap-lookups (Layer 96): hoist the per-column
    # work (string coercion + factorisation + per-column cardinality + the
    # marginal MI) OUT of the C(p, 2) pair loop and CACHE it. Each cat's dense
    # codes, its ``card + 1`` Horner radix, and its marginal ``I(cat; y)`` are
    # computed exactly ONCE and reused across every pair containing that cat.
    # Only ``I(cat_i, cat_j; y)`` (the joint) is inherently per-pair. Since
    # ``II = I(a,b;y) - I(a;y) - I(b;y)`` this caching removes 2/3 of the MI
    # estimator calls plus all redundant factorisation. CRITICAL: this is a
    # pure caching win -- NO pair is pruned by marginal MI (pure XOR has both
    # marginals ~0 yet large II; a marginal-floor filter would silently drop
    # the one synergy we exist to find). bench (p=30, n=5000, 2026-06-01):
    # naive per-pair recompute ~3036 ms -> cached ~240 ms = 12.6x.
    code_cache: dict[str, np.ndarray] = {}
    radix_cache: dict[str, int] = {}
    marginal_mi: dict[str, float] = {}

    def _codes(col: str) -> np.ndarray:
        """Memoized dense integer codes (and cached cardinality) for column ``col``, computed once and reused across every pair that references it."""
        if col not in code_cache:
            _, c = np.unique(_column_to_str(X[col]), return_inverse=True)
            c = c.astype(np.int64)
            code_cache[col] = c
            radix_cache[col] = (int(c.max()) + 1) if c.size else 1
        return code_cache[col]

    def _mi_col(col: str) -> float:
        """Memoized marginal MI ``I(col; y)``, reused across every pair involving ``col`` so it's computed at most once per column."""
        if col not in marginal_mi:
            marginal_mi[col] = float(_plug_in_mi(_codes(col), y_bin))
        return marginal_mi[col]

    rows = []
    for cat_i, cat_j in pairs:
        mi_i = _mi_col(cat_i)
        mi_j = _mi_col(cat_j)
        # Joint cell code: dense-factorise the (code_i, code_j) tuple stream.
        # Codes are already int64 (no per-pair astype copy); the Horner radix
        # is the cached per-column ``card + 1`` (no per-pair ``cj.max()``).
        ci = _codes(cat_i)
        cj = _codes(cat_j)
        joint_key = ci * radix_cache[cat_j] + cj
        _, joint_codes = np.unique(joint_key, return_inverse=True)
        mi_joint = float(_plug_in_mi(joint_codes, y_bin))
        ii = mi_joint - mi_i - mi_j
        rows.append({
            "cat_i": cat_i,
            "cat_j": cat_j,
            "engineered_col": engineered_name_cat_pair_cross(cat_i, cat_j),
            "mi_joint": mi_joint,
            "mi_i": mi_i,
            "mi_j": mi_j,
            "ii": ii,
        })
    out = pd.DataFrame(rows)
    return out.sort_values("ii", ascending=False, kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Auto-detection of categorical columns
# ---------------------------------------------------------------------------


def auto_detect_cat_pair_cols(
    X: pd.DataFrame,
    *,
    min_card: int = 2,
    max_card: int = 500,
    max_cols: int = 8,
) -> list[str]:
    """Pick categorical-ish columns suitable for crossing.

    Object / category / string dtype columns are always candidates when their
    cardinality is in ``[min_card, max_card]``. Low-cardinality INTEGER columns
    are also promoted (a cat x cat cross is exactly the case where an int-coded
    categorical like ``region`` should participate), matching the Layer 87/88
    int-as-cat detection shape. Float columns are excluded (continuous)."""
    if not isinstance(X, pd.DataFrame):
        return []
    out: list[str] = []
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_float_dtype(s):
            continue
        is_cat_dtype = s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s)
        is_int = pd.api.types.is_integer_dtype(s)
        if not (is_cat_dtype or is_int):
            continue
        try:
            card = int(s.nunique(dropna=True))
        except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed in _cat_pair_fe.py:337: %s", e)
            continue
        if min_card <= card <= max_card:
            out.append(str(col))
    return out[:max_cols]


# ---------------------------------------------------------------------------
# End-to-end pipeline: II pre-filter -> emit cross -> route TE / raw
# ---------------------------------------------------------------------------


def _kfold_target_encode_codes(
    codes: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int = 5,
    smoothing: float = 10.0,
    random_state: int = 0,
) -> tuple[np.ndarray, dict[int, float], float]:
    """K-fold OOF target encoding of an integer code array (Layer 33 discipline
    applied to the cross cell codes). Returns ``(oof_values, lookup, global_mean)``
    where ``lookup`` maps each cell code to its full-data smoothed mean-of-y.
    """
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = len(codes)
    global_mean = float(y_arr.mean()) if n else 0.0
    n_cells = int(codes.max()) + 1 if n else 0

    rng = np.random.default_rng(int(random_state))
    perm = rng.permutation(n)
    fold_ids = np.empty(n, dtype=np.int64)
    fold_ids[perm] = np.arange(n) % int(n_folds)

    oof = np.full(n, global_mean, dtype=np.float64)
    codes64 = codes.astype(np.int64, copy=False)
    for f in range(int(n_folds)):
        train_mask = fold_ids != f
        if not train_mask.any():
            continue
        cell_sum = np.zeros(n_cells, dtype=np.float64)
        cell_cnt = np.zeros(n_cells, dtype=np.float64)
        np.add.at(cell_sum, codes64[train_mask], y_arr[train_mask])
        np.add.at(cell_cnt, codes64[train_mask], 1.0)
        means = np.full(n_cells, global_mean, dtype=np.float64)
        nz = cell_cnt > 0.0
        if nz.any():
            raw = np.where(nz, cell_sum / np.maximum(cell_cnt, 1.0), global_mean)
            shrunk = (cell_cnt * raw + smoothing * global_mean) / (cell_cnt + smoothing)
            means = np.where(nz, shrunk, global_mean)
        test_mask = ~train_mask
        oof[test_mask] = means[codes64[test_mask]]

    full_sum = np.zeros(n_cells, dtype=np.float64)
    full_cnt = np.zeros(n_cells, dtype=np.float64)
    np.add.at(full_sum, codes64, y_arr)
    np.add.at(full_cnt, codes64, 1.0)
    lookup: dict[int, float] = {}
    for c in range(n_cells):
        if full_cnt[c] > 0.0:
            lookup[c] = _smooth(full_sum[c] / full_cnt[c], full_cnt[c], global_mean, smoothing)
        else:
            lookup[c] = global_mean
    return oof, lookup, global_mean


def hybrid_cat_pair_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cat_cols: Optional[Sequence[str]] = None,
    pairs: Optional[Sequence[tuple[str, str]]] = None,
    min_interaction_info: float = 0.001,
    top_k: int = 5,
    n_bins: int = 10,
    n_folds: int = 5,
    smoothing: float = 10.0,
    random_state: int = 0,
):
    """End-to-end cat x cat synergy-cross FE pipeline.

    1. Auto-detect ``cat_cols`` when not supplied.
    2. Compute interaction information ``II(cat_i, cat_j; y)`` for all pairs.
    3. Keep synergistic pairs (``II > min_interaction_info``), top ``top_k`` by II.
    4. For each survivor emit the integer cell cross. Apply the Layer 29
       cardinality pre-screen: if the cross has > ``0.5 * n`` distinct cells,
       route it through K-fold OOF target encoding (Layer 33); otherwise keep
       the raw integer cell code.
    5. Append survivors to X; return ``(X_aug, appended, recipes, scores)``.

    ``y`` is consumed only by the II gate and (for high-card crosses) the OOF
    target encoding. The persisted recipes carry no ``y`` reference, so
    ``transform()`` replay is leakage-free.
    """
    from .engineered_recipes import build_cat_pair_cross_recipe

    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"hybrid_cat_pair_fe: X must be a pandas DataFrame; got {type(X).__name__}")
    if cat_cols is None or len(cat_cols) == 0:
        cat_cols = auto_detect_cat_pair_cols(X)
    else:
        cat_cols = [c for c in cat_cols if c in X.columns]
    if len(cat_cols) < 2 and pairs is None:
        return X.copy(), [], [], pd.DataFrame()

    scores = score_cat_pairs_by_interaction_information(
        X, y, cat_cols, n_bins=n_bins, pairs=pairs,
    )
    if scores.empty:
        return X.copy(), [], [], scores

    keep = scores[scores["ii"] > float(min_interaction_info)]
    keep = keep.head(int(top_k))
    if keep.empty:
        return X.copy(), [], [], scores

    n = len(X)
    new_cols: dict[str, np.ndarray] = {}
    recipes = []
    appended: list[str] = []
    str_cache: dict[str, np.ndarray] = {}
    for _, row in keep.iterrows():
        cat_i = str(row["cat_i"])
        cat_j = str(row["cat_j"])
        name = engineered_name_cat_pair_cross(cat_i, cat_j)
        if cat_i not in str_cache:
            str_cache[cat_i] = _column_to_str(X[cat_i])
        if cat_j not in str_cache:
            str_cache[cat_j] = _column_to_str(X[cat_j])
        codes, mapping = _encode_pair(str_cache[cat_i], str_cache[cat_j])
        n_cells = len(mapping)
        if _cross_too_high_card(n_cells, n):
            # Route through K-fold OOF target encoding (Layer 33).
            oof, te_lookup, global_mean = _kfold_target_encode_codes(
                codes, y, n_folds=n_folds, smoothing=smoothing,
                random_state=random_state,
            )
            new_cols[name] = oof
            recipes.append(build_cat_pair_cross_recipe(
                name=name, cat_i=cat_i, cat_j=cat_j, mapping=mapping,
                encoding="target", te_lookup=te_lookup, global_mean=global_mean,
            ))
        else:
            new_cols[name] = codes.astype(np.float64)
            recipes.append(build_cat_pair_cross_recipe(
                name=name, cat_i=cat_i, cat_j=cat_j, mapping=mapping,
                encoding="raw",
            ))
        appended.append(name)

    if not appended:
        return X.copy(), [], [], scores
    new_df = pd.DataFrame(new_cols, index=X.index)
    X_aug = pd.concat([X, new_df], axis=1)
    return X_aug, appended, recipes, scores


# ---------------------------------------------------------------------------
# Transform-time replay
# ---------------------------------------------------------------------------


def apply_cat_pair_cross(
    X_test: pd.DataFrame,
    cat_i: str,
    cat_j: str,
    mapping: dict,
    *,
    encoding: str = "raw",
    te_lookup: Optional[dict] = None,
    global_mean: float = 0.0,
) -> np.ndarray:
    """Replay a cat-pair cross: look up each test row's (cat_i_val, cat_j_val)
    tuple in the stored ``mapping``.

    * ``encoding='raw'``: emit the integer cell code; unseen tuples -> sentinel
      code ``len(mapping)`` (a fresh bin distinct from any seen cell), as a float.
    * ``encoding='target'``: emit the per-cell mean-of-y from ``te_lookup``;
      unseen tuples (and seen-tuple codes absent from the lookup) -> ``global_mean``.

    No y reference at replay -- pure function of X.
    """
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_cat_pair_cross: X_test must be a DataFrame; got " f"{type(X_test).__name__}")
    cats_i = np.asarray(_column_to_str(X_test[cat_i]))
    cats_j = np.asarray(_column_to_str(X_test[cat_j]))
    n = len(cats_i)
    sentinel = len(mapping)
    lookup = te_lookup or {}

    def _value_for_pair(si, sj):
        """Per-row cross-cell value for a category pair: target-encoded lookup (falling back to the global mean for unseen pairs/codes) when ``encoding=="target"``, else the raw joint-cell integer code (sentinel for unseen pairs)."""
        # The exact per-row semantics, factored out so the vectorized and
        # fallback paths are provably identical.
        if encoding == "target":
            code = mapping.get((si, sj))
            return global_mean if code is None else float(lookup.get(code, global_mean))
        return float(mapping.get((si, sj), sentinel))

    if n == 0:
        return np.empty(0, dtype=np.float64)
    # bench-attempt-rejected (2026-06-02, n=200k, ki=kj=15): a pd.factorize-fold
    # vectorization (factorize cat_i + cat_j, fold the two dense codes into one
    # int key, factorize that, resolve mapping.get per distinct pair, gather)
    # measured 0.9x -- slightly SLOWER. The two 200k string-column factorizes cost
    # as much as the per-row tuple dict.get they replace, and the fold + combined
    # factorize add a third hashing pass that cancels the dedup benefit. Unlike
    # the single-column count/frequency encoders (one factorize replaces the
    # per-row get for a clean ~6x), the two-column tuple key has no cheap dense
    # factorize. Keep the per-row dict.get -- but iterate via ``.tolist()`` + zip:
    # ``.tolist()`` boxes each object array to Python objects in ONE C pass, so the
    # loop yields pre-boxed objects instead of paying a per-row numpy __getitem__
    # boxing on ``cats_i[r]`` / ``cats_j[r]``. Bit-identical (same values, order,
    # dict.get). bench (bench_cat_pair_cross_replay_dedup, 2026-06-23): ~1.17-1.22x
    # across n=5k..1M, ki/kj=15/50.
    return np.array(
        [_value_for_pair(si, sj) for si, sj in zip(cats_i.tolist(), cats_j.tolist())],
        dtype=np.float64,
    )
