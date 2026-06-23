"""Cat x cat x cat TRIPLE synergy cross via beam search (Layer 94, 2026-06-01).

Layer 89 crosses PAIRS of categoricals and pre-filters them by pairwise
interaction information ``II(a, b; y) = I(a, b; y) - I(a; y) - I(b; y)``. That
catches two-way synergy (the categorical XOR ``y = a XOR b``) but is blind to
GENUINE three-way synergy: the parity target ``y = a XOR b XOR c`` where EVERY
pair is marginally AND pairwise uninformative (``II(a, b; y) ~ II(a, c; y) ~
II(b, c; y) ~ 0``) yet the triple is fully predictive. The signal lives only at
the third order.

The three-way interaction information (co-information; McGill 1954) closes that
hole:

    II3(a, b, c; y) = I(a, b, c; y)
                      - [I(a, b; y) + I(a, c; y) + I(b, c; y)]
                      + [I(a; y) + I(b; y) + I(c; y)]

``II3 > 0`` = genuine three-way synergy that NO pair or single explains -- the
triple worth materialising. ``II3 ~ 0`` = no third-order structure. (Sign of
co-information is order-dependent in the general lattice, but for the synergy
question we ask -- "does the joint triple tell more than every pair predicts" --
a strongly positive II3 with near-zero pairwise II is the unambiguous parity
signature.)

Beam search (THE COST CONTROL)
------------------------------

Scoring all ``C(p, 3)`` triples is cubic. Instead we BEAM-search from the
Layer-89 pairwise-synergy ranking: take the top-K most-synergistic PAIRS (by
pairwise II), then extend EACH retained pair by the single best third cat, keep
a beam of width ``W``. Candidate triples evaluated is bounded by ``W * p`` (one
sweep of all remaining cats per beam slot) rather than ``C(p, 3)``. For the pure
three-way XOR the top pairwise II are near-zero AND near-tied, so the beam seeds
from arbitrary pairs -- but because the THIRD-cat sweep evaluates II3 (not
pairwise II), the genuine triple still surfaces: adding the missing parity
member is the only extension that drives II3 strongly positive.

Cardinality + TE (Layers 29 / 33)
---------------------------------

A triple cross can have up to ``card_a * card_b * card_c`` cells -- it explodes
faster than a pair. The Layer 29 pre-screen still applies: a cross with more
than ``0.5 * n`` distinct cells is refused as a raw factorize feature (cells
near-unique -> memorisation) and routed through K-fold OOF target encoding
(Layer 33), collapsing it to one dense numeric column. Low-card triples emit the
raw integer cell code.

Recipe replay
-------------

Recipe kind ``cat_triple_cross`` stores the (a, b, c) triple plus the fit-time
``(val_a, val_b, val_c) -> code`` mapping (and, for TE, the per-code mean-of-y
lookup + global mean). Replay reads ONLY X -- each test row's value-triple is
looked up; unseen triples map to a sentinel code (raw) or the global mean (TE).
No ``y`` reference is captured, so ``transform()`` is leakage-free.

NOT wired into ``MRMR.fit`` by default -- opt-in via
``fe_cat_triple_enable=True``.
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._target_encoding_fe import _column_to_str
from ._cat_pair_fe import (
    _bin_target,
    _cross_too_high_card,
    _kfold_target_encode_codes,
    score_cat_pairs_by_interaction_information,
)

logger = logging.getLogger(__name__)

__all__ = [
    "engineered_name_cat_triple_cross",
    "triple_interaction_information",
    "generate_cat_triple_crosses",
    "score_cat_triples_by_interaction_information",
    "hybrid_cat_triple_fe",
    "apply_cat_triple_cross",
]


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def engineered_name_cat_triple_cross(cat_a: str, cat_b: str, cat_c: str) -> str:
    """Stable engineered column name for the (cat_a, cat_b, cat_c) cross."""
    return f"cross3_{cat_a}_{cat_b}_{cat_c}"


# ---------------------------------------------------------------------------
# Discrete-MI primitives (reuse Layer 60 / Layer 19 plug-in family)
# ---------------------------------------------------------------------------


def _dense_codes(values) -> np.ndarray:
    """Dense int64 class codes for a stringified categorical column."""
    _, codes = np.unique(values, return_inverse=True)
    return codes.astype(np.int64)


def _join_codes(*code_arrays: np.ndarray) -> np.ndarray:
    """Dense int64 cell code for the joint of several integer code arrays.

    Horner-packs the per-column codes into a single key (multiply by per-column
    ``max+1``) then dense-renumbers via ``np.unique`` -- the cartesian space is
    never materialised, only the occupied cells survive.
    """
    if not code_arrays:
        return np.zeros(0, dtype=np.int64)
    key = np.zeros(len(code_arrays[0]), dtype=np.int64)
    for c in code_arrays:
        c64 = np.asarray(c, dtype=np.int64)
        cmax = int(c64.max()) + 1 if c64.size else 1
        key = key * cmax + c64
    _, codes = np.unique(key, return_inverse=True)
    return codes.astype(np.int64)


def triple_interaction_information(
    a, b, c, y, *, n_bins: int = 10,
) -> float:
    """Three-way interaction information (co-information) ``II3(a, b, c; y)``.

        II3 = I(a, b, c; y)
              - [I(a, b; y) + I(a, c; y) + I(b, c; y)]
              + [I(a; y) + I(b; y) + I(c; y)]

    ``a``, ``b``, ``c`` are categorical columns (object / int / string); ``y`` is
    the discrete target (regression y is quantile-binned via :func:`_bin_target`).
    Each categorical is dense-coded; joints are computed by renumbering the
    Horner-packed cell key so the cartesian product is never materialised. The
    plug-in MI estimator (Layer 19 / Layer 60 family) is reused for every term so
    the numbers are directly comparable to the Layer 89 pairwise II.

    Positive II3 = genuine three-way synergy not explained by any pair or single
    (the parity / XOR3 signature). Returns the value in nats.
    """
    from ._adaptive_nbins import _plug_in_mi

    ca = _dense_codes(_column_to_str(pd.Series(np.asarray(a))))
    cb = _dense_codes(_column_to_str(pd.Series(np.asarray(b))))
    cc = _dense_codes(_column_to_str(pd.Series(np.asarray(c))))
    y_bin = _bin_target(y, n_bins=n_bins)

    mi_a = float(_plug_in_mi(ca, y_bin))
    mi_b = float(_plug_in_mi(cb, y_bin))
    mi_c = float(_plug_in_mi(cc, y_bin))
    mi_ab = float(_plug_in_mi(_join_codes(ca, cb), y_bin))
    mi_ac = float(_plug_in_mi(_join_codes(ca, cc), y_bin))
    mi_bc = float(_plug_in_mi(_join_codes(cb, cc), y_bin))
    mi_abc = float(_plug_in_mi(_join_codes(ca, cb, cc), y_bin))

    return mi_abc - (mi_ab + mi_ac + mi_bc) + (mi_a + mi_b + mi_c)


# ---------------------------------------------------------------------------
# Triple-cross materialisation -- pure X-only kernel (no y reference)
# ---------------------------------------------------------------------------


def _encode_triple(
    cats_a: np.ndarray, cats_b: np.ndarray, cats_c: np.ndarray,
) -> tuple[np.ndarray, dict[tuple, int]]:
    """Factorise the (val_a, val_b, val_c) tuple stream into a dense int code
    per row. Returns ``(codes, mapping)`` where ``mapping`` maps each fit-time
    value-triple to its int cell code (replay-only payload).

    Codes are assigned in sorted-triple order (``np.unique`` sorts); absolute
    code values are arbitrary (consumed as a categorical / target-encoded) and
    replay maps via the stored value-triple lookup, so the ordering convention
    does not affect correctness.
    """
    n = len(cats_a)
    if n == 0:
        return np.empty(0, dtype=np.int64), {}
    uniq_a, inv_a = np.unique(cats_a, return_inverse=True)
    uniq_b, inv_b = np.unique(cats_b, return_inverse=True)
    uniq_c, inv_c = np.unique(cats_c, return_inverse=True)
    nb = len(uniq_b)
    nc = len(uniq_c)
    combined = (
        inv_a.astype(np.int64) * (nb * nc)
        + inv_b.astype(np.int64) * nc
        + inv_c.astype(np.int64)
    )
    uniq_keys, codes = np.unique(combined, return_inverse=True)
    codes = codes.astype(np.int64, copy=False)
    mapping: dict[tuple, int] = {}
    for code, key in enumerate(uniq_keys):
        k = int(key)
        ia = k // (nb * nc)
        rem = k % (nb * nc)
        ib = rem // nc
        ic = rem % nc
        mapping[(uniq_a[ia], uniq_b[ib], uniq_c[ic])] = code
    return codes, mapping


def generate_cat_triple_crosses(
    X: pd.DataFrame,
    cat_cols: Sequence[str],
    triples: Optional[Sequence[tuple[str, str, str]]] = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Materialise an integer-coded ``cross3_{a}_{b}_{c}`` column per triple.

    When ``triples`` is None, ALL unordered triples of ``cat_cols`` are
    generated. Returns ``(enc_df, raw_recipes)`` where ``raw_recipes`` maps each
    engineered name to ``{"cat_a", "cat_b", "cat_c", "mapping", "n_cells"}`` --
    the replay-only payload (no y reference).
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"generate_cat_triple_crosses: X must be a pandas DataFrame; got "
            f"{type(X).__name__}"
        )
    if len(X) == 0:
        raise ValueError("generate_cat_triple_crosses: X is empty")
    cat_cols = [c for c in cat_cols if c in X.columns]
    if triples is None:
        triples = list(combinations(cat_cols, 3))
    else:
        triples = [
            (a, b, c) for (a, b, c) in triples
            if a in X.columns and b in X.columns and c in X.columns
            and len({a, b, c}) == 3
        ]
    encoded: dict[str, np.ndarray] = {}
    raw_recipes: dict[str, dict] = {}
    str_cache: dict[str, np.ndarray] = {}

    def _strs(col: str) -> np.ndarray:
        if col not in str_cache:
            str_cache[col] = _column_to_str(X[col])
        return str_cache[col]

    for cat_a, cat_b, cat_c in triples:
        codes, mapping = _encode_triple(_strs(cat_a), _strs(cat_b), _strs(cat_c))
        name = engineered_name_cat_triple_cross(cat_a, cat_b, cat_c)
        encoded[name] = codes
        raw_recipes[name] = {
            "cat_a": str(cat_a),
            "cat_b": str(cat_b),
            "cat_c": str(cat_c),
            "mapping": mapping,
            "n_cells": int(len(mapping)),
        }
    enc_df = pd.DataFrame(encoded, index=X.index)
    return enc_df, raw_recipes


# ---------------------------------------------------------------------------
# Beam search over triples seeded by Layer-89 pairwise synergy
# ---------------------------------------------------------------------------


def score_cat_triples_by_interaction_information(
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: Sequence[str],
    *,
    n_bins: int = 10,
    beam_width: int = 3,
    top_k_pairs: int = 3,
    n_rounds: int = 2,
) -> pd.DataFrame:
    """Beam-search the triple lattice for three-way synergy.

    1. Rank all PAIRS by pairwise II (Layer 89
       :func:`score_cat_pairs_by_interaction_information`) and seed the beam with
       the top ``top_k_pairs`` pairs.
    2. ROUND: for each seed pair ``(a, b)``, sweep every remaining cat ``c`` and
       compute II3 ``(a, b, c; y)``; record every evaluated triple.
    3. REFINE: re-seed the next round from the SUB-PAIRS of the ``beam_width``
       best triples found so far, then repeat for ``n_rounds`` rounds.
    4. Return all evaluated triples with their II3, sorted descending.

    The refinement step is what lets the beam escape the parity trap: under a
    pure three-way XOR target, EVERY pairwise II is finite-sample noise, so the
    pairwise seeds (step 1) rarely contain two of the three signal cats. But once
    round 1 surfaces a partial triple whose II3 is lifted by containing two
    signal members, re-seeding round 2 from that triple's sub-pairs pulls in the
    missing third member -- driving II3 strongly positive. Candidate triples
    evaluated is bounded by ``n_rounds * max(top_k_pairs, beam_width) * (p - 2)``
    (a third-cat sweep per seed pair per round, deduplicated) rather than
    ``C(p, 3)``. The returned frame carries ``n_triples_evaluated`` /
    ``n_triples_exhaustive`` on ``.attrs`` for the efficiency contract.

    Columns: ``[cat_a, cat_b, cat_c, engineered_col, ii3]`` sorted by ``ii3``
    descending.
    """
    cat_cols = [c for c in cat_cols if c in X.columns]
    empty = pd.DataFrame(
        columns=["cat_a", "cat_b", "cat_c", "engineered_col", "ii3"]
    )
    if len(cat_cols) < 3:
        empty.attrs["n_triples_evaluated"] = 0
        empty.attrs["n_triples_exhaustive"] = 0
        return empty

    pair_scores = score_cat_pairs_by_interaction_information(
        X, y, cat_cols, n_bins=n_bins,
    )
    if pair_scores.empty:
        empty.attrs["n_triples_evaluated"] = 0
        empty.attrs["n_triples_exhaustive"] = 0
        return empty

    # Beam seeds (round 1): top pairs by pairwise II (already sorted descending).
    seed_pairs = [
        (str(r["cat_i"]), str(r["cat_j"]))
        for _, r in pair_scores.head(int(top_k_pairs)).iterrows()
    ]

    y_bin = _bin_target(y, n_bins=n_bins)

    # Cache dense codes + the discrete-MI primitive so repeated sweeps do not
    # recompute marginals / joints per candidate.
    code_cache: dict[str, np.ndarray] = {}

    def _codes(col: str) -> np.ndarray:
        if col not in code_cache:
            code_cache[col] = _dense_codes(_column_to_str(X[col]))
        return code_cache[col]

    from ._adaptive_nbins import _plug_in_mi

    mi_cache: dict[frozenset, float] = {}

    def _mi(cols: tuple[str, ...]) -> float:
        key = frozenset(cols)
        if key not in mi_cache:
            joint = _join_codes(*(_codes(c) for c in cols))
            mi_cache[key] = float(_plug_in_mi(joint, y_bin))
        return mi_cache[key]

    def _ii3(a: str, b: str, c: str) -> float:
        return (
            _mi((a, b, c))
            - (_mi((a, b)) + _mi((a, c)) + _mi((b, c)))
            + (_mi((a,)) + _mi((b,)) + _mi((c,)))
        )

    seen: set[frozenset] = set()
    triple_ii3: dict[frozenset, float] = {}
    n_evaluated = 0
    for _round in range(max(1, int(n_rounds))):
        if not seed_pairs:
            break
        for (a, b) in seed_pairs:
            for c in cat_cols:
                if c == a or c == b:
                    continue
                trip = frozenset((a, b, c))
                if trip in seen:
                    continue
                seen.add(trip)
                triple_ii3[trip] = _ii3(a, b, c)
                n_evaluated += 1
        # Refine: next round seeds = the sub-pairs of the best beam_width
        # triples found so far. This re-seeds the search from the strongest
        # partial-synergy neighbourhoods (the parity-escape step).
        best = sorted(
            triple_ii3.items(), key=lambda kv: kv[1], reverse=True,
        )[: max(1, int(beam_width))]
        next_seeds: list[tuple[str, str]] = []
        for trip, _ in best:
            for pr in combinations(sorted(trip), 2):
                if pr not in next_seeds:
                    next_seeds.append(pr)
        seed_pairs = next_seeds

    rows: list[dict] = []
    for trip, ii3 in triple_ii3.items():
        a_s, b_s, c_s = sorted(trip)
        rows.append({
            "cat_a": a_s,
            "cat_b": b_s,
            "cat_c": c_s,
            "engineered_col": engineered_name_cat_triple_cross(a_s, b_s, c_s),
            "ii3": float(ii3),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = (
            out.sort_values("ii3", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )
    n_exhaustive = len(list(combinations(cat_cols, 3)))
    out.attrs["n_triples_evaluated"] = int(n_evaluated)
    out.attrs["n_triples_exhaustive"] = int(n_exhaustive)
    return out


# ---------------------------------------------------------------------------
# End-to-end pipeline: beam -> II3 filter -> emit cross -> route TE / raw
# ---------------------------------------------------------------------------


def hybrid_cat_triple_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cat_cols: Optional[Sequence[str]] = None,
    min_interaction_info: float = 0.001,
    top_k: int = 3,
    beam_width: int = 3,
    top_k_pairs: int = 3,
    n_bins: int = 10,
    n_folds: int = 5,
    smoothing: float = 10.0,
    random_state: int = 0,
):
    """End-to-end cat x cat x cat triple-synergy-cross FE pipeline.

    1. Auto-detect ``cat_cols`` when not supplied (Layer 89 detector).
    2. Beam-search the triple lattice for three-way interaction information
       (:func:`score_cat_triples_by_interaction_information`).
    3. Keep synergistic triples (``II3 > min_interaction_info``), top ``top_k``.
    4. For each survivor emit the integer cell cross. Apply the Layer 29
       cardinality pre-screen: a cross with > ``0.5 * n`` distinct cells routes
       through K-fold OOF target encoding (Layer 33); otherwise the raw integer
       cell code is kept.
    5. Append survivors to X; return ``(X_aug, appended, recipes, scores)``.

    ``y`` is consumed only by the II3 gate and (for high-card crosses) the OOF
    target encoding. The persisted recipes carry no ``y`` reference, so
    ``transform()`` replay is leakage-free.
    """
    from .engineered_recipes import build_cat_triple_cross_recipe

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"hybrid_cat_triple_fe: X must be a pandas DataFrame; got "
            f"{type(X).__name__}"
        )
    if cat_cols is None or len(cat_cols) == 0:
        from ._cat_pair_fe import auto_detect_cat_pair_cols
        cat_cols = auto_detect_cat_pair_cols(X)
    else:
        cat_cols = [c for c in cat_cols if c in X.columns]
    if len(cat_cols) < 3:
        return X.copy(), [], [], pd.DataFrame()

    scores = score_cat_triples_by_interaction_information(
        X, y, cat_cols, n_bins=n_bins,
        beam_width=beam_width, top_k_pairs=top_k_pairs,
    )
    if scores.empty:
        return X.copy(), [], [], scores

    keep = scores[scores["ii3"] > float(min_interaction_info)]
    keep = keep.head(int(top_k))
    if keep.empty:
        return X.copy(), [], [], scores

    n = len(X)
    new_cols: dict[str, np.ndarray] = {}
    recipes = []
    appended: list[str] = []
    str_cache: dict[str, np.ndarray] = {}

    def _strs(col: str) -> np.ndarray:
        if col not in str_cache:
            str_cache[col] = _column_to_str(X[col])
        return str_cache[col]

    for _, row in keep.iterrows():
        cat_a = str(row["cat_a"])
        cat_b = str(row["cat_b"])
        cat_c = str(row["cat_c"])
        name = engineered_name_cat_triple_cross(cat_a, cat_b, cat_c)
        codes, mapping = _encode_triple(_strs(cat_a), _strs(cat_b), _strs(cat_c))
        n_cells = len(mapping)
        if _cross_too_high_card(n_cells, n):
            oof, te_lookup, global_mean = _kfold_target_encode_codes(
                codes, y, n_folds=n_folds, smoothing=smoothing,
                random_state=random_state,
            )
            new_cols[name] = oof
            recipes.append(build_cat_triple_cross_recipe(
                name=name, cat_a=cat_a, cat_b=cat_b, cat_c=cat_c, mapping=mapping,
                encoding="target", te_lookup=te_lookup, global_mean=global_mean,
            ))
        else:
            new_cols[name] = codes.astype(np.float64)
            recipes.append(build_cat_triple_cross_recipe(
                name=name, cat_a=cat_a, cat_b=cat_b, cat_c=cat_c, mapping=mapping,
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


def apply_cat_triple_cross(
    X_test: pd.DataFrame,
    cat_a: str,
    cat_b: str,
    cat_c: str,
    mapping: dict,
    *,
    encoding: str = "raw",
    te_lookup: Optional[dict] = None,
    global_mean: float = 0.0,
) -> np.ndarray:
    """Replay a cat-triple cross: look up each test row's value-triple in
    ``mapping``.

    * ``encoding='raw'``: emit the integer cell code; unseen triples -> sentinel
      code ``len(mapping)`` (a fresh bin distinct from any seen cell), as float.
    * ``encoding='target'``: emit the per-cell mean-of-y from ``te_lookup``;
      unseen triples (and seen codes absent from the lookup) -> ``global_mean``.

    No y reference at replay -- pure function of X.
    """
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            f"apply_cat_triple_cross: X_test must be a DataFrame; got "
            f"{type(X_test).__name__}"
        )
    cats_a = np.asarray(_column_to_str(X_test[cat_a]))
    cats_b = np.asarray(_column_to_str(X_test[cat_b]))
    cats_c = np.asarray(_column_to_str(X_test[cat_c]))
    n = len(cats_a)
    sentinel = len(mapping)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    lookup = te_lookup or {}
    is_target = encoding == "target"
    # Vectorized replay: factorize the three string columns (O(n) hashtable each, no sort), pack into a single joint
    # key, then resolve the (val_a, val_b, val_c) -> output value ONCE per DISTINCT joint cell and gather by code.
    # Bit-identical to the per-row mapping.get loop (same mapping/sentinel/te_lookup/global_mean per distinct triple).
    # Mirrors the count/freq/cat_num factorize replay paths; _column_to_str maps NaN -> "__nan__" so factorize never
    # emits its -1 sentinel here.
    codes_a, uniq_a = pd.factorize(cats_a)
    codes_b, uniq_b = pd.factorize(cats_b)
    codes_c, uniq_c = pd.factorize(cats_c)
    nb = len(uniq_b)
    nc = len(uniq_c)
    joint = (codes_a.astype(np.int64) * (nb * nc) + codes_b.astype(np.int64) * nc + codes_c.astype(np.int64))
    cell_codes, joint_uniques = pd.factorize(joint)
    vals = np.empty(len(joint_uniques), dtype=np.float64)
    for i, k in enumerate(joint_uniques):
        k = int(k)
        ia = k // (nb * nc)
        rem = k % (nb * nc)
        ib = rem // nc
        ic = rem % nc
        triple = (uniq_a[ia], uniq_b[ib], uniq_c[ic])
        if is_target:
            code = mapping.get(triple)
            vals[i] = global_mean if code is None else float(lookup.get(code, global_mean))
        else:
            vals[i] = float(mapping.get(triple, sentinel))
    return vals[cell_codes]
