"""K-way expansion + pair/k-way materialisation for ``cat_interactions``.

Split out of ``cat_interactions.py`` to keep the parent below the 1k-line
monolith threshold. Behaviour preserved bit-for-bit; the parent re-exports
the public-looking entry points so the orchestrator in
``run_cat_interaction_step`` continues to call them via the same names.

What lives here:
  - ``_greedy_expand_one_seed`` (synergy-seeded k-way growth)
  - ``_build_kway_chained_lookup`` / ``_materialize_kway``
  - ``_select_top_k_pairs``
  - ``_build_factorize_lookup`` / ``_materialize_pairs``
"""
from __future__ import annotations

import bisect
from typing import Literal, Optional

import numpy as np
from numba import njit

from .cat_fe_state import CatFEConfig
from .engineered_recipes import EngineeredRecipe
from .info_theory import compute_mi_from_classes, compute_mi_from_classes_weighted, merge_vars
# ``resolve_min_interaction_information`` lives in ``cat_interactions`` itself;
# imported lazily inside the function body to dodge the
# ``cat_interactions -> _cat_kway_materialize -> cat_interactions`` import cycle
# that an eager import would trigger.


def _build_merge_prefix_states(
    factors_data: np.ndarray,
    sorted_members: list,
    factors_nbins: np.ndarray,
    dtype,
    final_state: tuple | None = None,
) -> list:
    """Incremental ``merge_vars`` states after merging ``sorted_members[:i]`` (ascending order), for every ``i`` in ``0..len(sorted_members)``.

    ``merge_vars``'s dense renumbering is ORDER-SENSITIVE: merging the same variable set in a different order yields a bijective but numerically DIFFERENT
    ``final_classes`` encoding (verified empirically -- only the count of distinct classes is order-invariant, not the labels). A candidate variable inserted
    at some position among ``sorted_members`` therefore needs the merge to walk ``sorted_members[:pos] + [cand] + sorted_members[pos:]`` to stay bit-identical
    to a fresh ``merge_vars`` over the fully re-sorted tuple -- these prefix states let ``_merge_vars_sorted_insert`` splice a candidate in at its correct
    sorted position without re-scanning the members before it, for every candidate sharing that same insertion point.

    ``final_state`` lets the caller hand in an already-computed ``(classes, nclasses)`` for the FULL ``sorted_members`` merge (the parent state carried from
    the previous greedy-expansion round) instead of re-deriving it here.
    """
    n_rows = factors_data.shape[0]
    states = [(np.zeros(n_rows, dtype=dtype), 1)]
    upto = len(sorted_members) - 1 if final_state is not None else len(sorted_members)
    for i in range(1, upto + 1):
        prev_classes, prev_nclasses = states[-1]
        classes_i, _freqs_i, nclasses_i = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([sorted_members[i - 1]], dtype=np.int64),
            var_is_nominal=None, factors_nbins=factors_nbins,
            current_nclasses=prev_nclasses, final_classes=prev_classes.copy(), dtype=dtype,
        )
        states.append((classes_i, nclasses_i))
    if final_state is not None:
        states.append(final_state)
    return states


def _merge_vars_sorted_insert(
    factors_data: np.ndarray,
    prefix_states: list,
    sorted_members: list,
    cand_int: int,
    factors_nbins: np.ndarray,
    dtype,
) -> tuple:
    """``merge_vars`` over ``sorted(sorted_members + [cand_int])``, splicing ``cand_int`` into its correct sorted position via ``prefix_states`` instead of
    re-scanning the members before it. Bit-identical to a fresh full-tuple ``merge_vars`` call (verified end-to-end against the pre-fix algorithm across
    randomized trials incl. min/mid/max insertion positions and varying arities/cardinalities/row-counts -- see
    ``_benchmarks/bench_greedy_expand_seed_frozen_prefix.py``)."""
    ins = bisect.bisect_left(sorted_members, cand_int)
    prefix_classes, prefix_nclasses = prefix_states[ins]
    classes1, freqs1, nclasses1 = merge_vars(
        factors_data=factors_data, vars_indices=np.array([cand_int], dtype=np.int64),
        var_is_nominal=None, factors_nbins=factors_nbins,
        current_nclasses=prefix_nclasses, final_classes=prefix_classes.copy(), dtype=dtype,
    )
    suffix = sorted_members[ins:]
    if not suffix:
        return classes1, freqs1, nclasses1
    classes2, freqs2, nclasses2 = merge_vars(
        factors_data=factors_data, vars_indices=np.array(suffix, dtype=np.int64),
        var_is_nominal=None, factors_nbins=factors_nbins,
        current_nclasses=nclasses1, final_classes=classes1, dtype=dtype,
    )
    return classes2, freqs2, nclasses2


@njit(cache=True)
def _scatter_factorize_lookup(
    factors_data: np.ndarray,
    idx_a: int,
    idx_b: int,
    nbins_a: int,
    nbins_b: int,
    classes_pair_post: np.ndarray,
) -> np.ndarray:
    """Single-pass scatter of post-prune classes into the pre-prune code lookup.

    Builds ``lookup[a_val + b_val * nbins_a] = post_prune_class`` for every row,
    last-write-wins on duplicate codes -- identical to ``lookup[codes] = classes``
    numpy fancy-index assignment (which writes in ascending row order). Avoids the
    three length-n int64 temporaries (``vals_a``, ``vals_b``, ``pre_prune_codes``)
    the numpy form allocated per call. Returns the int64 lookup with -1 sentinel
    for codes never seen in the training data.
    """
    lookup = np.full(nbins_a * nbins_b, -1, dtype=np.int64)
    n = factors_data.shape[0]
    for r in range(n):
        code = factors_data[r, idx_a] + factors_data[r, idx_b] * nbins_a
        lookup[code] = classes_pair_post[r]
    return lookup


@njit(cache=True)
def _dense_renumber_codes(codes: np.ndarray, expected_size: int) -> tuple:
    """Prune empty bins out of a pre-combined joint code array and densely renumber the survivors, mirroring ``merge_vars``'s own per-step
    prune-then-renumber (same bincount + ascending-oldclass lookup-table scheme) but applied directly to an ALREADY-COMBINED single code array instead of
    accumulating it across multiple raw columns first.

    Used by ``_build_kway_chained_lookup``'s chain steps: ``codes`` there is ``pre_prune_codes`` (the growing prefix's running classes combined with the
    next raw column), which already carries everything a full ``merge_vars(idx_tuple[:step+1])`` call would re-derive from scratch. Bit-identical to
    ``merge_vars`` (verified: ``_benchmarks/bench_kway_chained_lookup_unique.py``) and faster (single O(n) pass over the pre-combined codes instead of
    re-walking every prefix column) -- an ``np.unique(codes, return_inverse=True)`` alternative was tried first and measured 3-4x SLOWER than
    ``merge_vars`` itself (sort-based dedup loses to njit's direct bincount at these join cardinalities), so it was rejected in favour of this kernel.
    """
    freqs = np.zeros(expected_size, dtype=np.int64)
    for i in range(codes.shape[0]):
        freqs[codes[i]] += 1
    nzeros = 0
    lookup_table = np.empty(expected_size, dtype=np.int64)
    for oldclass in range(expected_size):
        if freqs[oldclass] == 0:
            nzeros += 1
        lookup_table[oldclass] = oldclass - nzeros
    out = np.empty(codes.shape[0], dtype=np.int64)
    for i in range(codes.shape[0]):
        out[i] = lookup_table[codes[i]]
    return out, expected_size - nzeros


# ============================================================================
# K-way greedy expansion
#
# After top-K pairs are confirmed, greedily extend each surviving pair by ONE variable at a time up to ``cfg.max_kway_order``. For each candidate extension k:
#
#   delta_II = I(parent ∪ {k}; Y) - I(parent; Y) - I(X_k; Y)
#
# This is the 3-way Jakulin II between (parent_aggregate, X_k, Y) -- it measures whether adding X_k to the merged parent contributes information BEYOND what the parent
# and X_k separately give. Positive delta means X_k is genuinely synergistic with the parent group; <= 0 means X_k is redundant given parent.
#
# Naming: "incremental_interaction_information". NOT identical to higher-order Jakulin II (which is a 15-term inclusion-exclusion sum); closer to JMI (Yang & Moody 1999)
# than to CMIM (Fleuret 2004).
#
# Cost: O(top_k_pairs * (max_kway_order - 2) * N) merge_vars calls. The orchestrator caps ``max_kway_order`` to a small int (default 2 -> skip; typical use 3..5). Each
# greedy extension calls merge_vars once per candidate var.
# ============================================================================


def _greedy_expand_one_seed(
    factors_data: np.ndarray,
    seed_indices: tuple,  # (idx_a, idx_b) -- the seed pair
    candidate_pool: np.ndarray,  # indices eligible for extension
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    marginal_mi: np.ndarray,
    max_combined_nbins: int,
    max_kway_order: int,
    min_inc_ii: float,
    dtype,
    weights: Optional[np.ndarray] = None,
) -> tuple | None:
    """Greedily extend ``seed_indices`` up to ``max_kway_order`` by picking the variable with the largest incremental II at each step.

    Returns ``(final_indices_tuple, final_classes, final_n_uniq, final_joint_mi)`` or ``None`` if no extension cleared ``min_inc_ii`` (in which case the seed pair
    itself remains the best, no k-way emitted).

    Stops early when: no candidate var clears ``min_inc_ii`` (greedy local max reached); adding any candidate would violate the cardinality budget; or order reaches ``max_kway_order``.

    ``weights``, when given, route every joint MI through the weighted
    kernel so the k-way greedy expansion grows the SAME weighted seed the search phase found, instead
    of re-scoring extensions against an unweighted joint MI.
    """
    parent_set = set(seed_indices)
    parent_vi = np.array(sorted(parent_set), dtype=np.int64)
    parent_classes, parent_freqs, parent_nclasses = merge_vars(
        factors_data=factors_data, vars_indices=parent_vi,
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    if weights is not None:
        parent_mi = compute_mi_from_classes_weighted(parent_classes, classes_y, weights, dtype)
    else:
        parent_mi = compute_mi_from_classes(
            classes_x=parent_classes, freqs_x=parent_freqs,
            classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
        )

    for _order in range(len(parent_set) + 1, max_kway_order + 1):
        best_inc_ii = -np.inf
        best_var: int = -1
        best_classes = None
        best_nclasses = 0
        best_joint_mi = 0.0

        # ``parent_set``/``parent_classes``/``parent_nclasses`` are fixed for this WHOLE candidate sweep --
        # only the single best candidate gets accepted, and only AFTER the sweep finishes -- so the prefix
        # states can be built once here and reused for every candidate instead of re-merging the full parent
        # tuple from raw columns each time.
        parent_sorted = sorted(parent_set)
        prefix_states = _build_merge_prefix_states(
            factors_data, parent_sorted, nbins, dtype, final_state=(parent_classes, parent_nclasses),
        )

        for k in candidate_pool:
            k_int = int(k)
            if k_int in parent_set:
                continue
            new_card_estimate = parent_nclasses * int(nbins[k_int])
            if new_card_estimate > max_combined_nbins:
                continue
            if new_card_estimate >= 2**31:
                continue

            new_classes, new_freqs, new_n = _merge_vars_sorted_insert(
                factors_data, prefix_states, parent_sorted, k_int, nbins, dtype,
            )
            if weights is not None:
                new_joint_mi = compute_mi_from_classes_weighted(new_classes, classes_y, weights, dtype)
            else:
                new_joint_mi = compute_mi_from_classes(
                    classes_x=new_classes, freqs_x=new_freqs,
                    classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
                )
            # incremental II = new_joint_mi - parent_mi - marginal_mi[k] = how much extra info adding X_k brings, BEYOND what parent and X_k separately contribute.
            inc_ii = new_joint_mi - parent_mi - float(marginal_mi[k_int])
            if inc_ii > best_inc_ii:
                best_inc_ii = inc_ii
                best_var = k_int
                best_classes = new_classes
                best_nclasses = new_n
                best_joint_mi = new_joint_mi

        if best_var < 0 or best_inc_ii < min_inc_ii:
            break  # local maximum reached; no positive synergistic extension

        # Accept the extension
        parent_set.add(best_var)
        parent_classes = best_classes
        parent_nclasses = best_nclasses
        parent_mi = best_joint_mi

    if len(parent_set) <= 2:
        return None  # no extension survived; caller emits the original pair
    return (
        tuple(sorted(parent_set)),
        parent_classes,
        int(parent_nclasses),
        float(parent_mi),
    )


def _build_kway_chained_lookup(
    factors_data: np.ndarray,
    idx_tuple: tuple,  # k indices in sorted order
    nbins: np.ndarray,
    unknown_strategy: Literal["clip", "sentinel", "raise"],
    dtype,
) -> tuple:
    """Build a chain of ``k - 1`` pair lookup tables that together replay the full k-way merge on test data.

    ``merge_vars`` over k cols can be decomposed as ``merge_vars(merge_vars(...merge_vars(c1, c2), c3...), ck)`` -- a chain of pairwise merges with intermediate dense
    renumbering. We build the lookup for each step at fit time:

    Step 1: lookup_1[c1_val + c2_val * nbins_1] -> intermediate_class_1 (size nbins_1 * nbins_2; intermediate cardinality = n_uniq_step_1)
    Step 2: lookup_2[intermediate_1 + c3_val * n_uniq_step_1] -> intermediate_class_2
    ...
    Step k-1: lookup_{k-1}[intermediate_{k-2} + ck_val * n_uniq_step_{k-2}] -> final_class

    Returns:
    - ``lookups``: list of (k-1) int64 ndarrays, each a flat lookup table
    - ``intermediate_nuniqs``: list of (k-1) ints, cardinalities AFTER each step

    On transform: callers chain through this list to compute the final class from test-data column values.
    """
    k = len(idx_tuple)
    if k < 2:
        raise ValueError(f"chained lookup requires k>=2, got k={k}")

    lookups: list = []
    intermediate_nuniqs: list = []

    # First step: merge cols [idx_tuple[0], idx_tuple[1]]
    vi_2 = np.array([idx_tuple[0], idx_tuple[1]], dtype=np.int64)
    classes_step, _, n_uniq_step = merge_vars(
        factors_data=factors_data, vars_indices=vi_2,
        var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
    )
    nbins_a = int(nbins[idx_tuple[0]])
    nbins_b = int(nbins[idx_tuple[1]])
    lookup_1, _ = _build_factorize_lookup(
        factors_data=factors_data,
        idx_a=int(idx_tuple[0]), idx_b=int(idx_tuple[1]),
        nbins_a=nbins_a, nbins_b=nbins_b,
        classes_pair_post=classes_step,
        unknown_strategy=unknown_strategy,
    )
    lookups.append(lookup_1)
    intermediate_nuniqs.append(int(n_uniq_step))

    # Subsequent steps: merge (running classes, idx_tuple[step])
    running_classes = classes_step.astype(np.int64, copy=False)
    running_nuniq = int(n_uniq_step)

    for step in range(2, k):
        nxt_idx = int(idx_tuple[step])
        nxt_vals = factors_data[:, nxt_idx].astype(np.int64, copy=False)
        nxt_nbins = int(nbins[nxt_idx])
        # Pre-prune codes for this step: running + nxt_vals * running_nuniq
        pre_prune_codes = running_classes + nxt_vals * running_nuniq
        expected_size = running_nuniq * nxt_nbins
        # ``pre_prune_codes`` already carries everything the full-prefix merge_vars(idx_tuple[:step+1]) call
        # would derive from raw columns -- dense-renumber it directly instead of re-scanning every prefix
        # column from scratch (see ``_dense_renumber_codes``'s docstring for the bit-identity + perf story).
        cls_next, n_uniq_next = _dense_renumber_codes(pre_prune_codes, expected_size)
        lookup_step = np.full(expected_size, -1, dtype=np.int64)
        # Populate: each row's pre-prune code -> its post-prune class
        lookup_step[pre_prune_codes] = cls_next
        # Resolve unseen per unknown_strategy
        seen_mask = lookup_step >= 0
        if not seen_mask.all():
            if unknown_strategy == "clip":
                seen_max = int(lookup_step[seen_mask].max())
                lookup_step[~seen_mask] = seen_max
            elif unknown_strategy == "sentinel":
                seen_max = int(lookup_step[seen_mask].max())
                lookup_step[~seen_mask] = seen_max + 1
            # ``raise``: leave -1; apply_recipe surfaces an error.
        lookups.append(lookup_step)
        intermediate_nuniqs.append(int(n_uniq_next))
        running_classes = cls_next.astype(np.int64, copy=False)
        running_nuniq = int(n_uniq_next)

    return lookups, intermediate_nuniqs


def _materialize_kway(
    factors_data: np.ndarray,
    kway_results: list,  # list of (indices_tuple, classes, n_uniq, joint_mi)
    nbins: np.ndarray,
    cols: list,
    dtype,
    unknown_strategy: Literal["clip", "sentinel", "raise"],
) -> tuple:
    """Materialise greedy k-way survivors. Returns ``(new_data_block, new_names, new_nbins, new_recipes)`` mirroring ``_materialize_pairs``. K-way recipes have
    ``src_names`` of length k and ``factorize_nbins`` of length k.

    K-way recipes ship a CHAINED LOOKUP -- (k-1) pair lookup tables that ``apply_recipe`` walks sequentially on test data. Memory: sum of intermediate nbins products
    (typically O(k * max_combined_nbins)), NOT O(nbins^k). At k=3 with cardinalities (10, 10, 10): 100 + 10*n_uniq_step1 cells (post-prune n_uniq usually < 100), so
    ~200-1000 int64 cells per recipe -- negligible vs the pair lookup table cost.
    """
    if not kway_results:
        return (
            np.empty((factors_data.shape[0], 0), dtype=dtype),
            [], [], [],
        )
    n_samples = factors_data.shape[0]
    new_data_block = np.empty((n_samples, len(kway_results)), dtype=dtype)
    new_names: list = []
    new_nbins: list = []
    new_recipes: list = []

    for k_out, (idx_tuple, classes_arr, n_uniq, _) in enumerate(kway_results):
        src_names = tuple(cols[i] for i in idx_tuple)
        eng_name = f"kway({'__'.join(src_names)})"
        if eng_name in cols or eng_name in new_names:
            eng_name = f"kway_{k_out}({'__'.join(src_names)})"
        new_data_block[:, k_out] = classes_arr
        new_names.append(eng_name)
        new_nbins.append(int(n_uniq))

        # Build the chained lookup so transform() can replay this k-way on
        # test data. Memory: O(k * max_combined_nbins), bounded by config.
        chain_lookups, chain_nuniqs = _build_kway_chained_lookup(
            factors_data=factors_data,
            idx_tuple=idx_tuple,
            nbins=nbins,
            unknown_strategy=unknown_strategy,
            dtype=dtype,
        )

        new_recipes.append(
            EngineeredRecipe(
                name=eng_name,
                kind="factorize",
                src_names=src_names,
                factorize_nbins=tuple(int(nbins[i]) for i in idx_tuple),
                unknown_strategy=unknown_strategy,
                extra={
                    "kway_order": len(idx_tuple),
                    "n_uniq_post_prune": int(n_uniq),
                    # Chained-lookup payload for k-way replay. ``chain_lookups`` has len k-1; ``chain_nuniqs`` is the post-prune cardinality after each step
                    # (drives the multiplier for the NEXT step's pre-prune code).
                    "chain_lookups": chain_lookups,
                    "chain_nuniqs": chain_nuniqs,
                },
            )
        )
    return new_data_block, new_names, new_nbins, new_recipes


def _select_top_k_pairs(
    ii_arr: np.ndarray,
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    cfg: CatFEConfig,
    n_samples: int,
) -> np.ndarray:
    """Pick the top-K pair indices by score. Uses argpartition -- O(N) over heap's O(N log K) -- the flat II array is small.

    The score depends on ``cfg.select_on``:
    - ``"synergy"``: rank by ``ii_arr`` desc; keep where ``ii > floor``.
    - ``"redundancy"``: rank by ``-ii_arr`` desc; keep where ``ii < -floor``.
    - ``"absolute"``: rank by ``|ii_arr|`` desc; keep where ``|ii| > floor``.

    Returns int array of indices into ``pairs_a`` / ``pairs_b`` (ordered descending by score). Length ``<= cfg.top_k_pairs``.
    """
    from .cat_interactions import resolve_min_interaction_information  # lazy: import-cycle, see module top
    floor = resolve_min_interaction_information(cfg, n_samples)
    if cfg.select_on == "synergy":
        score = ii_arr
        eligible = score > floor
    elif cfg.select_on == "redundancy":
        score = -ii_arr
        eligible = -ii_arr > -floor  # i.e. ii < floor (negative side)
    elif cfg.select_on == "absolute":
        score = np.abs(ii_arr)
        eligible = score > abs(floor)
    else:
        raise ValueError(f"Unknown cfg.select_on: {cfg.select_on!r}")

    n_eligible = int(eligible.sum())
    if n_eligible == 0:
        return np.empty(0, dtype=np.int64)

    # If we have <= top_k eligible candidates, just return them sorted desc.
    if n_eligible <= cfg.top_k_pairs:
        idx_eligible = np.where(eligible)[0]
        order = np.argsort(-score[idx_eligible])
        return idx_eligible[order]

    # Otherwise argpartition on score then sort the top.
    # Wave 58 (2026-05-20): argpartition tie-break is impl-defined; switch to
    # full lexsort with pair-index secondary key so tied scores give the same
    # top-K pairs across runs.
    masked_score = np.where(eligible, score, -np.inf)
    top_idx = np.lexsort((np.arange(len(masked_score)), -masked_score))[: cfg.top_k_pairs]
    return np.asarray(top_idx)


def _build_factorize_lookup(
    factors_data: np.ndarray,
    idx_a: int,
    idx_b: int,
    nbins_a: int,
    nbins_b: int,
    classes_pair_post: np.ndarray,
    unknown_strategy: Literal["clip", "sentinel", "raise"],
) -> tuple:
    """Build the pre-prune -> post-prune lookup table that lets ``transform()`` replay the merge on test data.

    ``merge_vars`` densely renumbers post-prune so the engineered col stored in ``data`` only has values in ``[0, n_uniq)``. But the "code" before pruning is a
    deterministic function of the input: ``code = a_value + b_value * nbins_a``. Two training rows with the same ``(a, b)`` tuple produce the same pre-prune code, so a
    lookup table indexed by code works for any input that respects the original cardinalities.

    Unseen test combinations are resolved per ``unknown_strategy``:
    - ``"clip"``: cap at the highest seen class (collides unseen with the most frequent training combo's class -- safe, conservative).
    - ``"sentinel"``: dedicate one new class for "unseen" (inflates ``n_uniq`` by 1; preferable when downstream models can learn a special meaning for that class).
    - ``"raise"``: leave the lookup at -1 sentinel; ``apply_recipe`` raises a clear error on the first unseen value.

    Returns ``(lookup_table, n_uniq_effective)``:
    - ``lookup_table``: ``(nbins_a * nbins_b,)`` int64 array, ``[code] -> post_prune_class`` (or -1 for unseen if ``raise``).
    - ``n_uniq_effective``: ``n_uniq`` + 1 if ``sentinel`` and any unseen cells, else ``n_uniq``.
    """
    n_samples = factors_data.shape[0]
    # Single-pass njit scatter (no length-n int64 temporaries). Reads the two
    # columns directly and writes ``lookup[a + b*nbins_a] = post_prune_class`` in
    # row order -- last-write-wins on duplicate codes, EXACTLY as the prior numpy
    # fancy-index assignment did. Bit-identical by construction; ~4.5-16x faster
    # at n=10k..200k (this runs ``top_k_pairs`` times per fit). Bench:
    # _benchmarks/bench_factorize_lookup_njit_scatter.py.
    lookup = _scatter_factorize_lookup(
        factors_data, int(idx_a), int(idx_b), int(nbins_a), int(nbins_b),
        classes_pair_post,
    )

    seen_mask = lookup >= 0
    n_uniq_effective = int(classes_pair_post.max()) + 1 if n_samples > 0 else 0

    if not seen_mask.all():
        if unknown_strategy == "clip":
            seen_max = int(lookup[seen_mask].max())
            lookup[~seen_mask] = seen_max
        elif unknown_strategy == "sentinel":
            seen_max = int(lookup[seen_mask].max())
            lookup[~seen_mask] = seen_max + 1
            n_uniq_effective = seen_max + 2
        elif unknown_strategy == "raise":
            pass  # leave as -1; apply_recipe will raise
        else:
            raise ValueError(f"Unknown unknown_strategy: {unknown_strategy!r}")
    return lookup, n_uniq_effective


def _materialize_pairs(
    factors_data: np.ndarray,
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    selected_idx: np.ndarray,
    nbins: np.ndarray,
    cols: list,
    dtype,
    unknown_strategy: Literal["clip", "sentinel", "raise"] = "clip",
) -> tuple:
    """For each selected pair, run ``merge_vars`` to produce the ordinal-encoded engineered column, build the lookup table that enables transform-replay, then assemble
    the ``EngineeredRecipe(kind="factorize")``.

    Returns ``(new_data_block, new_names, new_nbins, new_recipes)``:
    - ``new_data_block``: ``(n_samples, len(selected_idx))`` ordinal array
    - ``new_names``: list of engineered column names
    - ``new_nbins``: list of post-merge cardinalities (pruned)
    - ``new_recipes``: list of ``EngineeredRecipe(kind="factorize")``, with the lookup table embedded in ``recipe.extra["lookup_table"]``.

    Caller is responsible for ``np.concatenate``-ing ``new_data_block`` onto ``data`` (single concat).
    """
    n_samples = factors_data.shape[0]
    n_pairs = len(selected_idx)
    new_data_block = np.empty((n_samples, n_pairs), dtype=dtype)
    new_names: list = []
    new_nbins: list = []
    new_recipes: list = []

    for k_out, k_in in enumerate(selected_idx):
        i = int(pairs_a[k_in])
        j = int(pairs_b[k_in])
        vi = np.array([i, j], dtype=np.int64)
        classes_pair, _, _n_uniq = merge_vars(
            factors_data=factors_data,
            vars_indices=vi,
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=dtype,
        )
        nbins_a = int(nbins[i])
        nbins_b = int(nbins[j])
        lookup, n_uniq_effective = _build_factorize_lookup(
            factors_data=factors_data,
            idx_a=i, idx_b=j,
            nbins_a=nbins_a, nbins_b=nbins_b,
            classes_pair_post=classes_pair,
            unknown_strategy=unknown_strategy,
        )
        # Names follow the cat-FE convention ``kway(c1__c2)``. The ``__`` separator collides with column names containing ``__`` -- the lineage filter uses
        # ``recipe.src_names`` directly rather than substring-parsing, so we just assert no collision.
        name_a = cols[i]
        name_b = cols[j]
        eng_name = f"kway({name_a}__{name_b})"
        if eng_name in cols:
            # Unlikely but possible if user pre-engineered the same name.
            # Disambiguate by appending the source indices.
            eng_name = f"kway({name_a}__{name_b})_pair{i}_{j}"
        new_data_block[:, k_out] = classes_pair
        new_names.append(eng_name)
        new_nbins.append(n_uniq_effective)
        new_recipes.append(
            EngineeredRecipe(
                name=eng_name,
                kind="factorize",
                src_names=(name_a, name_b),
                factorize_nbins=(nbins_a, nbins_b),
                unknown_strategy=unknown_strategy,
                # Lookup table is the load-bearing artefact for replay.
                # Stored as a plain ndarray in extra so the dataclass
                # frozen-field constraint is satisfied.
                extra={"lookup_table": lookup, "n_uniq_post_prune": n_uniq_effective},
            )
        )
    return new_data_block, new_names, new_nbins, new_recipes
