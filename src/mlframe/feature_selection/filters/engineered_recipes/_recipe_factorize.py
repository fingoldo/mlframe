"""Cat-FE factorize replay: pair (k=2) + chained k-way merge-class lookup.

``_apply_factorize`` handles the common pair case and delegates to
``_apply_factorize_kway`` when the recipe carries a chained-lookup payload.
Both read only ``X`` (no y) via the shared extraction helpers, so transform()
is leakage-free.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._recipe_core import EngineeredRecipe
from ._recipe_extract import _coerce_to_int_with_nan_handling, _extract_column


def _apply_factorize(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Cat-FE replay: look up each test row's ``(a, b)`` tuple (or k-way chain) in the fit-time lookup table(s) and emit the post-prune class.

    Pairs (k=2): single lookup maps ``a_value + b_value * nbins_a`` to post-prune class. K > 2: chained lookup via ``recipe.extra['chain_lookups']`` (a list
    of k-1 pair lookups); each step combines running intermediate class with next column's value via ``chain_nuniqs`` from previous step.

    Test values outside ``[0, nbins_i)`` are clipped to ``nbins_i - 1``. Combinations whose pre-prune code never appeared in training are resolved per
    ``recipe.unknown_strategy`` (already baked into each lookup at fit time, except for ``"raise"`` which keeps -1 sentinels and surfaces here).
    """
    if recipe.extra.get("chain_lookups"):
        return _apply_factorize_kway(recipe, X)

    # Defensive branch for old pickled k-way recipes that lack chained lookups.
    if recipe.extra.get("requires_refit_for_replay"):
        raise NotImplementedError(
            f"factorize recipe '{recipe.name}' is a legacy k-way recipe "
            f"(order {recipe.extra.get('kway_order', '?')}) lacking a chained-lookup payload. "
            "Re-fit MRMR to materialise the chained-lookup version for replay."
        )
    if len(recipe.src_names) != 2 or len(recipe.factorize_nbins) != 2:
        raise ValueError(
            f"factorize recipe '{recipe.name}' requires exactly 2 src_names "
            f"and 2 factorize_nbins; got {len(recipe.src_names)} / "
            f"{len(recipe.factorize_nbins)}"
        )
    if "lookup_table" not in recipe.extra:
        raise KeyError(
            f"factorize recipe '{recipe.name}' is missing the 'lookup_table' "
            f"in recipe.extra. This usually means the recipe was built before "
            f"the cat-FE replay PR landed; refit the MRMR estimator."
        )

    name_a, name_b = recipe.src_names
    nbins_a, nbins_b = recipe.factorize_nbins
    lookup: np.ndarray = recipe.extra["lookup_table"]

    vals_a = _extract_column(X, name_a)
    vals_b = _extract_column(X, name_b)
    # Handle NaN / non-integer values per unknown_strategy. ``cat_code_maps`` (per-source
    # ``raw_value -> fit_code`` tables) reproduces fit-time discretiser codes for categorical /
    # string sources; absent (numeric sources / legacy pickles) the int-cast path is used.
    _cat_maps = recipe.extra.get("cat_code_maps") or {}
    _edges = recipe.extra.get("src_bin_edges") or {}
    vals_a_i = _coerce_to_int_with_nan_handling(vals_a, nbins_a, recipe.name, name_a, recipe.unknown_strategy, _cat_maps.get(name_a), _edges.get(name_a))
    vals_b_i = _coerce_to_int_with_nan_handling(vals_b, nbins_b, recipe.name, name_b, recipe.unknown_strategy, _cat_maps.get(name_b), _edges.get(name_b))

    # Clip out-of-range to nbins-1. Without this, a test value of ``nbins_a + 1`` would index past the lookup buffer end. Per ``unknown_strategy="clip"``
    # semantics (default), unseen values map to the highest seen class -- already encoded in the lookup; here we just guard the buffer.
    vals_a_i = np.clip(vals_a_i, 0, nbins_a - 1)
    vals_b_i = np.clip(vals_b_i, 0, nbins_b - 1)

    pre_prune_codes = vals_a_i + vals_b_i * nbins_a
    out = lookup[pre_prune_codes]

    # ``raise`` strategy left -1 sentinels in the lookup -- anything negative here is a test combo never seen in training.
    if recipe.unknown_strategy == "raise" and (out < 0).any():
        n_unseen = int((out < 0).sum())
        raise ValueError(
            f"factorize recipe '{recipe.name}': {n_unseen} row(s) have "
            f"(X[{name_a}], X[{name_b}]) combinations not seen during fit. "
            f"Set unknown_strategy='clip' or 'sentinel' to handle these "
            f"silently."
        )
    return out


def _apply_factorize_kway(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """K-way replay via the chained-lookup payload. Each ``chain_lookups[step]`` is a flat int64 table indexed by ``running_intermediate + col_value *
    running_nuniq``. We walk through all (k-1) steps, refreshing ``running_intermediate`` and ``running_nuniq`` from each step's output.

    Per-column test values are clipped to ``[0, factorize_nbins[i])``. Unseen combinations resolve per ``recipe.unknown_strategy`` (already encoded at fit
    time, except ``"raise"`` which leaves -1 sentinels and surfaces here).
    """
    src_names = recipe.src_names
    nbins_tuple = recipe.factorize_nbins
    chain_lookups: list = recipe.extra["chain_lookups"]
    chain_nuniqs: list = recipe.extra["chain_nuniqs"]
    k = len(src_names)
    if len(chain_lookups) != k - 1 or len(chain_nuniqs) != k - 1:
        raise ValueError(
            f"k-way recipe '{recipe.name}' chain payload size mismatch: "
            f"k={k}, chain_lookups={len(chain_lookups)}, "
            f"chain_nuniqs={len(chain_nuniqs)} (expected {k-1} each)."
        )

    _cat_maps = recipe.extra.get("cat_code_maps") or {}
    _edges = recipe.extra.get("src_bin_edges") or {}
    # Step 1: build running from first two columns
    vals_0 = _coerce_to_int_with_nan_handling(
        _extract_column(X, src_names[0]), int(nbins_tuple[0]),
        recipe.name, src_names[0], recipe.unknown_strategy,
        _cat_maps.get(src_names[0]), _edges.get(src_names[0]),
    )
    vals_1 = _coerce_to_int_with_nan_handling(
        _extract_column(X, src_names[1]), int(nbins_tuple[1]),
        recipe.name, src_names[1], recipe.unknown_strategy,
        _cat_maps.get(src_names[1]), _edges.get(src_names[1]),
    )
    vals_0 = np.clip(vals_0, 0, int(nbins_tuple[0]) - 1)
    vals_1 = np.clip(vals_1, 0, int(nbins_tuple[1]) - 1)
    pre_prune = vals_0 + vals_1 * int(nbins_tuple[0])
    running = chain_lookups[0][pre_prune]
    running_nuniq = chain_nuniqs[0]
    # 2026-05-30 Wave 9.1 fix (loop iter 13): under unknown_strategy='raise'
    # we MUST raise BEFORE the next chain step uses ``running`` as an index.
    # If we don't, ``pre_prune = running + vals_next * running_nuniq`` with
    # ``running[i] == -1`` (unseen prefix) computes a negative-or-small
    # index that Python wraps to the tail of ``chain_lookups[step-1]``,
    # silently returning a real class code. The post-loop ``(running < 0)``
    # check then sees no -1 and fails to raise. Confirmed live: a 3-way
    # recipe with raise mode + unseen prefix returned [1] instead of
    # raising ValueError.
    if recipe.unknown_strategy == "raise" and (running < 0).any():
        n_unseen = int((running < 0).sum())
        raise ValueError(
            f"k-way factorize recipe '{recipe.name}': {n_unseen} row(s) "
            f"hit unseen prefix at chain step 1. Set unknown_strategy="
            f"'clip' or 'sentinel' to handle silently."
        )

    # Steps 2..k-1: chain forward
    for step in range(2, k):
        vals_next = _coerce_to_int_with_nan_handling(
            _extract_column(X, src_names[step]), int(nbins_tuple[step]),
            recipe.name, src_names[step], recipe.unknown_strategy,
            _cat_maps.get(src_names[step]), _edges.get(src_names[step]),
        )
        vals_next = np.clip(vals_next, 0, int(nbins_tuple[step]) - 1)
        pre_prune = running + vals_next * running_nuniq
        running = chain_lookups[step - 1][pre_prune]
        running_nuniq = chain_nuniqs[step - 1]
        # Same guard at every intermediate step: any -1 here would get
        # silently overwritten by the next negative-index wrap.
        if recipe.unknown_strategy == "raise" and (running < 0).any():
            n_unseen = int((running < 0).sum())
            raise ValueError(
                f"k-way factorize recipe '{recipe.name}': {n_unseen} row(s) "
                f"hit unseen prefix at chain step {step}. Set "
                f"unknown_strategy='clip' or 'sentinel' to handle silently."
            )

    if recipe.unknown_strategy == "raise" and (running < 0).any():
        n_unseen = int((running < 0).sum())
        raise ValueError(
            f"k-way factorize recipe '{recipe.name}': {n_unseen} row(s) "
            f"have combinations not seen during fit. Set "
            f"unknown_strategy='clip' or 'sentinel' to handle silently."
        )
    return running
