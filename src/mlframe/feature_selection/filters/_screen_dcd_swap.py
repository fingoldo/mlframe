"""Dynamic Cluster Discovery (DCD) discover/swap block carved out of
``screen_predictors`` (``_screen_predictors.py``).

This is the inline ``if dcd_state is not None:`` block that previously lived
inside the select-loop's ``for var in best_candidate:`` body. It shares many
mutable locals with the orchestration function, so the carve threads every
local it reads/writes EXPLICITLY (no closure capture) and RETURNS the four
loop-locals it reassigns on a committed swap (``factors_data`` /
``factors_nbins`` / ``factors_names`` / ``data_copy``). ``dcd_state`` /
``selected_vars`` / ``predictors`` / ``entropy_cache`` / ``cached_MIs`` / ``ctx``
are mutated in place by the DCD primitives, so they need no return.

Behaviour is byte-for-byte identical to the pre-carve inline block.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def screen_dcd_discover_and_swap(
    *,
    dcd_state,
    var,
    factors_data,
    factors_nbins,
    factors_names,
    data_copy,
    selected_vars,
    entropy_cache,
    cached_MIs,
    full_npermutations,
    y,
    engineered_recipes,
    predictors,
    ctx,
    verbose,
):
    """DCD discover + anchor->PC1 swap step (verbatim carve of the select-loop block).

    Returns the (possibly swap-extended) loop-local matrix refs:
    ``(factors_data, factors_nbins, factors_names, data_copy)``.
    """
    try:
        from ._dynamic_cluster_discovery import (
            discover_cluster_members as _dcd_discover,
            evaluate_swap_candidate as _dcd_eval_swap,
            commit_swap as _dcd_commit_swap,
        )
        # candidate_pool = surviving non-pruned non-selected indices
        _pool = [
            i for i in range(factors_data.shape[1])
            if i != var
            and not dcd_state.pool_pruned_mask[i]
            and i not in selected_vars
        ]
        _dcd_discover(
            dcd_state, var, _pool,
            entropy_cache=entropy_cache,
            factors_data=factors_data,
            factors_nbins=factors_nbins,
            selected_vars=selected_vars,
        )
        # 2026-05-30 Wave 9.1 — anchor → PC1 swap.
        # When the freshly-grown cluster reaches
        # ``cluster_size_threshold``, evaluate a
        # PC1 aggregate. If ``conditional_mi(rep ;
        # y | Selected − anchor)`` beats anchor's
        # by ``swap_gain_threshold``, commit_swap
        # extends ``factors_data`` and replaces
        # ``var`` in ``selected_vars`` atomically.
        _cluster_members = dcd_state.cluster_anchors.get(int(var), set())
        if len(_cluster_members) >= int(dcd_state.cluster_size_threshold):
            _decision = _dcd_eval_swap(
                dcd_state, int(var), selected_vars,
                target_y=y,
                factors_data=factors_data,
                factors_nbins=factors_nbins,
                entropy_cache=entropy_cache,
                cached_MIs=cached_MIs,
                full_npermutations=int(full_npermutations or 0),
            )
            if _decision.accept:
                _data_ref = {}
                # 2026-05-31 Layer 43 (PART A) fix:
                # pass the host MRMR's engineered_recipes
                # dict so commit_swap registers the PC1
                # aggregate as an EngineeredRecipe. Pre-
                # fix this was hardcoded to None and the
                # aggregate name was dropped by the
                # _mrmr_fit_impl.py remap that copies
                # engineered_recipes.get(name) into
                # self._engineered_recipes_. Net result:
                # support_ silently shrank when swap fired.
                _new_idx = _dcd_commit_swap(
                    dcd_state, int(var), _decision,
                    selected_vars=selected_vars,
                    data_ref=_data_ref,
                    engineered_recipes=engineered_recipes,
                    predictors_log=predictors,
                )
                # Re-bind the loop-local matrix refs
                # so subsequent iterations see the
                # extended data / cols / nbins.
                factors_data = _data_ref.get("data", factors_data)
                factors_nbins = _data_ref.get("nbins", factors_nbins)
                factors_names = _data_ref.get("cols", factors_names)
                # Re-alias data_copy to the (post-swap, extended) factors_data for the next confirm cycle; the Fleuret permutation njit
                # saves+restores the columns it shuffles, so no whole-matrix copy is needed here either.
                data_copy = factors_data
                # 2026-05-30 Wave 9.1 fix (loop iter 2):
                # confirm_one_predictor reads ctx.factors_data
                # and ctx.data_copy at the top of every call;
                # without these writes, subsequent confirmations
                # in this screen invocation index into the OLD
                # matrix with selected_vars holding the NEW
                # post-swap index -> silent OOB under
                # numba boundscheck=False, IndexError otherwise.
                ctx.factors_data = factors_data
                ctx.factors_nbins = factors_nbins
                ctx.factors_names = factors_names
                ctx.data_copy = data_copy
                if verbose:
                    logger.info(
                        "DCD swap: anchor %s -> aggregate idx %d (%d members)",
                        var, _new_idx, len(_cluster_members),
                    )
    except (IndexError, AttributeError, KeyError, TypeError) as _dcd_exc:
        # 2026-05-30 Wave 9.1 fix (loop iter 2):
        # programming errors MUST surface. Silently
        # swallowing IndexError under verbose=0 is
        # how the matrix-propagation gap slipped
        # past testing in the first place. Numerical
        # / binning edge cases (NaN/Inf in SU, SVD
        # convergence) are caught one level down in
        # the DCD module itself.
        raise RuntimeError(
            f"DCD discover/swap raised a programming "
            f"error -- this indicates an mlframe bug, "
            f"not a data issue: {_dcd_exc!r}"
        ) from _dcd_exc
    except Exception as _dcd_exc:
        # Genuinely best-effort -- numeric / fitting
        # failures inside DCD (e.g. all-constant
        # cluster, degenerate PC1) should not break
        # the screen. These ARE expected on pathologic
        # inputs; surface with a warning regardless of
        # verbose level.
        logger.warning(
            "DCD discover/swap step failed: %r",
            _dcd_exc,
        )

    return factors_data, factors_nbins, factors_names, data_copy
