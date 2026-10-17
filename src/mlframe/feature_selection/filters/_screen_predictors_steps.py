"""Interaction-order enumeration step of ``screen_predictors``.

Carved out of ``_screen_predictors`` (module size limit); the parent re-exports its names.
"""

from __future__ import annotations

import logging
from itertools import combinations
from os.path import exists
from pyutilz.system import tqdmu
from ._confirm_predictor import confirm_one_predictor
from ._screen_predictors_gate import compute_selection_gate
from .evaluation import get_candidate_name
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)


def _short_name(name, maxlen: int = 28) -> str:
    """Truncate a (possibly long engineered) feature name for live progress-bar display.

    Keeps the head and tail so both the operation and the operand stay legible
    (``mul(log(c),sin(d))`` -> ``mul(log(..in(d))``). Robust to non-str input.
    """
    try:
        s = str(name)
    except Exception as e:
        log_throttle(logger, "_screen_predictors_steps._short_name", logging.WARNING, "_short_name: str(name) failed for progress-display truncation, showing '?': %s", e)
        return "?"
    if len(s) <= maxlen:
        return s
    head = (maxlen - 2) // 2
    tail = maxlen - 2 - head
    return s[:head] + ".." + s[-tail:]


def _screen_predictors_step2_interactions_order_subsets(
    subsets,
    verbose,
    st,
    num_possible_candidates,
    ctx,
    stop_file,
    patience_triggered,
    min_relevance_gain,
    cardinality_bias_correction,
    factors_data,
    y,
    factors_nbins,
    min_relevance_gain_relative_to_first,
    selected_vars,
    predictors,
    _fdr_gain_floor,
    cached_MIs,
    selected_interactions_vars,
    dcd_state,
    factors_names,
    data_copy,
    entropy_cache,
    full_npermutations,
    engineered_recipes,
    ndigits,
):
    """Step 2 of screen_predictors: lines starting at ``for interactions_order in (subsets_pbar := tqdmu(subsets, desc="Intera``."""
    for interactions_order in (subsets_pbar := tqdmu(subsets, desc="Interactions order", leave=False, disable=not verbose)):

        if st.run_out_of_time:
            break
        subsets_pbar.set_description(f"{interactions_order}-way interactions")

        # ---------------------------------------------------------------------------------------------------------------
        # Generate candidates
        # ---------------------------------------------------------------------------------------------------------------

        candidates = [tuple(el) for el in combinations(st.x, interactions_order)]

        num_possible_candidates += len(candidates)

        # ---------------------------------------------------------------------------------------------------------------
        # Subset level inits
        # ---------------------------------------------------------------------------------------------------------------

        total_disproved = 0
        total_checked = 0
        partial_gains: dict = {}
        added_candidates: set = set()
        failed_candidates: set = set()
        nconsec_unconfirmed = 0

        # Refresh the confirmation context for this interactions order.
        ctx.candidates = candidates
        ctx.interactions_order = interactions_order
        ctx.partial_gains = partial_gains
        ctx.added_candidates = added_candidates
        ctx.failed_candidates = failed_candidates
        ctx.num_possible_candidates = num_possible_candidates

        # Running leader for the live "Confirmed predictors" postfix (reset per
        # interactions_order). ``_best_confirmed_gain`` starts at -inf so the first
        # confirmed feature - even a negative-gain one - becomes the displayed top.
        _best_confirmed_gain = float("-inf")
        _best_confirmed_name = None
        for _n_confirmed_predictors in (predictors_pbar := tqdmu(range(len(candidates)), leave=False, desc="Confirmed predictors", disable=not verbose)):
            if st.run_out_of_time:
                break
            if stop_file and exists(stop_file):
                log_throttle(logger, "screen_predictors_stop_file_detected", logging.WARNING, "Stop file %s detected, quitting.", stop_file)
                break

            # The full single-predictor confirmation cycle (score all candidates, then permutation-confirm in
            # expected-gain order with partial-gain recompute/retry + patience accounting) lives in the
            # ``confirm_one_predictor`` primitive (``_confirm_predictor.py``), keeping this file below the
            # 1k-line monolith threshold and the frequently patched confirmation math in one place.
            (
                best_candidate,
                best_gain,
                confidence,
                st.run_out_of_time,
                nconsec_unconfirmed,
                total_checked,
                total_disproved,
                patience_triggered,
            ) = confirm_one_predictor(
                ctx,
                nconsec_unconfirmed=nconsec_unconfirmed,
                total_checked=total_checked,
                total_disproved=total_disproved,
                patience_triggered=patience_triggered,
            )

            # ---------------------------------------------------------------------------------------------------------------
            # Add best candidate to the list, if criteria are met, or proceed to the next interactions_order
            # ---------------------------------------------------------------------------------------------------------------

            # Abs/relative/maxT-FDR gain-floor gate: Miller-Madow correction, diminishing-
            # returns relative floor, and the maxT permutation-null floor all live in
            # ``compute_selection_gate`` (``_screen_predictors_gate.py``).
            _gate_passed, _best_gain_for_gate = compute_selection_gate(
                min_relevance_gain=min_relevance_gain,
                interactions_order=interactions_order,
                best_candidate=best_candidate,
                best_gain=best_gain,
                cardinality_bias_correction=cardinality_bias_correction,
                factors_data=factors_data,
                y=y,
                factors_nbins=factors_nbins,
                min_relevance_gain_relative_to_first=min_relevance_gain_relative_to_first,
                selected_vars=selected_vars,
                predictors=predictors,
                fdr_gain_floor=_fdr_gain_floor,
                cached_MIs=cached_MIs,
            )
            if _gate_passed:
                for var in best_candidate:
                    if var not in selected_vars:
                        selected_vars.append(var)
                        if interactions_order > 1:
                            selected_interactions_vars.append(var)
                        # Dynamic Cluster Discovery hook.
                        # After each accepted predictor, prune the Pool by
                        # SU(c, var) > tau_cluster. Mutates ``pool_pruned_mask``
                        # in-place (Critic1/B-1: NO mutation of candidates
                        # list — uses ``should_skip_candidate``'s mask check).
                        if dcd_state is not None:
                            # DCD discover/swap block carved into ``_screen_dcd_swap.py``
                            # (Tier E). The helper threads the loop-locals it reads/writes
                            # explicitly and RETURNS the four matrix refs it reassigns on a
                            # committed swap; ``dcd_state`` / ``selected_vars`` / ``predictors``
                            # / ``ctx`` / the caches are mutated in place. Behaviour is
                            # byte-for-byte identical to the prior inline block.
                            from mlframe.feature_selection.filters._screen_dcd_swap import screen_dcd_discover_and_swap

                            factors_data, factors_nbins, factors_names, data_copy = screen_dcd_discover_and_swap(
                                dcd_state=dcd_state,
                                var=var,
                                factors_data=factors_data,
                                factors_nbins=factors_nbins,
                                factors_names=factors_names,
                                data_copy=data_copy,
                                selected_vars=selected_vars,
                                entropy_cache=entropy_cache,
                                cached_MIs=cached_MIs,
                                full_npermutations=full_npermutations,
                                y=y,
                                engineered_recipes=engineered_recipes,
                                predictors=predictors,
                                ctx=ctx,
                                verbose=verbose,
                            )
                cand_name = get_candidate_name(best_candidate, factors_names=factors_names)

                res = {"name": cand_name, "indices": best_candidate, "gain": best_gain}
                if full_npermutations:
                    res["confidence"] = confidence
                predictors.append(res)

                # Live progress: surface the winning feature + its mrmr_gain on the
                # "Confirmed predictors" bar. ``best_gain`` was just computed by
                # confirm_one_predictor - display-only, zero extra MI work. Track the
                # strongest-confirmed so the postfix always names the current leader.
                if verbose:
                    try:
                        _g = float(best_gain)
                        if _g > _best_confirmed_gain:
                            _best_confirmed_gain = _g
                            _best_confirmed_name = cand_name
                        _pf = {"last": _short_name(cand_name), "gain": f"{_g:.{ndigits}f}"}
                        if _best_confirmed_name is not None and _best_confirmed_name != cand_name:
                            _pf["top"] = f"{_short_name(_best_confirmed_name)}={_best_confirmed_gain:.{ndigits}f}"
                        predictors_pbar.set_postfix(_pf, refresh=False)
                    except (TypeError, ValueError):
                        pass

                if verbose >= 2:
                    mes = f"Added new predictor {cand_name} to the list with expected gain={best_gain:.{ndigits}f}"
                    if full_npermutations:
                        mes += f" and confidence={confidence:.3f}"
                    # Surface the pool_size x |Z| scaling this predictor's confirmation cost right
                    # alongside the pick itself - pool shrinks as candidates fail while |Z| only grows, so a
                    # later predictor with a SMALLER remaining pool can still be slower than an earlier one.
                    mes += f" [pool={len(candidates)} |Z|={len(selected_vars)} " f"cum_score_candidates_wall={ctx.sc_wall:.1f}s calls={ctx.sc_calls}]"
                    logger.info(mes)

            else:
                if verbose >= 2:
                    if total_checked > 0:
                        details = f" Total candidates disproved: {total_disproved:_}/{total_checked:_} ({total_disproved*100/total_checked:.2f}%)"
                    else:
                        details = ""
                    logger.info("Can't add anything valuable anymore for interactions_order=%s.%s", interactions_order, details)
                predictors_pbar.total = len(candidates)
                predictors_pbar.close()
                break
    return factors_nbins, patience_triggered
