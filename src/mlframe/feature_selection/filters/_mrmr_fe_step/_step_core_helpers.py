"""Helpers carved out of ``_step_core`` to keep that module under its size budget."""
from __future__ import annotations

import logging

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def _run_fe_step_impl_full_key_raw_vars(self, _synergy_added_idx, prospective_pairs, verbose):
    """Block of _run_fe_step_impl starting at ``if _synergy_added_idx:``."""
    if _synergy_added_idx:
        _synergy_budget = int(getattr(self, "fe_synergy_max_pairs", 16) or 0)
        _synergy_keys = [k for k in prospective_pairs if (k[0][0] in _synergy_added_idx or k[0][1] in _synergy_added_idx)]
        if _synergy_budget >= 0 and len(_synergy_keys) > _synergy_budget:
            _keep_synergy = set(sorted(_synergy_keys, key=lambda k: k[1], reverse=True)[:_synergy_budget])
            _dropped = 0
            for k in _synergy_keys:
                if k not in _keep_synergy:
                    del prospective_pairs[k]
                    _dropped += 1
            if verbose and _dropped:
                logger.info(
                    "MRMR FE synergy bootstrap: kept top %d synergy pairs by joint MI, "
                    "dropped %d below budget (fe_synergy_max_pairs) to bound FE search cost.",
                    min(_synergy_budget, len(_synergy_keys)), _dropped,
                )


def _run_fe_step_impl_below_fe_rung_min(self, prospective_pairs, data, verbose):
    """Block of _run_fe_step_impl starting at ``if bool(getattr(self, "fe_rung_schedule_enable", True)) and len(prospe``."""
    if bool(getattr(self, "fe_rung_schedule_enable", True)) and len(prospective_pairs) >= int(getattr(self, "fe_rung_min_pairs", 6)):
        from mlframe.feature_selection.filters._fe_rung_schedule import apply_rung_schedule
        _rung_n_rows = int(data.shape[0]) if hasattr(data, "shape") else 0
        prospective_pairs, _rung_info = apply_rung_schedule(
            prospective_pairs,
            n_rows=_rung_n_rows,
            keep_frac=getattr(self, "fe_rung_keep_frac", None),
            rel_floor=float(getattr(self, "fe_rung_rel_floor", 0.40)),
            min_pairs=int(getattr(self, "fe_rung_min_pairs", 6)),
            verbose=verbose,
        )
        # apply_rung_schedule's own docstring documents ``info`` as "for logging / tests" -
        # it was computed and silently discarded here with no consumer. Surface it at verbose>=1.
        if verbose and _rung_info.get("applied"):
            logger.info(
                "mrmr: rung-0 pair screen kept %d/%d prospective pairs (keep_frac=%.2f, rel_floor=%.2f).",
                _rung_info.get("n_kept"), _rung_info.get("n_in"), _rung_info.get("keep_frac", 0.0), _rung_info.get("rel_floor", 0.0),
            )
    return prospective_pairs


def _run_fe_step_impl_pure_noise_risk(_synergy_added_idx, _screening_returned_empty, prospective_pairs, _prospective_for_polynom):
    """Block of _run_fe_step_impl starting at ``if _synergy_added_idx and not _screening_returned_empty:``."""
    if _synergy_added_idx and not _screening_returned_empty:
        _filtered_for_polynom = {k: v for k, v in prospective_pairs.items() if not (k[0][0] in _synergy_added_idx or k[0][1] in _synergy_added_idx)}
        # (see default_filtering.py:165): apply the
        # speculative-synergy exclusion ONLY if it leaves a non-empty pool.
        # When the selected pool is too small to form any NON-synergy pair
        # (screening kept 0-1 features on an interaction-only target, so
        # every surviving pair has a synergy-added operand), excluding them
        # would withhold EVERY pair and silently disable the polynom search
        # - yet those pairs ARE the signal. Keep them in that case; the
        # synergy max-pairs cap + the downstream pair-MI / engineered-MI /
        # uplift gates already bound the pure-noise risk.
        if _filtered_for_polynom:
            _prospective_for_polynom = _filtered_for_polynom
    return _prospective_for_polynom


def _run_fe_step_impl_handles_any_internal_subsample(self, _prewarp_enable, classes_y, _prewarp_y_cont):
    """Block of _run_fe_step_impl starting at ``if _prewarp_enable:``."""
    if _prewarp_enable:
        _pwc = getattr(self, "_fe_prewarp_y_continuous_", None)
        if _pwc is not None and len(_pwc) == len(classes_y):
            _prewarp_y_cont = _pwc
    return _prewarp_y_cont


def _run_fe_step_impl_key_value_prospective_pairs(prospective_pairs, nitems, cur_dict, desired_nitems, jobs_list):
    """Block of _run_fe_step_impl starting at ``for key, value in prospective_pairs.items():``."""
    for key, value in prospective_pairs.items():
        nitems += 1
        cur_dict[key] = value
        if nitems >= desired_nitems:
            jobs_list.append(cur_dict)
            nitems = 0
            cur_dict = {}
    return cur_dict


def _run_fe_step_impl_accumulators_during_merge_loop(self, dicts, _prewarp_specs, _gate_med_specs, num_fs_steps, prospective_additions):
    """Block of _run_fe_step_impl starting at ``for next_dict in dicts:``."""
    from mlframe.feature_selection.filters._feature_engineering_pairs import _PREWARP_SPECS_RESULT_KEY, _GATE_MED_SPECS_RESULT_KEY, _FE_REJECTION_RESULT_KEY

    for next_dict in dicts:
        _pw_chunk = next_dict.pop(_PREWARP_SPECS_RESULT_KEY, None)
        if _pw_chunk:
            _prewarp_specs.update(_pw_chunk)
        _gm_chunk = next_dict.pop(_GATE_MED_SPECS_RESULT_KEY, None)
        if _gm_chunk:
            _gate_med_specs.update(_gm_chunk)
        # REJECTION LEDGER (additive): drain each chunk's per-pair-gate drops.
        _rej_chunk = next_dict.pop(_FE_REJECTION_RESULT_KEY, None)
        if _rej_chunk:
            for _rr in _rej_chunk:
                _record_fe_rejection(
                    self, gate=_rr.get("gate", "engineered_mi_prevalence"),
                    candidate=_rr.get("candidate"), operands=_rr.get("operands"),
                    operator=_rr.get("operator"),
                    observed=_rr.get("observed", float("nan")),
                    threshold=_rr.get("threshold", float("nan")),
                    reason=_rr.get("reason", ""), step=int(num_fs_steps),
                )
        prospective_additions.update(next_dict)


def _run_fe_step_impl_joblib_branch_already_drained(self, _rej_from_res, num_fs_steps):
    """Block of _run_fe_step_impl starting at ``if _rej_from_res:``."""
    if _rej_from_res:
        for _rr in _rej_from_res:
            _record_fe_rejection(
                self, gate=_rr.get("gate", "engineered_mi_prevalence"),
                candidate=_rr.get("candidate"), operands=_rr.get("operands"),
                operator=_rr.get("operator"),
                observed=_rr.get("observed", float("nan")),
                threshold=_rr.get("threshold", float("nan")),
                reason=_rr.get("reason", ""), step=int(num_fs_steps),
            )
