"""Pure-helper extractions from ``_ensembling_score.py``.

Each helper here is a value-in / value-out transform whose body was lifted verbatim from the original ``score_ensemble`` so behavioural equivalence is preserved by construction: every call site stitches the helper's return tuple back into the same locals the original code used.

The helpers live here so the parent function body shrinks toward the <500 LOC budget without introducing a state dataclass (which would touch every mutable). Order follows the original control flow of ``score_ensemble``.
"""
from __future__ import annotations

import logging
import re as _re_mod
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("mlframe.models.ensembling")


def build_member_tag_lists(
    level_models_and_predictions: Sequence[Any],
) -> Tuple[List[str], List[str]]:
    """Build the two parallel tag lists used downstream in ``score_ensemble``.

    ``_ensemble_member_tags`` carries shim-stripped class / model names for the per-member quality-gate log line (operators want to see which exact model class was excluded). ``_ensemble_short_tags`` collapses to the short tag (``cb`` / ``xgb`` / ``lgb`` / ``hgb`` / non-tree class name) used in the rebuilt ensemble label after the gate (full class names would bloat chart titles + break the original short-label contract from core.py).
    """
    from mlframe.training import (
        short_model_tag as _short_tag,
        strip_shim_suffix as _strip_shim,
    )

    _ensemble_member_tags: List[str] = []
    _ensemble_short_tags: List[str] = []
    for _m in level_models_and_predictions:
        _name_attr = getattr(_m, "model_name", None) or getattr(_m, "name", None)
        _model_obj = getattr(_m, "model", _m)
        if _name_attr:
            _ensemble_member_tags.append(_strip_shim(str(_name_attr)))
        else:
            _ensemble_member_tags.append(_strip_shim(type(_model_obj).__name__))
        # short-tag ALWAYS derived from the underlying CLASS, not from ``model_name`` which carries augmentations like ``"target MTTR=11497.66"`` that would defeat the startswith() prefix checks (``startswith("CatBoost")`` etc.).
        _ensemble_short_tags.append(_short_tag(_model_obj))
    return _ensemble_member_tags, _ensemble_short_tags


def apply_diversity_drop(
    level_models_and_predictions: list,
    _ensemble_member_tags: List[str],
    _ensemble_short_tags: List[str],
    _high_corr_pairs: list,
    _gate_stats: Any,
    auto_drop_diversity_above: Optional[float],
    diversity_corr_warn_threshold: float,
    _div_split_used: Any,
    res: dict,
) -> Tuple[list, List[str], List[str], list]:
    """Apply auto-drop on high-correlation member pairs and stamp the diversity report.

    Picks the loser per pair via per-member MAE (higher loses); ties + missing stats fall back to alphabetical tag for deterministic tiebreak. Pre-fix the implementation picked by ``_ensemble_member_tags.index(...)`` -- purely the member's POSITION in the suite, with no quality signal.

    Stamps ``res["_diversity"]`` with high-correlation pairs + (when fired) the auto-dropped tag list.
    """
    _drop_floor = auto_drop_diversity_above
    _auto_dropped: list[str] = []
    for _pair in _high_corr_pairs:
        if _drop_floor is not None and abs(_pair["corr"]) >= float(_drop_floor):
            _would_drop_msg = f" auto-drop active (floor={_drop_floor:.3f}) -- one of {_pair['m1']}/{_pair['m2']} will be dropped"
        else:
            _would_drop_msg = " (would auto-drop one member if auto_drop_diversity_above set <= corr)"
        logger.warning(
            "[ensemble] high-correlation member pair (split=%s): %s vs %s -- Pearson corr=%.4f > threshold=%.4f.%s",
            _div_split_used,
            _pair["m1"],
            _pair["m2"],
            _pair["corr"],
            diversity_corr_warn_threshold,
            _would_drop_msg,
        )
    # Auto-drop one member from each high-corr pair when the knob is set. The docstring on the
    # ``auto_drop_diversity_above`` kwarg promises "MEMBER WITH HIGHER MEAN ABSOLUTE GATE-METRIC
    # (mae from the gate) is dropped, so the surviving member is the one closer to the median".
    # Pre-fix the implementation picked by ``_ensemble_member_tags.index(...)`` -- purely the
    # member's POSITION in the suite, with no quality signal. Now use the per-member MAE stamped
    # by ``compute_member_quality_gate`` earlier when available; the higher-MAE member loses.
    # When two members have identical MAE (or stats are unavailable), fall back to alphabetical
    # tag order so the tiebreak is deterministic across runs / BLAS thread counts.
    if _drop_floor is not None and _high_corr_pairs:
        _to_drop_tags: set[str] = set()
        _gate_per_mae = None
        try:
            _gate_per_mae = _gate_stats.get("per_member_mae") if isinstance(_gate_stats, dict) else None
        except NameError:
            _gate_per_mae = None
        # Map tag -> MAE on the pre-gate-drop member list. ``_ensemble_member_tags`` is the
        # POST-gate tag list (sliced if the gate dropped members), so realign by looking up the
        # member's current position in the sliced list and using that index into per_member_mae
        # only when the lists are the same length; otherwise fall back to alphabetical-only.
        _tag_to_mae: dict[str, float] = {}
        if _gate_per_mae is not None and len(_gate_per_mae) == len(_ensemble_member_tags):
            for _i, _t in enumerate(_ensemble_member_tags):
                try:
                    _tag_to_mae[_t] = float(_gate_per_mae[_i])
                except (TypeError, ValueError):
                    pass
        for _pair in _high_corr_pairs:
            if abs(_pair["corr"]) < float(_drop_floor):
                continue
            _m1, _m2 = _pair["m1"], _pair["m2"]
            try:
                _ensemble_member_tags.index(_m1)
                _ensemble_member_tags.index(_m2)
            except ValueError:
                continue
            _mae1 = _tag_to_mae.get(_m1, float("nan"))
            _mae2 = _tag_to_mae.get(_m2, float("nan"))
            # Drop the worse (higher MAE) member; ties (including all-NaN) -> alphabetical drop.
            if np.isfinite(_mae1) and np.isfinite(_mae2) and _mae1 != _mae2:
                _drop_tag = _m1 if _mae1 > _mae2 else _m2
            else:
                _drop_tag = max(_m1, _m2)
            _to_drop_tags.add(_drop_tag)
        if _to_drop_tags:
            _kept_pairs = [
                (i, m) for i, m in enumerate(level_models_and_predictions)
                if _ensemble_member_tags[i] not in _to_drop_tags
            ]
            if len(_kept_pairs) >= 1:
                level_models_and_predictions = [m for _, m in _kept_pairs]
                _kept_indices = [i for i, _ in _kept_pairs]
                _ensemble_member_tags = [_ensemble_member_tags[i] for i in _kept_indices]
                _ensemble_short_tags = [_ensemble_short_tags[i] for i in _kept_indices]
                _auto_dropped = sorted(_to_drop_tags)
                logger.warning(
                    "[ensemble] auto_drop_diversity_above=%.3f dropped %d duplicate member(s): %s",
                    float(_drop_floor), len(_auto_dropped), _auto_dropped,
                )
    if _high_corr_pairs or _auto_dropped:
        res["_diversity"] = {
            "high_correlation_pairs": _high_corr_pairs,
            "threshold": diversity_corr_warn_threshold,
            "split_used": _div_split_used,
            "auto_drop_floor": float(_drop_floor) if _drop_floor is not None else None,
            "auto_dropped_members": _auto_dropped,
        }
    return level_models_and_predictions, _ensemble_member_tags, _ensemble_short_tags, _auto_dropped


def filter_sign_sensitive_flavours(
    ensembling_methods: list,
    is_regression: bool,
    level_models_and_predictions: Sequence[Any],
    verbose: bool,
) -> list:
    """Gate out harm / geo / quad flavours when ANY member's predictions contain near-zero values or sign changes.

    I2 (2026-05-11): for regression, harmonic mean = N / sum(1/p) and geometric mean = exp(mean(log p)) both diverge / are undefined on signals that cross zero. Symptom seen in the prod log: ``EnsHARM ... RMSE=178.84 MaxError=55206`` and ``RMSE=1299.55 MaxError=920165`` on composite residuals which cluster around zero by construction.

    Also gate-out QUAD (quadratic mean = sqrt(mean(p^2))) on sign-changing targets. Squaring loses the sign of the input by construction, so QUAD ALWAYS emits non-negative predictions -- catastrophic for a target spanning both signs (observed in a prod chart for ``EnsQUAD ... target__monotonic_residual__Y``: R2=-9.97 with all predictions in [0, 2000] vs true values in [-2200, 500]). QUBE (cube root) is sign-preserving so it stays in.
    """
    if not (is_regression and ensembling_methods):
        return ensembling_methods
    _has_zero_crossing = False
    _sign_sensitive_in_methods = any(m in ensembling_methods for m in ("harm", "geo", "quad"))
    if not _sign_sensitive_in_methods:
        return ensembling_methods
    # ENS-P2-4 vectorised zero-crossing scan: flatten every member's train/val/test pred arrays into
    # one stacked float view and call np.nanmin / np.any once instead of looping per (member, split).
    _flat_arrays: list[np.ndarray] = []
    for _m in level_models_and_predictions:
        for _attr in ("val_preds", "test_preds", "train_preds"):
            _arr = getattr(_m, _attr, None)
            if _arr is None:
                continue
            _arr_f = np.asarray(_arr, dtype=np.float64).ravel()
            if _arr_f.size:
                _flat_arrays.append(_arr_f)
    if _flat_arrays:
        _stacked = np.concatenate(_flat_arrays)
        # NaN-safe: nanmin of abs handles fully-NaN arrays gracefully.
        with np.errstate(invalid="ignore"):
            _abs_min = float(np.nanmin(np.abs(_stacked))) if np.isfinite(_stacked).any() else np.inf
            _has_neg = bool(np.nanmin(_stacked) < 0) if np.isfinite(_stacked).any() else False
            _has_pos = bool(np.nanmax(_stacked) > 0) if np.isfinite(_stacked).any() else False
        if _abs_min < 1e-6 or (_has_neg and _has_pos):
            _has_zero_crossing = True
    if _has_zero_crossing:
        _filtered_methods = [m for m in ensembling_methods if m not in ("harm", "geo", "quad")]
        if verbose and len(_filtered_methods) != len(ensembling_methods):
            _dropped = [m for m in ensembling_methods if m not in _filtered_methods]
            logger.info(
                "[ensemble] gating out %s flavour(s): member "
                "predictions contain near-zero / sign-changing "
                "values (e.g. composite residual targets). "
                "Harmonic / geometric diverge near zero; quadratic "
                "loses input sign (sqrt(mean(p^2)) >= 0 always).",
                "/".join(_dropped),
            )
        ensembling_methods = _filtered_methods
    return ensembling_methods


def collapse_to_single_flavour_if_identical(
    ensembling_methods: list,
    early_exit_if_identical: bool,
    _gate_preds_for_check: Optional[Sequence[np.ndarray]],
    level_models_and_predictions: Sequence[Any],
    _gate_source_split: Any,
    res: dict,
    verbose: bool,
) -> list:
    """Collapse to a single 'arithm' flavour when every kept member's gate-source predictions are numerically identical.

    NO-GUARD-IDENTICAL: if every kept member's gate-source predictions are numerically identical (Pearson corr == 1.0 within atol AND elementwise close), every flavour collapses to the same arithmetic-mean output. Run just one flavour (arithm) and return early when explicitly enabled.

    P2-7: surface the collapse into res["_diversity"] so finalize can persist it. Operators reading ``ensembles_chosen[tt][tname]=='arithm'`` can then distinguish "arithm won by metric" from "all members were duplicates and arithm was the only viable flavour".
    """
    if not (early_exit_if_identical and _gate_preds_for_check is not None and len(level_models_and_predictions) > 1):
        return ensembling_methods
    try:
        _stack = np.vstack([np.asarray(p, dtype=np.float64).ravel() for p in _gate_preds_for_check])
        _ref = _stack[0]
        _all_close = all(np.allclose(_stack[i], _ref, atol=1e-9, rtol=1e-9) for i in range(1, _stack.shape[0]))
    except Exception:  # pragma: no cover -- defensive
        _all_close = False
    if _all_close:
        if verbose:
            logger.info("[ensemble] all members produce numerically identical predictions on split=%s; collapsing to a single 'arithm' flavour.", _gate_source_split)
        ensembling_methods = ["arithm"] if "arithm" in ensembling_methods else (ensembling_methods[:1] if ensembling_methods else [])
        # P2-7: surface the collapse into res["_diversity"] so finalize can persist it. Operators
        # reading ``ensembles_chosen[tt][tname]=='arithm'`` can then distinguish "arithm won by
        # metric" from "all members were duplicates and arithm was the only viable flavour".
        _div_block = res.setdefault("_diversity", {})
        _div_block["all_members_identical"] = True
        _div_block["collapsed_to_single_flavour"] = ensembling_methods[0] if ensembling_methods else None
        _div_block["split_used"] = _gate_source_split
    return ensembling_methods


def run_stacking_aware_gate(
    enable_stacking_aware_gate: bool,
    _gate_preds_for_check: Optional[Sequence[np.ndarray]],
    target_arr: Optional[np.ndarray],
    level_models_and_predictions: Sequence[Any],
    _ensemble_member_tags: List[str],
    stacking_gate_min_weight: float,
    use_nnls_weights: bool,
    res: dict,
    verbose: bool,
) -> Optional[np.ndarray]:
    """Run the composite_stacking NNLS-weight gate observationally; persist survivors/weights to ``res["_stacking_gate"]``.

    Returns the aligned weight vector for the downstream blend when ``use_nnls_weights`` is True (else None). AP7: assemble length-M weight vector aligned with ``level_models_and_predictions``. Members not in the survivor set get weight 0. Members whose gate-source was missing also get 0.
    """
    _nnls_weights_for_blend: Optional[np.ndarray] = None
    if not (enable_stacking_aware_gate and _gate_preds_for_check is not None and target_arr is not None):
        return _nnls_weights_for_blend
    try:
        from mlframe.training.composite import stacking_aware_gate as _saw_gate

        _saw_y = np.asarray(target_arr).reshape(-1)
        # Keep two parallel views: a member-tag -> array dict for the gate, plus a list of
        # tags in the same order as ``level_models_and_predictions`` so we can later assemble
        # a length-M weight vector. Duplicate tags fall back to suffix-disambiguation (rare;
        # short_model_tag collapses CB+CB_v2 to "cb" but the underlying tags from the suite
        # builder are typically unique).
        _ordered_tags: list[str] = []
        _seen_tags: set[str] = set()
        _saw_preds: dict[str, np.ndarray] = {}
        for i, p in enumerate(_gate_preds_for_check):
            _p_arr = np.asarray(p).reshape(-1)
            if _p_arr.shape[0] != _saw_y.shape[0]:
                _ordered_tags.append("")  # placeholder: this member was skipped in the gate
                continue
            _tag = _ensemble_member_tags[i]
            if _tag in _seen_tags:
                _tag = f"{_tag}#{i}"
            _seen_tags.add(_tag)
            _ordered_tags.append(_tag)
            _saw_preds[_tag] = _p_arr.astype(np.float64)
        if _saw_preds:
            _saw_survivors, _saw_weights = _saw_gate(_saw_preds, _saw_y, min_weight=stacking_gate_min_weight)
            res["_stacking_gate"] = {
                "survivors": list(_saw_survivors),
                "weights": dict(_saw_weights),
                "min_weight": float(stacking_gate_min_weight),
            }
            # AP7: assemble length-M weight vector aligned with ``level_models_and_predictions``.
            # Members not in the survivor set get weight 0. Members whose gate-source was missing
            # also get 0. When use_nnls_weights=False this is computed but unused (stamped only
            # for the observational report).
            if use_nnls_weights and _saw_survivors:
                _w_aligned = np.zeros(len(level_models_and_predictions), dtype=np.float64)
                _surv_set = set(_saw_survivors)
                for i, _tag in enumerate(_ordered_tags):
                    if _tag and _tag in _surv_set:
                        _w_aligned[i] = float(_saw_weights.get(_tag, 0.0))
                _wsum = float(_w_aligned.sum())
                if _wsum > 0.0:
                    _w_aligned = _w_aligned / _wsum
                    _nnls_weights_for_blend = _w_aligned
                    res["_stacking_gate"]["aligned_weights"] = _w_aligned.tolist()
                    res["_stacking_gate"]["applied_to_blend"] = True
                    if verbose:
                        logger.info(
                            "[ensemble] NNLS weights applied to blend: %s",
                            {_ensemble_member_tags[i]: float(_w_aligned[i]) for i in range(len(_w_aligned))},
                        )
                else:
                    res["_stacking_gate"]["applied_to_blend"] = False
            else:
                res["_stacking_gate"]["applied_to_blend"] = False
    except Exception as _saw_err:  # pragma: no cover -- defensive
        logger.warning("[ensemble] stacking_aware_gate failed: %s", _saw_err)
    return _nnls_weights_for_blend


def apply_quality_gate_kn(
    level_models_and_predictions: list,
    _gate_preds_for_check: Optional[Sequence[np.ndarray]],
    _gate_source_split: Any,
    _ensemble_member_tags: List[str],
    _ensemble_short_tags: List[str],
    ensemble_name: str,
    max_mae: float,
    max_std: float,
    max_mae_relative: float,
    max_std_relative: float,
    sample_weight: Optional[np.ndarray],
    group_ids: Optional[np.ndarray],
    ensembling_methods: list,
    res: dict,
    verbose: bool,
    compute_member_quality_gate_fn: Any = None,
) -> Tuple[list, List[str], List[str], str, float, float, float, float, Any]:
    """K>2 quality gate: drop members whose per-member MAE exceeds the relative/absolute thresholds.

    When the gate fires + members are dropped, rebuilds the ensemble name from surviving short-tags ([cb+xgb+lgb] for <=4, [N=K] otherwise) and disables the embedded per-flavour filter (set thresholds to 0) so the downstream flavour loop doesn't reprint the same exclusion line per flavour.

    Returns the (possibly reduced) member list + tag lists + the rebuilt ensemble_name + zeroed max_* thresholds + ``_gate_stats`` (None when the gate did not fire).

    ``compute_member_quality_gate_fn`` is injected by the caller so existing tests that monkey-patch ``_ensembling_score.compute_member_quality_gate`` see their patch hit -- without injection the helper would resolve the function via its own top-level import and silently bypass the test's spy. When ``None`` (the default), falls back to ``_ensembling_quality_gate.compute_member_quality_gate``.
    """
    if compute_member_quality_gate_fn is None:
        from .quality_gate import compute_member_quality_gate as compute_member_quality_gate_fn  # noqa: E501

    _gate_stats: Any = None
    if not (_gate_preds_for_check is not None and len(_gate_preds_for_check) > 2):
        return level_models_and_predictions, _ensemble_member_tags, _ensemble_short_tags, ensemble_name, max_mae, max_std, max_mae_relative, max_std_relative, _gate_stats
    _kept_idx, _excluded, _gate_stats = compute_member_quality_gate_fn(
        _gate_preds_for_check,
        max_mae=max_mae,
        max_std=max_std,
        max_mae_relative=max_mae_relative,
        max_std_relative=max_std_relative,
        sample_weight=sample_weight,
        group_ids=group_ids,
    )
    if verbose:
        # Per-member visual table: tag + MAE-vs-median + check/cross + reason
        _per_mae = _gate_stats.get("per_member_mae", [])
        _med_mae = _gate_stats.get("median_mae", 0.0)
        _excl_idx = {i for i, _ in _excluded}
        _kept_lbls = [f"{_ensemble_member_tags[i]} (MAE={float(_per_mae[i]):.4f})" for i in _kept_idx]
        _excl_lbls = [f"{_ensemble_member_tags[i]} (MAE={float(_per_mae[i]):.4f}, >{max_mae_relative:g}x median={_med_mae:.4f})" for i in _excl_idx]
        logger.info(
            "[ensemble] member quality gate (split=%s): kept %d/%d -- %s%s",
            _gate_source_split,
            len(_kept_idx),
            len(_gate_preds_for_check),
            ", ".join(_kept_lbls) if _kept_lbls else "(none)",
            ("; excluded: " + ", ".join(_excl_lbls)) if _excl_lbls else "",
        )
        if _excluded:
            # Approximate downstream-saved-work reporting so the user
            # sees ROI of the gate.
            _est_skipped_iters = len(_excluded) * len(ensembling_methods) * 2
            logger.info(
                "[ensemble] gate saves ~%d redundant per-flavor x per-split ensemble computations on these excluded members",
                _est_skipped_iters,
            )
        if _gate_stats.get("filter_too_restrictive"):
            logger.warning("[ensemble] gate would have excluded ALL members; falling back to original list (filter too restrictive for this combo)")
            # Low-5: stamp the bypass into the returned res so finalize / metadata reflect that
            # the gate was bypassed -- operators reading "all members kept" can now distinguish
            # the "all members within threshold" case from the "filter too restrictive" case.
            res["_gate_bypassed"] = {
                "reason": "filter_too_restrictive",
                "source_split": _gate_source_split,
                "would_have_excluded": [i for i, _ in (_excluded or [])],
            }
    if _excluded and not _gate_stats.get("filter_too_restrictive"):
        level_models_and_predictions = [level_models_and_predictions[i] for i in _kept_idx]
        # 2026-05-11: refresh ``ensemble_name`` to reflect the kept
        # members so downstream model_name_prefix / report titles
        # show [cb+xgb+lgb] (gate-survivors) instead of the original
        # [cb+xgb+lgb+linear] which advertises members that didn't
        # actually contribute to the ensemble. The caller stamped
        # the label assuming all members participate; we rebuild it
        # from the surviving tag list using the same caller-side
        # format ([cb+xgb+lgb] for <=4, [N=K] otherwise).
        try:
            # F2 fix (2026-05-11): use the SHORT tag list (cb / xgb / lgb / ...) for the rebuilt ensemble label rather than the full class names; matches the original short-label contract from core.py:5483 and keeps chart titles compact.
            _kept_tags = [_ensemble_short_tags[i] for i in _kept_idx]
            _re_label = "[" + "+".join(_kept_tags) + "]" if len(_kept_tags) <= 4 else f"[N={len(_kept_tags)}]"
            # Replace any [...] / [N=k] in ``ensemble_name`` with
            # the new label. The caller pattern is
            # ``f"{pre_pipeline}{_members_label} "`` so we look for
            # the first bracketed substring and substitute.
            if _re_mod.search(r"\[[^\]]+\]", ensemble_name):
                # Callable replacement -- re.sub does NOT interpret backreferences in
                # the return value of a callable, so any incidental ``\1`` / ``\g<...>`` /
                # backslash inside a model tag round-trips verbatim. A plain string
                # replacement would either crash on "invalid group reference" or silently
                # inject backslashes into the ensemble label.
                _label_value = _re_label
                ensemble_name = _re_mod.sub(
                    r"\[[^\]]+\]",
                    lambda _m, _v=_label_value: _v,
                    ensemble_name,
                    count=1,
                )
            else:
                # REGEX-RELABEL: caller passed an unbracketed name (or already-stripped one);
                # don't silently lose the new short label -- prepend it so log lines show the
                # surviving members instead of advertising the original full member list.
                ensemble_name = f"{_re_label} {ensemble_name}".rstrip() if ensemble_name else _re_label
        except Exception:  # pragma: no cover -- defensive
            pass
        # Disable the embedded per-flavor filter -- members are already
        # gated, so re-running it would just reprint the same exclusion
        # line per flavor (the noise this commit set out to eliminate).
        max_mae = 0.0
        max_std = 0.0
        max_mae_relative = 0.0
        max_std_relative = 0.0
    # Tag lists left unchanged when the gate didn't materially fire (caller's references stay valid).
    # When the gate dropped members we deliberately do NOT slice them here -- the caller's downstream
    # diversity / stacking / for-loop code reads tags using the same indices as level_models_and_predictions
    # AFTER the slice, but the original score_ensemble body kept the pre-gate tag lists intact and the
    # downstream code already realigns via _ensemble_member_tags.index(...). Returning them unchanged
    # preserves the original behavioural contract.
    return (
        level_models_and_predictions,
        _ensemble_member_tags,
        _ensemble_short_tags,
        ensemble_name,
        max_mae,
        max_std,
        max_mae_relative,
        max_std_relative,
        _gate_stats,
    )


def maybe_build_votenrank_leaderboard(
    res: dict,
    is_regression: bool,
    build_votenrank_leaderboard_flag: bool,
) -> None:
    """Build the votenrank Leaderboard over per-flavour metrics and stamp it under ``res["_leaderboard"]``.

    VOTENRANK: one table per (flavour x split.metric) cell; regression suites still get a leaderboard but with regression-appropriate columns only. Defined at the tail of ``ensembling.py`` (after the ``from .score import score_ensemble`` re-export), so import lazily.
    """
    if not build_votenrank_leaderboard_flag:
        return
    try:
        from .ensembling import _build_votenrank_leaderboard_from_results
        _lb_obj = _build_votenrank_leaderboard_from_results(res, is_regression=is_regression)
        if _lb_obj is not None:
            res["_leaderboard"] = _lb_obj
    except Exception as _lb_err:  # pragma: no cover -- defensive
        logger.warning("[ensemble] votenrank leaderboard build failed: %s", _lb_err)
