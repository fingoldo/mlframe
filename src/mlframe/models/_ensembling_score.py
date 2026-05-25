"""Main ``score_ensemble`` entry point for ``mlframe.models.ensembling``.

Split out of ``ensembling.py`` to keep the parent below the 1k-line monolith
threshold. The parent re-exports ``score_ensemble`` so historical
``from mlframe.models.ensembling import score_ensemble`` imports continue
to resolve.

Most of the heavy work happens via parent-module helpers
(``compute_member_quality_gate``, ``_process_single_ensemble_method``,
``ensemble_probabilistic_predictions``, ``compute_high_correlation_pairs``,
``_build_votenrank_leaderboard_from_results``); imported lazily here to
dodge the ``ensembling -> _ensembling_score -> ensembling`` import cycle.
"""
from __future__ import annotations

import copy
import logging
import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Shared helpers + constants come from the leaf; sibling-defined dispatchers
# (quality_gate / predict / process_method) come from their own siblings.
# No ``from .ensembling import`` here: the parent re-imports this module at
# its bottom; routing every dependency through leaves breaks the cycle.
from ._ensembling_base import (  # noqa: F401
    SIMPLE_ENSEMBLING_METHODS,
    compute_high_correlation_pairs,
)
from ._ensembling_quality_gate import compute_member_quality_gate
from ._ensembling_predict import ensemble_probabilistic_predictions
from ._ensembling_process_method import _process_single_ensemble_method
from ._ensembling_score_validate import _validate_score_ensemble_inputs
from ._ensembling_score_gate import (
    catastrophic_drop_k2,
    catastrophic_drop_kn,
    select_gate_source_split,
)
# ``_build_votenrank_leaderboard_from_results`` lives in ``ensembling.py``
# (defined after this sibling is loaded), so it can only be imported lazily
# inside the call site that uses it.
from joblib import delayed  # noqa: F401
from pyutilz.parallel import cpu_count_physical, parallel_run  # noqa: F401
from pyutilz.pythonlib import is_jupyter_notebook  # noqa: F401

# Use the parent module's logger name so caplog filters on
# ``"mlframe.models.ensembling"`` continue to capture our records.
# The sibling lives at ``mlframe.models._ensembling_score`` but the public
# API surface (and the tests that assert on log lines) all reference the
# parent module name.
logger = logging.getLogger("mlframe.models.ensembling")


def score_ensemble(
    models_and_predictions: Sequence,
    ensemble_name: str,
    target: pd.Series = None,
    train_idx: np.ndarray = None,
    test_idx: np.ndarray = None,
    val_idx: np.ndarray = None,
    df: pd.DataFrame = None,
    train_target: pd.Series = None,
    test_target: pd.Series = None,
    val_target: pd.Series = None,
    target_label_encoder: object = None,
    # Outlier-member-filter thresholds. The historical absolute defaults
    # (``max_mae=0.05``, ``max_std=0.06``) excluded all 6 members of a
    # uniform tree-model suite (CB / XGB / LGB x 2 weight schemas) on
    # the 2026-04-24 prod log -- turning the filter into a no-op + 36
    # noisy WARN lines per ensemble. Defaults flipped to relative
    # (``2.5xmedian``); pass non-zero ``max_mae`` / ``max_std`` to keep
    # the legacy behaviour.
    max_mae: float = 0.0,
    max_std: float = 0.0,
    max_mae_relative: float = 2.5,
    max_std_relative: float = 2.5,
    ensure_prob_limits: bool = True,
    nbins: int = 100,
    ensembling_methods=SIMPLE_ENSEMBLING_METHODS,
    uncertainty_quantile: float = 0.1,
    normalize_stds_by_mean_preds: bool = False,
    custom_ice_metric: Callable = None,
    custom_rice_metric: Callable = None,
    subgroups: dict = None,
    max_ensembling_level: int = 1,
    n_features: int = None,
    n_jobs: int = None,
    min_samples_for_parallel: int = 10_000_000,
    verbose: bool = True,
    flag_degenerate_conf_subset: bool = True,
    degenerate_class_ratio: float = 0.01,
    diversity_corr_warn_threshold: float = 0.98,
    # NO-SW / NO-GROUPS: per-row weights and group identifiers, plumbed through the quality gate,
    # diversity check, member-quality metric aggregation, and downstream weight-fit. Both default
    # to None to preserve legacy unweighted-i.i.d. semantics; ctx auto-passes when available.
    sample_weight: Optional[np.ndarray] = None,
    group_ids: Optional[np.ndarray] = None,
    rrf_k: int = 60,
    # NO-GUARD-IDENTICAL: short-circuit when every member's predictions on the gate split match
    # numerically (Pearson corr == 1.0 AND elementwise close). One arithmetic-mean ensemble is
    # returned to skip every redundant flavour. Disabled by default so legacy reports keep their
    # shape; opt in via the suite caller.
    early_exit_if_identical: bool = False,
    # GATE-DOUBLE-DIP: when True, the quality-gate source is restricted to OOF predictions; legacy
    # callers that only stamped val_/test_/train_ preds fall through to the disabled gate path.
    # C-P1-1: default flipped to True. Pre-fix the gate silently fell through to ``val_preds``
    # (the same surface early-stopping already burned) for any member without OOF, biasing the
    # gate-survivors selection. The suite caller (_phase_train_one_target.py) NEVER overrode this
    # so every default suite ran with a val-biased gate. Setting to False explicitly re-enables
    # the legacy fallback chain (oof -> val -> test -> train) when the suite has not stamped OOF.
    require_oof_for_gate: bool = True,
    # COARSE-GATE-FALLBACK: when require_oof_for_gate=True AND OOF is unavailable, the strict gate
    # skips entirely. That's the right call for FINE thresholds (2.5x median), but it lets
    # CATASTROPHIC outliers survive: 2026-05-21 prod log had an MLP with R^2=-4.75 sitting in the
    # ensemble alongside three R^2~0.99 members because no member stamped OOF. This fallback runs
    # a SECOND gate at a much higher relative threshold (5x median by default) against the
    # val/test/train fallback chain -- enough to drop the catastrophic disasters while leaving
    # honest near-median members alone. Setting to <=0 disables the coarse fallback entirely.
    coarse_gate_max_mae_relative: float = 5.0,
    coarse_gate_max_std_relative: float = 5.0,
    # K2-CATASTROPHIC-DROPOUT: when K == 2, the peer-median gate is symmetric
    # (both members are equidistant from (a+b)/2 by construction), so the
    # legacy K=2 branch returned kept-all unconditionally. TVT-2026-05-21 had
    # Ridge MAE=7.89 alongside MLP MAE=11442 (ratio = 1450x); the ensemble
    # arithm-mean was MAE=5720 -- half-broken. When true target is available
    # for the gate-source split, this NEW gate compares per-member MAE-to-target
    # directly and drops the obvious catastrophic outlier (ratio >= threshold).
    # Conservative default 20.0 -- only catches disasters, not normal variance
    # between honest models (Ridge vs LightGBM typically differ by <2x MAE).
    # Set <= 1.0 to disable.
    k2_catastrophic_mae_ratio: float = 20.0,
    # VOTENRANK: build a votenrank.Leaderboard over the resulting per-flavour metrics and stamp it
    # in the returned dict under ``_leaderboard``. Defaults True for classification; regression-only
    # flavours skip rank-based methods automatically.
    build_votenrank_leaderboard: bool = True,
    # Stacking-aware gate hook. When True, runs the NNLS-weight gate from composite_stacking on the
    # ensemble's OOF predictions and persists the survivors / weights under ``_stacking_gate``. The
    # gate is observational unless the suite caller wires it into a follow-up linear stack.
    # C-P1-5: default flipped to True. The gate is observational (does not drop members) and
    # surfaces per-member NNLS-weight info operators want for audit; previously this required the
    # suite caller to opt in, so the default suite path lost the diagnostics entirely.
    enable_stacking_aware_gate: bool = True,
    stacking_gate_min_weight: float = 0.05,
    # AP7: when True, the NNLS weights computed by ``stacking_aware_gate`` are fed into
    # ``combine_probs`` as ``precomputed_weights`` (replacing the uniform 1/M weight on
    # arithm / harm / quad / qube / geo flavours). Default True -- the gate already runs
    # observationally so wiring its output into the blend is the natural finalisation. Set
    # False to restore the legacy uniform-mean behaviour while keeping the NNLS diagnostic.
    use_nnls_weights: bool = True,
    # P1-7: optional auto-drop of one member from each high-correlation pair.
    # ``None`` preserves the observational-only default (just WARNs + stamps to _diversity);
    # passing a float in (0, 1] activates auto-drop when any pair's |corr| exceeds the floor.
    # The MEMBER WITH HIGHER MEAN ABSOLUTE GATE-METRIC (mae from the gate) is dropped, so the
    # surviving member is the one closer to the median.
    auto_drop_diversity_above: Optional[float] = None,
    **kwargs,
):
    """Compares different ensembling methods for a list of models.

    Parameters
    ----------
    n_jobs : int, optional
        Number of parallel jobs. If None, automatically determined based on
        sample count and min_samples_for_parallel. Use 1 for sequential processing.
    min_samples_for_parallel : int, default=1_000_000
        Minimum number of samples required to enable parallel processing when n_jobs is None.
    """

    level_models_and_predictions = models_and_predictions
    res, is_regression, ensembling_methods, ensure_prob_limits = _validate_score_ensemble_inputs(
        level_models_and_predictions=level_models_and_predictions,
        ensembling_methods=ensembling_methods,
        ensure_prob_limits=ensure_prob_limits,
        max_ensembling_level=max_ensembling_level,
        verbose=verbose,
    )
    if res:
        return res

    # Determine sample count for parallelization decision
    first_pred = level_models_and_predictions[0]
    if first_pred.val_probs is not None:
        n_samples = len(first_pred.val_probs)
    elif first_pred.val_preds is not None:
        n_samples = len(first_pred.val_preds)
    else:
        n_samples = 0

    # Determine n_jobs if not specified
    effective_n_jobs = n_jobs
    if effective_n_jobs is None:
        if n_samples >= min_samples_for_parallel and not is_jupyter_notebook():
            effective_n_jobs = min(len(ensembling_methods), cpu_count_physical())
        else:
            effective_n_jobs = 1

    # Convert pandas Series to numpy arrays before parallel section to avoid pickling issues
    train_target_arr = train_target.to_numpy() if isinstance(train_target, pd.Series) else train_target
    test_target_arr = test_target.to_numpy() if isinstance(test_target, pd.Series) else test_target
    val_target_arr = val_target.to_numpy() if isinstance(val_target, pd.Series) else val_target
    target_arr = target.to_numpy() if isinstance(target, pd.Series) else target

    # ONE-pass member quality gate before iterating ensemble flavors. The previous behaviour ran the same outlier
    # filter inside ``ensemble_probabilistic_predictions`` once per flavor x split, which on a 4-model x 5-flavor x
    # (full+conf) x 2-split layout printed the same "ens member N excluded ..." line ~20x per suite call. Compute
    # ONCE here, log the decision once, then pass only kept members to the flavor loop and disable the embedded
    # filter so no duplicate prints fire.
    #
    # Source ordering: OOF preds/probs come FIRST -- the gate's job is to drop members whose preds are outliers vs
    # the ensemble median, and val_preds are already burned for early-stopping (gating on them double-dips val).
    # OOF preds are the only honest train-side signal (cross_val_predict held-out rows). Fallback chain: oof_* ->
    # val_* -> test_* -> train_* preserves the legacy behaviour for members trained without oof_n_splits.
    (
        _gate_preds_for_check,
        _gate_source_split,
        _coarse_gate_active,
        max_mae,
        max_std,
        max_mae_relative,
        max_std_relative,
    ) = select_gate_source_split(
        level_models_and_predictions=level_models_and_predictions,
        require_oof_for_gate=require_oof_for_gate,
        coarse_gate_max_mae_relative=coarse_gate_max_mae_relative,
        coarse_gate_max_std_relative=coarse_gate_max_std_relative,
        max_mae=max_mae,
        max_std=max_std,
        max_mae_relative=max_mae_relative,
        max_std_relative=max_std_relative,
        verbose=verbose,
    )

    # 2026-05-11 (user request): TWO tag lists:
    # 1. ``_ensemble_member_tags`` -- full (shim-stripped) class / model names for the per-member quality-gate log line (operators want to see which exact model class was excluded).
    # 2. ``_ensemble_short_tags`` -- collapsed short tags (``cb`` / ``xgb`` / ``lgb`` / ``hgb`` / non-tree class name) for the rebuilt ensemble label after the gate. Without the short-collapse, the rebuilt label reads ``[CatBoostRegressor+XGBRegressor+LGBMRegressor]`` (38 chars) instead of ``[cb+xgb+lgb]`` (12 chars) -- bloated chart titles + breaks the original short-label contract from core.py.
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
        # F2 fix (2026-05-11): short-tag ALWAYS derived from the underlying CLASS, not from ``model_name`` which carries augmentations like ``"TVT MTTR=11497.66"`` that would defeat the startswith() prefix checks (``startswith("CatBoost")`` etc.).
        _ensemble_short_tags.append(_short_tag(_model_obj))

    (
        level_models_and_predictions,
        _gate_preds_for_check,
        _ensemble_member_tags,
        _ensemble_short_tags,
    ) = catastrophic_drop_kn(
        level_models_and_predictions=level_models_and_predictions,
        _gate_preds_for_check=_gate_preds_for_check,
        _gate_source_split=_gate_source_split,
        _ensemble_member_tags=_ensemble_member_tags,
        _ensemble_short_tags=_ensemble_short_tags,
        train_target_arr=train_target_arr,
        val_target_arr=val_target_arr,
        test_target_arr=test_target_arr,
        k2_catastrophic_mae_ratio=k2_catastrophic_mae_ratio,
        verbose=verbose,
        res=res,
    )

    (
        level_models_and_predictions,
        _ensemble_member_tags,
        _ensemble_short_tags,
        ensemble_name,
        _k2_early_return,
    ) = catastrophic_drop_k2(
        level_models_and_predictions=level_models_and_predictions,
        _gate_preds_for_check=_gate_preds_for_check,
        _gate_source_split=_gate_source_split,
        _ensemble_member_tags=_ensemble_member_tags,
        _ensemble_short_tags=_ensemble_short_tags,
        ensemble_name=ensemble_name,
        train_target_arr=train_target_arr,
        val_target_arr=val_target_arr,
        test_target_arr=test_target_arr,
        k2_catastrophic_mae_ratio=k2_catastrophic_mae_ratio,
        verbose=verbose,
        res=res,
    )
    if _k2_early_return:
        return res

    if _gate_preds_for_check is not None and len(_gate_preds_for_check) > 2:
        _kept_idx, _excluded, _gate_stats = compute_member_quality_gate(
            _gate_preds_for_check,
            max_mae=max_mae,
            max_std=max_std,
            max_mae_relative=max_mae_relative,
            max_std_relative=max_std_relative,
            sample_weight=sample_weight,
            group_ids=group_ids,
        )
        if verbose:
            # Per-member visual table: tag + MAE-vs-median + вњ“/вњ— + reason
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
                import re as _re_mod

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

    # Observational diversity check: pairs of kept members whose val-pred Pearson correlation exceeds the threshold are
    # surfaced via WARN + persisted to the returned dict under ``_diversity.high_correlation_pairs``. Defaults to
    # observational-only (no member removed); pass ``auto_drop_diversity_above`` to actually drop one of each pair.
    _high_corr_pairs, _div_split_used = compute_high_correlation_pairs(
        level_models_and_predictions,
        _ensemble_member_tags,
        threshold=diversity_corr_warn_threshold,
    )
    _drop_floor = auto_drop_diversity_above
    _auto_dropped: list[str] = []
    for _pair in _high_corr_pairs:
        _would_drop_msg = ""
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

    # I2 (2026-05-11): for regression, gate-out harmonic / geometric ensemble flavours when ANY member's predictions contain near-zero values or sign changes. Harmonic mean = N / sum(1/p) and geometric mean = exp(mean(log p)) both diverge / are undefined on signals that cross zero. Symptom seen in the prod log: ``EnsHARM ... RMSE=178.84 MaxError=55206`` and ``RMSE=1299.55 MaxError=920165`` on composite residuals which cluster around zero by construction.
    #
    # 2026-05-12 (user feedback): also gate-out QUAD (quadratic mean =
    # sqrt(mean(p^2))) on sign-changing targets. Squaring loses the sign of
    # the input by construction, so QUAD ALWAYS emits non-negative
    # predictions -- catastrophic for a target spanning both signs (the
    # prod chart for ``EnsQUAD ... TVT__monotonic_residual__Y`` showed
    # R2=-9.97 with all predictions in [0, 2000] vs true values in
    # [-2200, 500]). QUBE (cube root) is sign-preserving so it stays in.
    if is_regression and ensembling_methods:
        _has_zero_crossing = False
        _sign_sensitive_in_methods = any(m in ensembling_methods for m in ("harm", "geo", "quad"))
        if _sign_sensitive_in_methods:
            # ENS-P2-4 vectorised zero-crossing scan: flatten every member's
            # train/val/test pred arrays into one stacked float view and call
            # np.nanmin / np.any once instead of looping per (member, split).
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

    # NO-GUARD-IDENTICAL: if every kept member's gate-source predictions are numerically identical
    # (Pearson corr == 1.0 within atol AND elementwise close), every flavour collapses to the same
    # arithmetic-mean output. Run just one flavour (arithm) and return early when explicitly enabled.
    if early_exit_if_identical and _gate_preds_for_check is not None and len(level_models_and_predictions) > 1:
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

    # Stacking-aware gate (composite_stacking.stacking_aware_gate). Observational by default: runs
    # NNLS over member OOF preds, persists survivors / weights on ``res["_stacking_gate"]``. The
    # caller can choose to feed the survivors into a follow-up linear stack at the suite level.
    _nnls_weights_for_blend: Optional[np.ndarray] = None
    if enable_stacking_aware_gate and _gate_preds_for_check is not None and target_arr is not None:
        try:
            from mlframe.training.composite_stacking import stacking_aware_gate as _saw_gate

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

    for ensembling_level in range(max_ensembling_level):

        next_level_models_and_predictions = []

        # Common parameters for all ensemble methods
        common_params = dict(
            level_models_and_predictions=level_models_and_predictions,
            is_regression=is_regression,
            ensembling_level=ensembling_level,
            ensemble_name=ensemble_name,
            target=target_arr,
            train_idx=train_idx,
            test_idx=test_idx,
            val_idx=val_idx,
            train_target=train_target_arr,
            test_target=test_target_arr,
            val_target=val_target_arr,
            target_label_encoder=target_label_encoder,
            max_mae=max_mae,
            max_std=max_std,
            max_mae_relative=max_mae_relative,
            max_std_relative=max_std_relative,
            ensure_prob_limits=ensure_prob_limits,
            nbins=nbins,
            uncertainty_quantile=uncertainty_quantile,
            normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
            custom_ice_metric=custom_ice_metric,
            custom_rice_metric=custom_rice_metric,
            subgroups=subgroups,
            n_features=n_features,
            verbose=verbose,
            kwargs=kwargs,
            flag_degenerate_conf_subset=flag_degenerate_conf_subset,
            degenerate_class_ratio=degenerate_class_ratio,
            sample_weight=sample_weight,
            rrf_k=rrf_k,
            precomputed_weights=_nnls_weights_for_blend,
        )

        if len(ensembling_methods) > 1 and effective_n_jobs > 1:
            # loky pickles kwargs across worker boundaries; closure-captured metrics/lambdas
            # blow up in workers. Pre-check so we can fall back to sequential with a clear warning.
            try:
                import pickle

                pickle.dumps((custom_ice_metric, custom_rice_metric, kwargs))
            except (pickle.PicklingError, AttributeError, TypeError) as exc:
                logger.warning(
                    "ensembling: falling back to sequential -- one of " "custom_ice_metric / custom_rice_metric / kwargs is not picklable: %s",
                    exc,
                )
                effective_n_jobs = 1

        if len(ensembling_methods) > 1 and effective_n_jobs > 1:
            # Parallel processing -- loky + tiny max_nbytes keeps arrays in-memory (no spill) per pre-existing tuning
            results = parallel_run(
                [delayed(_process_single_ensemble_method)(ensemble_method=method, **common_params) for method in ensembling_methods],
                n_jobs=effective_n_jobs,
                backend="loky",
                max_nbytes="1K",
                verbose=0,
            )
            for internal_method, next_ens_results, conf_results in results:
                res[internal_method] = next_ens_results
                next_level_models_and_predictions.append(next_ens_results)
                if conf_results is not None:
                    res[internal_method + " conf"] = conf_results
        else:
            # Sequential processing
            for ensemble_method in ensembling_methods:
                internal_method, next_ens_results, conf_results = _process_single_ensemble_method(ensemble_method=ensemble_method, **common_params)
                res[internal_method] = next_ens_results
                next_level_models_and_predictions.append(next_ens_results)
                if conf_results is not None:
                    res[internal_method + " conf"] = conf_results

        level_models_and_predictions = next_level_models_and_predictions

    # VOTENRANK: build a Leaderboard over the per-flavour metrics for downstream rank-aggregation
    # diagnostics (Borda, Copeland, Dowdall, mean ranking). One table per (flavour x split.metric)
    # cell; regression suites still get a leaderboard but with regression-appropriate columns
    # only. The result is stamped under ``res["_leaderboard"]`` and exposes a ``to_csv`` helper
    # for the F4b main.py wiring to write to ``output_config.data_dir/<suite>.leaderboard.csv``.
    if build_votenrank_leaderboard:
        try:
            # Lazy import: defined at the tail of ``ensembling.py`` (after the
            # ``from ._ensembling_score import score_ensemble`` re-export), so
            # it is not yet bound when this sibling is initially loaded.
            from .ensembling import _build_votenrank_leaderboard_from_results
            _lb_obj = _build_votenrank_leaderboard_from_results(res, is_regression=is_regression)
            if _lb_obj is not None:
                res["_leaderboard"] = _lb_obj
        except Exception as _lb_err:  # pragma: no cover -- defensive
            logger.warning("[ensemble] votenrank leaderboard build failed: %s", _lb_err)
    return res

