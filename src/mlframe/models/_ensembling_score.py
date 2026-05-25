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

    res = {}
    level_models_and_predictions = models_and_predictions

    # SINGLE-MEMBER: short-circuit when only one member is supplied. There is no ensemble to score;
    # historically the caller filtered K==1 but score_ensemble itself silently iterated every flavour
    # over a 1-member tensor (the rrf/median/harm reduction is a no-op). Returning a sentinel-only
    # dict ({"_reason": "single_member"}) signals "no ensemble built" to the caller without raising,
    # AND lets finalize / metadata distinguish "single-member suite" from "ensemble failed silently"
    # (Low-9). The sentinel key starts with ``_`` so it is filtered out by the ensemble-iteration
    # logic in callers that iterate ``res.items()`` for real flavours.
    if len(level_models_and_predictions) < 2:
        if verbose and len(level_models_and_predictions) == 1:
            logger.info("[ensemble] only one member supplied; nothing to ensemble. Returning sentinel result.")
        if len(level_models_and_predictions) == 1:
            res["_reason"] = "single_member"
            res["_n_members"] = 1
        else:
            res["_reason"] = "no_members"
            res["_n_members"] = 0
        return res

    # Uniformity gate: mixing a classifier (probs available) with a regressor (probs == None)
    # in one ensemble silently miscategorises the suite. The historical dispatch only
    # inspected member[0]; member[1] could disagree with no error. Validate up front.
    if level_models_and_predictions:
        def _has_probs(m) -> bool:
            # ``oof_probs`` MUST be inspected too: a member with val_probs=None but oof_probs
            # populated (rare: trainer stamped OOF but disabled val-metric computation; or
            # cross_val_predict-only fits) is classifier-like, not regressor-like. Pre-fix the
            # check skipped oof_probs and mis-classified those members as regression.
            return any(getattr(m, attr, None) is not None for attr in ("oof_probs", "val_probs", "test_probs", "train_probs"))

        _probs_flags = [_has_probs(m) for m in level_models_and_predictions]
        if len(set(_probs_flags)) > 1:
            _clf_idx = [i for i, f in enumerate(_probs_flags) if f]
            _reg_idx = [i for i, f in enumerate(_probs_flags) if not f]
            raise ValueError(
                "score_ensemble requires uniform member types: got a mix of classifier-like "
                f"(probs available, indices {_clf_idx}) and regressor-like (no probs, indices "
                f"{_reg_idx}) members. Split the suite into per-task lists before calling."
            )

    _first = level_models_and_predictions[0]
    if (
        getattr(_first, "oof_probs", None) is not None
        or _first.val_probs is not None
        or _first.test_probs is not None
        or _first.train_probs is not None
    ):
        is_regression = False
    else:
        is_regression = True
        ensure_prob_limits = False

    # RRF is a rank-fusion flavour that only makes sense on classifier
    # probabilities (where per-row ranks across the n_samples axis encode
    # "confidence ordering"). For regression there is no analogous per-sample
    # rank operation, so drop "rrf" silently from the candidate list rather
    # than fail late inside _process_single_ensemble_method.
    if is_regression and ensembling_methods:
        _pre = list(ensembling_methods)
        ensembling_methods = [m for m in ensembling_methods if m != "rrf"]
        if verbose and len(ensembling_methods) != len(_pre):
            logger.info(
                "[ensemble] target_type=REGRESSION: skipping rrf candidate (rank-fusion only meaningful on classifier probabilities)."
            )

    # Multi-level stacking requires OOF predictions on EVERY member: the level-2 (and deeper) meta-learner consumes
    # level-1 ensemble outputs as features, and if any member contributes an in-sample ``train_preds`` row instead of
    # a ``cross_val_predict`` OOF row the meta-learner sees leaked targets. Fail fast rather than silently fold the
    # leakage forward. Single-level (``max_ensembling_level == 1``) aggregation tolerates missing OOF by falling back
    # to ``train_*`` because no downstream meta-learner consumes the train slice in that case. Membership uses
    # ``isinstance(..., np.ndarray)`` for the same reason as ``_oof_or_train``: MagicMock test doubles fabricate
    # any attribute on access, so ``is None`` would never fire on a real-world stub.
    if max_ensembling_level > 1:
        _oof_attr = "oof_probs" if not is_regression else "oof_preds"
        _missing_oof = [
            i for i, m in enumerate(level_models_and_predictions)
            if not isinstance(getattr(m, _oof_attr, None), np.ndarray)
        ]
        if _missing_oof:
            raise ValueError(
                f"score_ensemble(max_ensembling_level={max_ensembling_level}) requires {_oof_attr} on every member; "
                f"members at indices {_missing_oof} are missing OOF. Re-train with oof_n_splits>=2 so cross_val_predict "
                f"OOFs are stamped on each model, or call with max_ensembling_level=1."
            )

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
    _gate_source_split = None
    _gate_preds_for_check: Optional[List[np.ndarray]] = None
    # GATE-DOUBLE-DIP / GATE-NO-OOF: prefer oof_* exclusively; fall back to val/test/train only
    # when require_oof_for_gate is False. When True and any member lacks OOF we WARN and skip the
    # gate entirely (better to run all members than to gate on the same surface the early-stopper /
    # test-set selector burned). The "all members must share a split" condition stays the same --
    # mixing splits across members would compare incomparable rows.
    _candidate_attrs = (
        ("oof_preds", "oof"),
        ("oof_probs", "oof"),
    )
    if not require_oof_for_gate:
        _candidate_attrs = _candidate_attrs + (
            ("val_preds", "val"),
            ("test_preds", "test"),
            ("train_preds", "train"),
            ("val_probs", "val"),
            ("test_probs", "test"),
            ("train_probs", "train"),
        )
    for _attr, _label in _candidate_attrs:
        _candidate = [getattr(m, _attr, None) for m in level_models_and_predictions]
        # MagicMock test doubles fabricate any attribute access, so ``p is not None`` would always pass; require an
        # actual numpy array to gate the source-split selection.
        if all(isinstance(p, np.ndarray) for p in _candidate):
            _gate_preds_for_check = _candidate
            _gate_source_split = _label
            break
    # COARSE-GATE-FALLBACK: when OOF is unavailable AND require_oof_for_gate=True, the fine-grained
    # 2.5x gate is intentionally skipped to avoid double-dipping on val. But that lets catastrophic
    # outliers (R^2=-4.75 alongside R^2=0.99 members) survive into the ensemble. Run a SECOND
    # coarse-threshold pass against the val/test/train fallback chain to catch only the disasters.
    # Marked separately in logs (split=val-coarse) so it can't be confused with the strict gate.
    _coarse_gate_active = False
    if (
        require_oof_for_gate
        and _gate_preds_for_check is None
        and (coarse_gate_max_mae_relative > 0.0 or coarse_gate_max_std_relative > 0.0)
    ):
        _coarse_fallback_attrs = (
            ("val_preds", "val-coarse"),
            ("test_preds", "test-coarse"),
            ("train_preds", "train-coarse"),
            ("val_probs", "val-coarse"),
            ("test_probs", "test-coarse"),
            ("train_probs", "train-coarse"),
        )
        for _attr, _label in _coarse_fallback_attrs:
            _candidate = [getattr(m, _attr, None) for m in level_models_and_predictions]
            if all(isinstance(p, np.ndarray) for p in _candidate):
                _gate_preds_for_check = _candidate
                _gate_source_split = _label
                _coarse_gate_active = True
                # Coarse pass replaces fine thresholds for the single call below; the embedded
                # per-flavor filter has already been zeroed before this block runs (or will be
                # zeroed in the kept-survivors branch below), so this only governs the single
                # compute_member_quality_gate invocation that follows.
                max_mae = 0.0
                max_std = 0.0
                max_mae_relative = float(coarse_gate_max_mae_relative)
                max_std_relative = float(coarse_gate_max_std_relative)
                break
        if verbose:
            if _coarse_gate_active:
                logger.warning(
                    "[ensemble] OOF unavailable; running COARSE gate on %s at %.1fx median MAE / %.1fx median STD (catches catastrophic outliers only; theoretical val double-dip risk acknowledged).",
                    _gate_source_split, max_mae_relative, max_std_relative,
                )
            else:
                logger.warning(
                    "[ensemble] require_oof_for_gate=True but at least one member lacks OOF preds AND no val/test/train fallback available; skipping quality gate entirely."
                )
    elif require_oof_for_gate and _gate_preds_for_check is None and verbose:
        logger.warning(
            "[ensemble] require_oof_for_gate=True but at least one member lacks OOF preds; coarse-gate disabled (coarse_gate_max_mae_relative<=0); skipping quality gate."
        )

    # 2026-05-11 (user request): TWO tag lists:
    # 1. ``_ensemble_member_tags`` -- full (shim-stripped) class / model names for the per-member quality-gate log line (operators want to see which exact model class was excluded).
    # 2. ``_ensemble_short_tags`` -- collapsed short tags (``cb`` / ``xgb`` / ``lgb`` / ``hgb`` / non-tree class name) for the rebuilt ensemble label after the gate. Without the short-collapse, the rebuilt label reads ``[CatBoostRegressor+XGBRegressor+LGBMRegressor]`` (38 chars) instead of ``[cb+xgb+lgb]`` (12 chars) -- bloated chart titles + breaks the original short-label contract from core.py.
    from mlframe.training._format import (
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

    # E4.1 (2026-05-21): K>2 absolute target-MAE catastrophic check. The legacy
    # peer-median gate at line 2106 below uses median MAE across members for K>2
    # but the median IS THE GROUP being judged -- if 2 of 4 members are catastrophic
    # the median MAE is HALF-catastrophic and the relative threshold (2.5x median)
    # may still let the disasters through. This block runs FIRST when target is
    # available for the gate-source split: any member whose MAE-to-target exceeds
    # the catastrophic ratio relative to the BEST member is removed before the
    # peer-median gate sees it. Conservative default (same 20x as K=2) -- only
    # catches absolute disasters, not honest variance.
    if (
        _gate_preds_for_check is not None
        and len(_gate_preds_for_check) > 2
        and k2_catastrophic_mae_ratio > 1.0
    ):
        _gate_target_arr_kn = None
        if _gate_source_split in ("test", "test-coarse"):
            _gate_target_arr_kn = test_target_arr
        elif _gate_source_split in ("val", "val-coarse"):
            _gate_target_arr_kn = val_target_arr
        elif _gate_source_split in ("train", "train-coarse"):
            _gate_target_arr_kn = train_target_arr
        elif _gate_source_split == "oof":
            _gate_target_arr_kn = train_target_arr
        if isinstance(_gate_target_arr_kn, np.ndarray) and _gate_target_arr_kn.size > 0:
            try:
                _t_kn = _gate_target_arr_kn.reshape(-1).astype(np.float64)
                _maes = []
                for _p in _gate_preds_for_check:
                    _p_arr = np.asarray(_p).reshape(-1).astype(np.float64)
                    if _p_arr.shape == _t_kn.shape:
                        _maes.append(float(np.nanmean(np.abs(_p_arr - _t_kn))))
                    else:
                        _maes.append(float("nan"))
                _maes_arr = np.asarray(_maes)
                # Defensive: if every per-member MAE was NaN (shape mismatch
                # on every member, or t_kn itself NaN), ``np.nanmin`` emits
                # ``RuntimeWarning: All-NaN slice encountered`` and returns
                # NaN. The subsequent ``> 0.0`` short-circuits the diagnostic
                # block so the warning is the only observable -- but it
                # pollutes the log with no actionable signal. Short-circuit
                # cleanly when all-NaN: skip the blowout diagnostic block.
                if np.all(np.isnan(_maes_arr)):
                    _best_mae = float("nan")
                else:
                    _best_mae = float(np.nanmin(_maes_arr))
                if _best_mae > 0.0 and math.isfinite(_best_mae):
                    _ratios = _maes_arr / _best_mae
                    # E4.3 (2026-05-21): surface a borderline-member diagnostic for operators.
                    # Any member whose target-MAE exceeds 0.5 * catastrophic_ratio (default
                    # 10x best) but stays below the catastrophic_ratio (20x best) is
                    # "borderline" -- not dropped, but worth knowing in metadata for the
                    # next-session tune of model selection.
                    _blowout_floor = 0.5 * float(k2_catastrophic_mae_ratio)
                    _blowout_idx = [
                        i for i in range(len(_gate_preds_for_check))
                        if math.isfinite(_ratios[i])
                        and _blowout_floor <= _ratios[i] < float(k2_catastrophic_mae_ratio)
                    ]
                    if _blowout_idx:
                        res["_diagnostic_mae_blowout"] = {
                            "split": _gate_source_split,
                            "borderline_idx": _blowout_idx,
                            "borderline_member_tags": [_ensemble_member_tags[i] for i in _blowout_idx],
                            "per_member_target_mae": _maes,
                            "best_mae": _best_mae,
                            "borderline_floor_ratio": _blowout_floor,
                            "catastrophic_ratio_threshold": float(k2_catastrophic_mae_ratio),
                        }
                        if verbose:
                            logger.warning(
                                "[ensemble] K>%d borderline-MAE diagnostic (split=%s): %d member(s) "
                                "between %.1fx and %.1fx best MAE -- not dropped, but worth review: %s",
                                2, _gate_source_split, len(_blowout_idx),
                                _blowout_floor, float(k2_catastrophic_mae_ratio),
                                [_ensemble_member_tags[i] for i in _blowout_idx],
                            )

                    _drop_mask = (_ratios >= float(k2_catastrophic_mae_ratio)) | ~np.isfinite(_maes_arr)
                    _kept_idx_kn = [i for i in range(len(_gate_preds_for_check)) if not _drop_mask[i]]
                    _excl_kn = [(i, f"target_mae={_maes[i]:.4f} vs best={_best_mae:.4f} ratio={_ratios[i]:.1f}x") for i in range(len(_gate_preds_for_check)) if _drop_mask[i]]
                    # E4.2 (2026-05-21): all-members-catastrophic sentinel. If every member's
                    # target-MAE either NaNs or exceeds the catastrophic threshold relative to the
                    # best one (rare edge case: best is barely-finite, others are way worse), the
                    # peer-median fallback would just keep everyone. Detect explicitly and signal
                    # the suite to skip ensembling for this target.
                    if len(_kept_idx_kn) < 2 and len(_excl_kn) >= 1:
                        # Either 1 or 0 survivors -- treat as "no honest ensemble possible".
                        # Use a separate sentinel so the caller (suite finalize) can decide:
                        # ship-best-single-member or fall back to dummy baseline.
                        res["_all_members_catastrophic"] = {
                            "split": _gate_source_split,
                            "n_survivors": len(_kept_idx_kn),
                            "n_members": len(_gate_preds_for_check),
                            "best_mae": _best_mae,
                            "per_member_target_mae": _maes,
                            "ratio_threshold": float(k2_catastrophic_mae_ratio),
                        }
                        if verbose:
                            logger.warning(
                                "[ensemble] K>2 all-members-catastrophic (split=%s): only %d/%d "
                                "members within %.1fx of best MAE=%.4f. Caller should skip "
                                "ensembling and ship the single survivor (or dummy baseline if 0).",
                                _gate_source_split, len(_kept_idx_kn),
                                len(_gate_preds_for_check), float(k2_catastrophic_mae_ratio),
                                _best_mae,
                            )
                    if len(_kept_idx_kn) >= 1 and len(_excl_kn) >= 1:
                        if verbose:
                            _drop_tags = [_ensemble_member_tags[i] for i, _ in _excl_kn]
                            _kept_tags_log = [_ensemble_member_tags[i] for i in _kept_idx_kn]
                            logger.warning(
                                "[ensemble] K>2 absolute-MAE catastrophic-drop (split=%s): dropping %s; keeping %s (ratio threshold=%.1fx, best MAE=%.4f).",
                                _gate_source_split, _drop_tags, _kept_tags_log,
                                float(k2_catastrophic_mae_ratio), _best_mae,
                            )
                        # Slice members + tags down to the survivor set; peer-median gate below
                        # then runs on the cleaner pool.
                        level_models_and_predictions = [level_models_and_predictions[i] for i in _kept_idx_kn]
                        _gate_preds_for_check = [_gate_preds_for_check[i] for i in _kept_idx_kn]
                        _ensemble_member_tags = [_ensemble_member_tags[i] for i in _kept_idx_kn]
                        _ensemble_short_tags = [_ensemble_short_tags[i] for i in _kept_idx_kn]
                        # Stamp diagnostic into the result dict so finalize / metadata reflect the
                        # pre-median-gate purge. Key uses ``_`` prefix per the sentinel-key contract
                        # (P0 #4 follow-up: callers filter ``_``-prefixed keys when building the
                        # per-target model list).
                        res["_kn_catastrophic_dropped"] = {
                            "split": _gate_source_split,
                            "dropped_idx": [i for i, _ in _excl_kn],
                            "best_mae": _best_mae,
                            "ratio_threshold": float(k2_catastrophic_mae_ratio),
                            "per_member_target_mae": _maes,
                        }
            except Exception as _kn_err:
                if verbose:
                    logger.warning("[ensemble] K>2 catastrophic-drop check raised %s; proceeding without drop.", _kn_err)

    # K2-CATASTROPHIC-DROPOUT (2026-05-21): when K == 2, the legacy peer-median gate
    # is symmetric (both members equidistant from (a+b)/2). When TARGET is available
    # for the gate-source split, this branch compares per-member MAE-to-target
    # directly and drops the obvious catastrophic outlier (e.g. Ridge MAE=7.89 vs
    # MLP MAE=11442 on the TVT-2026-05-21 run -- ratio 1450x, well above the 20x
    # default threshold). Without this branch the EnsARITHM blend on the same split
    # was MAE=5720 (half-broken).
    if (
        _gate_preds_for_check is not None
        and len(_gate_preds_for_check) == 2
        and k2_catastrophic_mae_ratio > 1.0
    ):
        # Match the gate source to its target array. ``oof_*`` aligns with train rows;
        # ``val/test/train`` (and their *-coarse variants) align with the matching
        # target_arr that score_ensemble already produced above.
        _gate_target_arr = None
        if _gate_source_split in ("test", "test-coarse"):
            _gate_target_arr = test_target_arr
        elif _gate_source_split in ("val", "val-coarse"):
            _gate_target_arr = val_target_arr
        elif _gate_source_split in ("train", "train-coarse"):
            _gate_target_arr = train_target_arr
        elif _gate_source_split == "oof":
            _gate_target_arr = train_target_arr
        if isinstance(_gate_target_arr, np.ndarray) and _gate_target_arr.size > 0:
            try:
                _t = _gate_target_arr.reshape(-1).astype(np.float64)
                _p0 = np.asarray(_gate_preds_for_check[0]).reshape(-1).astype(np.float64)
                _p1 = np.asarray(_gate_preds_for_check[1]).reshape(-1).astype(np.float64)
                if _p0.shape == _t.shape and _p1.shape == _t.shape:
                    # Use nanmean so a NaN-poisoned member doesn't make the gate skip.
                    _mae_a = float(np.nanmean(np.abs(_p0 - _t)))
                    _mae_b = float(np.nanmean(np.abs(_p1 - _t)))
                    _worse = max(_mae_a, _mae_b)
                    _better = min(_mae_a, _mae_b)
                    if _better > 0.0 and math.isfinite(_worse / _better) and _worse / _better >= float(k2_catastrophic_mae_ratio):
                        # Deterministic tiebreak: when both members are within fp64 noise of each
                        # other (``_mae_a == _mae_b``), pick the alphabetically-larger tag as the
                        # "worse" so removal is reproducible regardless of BLAS thread count /
                        # input ordering. Without this, an exact tie always drops index 1 (silent
                        # bias toward whichever member happened to be supplied second).
                        if _mae_a > _mae_b:
                            _worse_idx = 0
                        elif _mae_b > _mae_a:
                            _worse_idx = 1
                        else:
                            _tag_a = _ensemble_member_tags[0]
                            _tag_b = _ensemble_member_tags[1]
                            _worse_idx = 0 if _tag_a > _tag_b else 1
                        _kept_idx = [1 - _worse_idx]
                        # Capture the dropped tag BEFORE the slicing below overwrites the tag list.
                        _dropped_tag = _ensemble_member_tags[_worse_idx]
                        _kept_tag = _ensemble_member_tags[1 - _worse_idx]
                        if verbose:
                            logger.warning(
                                "[ensemble] K=2 catastrophic-dropout (split=%s): dropping %s (MAE=%.4f), keeping %s (MAE=%.4f); ratio=%.1fx >= %.1fx threshold.",
                                _gate_source_split,
                                _dropped_tag, _worse,
                                _kept_tag, _better,
                                _worse / _better, float(k2_catastrophic_mae_ratio),
                            )
                        level_models_and_predictions = [level_models_and_predictions[1 - _worse_idx]]
                        _ensemble_member_tags = [_kept_tag]
                        _ensemble_short_tags = [_ensemble_short_tags[1 - _worse_idx]]
                        # Refresh ensemble_name to reflect the surviving single member.
                        try:
                            import re as _re_mod2
                            _kept_tags = _ensemble_short_tags
                            _re_label = "[" + "+".join(_kept_tags) + "]"
                            if _re_mod2.search(r"\[[^\]]+\]", ensemble_name):
                                _label_value = _re_label
                                ensemble_name = _re_mod2.sub(
                                    r"\[[^\]]+\]",
                                    lambda _m, _v=_label_value: _v,
                                    ensemble_name, count=1,
                                )
                            else:
                                ensemble_name = f"{_re_label} {ensemble_name}".rstrip() if ensemble_name else _re_label
                        except Exception:
                            pass
                        # Drop into single-member short-circuit: with K=1 surviving,
                        # there's no ensemble to score. Return the sentinel-only dict
                        # exactly as the upstream "single_member" branch (line ~1785)
                        # does. Operators see "_reason: k2_catastrophic_dropout" in
                        # the result and can choose to keep just the surviving member.
                        res["_reason"] = "k2_catastrophic_dropout"
                        res["_n_members"] = 1
                        res["_dropped_member"] = _dropped_tag
                        res["_kept_member"] = _kept_tag
                        res["_k2_mae_ratio"] = _worse / _better
                        return res
            except Exception as _k2_err:
                if verbose:
                    logger.warning("[ensemble] K=2 catastrophic-dropout check raised %s; proceeding without drop.", _k2_err)

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

