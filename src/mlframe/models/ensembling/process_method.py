"""Per-method ensemble dispatcher for ``mlframe.models.ensembling``.

Split out of ``ensembling.py`` to keep the parent below the 1k-line monolith
threshold. ``_process_single_ensemble_method`` is the per-method worker that
``score_ensemble`` fans out via joblib; the parent re-exports it so
``score_ensemble``'s call sites continue to resolve identity-equal.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .base import build_predictive_kwargs
from .predict import ensemble_probabilistic_predictions

logger = logging.getLogger("mlframe.models.ensembling")


def _select_member_probs(member, split: str, use_calibrated: bool) -> Optional[np.ndarray]:
    """Return AP12-calibrated probs when available + opt-in, else raw probs.

    AP12 ``post_calibrate_model`` stamps ``.calibrated_<split>_probs`` on the
    inner model object when ``CalibrationConfig.policy_auto_pick=True`` (Wave 9
    / 13 / 16). When opt-out OR no AP12 stamp on the member, falls back to raw
    ``.<split>_probs``. The RRF flavour is rank-based and intentionally bypasses
    this helper (scale-invariant by design).

    Lookup order on the member: ``calibrated_<split>_probs`` direct attribute,
    then ``<inner_model>.calibrated_<split>_probs``, then raw ``<split>_probs``.
    SimpleNamespace members carried by the suite expose the raw split arrays at
    top level and may also expose the calibrated mirror after post-calibration
    propagates the stamp.
    """
    if use_calibrated:
        cal_attr = f"calibrated_{split}_probs"
        cal = getattr(member, cal_attr, None)
        if cal is None:
            inner = getattr(member, "model", None)
            if inner is not None:
                cal = getattr(inner, cal_attr, None)
        if isinstance(cal, np.ndarray):
            return cal
    return getattr(member, f"{split}_probs", None)


def _process_single_ensemble_method(
    ensemble_method: str,
    level_models_and_predictions: Sequence,
    is_regression: bool,
    ensembling_level: int,
    ensemble_name: str,
    target: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    val_idx: np.ndarray,
    train_target: np.ndarray,
    test_target: np.ndarray,
    val_target: np.ndarray,
    target_label_encoder: object,
    max_mae: float,
    max_std: float,
    max_mae_relative: float,
    max_std_relative: float,
    ensure_prob_limits: bool,
    nbins: int,
    uncertainty_quantile: float,
    normalize_stds_by_mean_preds: bool,
    custom_ice_metric: Callable,
    custom_rice_metric: Callable,
    subgroups: dict,
    n_features: int,
    verbose: bool,
    kwargs: dict,
    flag_degenerate_conf_subset: bool = True,
    degenerate_class_ratio: float = 0.01,
    sample_weight: Optional[np.ndarray] = None,
    rrf_k: int = 60,
    precomputed_weights: Optional[np.ndarray] = None,
    use_ap12_calibrated_probs: bool = True,
) -> tuple:
    """Process a single ensemble method. Returns (method_name, results, conf_results, next_level_pred)."""
    # E1.1 opt-out: these lazy imports break a circular dep with mlframe.training.
    # The Parallel dispatch at line 2539 uses ``backend="loky"`` (process workers,
    # not threads), so each fork has its own Python interpreter -- no shared
    # import state to race against. Safe to keep lazy here.
    from mlframe.training import train_and_evaluate_model  # joblib-import-race-ok
    from mlframe.training.trainer import _build_configs_from_params  # joblib-import-race-ok

    # 2026-05-13 (bug fix): val_preds / test_preds may be ``None`` when the
    # corresponding split metric computation was disabled at suite level
    # (``ReportingConfig.compute_valset_metrics=False`` /
    # ``compute_testset_metrics=False``). Pre-fix the bare
    # ``el.val_preds.reshape(-1, 1)`` raised AttributeError on the first
    # ``None``-valued member. Mirrors the existing train-side guard at
    # the bottom of this function (line ~870). When NO members have val
    # preds, the ensemble call gets an empty tuple; downstream
    # ``ensemble_probabilistic_predictions`` already returns
    # ``(None, None, None)`` for that case (line ~438).
    # NNLS-weight alignment: when ``precomputed_weights`` is supplied it is length-M aligned to
    # ``level_models_and_predictions``. Each per-split branch below filters None members so we
    # build a parallel mask + slice the weights before handing them to ``ensemble_probabilistic_predictions``.
    _pcw_full = (
        np.asarray(precomputed_weights, dtype=np.float64).reshape(-1)
        if precomputed_weights is not None else None
    )

    def _slice_weights(mask: list[bool]) -> Optional[np.ndarray]:
        if _pcw_full is None:
            return None
        if _pcw_full.shape[0] != len(mask):
            return None
        return _pcw_full[np.asarray(mask, dtype=bool)]

    # RRF is rank-based and scale-invariant by construction (per ``rrf_ensemble`` docstring at
    # _ensembling_base.py:493). Calibrating before rank fusion is a no-op on rank order and a small loss on the
    # bottom-of-ladder discrimination because isotonic / Platt compress the tails; force the raw-probs read for
    # RRF regardless of the knob so the historical RRF semantics are preserved bit-for-bit.
    _use_cal = use_ap12_calibrated_probs and ensemble_method != "rrf"

    if not is_regression:
        _val_probs_list = [_select_member_probs(el, "val", _use_cal) for el in level_models_and_predictions]
        _val_mask = [p is not None for p in _val_probs_list]
        _val_preds = [p for p in _val_probs_list if p is not None]
        predictions = iter(_val_preds)
    else:
        _val_mask = [el.val_preds is not None for el in level_models_and_predictions]
        _val_preds = [el.val_preds for el in level_models_and_predictions if el.val_preds is not None]
        # reshape(shape[0], -1) keeps the ROW axis: (N,)->(N,1) for single-target regression
        # (bit-identical to the old reshape(-1,1)), (N,K)->(N,K) for multi_target_regression.
        # The old reshape(-1,1) flattened (N,K)->(N*K,1), so ensemble_probabilistic_predictions
        # saw N*K "rows" and returned confident_indices up to N*K, which then IndexError'd when
        # indexing the per-row test_idx / *_target / *_ensembled_predictions arrays (size N).
        predictions = (p.reshape(p.shape[0], -1) for p in _val_preds)

    # No member carries this split's preds (e.g. compute_valset_metrics=False): skip the ensemble call.
    # ensemble_probabilistic_predictions now raises on empty input, so the empty case is guarded here.
    if len(_val_preds) == 0:
        val_ensembled_predictions, val_confident_indices = None, None
    else:
        val_ensembled_predictions, _, val_confident_indices = ensemble_probabilistic_predictions(
            *predictions,
            ensemble_method=ensemble_method,
            max_mae=max_mae,
            max_std=max_std,
            max_mae_relative=max_mae_relative,
            max_std_relative=max_std_relative,
            ensure_prob_limits=ensure_prob_limits,
            uncertainty_quantile=uncertainty_quantile,
            normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
            verbose=verbose,
            sample_weight=sample_weight,
            rrf_k=rrf_k,
            precomputed_weights=_slice_weights(_val_mask),
        )

    # 2026-05-13: same ``None``-guard for test_preds / test_probs.
    if not is_regression:
        _test_probs_list = [_select_member_probs(el, "test", _use_cal) for el in level_models_and_predictions]
        _test_mask = [p is not None for p in _test_probs_list]
        _test_preds = [p for p in _test_probs_list if p is not None]
        predictions = iter(_test_preds)
    else:
        _test_mask = [el.test_preds is not None for el in level_models_and_predictions]
        _test_preds = [el.test_preds for el in level_models_and_predictions if el.test_preds is not None]
        predictions = (p.reshape(p.shape[0], -1) for p in _test_preds)  # preserve row axis for multi-target (see val-branch note)

    if len(_test_preds) == 0:
        test_ensembled_predictions, test_confident_indices = None, None
    else:
        test_ensembled_predictions, _, test_confident_indices = ensemble_probabilistic_predictions(
            *predictions,
            ensemble_method=ensemble_method,
            max_mae=max_mae,
            max_std=max_std,
            max_mae_relative=max_mae_relative,
            max_std_relative=max_std_relative,
            ensure_prob_limits=ensure_prob_limits,
            uncertainty_quantile=uncertainty_quantile,
            normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
            verbose=verbose,
            sample_weight=sample_weight,
            rrf_k=rrf_k,
            precomputed_weights=_slice_weights(_test_mask),
        )

    # Level-1 aggregation reads OOF predictions when present (the K-fold cross_val_predict output stamped by the trainer), not in-sample ``train_preds``/``train_probs`` -- the latter are produced by predicting on rows the model already saw during fit and leak that fit's residual structure into a meta-learner. Falling back to ``train_*`` keeps the path alive when OOF was unavailable (e.g. tiny datasets, multi-output paths where ``cross_val_predict`` skipped); a higher-level guard in ``score_ensemble`` raises when ``max_ensembling_level > 1`` AND any member is missing OOF.
    #
    # Existence check uses ``isinstance(..., np.ndarray)`` rather than ``is not None`` because MagicMock-based test doubles auto-fabricate any attribute access (mock.oof_probs returns a MagicMock, never None); falling back on the array-instance check keeps the production path identical while letting test mocks still hit the train_*/branch when they only stamp train_*/val_* arrays.
    #
    # ``_fallback_used`` collects member indices where the OOF attribute was absent and we silently consumed ``train_*``. Surfaced as a single WARN below so operators can audit suites running with cross_val_predict disabled -- pre-fix shape was completely silent and the level-1 "train" branch of every ensemble flavour got evaluated on leaked rows.
    _fallback_used: list[int] = []

    def _oof_or_train(el, oof_attr, train_attr, _idx, _prefer_calibrated: bool = False):
        # Prefer AP12-calibrated OOF / train probs when the knob is on; transparent fall-through to raw when the
        # stamp is missing (legacy / opt-out callers). OOF is the calibrator's fit source so most members will not
        # carry calibrated_oof_probs -- that's expected; the lookup is a no-op for them.
        if _prefer_calibrated:
            _cal_oof = getattr(el, f"calibrated_{oof_attr}", None)
            if isinstance(_cal_oof, np.ndarray):
                return _cal_oof
        _oof = getattr(el, oof_attr, None)
        if isinstance(_oof, np.ndarray):
            return _oof
        if _prefer_calibrated:
            _cal_train = getattr(el, f"calibrated_{train_attr}", None)
            if isinstance(_cal_train, np.ndarray):
                _fallback_used.append(_idx)
                return _cal_train
        _train = getattr(el, train_attr, None)
        if isinstance(_train, np.ndarray):
            _fallback_used.append(_idx)
        return _train

    if not is_regression:
        predictions = [_oof_or_train(el, "oof_probs", "train_probs", _i, _prefer_calibrated=_use_cal) for _i, el in enumerate(level_models_and_predictions)]
    elif (_oof_or_train(level_models_and_predictions[0], "oof_preds", "train_preds", 0) is not None):
        # Re-walk so every member's fallback decision is recorded (probe call above counts index 0 only if it fell back; clear and re-probe symmetrically across all members).
        _fallback_used.clear()
        predictions = [_oof_or_train(el, "oof_preds", "train_preds", _i) for _i, el in enumerate(level_models_and_predictions)]
        predictions = [el.reshape(el.shape[0], -1) if (el is not None) else el for el in predictions]  # preserve row axis for multi-target (see val-branch note)
    else:
        predictions = []

    if _fallback_used:
        logger.warning(
            "ensemble '%s' L%d: OOF preds missing on members %s; silently fell back to in-sample train_* arrays. "
            "Train-branch metrics for this ensemble are computed on LEAKED rows and will look optimistically biased. "
            "Re-train with oof_n_splits>=2 (cross_val_predict) so OOF preds are stamped on every model, or call "
            "score_ensemble(max_ensembling_level=1) accepting the train-fallback explicitly.",
            ensemble_method, ensembling_level, _fallback_used,
        )

    # train branch: ``predictions`` was built via ``_oof_or_train`` which preserves the per-member
    # order but may contain ``None`` entries when neither OOF nor train_* was available. Build the
    # alignment mask to slice precomputed_weights consistently with the per-split branches above.
    _train_mask = [(p is not None) for p in predictions] if predictions else []
    if not any(_train_mask):
        train_ensembled_predictions, train_confident_indices = None, None
    else:
        train_ensembled_predictions, _, train_confident_indices = ensemble_probabilistic_predictions(
            *predictions,
            ensemble_method=ensemble_method,
            max_mae=max_mae,
            max_std=max_std,
            max_mae_relative=max_mae_relative,
            max_std_relative=max_std_relative,
            ensure_prob_limits=ensure_prob_limits,
            uncertainty_quantile=uncertainty_quantile,
            normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
            verbose=verbose,
            sample_weight=sample_weight,
            rrf_k=rrf_k,
            precomputed_weights=_slice_weights(_train_mask) if _train_mask and _pcw_full is not None and len(_train_mask) == _pcw_full.shape[0] else None,
        )

    internal_ensemble_method = f"{ensemble_method} L{ensembling_level}" if ensembling_level > 0 else ensemble_method

    predictive_kwargs = build_predictive_kwargs(
        train_data=train_ensembled_predictions, test_data=test_ensembled_predictions, val_data=val_ensembled_predictions, is_regression=is_regression
    )

    if target is not None:
        target_kwargs = dict(target=target)
    else:
        target_kwargs = dict(train_target=train_target, test_target=test_target, val_target=val_target)

    # Pop params not accepted by _build_configs_from_params (they come from common_params in core.py)
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop("trainset_features_stats", None)
    kwargs_copy.pop("train_od_idx", None)
    kwargs_copy.pop("val_od_idx", None)
    # 2026-04-23 (coverage-gap test_ensembles_enabled_produces_ensemble_log):
    # ``common_params`` frequently carries ``drop_columns`` when the user
    # passes ``init_common_params={"drop_columns": [...]}``. The literal
    # ``drop_columns=[]`` below then collides with the ``**kwargs_copy``
    # splat two positions later, raising
    # ``TypeError: dict() got multiple values for keyword argument 'drop_columns'``.
    # Pop the caller's value -- the ensemble scorer intentionally sets
    # ``drop_columns=[]`` to avoid dropping anything its sub-models
    # already trained on (columns already stripped upstream).
    kwargs_copy.pop("drop_columns", None)
    # 2026-04-24 (fuzz extension): init_common_params is a prod
    # convention for passing PIPELINE COMPONENTS (not training
    # hyperparams), e.g.:
    #     init_common_params = {
    #         "category_encoder": ce.CatBoostEncoder(),
    #         "scaler": StandardScaler(),
    #         "imputer": SimpleImputer(strategy="mean"),
    #     }
    # Suite threads these into common_params so per-model pre_pipeline
    # builders pick them up. But the ensemble-scoring helper calls
    # ``_build_configs_from_params(**kwargs_copy)`` -- a function with a
    # declared signature that raises TypeError on any kwarg it doesn't
    # know about. Pop pipeline-component kwargs here so the ensemble
    # path doesn't leak them into the config builder. This isn't
    # feature loss: sub-models have already been fitted BEFORE the
    # ensemble scorer runs; we don't re-apply encoder/scaler/imputer
    # inside ensemble scoring.
    for _pipeline_kwarg in ("category_encoder", "scaler", "imputer"):
        kwargs_copy.pop(_pipeline_kwarg, None)

    # 2026-04-27 typed-config refactor: ``compute_{trainset,valset,testset}_-
    # metrics`` were lifted from trainer-internal to ``ReportingConfig``.
    # core.py now seeds ``common_params_dict`` from ``reporting_config.model_dump()``,
    # which makes those fields part of ``kwargs_copy``. Preserve the caller's
    # switches before popping them to avoid duplicate-key ``dict(...)`` splats;
    # ensemble scoring must respect the same reporting contract as single
    # models.
    _caller_compute_trainset_metrics = bool(
        kwargs_copy.pop("compute_trainset_metrics", False)
    )
    _caller_compute_valset_metrics = bool(
        kwargs_copy.pop("compute_valset_metrics", True)
    )
    _caller_compute_testset_metrics = bool(
        kwargs_copy.pop("compute_testset_metrics", True)
    )

    def _has_split_predictions(_kwargs: dict, _split: str) -> bool:
        return (
            _kwargs.get(f"{_split}_preds") is not None
            or _kwargs.get(f"{_split}_probs") is not None
        )

    # Build config objects from flat params
    flat_params = dict(
        df=None,
        drop_columns=[],
        model_name_prefix=f"Ens{internal_ensemble_method.upper()} {ensemble_name}",
        train_idx=train_idx,
        test_idx=test_idx,
        val_idx=val_idx,
        target_label_encoder=target_label_encoder,
        compute_trainset_metrics=(
            _caller_compute_trainset_metrics
            and _has_split_predictions(predictive_kwargs, "train")
        ),
        compute_valset_metrics=(
            _caller_compute_valset_metrics
            and _has_split_predictions(predictive_kwargs, "val")
        ),
        compute_testset_metrics=(
            _caller_compute_testset_metrics
            and _has_split_predictions(predictive_kwargs, "test")
        ),
        nbins=nbins,
        custom_ice_metric=custom_ice_metric,
        custom_rice_metric=custom_rice_metric,
        subgroups=subgroups,
        n_features=n_features,
        **target_kwargs,
        **predictive_kwargs,
        **kwargs_copy,
    )
    data, control, metrics_cfg, reporting_cfg, naming, confidence, predictions_cfg, output_cfg = _build_configs_from_params(**flat_params)
    next_ens_results = train_and_evaluate_model(
        model=None,
        data=data,
        control=control,
        metrics=metrics_cfg,
        reporting=reporting_cfg,
        naming=naming,
        output=output_cfg,
        confidence=confidence,
        predictions=predictions_cfg,
    )

    # Stash the per-member test-split predictions as an (n_rows, n_members) matrix so the suite finalize can render the
    # ensemble prediction-stability (member-disagreement) panel. Test arrays already materialised above; for
    # classification members carry probs (positive-class column), for regression the point preds.
    try:
        _cols = []
        for _p in _test_preds:
            if _p is None:
                continue
            _a = np.asarray(_p, dtype=np.float64)
            if _a.ndim == 2 and _a.shape[1] == 2:
                _a = _a[:, 1]
            _cols.append(_a.ravel())
        if len(_cols) >= 2:
            _w = min(c.shape[0] for c in _cols)
            # train_and_evaluate_model returns (entry_namespace, train_df, val_df, test_df); stamp the namespace.
            _entry_ns = next_ens_results[0] if isinstance(next_ens_results, tuple) else next_ens_results
            _entry_ns.member_test_preds = np.column_stack([c[:_w] for c in _cols])
    except Exception:
        pass

    conf_results = None
    if uncertainty_quantile:
        if target is not None:
            conf_target_kwargs = dict(target=target)
        else:
            conf_target_kwargs = dict(
                train_target=train_target[train_confident_indices] if (train_target is not None and train_confident_indices is not None) else None,
                test_target=test_target[test_confident_indices] if (test_target is not None and test_confident_indices is not None) else None,
                val_target=val_target[val_confident_indices] if (val_target is not None and val_confident_indices is not None) else None,
            )

        conf_predictive_kwargs = build_predictive_kwargs(
            train_data=(
                train_ensembled_predictions[train_confident_indices]
                if (train_ensembled_predictions is not None and train_confident_indices is not None)
                else None
            ),
            test_data=(
                test_ensembled_predictions[test_confident_indices] if (test_ensembled_predictions is not None and test_confident_indices is not None) else None
            ),
            val_data=(
                val_ensembled_predictions[val_confident_indices] if (val_ensembled_predictions is not None and val_confident_indices is not None) else None
            ),
            is_regression=is_regression,
        )

        # Report the confidence-filter coverage right in the model name so
        # log-grep immediately shows that e.g. "Conf Ensemble ... [VAL
        # COV=10%]" is computed on just 10 % of VAL rows -- previously the
        # 99.77 % accuracy number in the Conf Ensemble block was easy to
        # misread as a headline, because coverage only appeared inside the
        # calibration subsection as ``COV=XX%`` (2026-04-23 review finding).
        # Prefer VAL coverage as the headline (early-stopping + calibration
        # both key on VAL); fall back to TEST coverage then TRAIN.
        # Evaluate (label, full_preds, conf_idx, target_for_label) for each split
        # in priority order. We need the target slice as well because the
        # degenerate-class-balance check (below) operates on the filtered target,
        # not on the prediction array.
        _cov_src = None
        _conf_target = None
        for _label, _full, _conf, _full_target in (
            ("VAL", val_ensembled_predictions, val_confident_indices, val_target),
            ("TEST", test_ensembled_predictions, test_confident_indices, test_target),
            ("TRAIN", train_ensembled_predictions, train_confident_indices, train_target),
        ):
            if _full is not None and _conf is not None and len(_full) > 0:
                _cov_src = (_label, 100.0 * len(_conf) / len(_full))
                if _full_target is not None and len(_full_target) == len(_full):
                    _conf_target = _full_target[_conf]
                break

        # Degenerate-class-balance check on the filtered target. A confidence
        # filter that "keeps the rows the ensemble agrees on" tends to keep
        # almost-all-positive (or almost-all-negative) subsets on imbalanced
        # data РІР‚вЂќ one prod log showed 21 negatives vs 81_815 positives in the
        # 10 % VAL slice, and the resulting ``BR=0.026 %`` looked like a
        # headline win until you noticed it was reporting on a degenerate
        # split. Marker is binary-classification only; regression has no
        # class balance to check.
        _degenerate_marker = ""
        if flag_degenerate_conf_subset and not is_regression and _conf_target is not None and len(_conf_target) > 0:
            _ct = np.asarray(_conf_target)
            if _ct.ndim == 1:
                # Count positives via boolean comparison so float / bool / int
                # targets all behave the same. Ratio is min/max regardless of
                # which class is the minority.
                _n_pos = int((_ct == 1).sum())
                _n_neg = int(_ct.shape[0] - _n_pos)
                _hi = max(_n_pos, _n_neg)
                _lo = min(_n_pos, _n_neg)
                if _hi > 0 and (_lo / _hi) < degenerate_class_ratio:
                    _degenerate_marker = "[DEGENERATE] "

        # Trailing space so the downstream concat ``f"...{ensemble_name}{_cov_tag}"``
        # doesn't slam the next token onto the closing bracket -- the 2026-04-24
        # prod log showed ``[VAL COV=10%]notext prod_jobsdetails ...`` (no space
        # before "notext"). Empty tag stays empty (no double-space when off).
        _cov_tag = f" {_degenerate_marker}[{_cov_src[0]} COV={_cov_src[1]:.0f}%] " if _cov_src else ""

        # Build config objects from flat params for confidence ensemble
        conf_flat_params = dict(
            df=None,
            drop_columns=[],
            model_name_prefix=f"Conf Ensemble {internal_ensemble_method} {ensemble_name}{_cov_tag}",
            train_idx=train_idx[train_confident_indices] if (train_idx is not None and train_confident_indices is not None) else None,
            test_idx=test_idx[test_confident_indices] if (test_idx is not None and test_confident_indices is not None) else None,
            val_idx=val_idx[val_confident_indices] if (val_idx is not None and val_confident_indices is not None) else None,
            target_label_encoder=target_label_encoder,
            compute_trainset_metrics=(
                _caller_compute_trainset_metrics
                and _has_split_predictions(conf_predictive_kwargs, "train")
            ),
            compute_valset_metrics=(
                _caller_compute_valset_metrics
                and _has_split_predictions(conf_predictive_kwargs, "val")
            ),
            compute_testset_metrics=(
                _caller_compute_testset_metrics
                and _has_split_predictions(conf_predictive_kwargs, "test")
            ),
            nbins=nbins,
            custom_ice_metric=custom_ice_metric,
            custom_rice_metric=custom_rice_metric,
            subgroups=subgroups,
            n_features=n_features,
            **conf_predictive_kwargs,
            **conf_target_kwargs,
            **kwargs_copy,
        )
        conf_data, conf_control, conf_metrics, conf_reporting, conf_naming, conf_confidence, conf_predictions, conf_output = _build_configs_from_params(
            **conf_flat_params
        )
        conf_results = train_and_evaluate_model(
            model=None,
            data=conf_data,
            control=conf_control,
            metrics=conf_metrics,
            reporting=conf_reporting,
            naming=conf_naming,
            output=conf_output,
            confidence=conf_confidence,
            predictions=conf_predictions,
        )

    return (internal_ensemble_method, next_ens_results, conf_results)


