# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from typing import *  # noqa: F401 pylint: disable=wildcard-import,unused-wildcard-import

import copy
import psutil
from joblib import delayed
import pandas as pd, numpy as np

from pyutilz.parallel import parallel_run, cpu_count_physical
from pyutilz.pythonlib import is_jupyter_notebook
from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names, compute_numerical_aggregates_numba, get_basic_feature_names

SIMPLE_ENSEMBLING_METHODS: list = "arithm harm median quad qube geo".split()

_MEANS_COLS: list = "arimean,quadmean,qubmean,geomean,harmmean".split(",")

basic_features_names = get_basic_feature_names(
    return_drawdown_stats=False,
    return_profit_factor=False,
    whiten_means=False,
)

# *****************************************************************************************************************************************************
# Core ensembling functionality
# *****************************************************************************************************************************************************


def batch_numaggs(predictions: np.ndarray, get_numaggs_names_len: int, numaggs_kwds: dict, means_only: bool = True) -> np.ndarray:
    row_features = np.empty(shape=(len(predictions), get_numaggs_names_len), dtype=np.float32)
    for i in range(len(predictions)):
        arr = predictions[i, :]
        if means_only:
            numerical_features = compute_numerical_aggregates_numba(arr, geomean_log_mode=False, directional_only=False)
        else:
            numerical_features = compute_numaggs(arr=arr, **numaggs_kwds)
        row_features[i, :] = numerical_features
    return row_features


def enrich_ensemble_preds_with_numaggs(
    predictions: np.ndarray,
    models_names: Sequence = None,
    means_only: bool = False,
    keep_probs: bool = True,
    numaggs_kwds: dict = None,
    n_jobs: int = 1,
    only_physical_cores: bool = True,
) -> pd.DataFrame:
    """Probs are non-negative that allows more averages to be applied"""

    if models_names is None:
        models_names = []
    if numaggs_kwds is None:
        numaggs_kwds = {"whiten_means": False}

    if predictions.shape[1] >= 10:
        numaggs_kwds.update(dict(directional_only=False, return_hurst=True, return_entropy=True))
    else:
        numaggs_kwds.update(dict(directional_only=False, return_hurst=False, return_entropy=False))

    if means_only:
        numaggs_names = basic_features_names
    else:
        numaggs_names = list(get_numaggs_names(**numaggs_kwds))

    if keep_probs:
        probs_fields_names = models_names if models_names else [f"p{i}" for i in range(predictions.shape[1])]
    else:
        probs_fields_names = []

    if n_jobs == -1:
        n_jobs = cpu_count_physical() if only_physical_cores else (psutil.cpu_count(logical=True) or 1)

    if n_jobs and n_jobs != 1:
        batch_numaggs_results = parallel_run(
            [
                delayed(batch_numaggs)(predictions=arr, means_only=means_only, get_numaggs_names_len=len(numaggs_names), numaggs_kwds=numaggs_kwds)
                for arr in np.array_split(predictions, n_jobs)
            ],
            backend=None,
        )
        if keep_probs:
            idx = predictions.shape[1]
        else:
            idx = 0
        row_features = np.empty(shape=(len(predictions), len(numaggs_names) + idx), dtype=np.float32)
        row_features[:, idx:] = np.concatenate(batch_numaggs_results)
        if keep_probs:
            row_features[:, :idx] = predictions
    else:
        row_features = []
        for i in range(len(predictions)):
            arr = predictions[i, :]
            if means_only:
                numerical_features = compute_numerical_aggregates_numba(arr, geomean_log_mode=False, directional_only=False, whiten_means=False)
            else:
                numerical_features = compute_numaggs(arr=arr, **numaggs_kwds)
            if keep_probs:
                line = arr.tolist()
            else:
                line = []
            line.extend(numerical_features)

            row_features.append(line)

    columns = probs_fields_names + numaggs_names

    res = pd.DataFrame(data=row_features, columns=columns)
    if means_only:
        return res[probs_fields_names + _MEANS_COLS]
    else:
        return res


def ensemble_probabilistic_predictions(
    *preds,
    ensemble_method="harm",
    ensure_prob_limits: bool = True,
    max_mae: float = 0.0,
    max_std: float = 0.0,
    max_mae_relative: float = 2.5,
    max_std_relative: float = 2.5,
    uncertainty_quantile: float = 0,
    normalize_stds_by_mean_preds: bool = False,
    verbose: bool = True,
) -> tuple:
    """Ensembles probabilistic predictions. All elements of the preds tuple must have the same shape.
    uncertainty_quantile>0 produces separate charts for points where the models are confident (agree).

    Outlier-member filter (when len(preds) > 2):
        Each member's distance from the cross-member median is summarised
        by per-column MAE and STD (averaged across columns). A member is
        excluded if either is "too large".

        Two threshold styles are supported and **applied with OR-semantics**
        (a member is excluded if ANY active threshold is exceeded):

        1. ``max_mae`` / ``max_std`` — absolute thresholds in probability
           units. Default 0.0 ⇒ disabled. Use when you know an upper-bound
           on acceptable per-row drift in your domain (e.g. calibrated
           classifiers within 5 pp).

        2. ``max_mae_relative`` / ``max_std_relative`` — multiples of the
           **median MAE / STD** across all members. Default 2.5 ⇒ exclude a
           member whose distance is more than 2.5× the typical member's.
           Default 0.0 disables.

           Adaptive to suite composition: a 6-tree-model suite (CB / XGB /
           LGB × 2 weight schemas) where every member has MAE 0.025-0.054
           against median had max_mae=0.04 absolute trigger excluding all
           6 members (2026-04-24 prod log) — making the filter a no-op +
           36 noisy WARN lines per ensemble. Relative threshold 2.5 keeps
           the typical members and excludes a true outlier (e.g. a single
           MLP that's 5× off).

    The previous defaults (``max_mae=0.04`` / ``max_std=0.06`` absolute) are
    kept reachable by passing them explicitly; defaults are now relative.
    """
    assert ensemble_method in SIMPLE_ENSEMBLING_METHODS
    confident_indices = None

    preds = [p for p in preds if p is not None]
    if len(preds) == 0:
        return None, None, None

    # 2026-04-24: dedup memory churn. Pre-2026-04-24, this function called
    # `np.array(preds)` ~9 times across the various ensemble methods,
    # outlier-filter, and confidence paths — each call materialised a full
    # (M, N, K) tensor, peaking RAM at ~9× the steady-state cost. On
    # multi_5 × ensembles × 2-weight_schemas (M=6, N=600, K=5) the peak hit
    # native C++ allocator's Win32 4GB ceiling and OOM'd. Materialising
    # ONCE here eliminates that churn — full Welford-streaming refactor
    # for the big-N case (N=9M+) is tracked as TODO below.
    #
    # TODO(future): For N*K*M*8 > EnsemblingConfig.quantile_budget_bytes,
    # switch to streaming Welford accumulators (mean/std/geomean via
    # log-mean/M2) + P²-Quantile sketch for median/quantile aggregations.
    # Estimated gain: ~5× peak-memory drop on prod-sized frames; not
    # needed for fuzz-sized data.
    _preds_arr = np.asarray(preds, dtype=np.float64)

    if len(preds) > 2:

        skipped_preds_indices = set()

        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        # Disregard whole predictions deviating from the median too much
        # -----------------------------------------------------------------------------------------------------------------------------------------------------

        # compute median preds first
        median_preds = np.quantile(_preds_arr, 0.5, axis=0)

        # Per-member distance summary (vectorised over all columns at once).
        # Old code did a Python loop over columns; on 6-member × 1-column
        # ensembles that's only 6 iterations, but on multi-output cases this
        # is cleaner and ~free.
        per_member_mae = np.empty(len(preds), dtype=np.float64)
        per_member_std = np.empty(len(preds), dtype=np.float64)
        for i, pred in enumerate(preds):
            diffs = np.abs(pred - median_preds)
            mae_per_col = diffs.mean(axis=0)                     # (n_cols,)
            std_per_col = np.sqrt(((diffs - mae_per_col) ** 2).mean(axis=0))
            per_member_mae[i] = mae_per_col.mean()
            per_member_std[i] = std_per_col.mean()

        # Resolve the relative thresholds against the **median across
        # members** (robust to a single outlier; using mean would let one
        # bad member drag the threshold up and shield itself).
        median_mae = float(np.median(per_member_mae))
        median_std = float(np.median(per_member_std))
        rel_mae_threshold = (
            max_mae_relative * median_mae if max_mae_relative > 0 else 0.0
        )
        rel_std_threshold = (
            max_std_relative * median_std if max_std_relative > 0 else 0.0
        )

        for i in range(len(preds)):
            tot_mae = float(per_member_mae[i])
            tot_std = float(per_member_std[i])
            abs_violation = (
                (max_mae > 0 and tot_mae > max_mae)
                or (max_std > 0 and tot_std > max_std)
            )
            rel_violation = (
                (rel_mae_threshold > 0 and tot_mae > rel_mae_threshold)
                or (rel_std_threshold > 0 and tot_std > rel_std_threshold)
            )
            if abs_violation or rel_violation:
                if verbose:
                    reason_parts = []
                    if abs_violation:
                        reason_parts.append(
                            f"abs(mae>{max_mae}|std>{max_std})"
                        )
                    if rel_violation:
                        reason_parts.append(
                            f"rel(mae>{rel_mae_threshold:.4f}|std>{rel_std_threshold:.4f}; "
                            f"median_mae={median_mae:.4f},median_std={median_std:.4f})"
                        )
                    print(
                        f"ens member {i} excluded due to high distance from the median: "
                        f"mae={tot_mae:.4f}, std={tot_std:.4f} [{'; '.join(reason_parts)}]"
                    )
                skipped_preds_indices.add(i)
        if skipped_preds_indices:
            if len(skipped_preds_indices) < len(preds):
                preds = [el for i, el in enumerate(preds) if i not in skipped_preds_indices]
                if verbose:
                    print(f"Using {len(preds)} members of ensemble")
                # Members were dropped — re-materialise the cached tensor
                # so downstream aggregations see only kept members.
                _preds_arr = np.asarray(preds, dtype=np.float64)
            else:
                if verbose:
                    print(f"ensemble_probabilistic_predictions filters too restrictive ({len(skipped_preds_indices)} vs {len(preds)}), skipping them")

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Actual ensembling
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    if ensemble_method == "harm":
        # Harmonic mean: if any model predicts exactly 0, HM is defined as 0.
        # Plain ``1 / pred`` triggers RuntimeWarning ("divide by zero") and
        # produces ``inf``, which ``1/mean(...)`` then maps back to 0 — correct
        # numerically but noisy in logs (observed 2026-04-23 prod run). Mask the
        # zeros explicitly so the common path stays warning-free.
        any_zero = (_preds_arr == 0).any(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_sum = np.sum(1.0 / _preds_arr, axis=0)
            ensembled_predictions = len(_preds_arr) / inv_sum
        if any_zero.any():
            ensembled_predictions = np.where(any_zero, 0.0, ensembled_predictions)
    elif ensemble_method == "arithm":
        ensembled_predictions = np.mean(_preds_arr, axis=0)
    elif ensemble_method == "median":
        ensembled_predictions = np.quantile(_preds_arr, 0.5, axis=0)
    elif ensemble_method == "quad":
        ensembled_predictions = np.sqrt(np.mean(_preds_arr ** 2, axis=0))
    elif ensemble_method == "qube":
        ensembled_predictions = np.cbrt(np.mean(_preds_arr ** 3, axis=0))
    elif ensemble_method == "geo":
        # Use log-sum-exp via log-mean for numerical stability on large M.
        # Floor at 1e-300 (smallest safe float64) instead of 1e-12 to
        # preserve precision for legitimately rare events from well-
        # calibrated boosted trees.
        with np.errstate(divide="ignore"):
            ensembled_predictions = np.exp(np.mean(np.log(np.clip(_preds_arr, 1e-300, None)), axis=0))

    # Replace non-finite values (NaN, inf) with arithmetic mean fallback
    non_finite_mask = ~np.isfinite(ensembled_predictions)
    if non_finite_mask.any():
        arith_mean = np.mean(_preds_arr, axis=0)
        n_replaced = np.sum(non_finite_mask)
        if verbose:
            logger.info(f"{n_replaced} non-finite values replaced with arithmetic mean")
        ensembled_predictions = np.where(non_finite_mask, arith_mean, ensembled_predictions)

    if ensure_prob_limits:
        ensembled_predictions = np.clip(ensembled_predictions, 0.0, 1.0)

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Confidence estimates
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    if uncertainty_quantile:

        std_preds = np.std(_preds_arr, axis=0)
        if normalize_stds_by_mean_preds:
            mean_preds = np.mean(_preds_arr, axis=0)
            uncertainty = (std_preds / mean_preds).mean(axis=1)
        else:
            uncertainty = std_preds.mean(axis=1)

        threshold = np.quantile(uncertainty, uncertainty_quantile)
        confident_indices = np.where(uncertainty <= threshold)[0]
    else:
        uncertainty = None

    return ensembled_predictions, uncertainty, confident_indices


def build_predictive_kwargs(train_data, test_data, val_data, is_regression: bool):
    """
    Build predictive_kwargs dict for classification or regression tasks.

    Parameters
    ----------
    train_data, test_data, val_data : tuple | np.ndarray | None
        Either a tuple (predictions, indices), or just predictions, or None.
    is_regression : bool
        Whether the task is regression (True) or classification (False).

    Returns
    -------
    dict
        predictive_kwargs containing appropriately filtered and flattened arrays.
    """

    def process(data, flatten=False):
        # Case 1: None → None
        if data is None:
            return None

        # Case 2: Tuple or list → try to unpack (preds, indices)
        if isinstance(data, (tuple, list)):
            if len(data) == 2:
                preds, indices = data
                if preds is None:
                    return None
                if indices is None:
                    result = preds
                else:
                    result = preds[indices]
            else:
                # Unexpected tuple/list length
                result = data[0]
        else:
            # Case 3: raw ndarray (no indices)
            result = data

        return result.flatten() if (flatten and result is not None) else result

    if not is_regression:
        return dict(
            train_probs=process(train_data),
            test_probs=process(test_data),
            val_probs=process(val_data),
        )
    else:
        return dict(
            train_preds=process(train_data, flatten=True),
            test_preds=process(test_data, flatten=True),
            val_preds=process(val_data, flatten=True),
        )


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
) -> tuple:
    """Process a single ensemble method. Returns (method_name, results, conf_results, next_level_pred)."""
    from mlframe.training import train_and_evaluate_model
    from mlframe.training.trainer import _build_configs_from_params

    if not is_regression:
        predictions = (el.val_probs for el in level_models_and_predictions)
    else:
        predictions = (el.val_preds.reshape(-1, 1) for el in level_models_and_predictions)

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
    )

    if not is_regression:
        predictions = (el.test_probs for el in level_models_and_predictions)
    else:
        predictions = (el.test_preds.reshape(-1, 1) for el in level_models_and_predictions)

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
    )

    if not is_regression:
        predictions = (el.train_probs for el in level_models_and_predictions)
    elif level_models_and_predictions[0].train_preds is not None:
        predictions = (el.train_preds for el in level_models_and_predictions)
        predictions = (el.reshape(-1, 1) if (el is not None) else el for el in predictions)
    else:
        predictions = ()

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
    # Pop the caller's value — the ensemble scorer intentionally sets
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
    # ``_build_configs_from_params(**kwargs_copy)`` — a function with a
    # declared signature that raises TypeError on any kwarg it doesn't
    # know about. Pop pipeline-component kwargs here so the ensemble
    # path doesn't leak them into the config builder. This isn't
    # feature loss: sub-models have already been fitted BEFORE the
    # ensemble scorer runs; we don't re-apply encoder/scaler/imputer
    # inside ensemble scoring.
    for _pipeline_kwarg in ("category_encoder", "scaler", "imputer"):
        kwargs_copy.pop(_pipeline_kwarg, None)

    # Build config objects from flat params
    flat_params = dict(
        df=None,
        drop_columns=[],
        model_name_prefix=f"Ens{internal_ensemble_method.upper()} {ensemble_name}",
        train_idx=train_idx,
        test_idx=test_idx,
        val_idx=val_idx,
        target_label_encoder=target_label_encoder,
        compute_valset_metrics=True,
        nbins=nbins,
        custom_ice_metric=custom_ice_metric,
        custom_rice_metric=custom_rice_metric,
        subgroups=subgroups,
        n_features=n_features,
        **target_kwargs,
        **predictive_kwargs,
        **kwargs_copy,
    )
    data, control, metrics_cfg, display, naming, confidence, predictions_cfg = _build_configs_from_params(**flat_params)
    next_ens_results = train_and_evaluate_model(
        model=None,
        data=data,
        control=control,
        metrics=metrics_cfg,
        display=display,
        naming=naming,
        confidence=confidence,
        predictions=predictions_cfg,
    )

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
        # COV=10%]" is computed on just 10 % of VAL rows — previously the
        # 99.77 % accuracy number in the Conf Ensemble block was easy to
        # misread as a headline, because coverage only appeared inside the
        # calibration subsection as ``COV=XX%`` (2026-04-23 review finding).
        # Prefer VAL coverage as the headline (early-stopping + calibration
        # both key on VAL); fall back to TEST coverage then TRAIN.
        _cov_src = None
        for _label, _full, _conf in (
            ("VAL", val_ensembled_predictions, val_confident_indices),
            ("TEST", test_ensembled_predictions, test_confident_indices),
            ("TRAIN", train_ensembled_predictions, train_confident_indices),
        ):
            if _full is not None and _conf is not None and len(_full) > 0:
                _cov_src = (_label, 100.0 * len(_conf) / len(_full))
                break
        # Trailing space so the downstream concat ``f"...{ensemble_name}{_cov_tag}"``
        # doesn't slam the next token onto the closing bracket — the 2026-04-24
        # prod log showed ``[VAL COV=10%]notext prod_jobsdetails ...`` (no space
        # before "notext"). Empty tag stays empty (no double-space when off).
        _cov_tag = f" [{_cov_src[0]} COV={_cov_src[1]:.0f}%] " if _cov_src else ""

        # Build config objects from flat params for confidence ensemble
        conf_flat_params = dict(
            df=None,
            drop_columns=[],
            model_name_prefix=f"Conf Ensemble {internal_ensemble_method} {ensemble_name}{_cov_tag}",
            train_idx=train_idx[train_confident_indices] if (train_idx is not None and train_confident_indices is not None) else None,
            test_idx=test_idx[test_confident_indices] if (test_idx is not None and test_confident_indices is not None) else None,
            val_idx=val_idx[val_confident_indices] if (val_idx is not None and val_confident_indices is not None) else None,
            target_label_encoder=target_label_encoder,
            compute_valset_metrics=True,
            nbins=nbins,
            custom_ice_metric=custom_ice_metric,
            custom_rice_metric=custom_rice_metric,
            subgroups=subgroups,
            n_features=n_features,
            **conf_predictive_kwargs,
            **conf_target_kwargs,
            **kwargs_copy,
        )
        conf_data, conf_control, conf_metrics, conf_display, conf_naming, conf_confidence, conf_predictions = _build_configs_from_params(**conf_flat_params)
        conf_results = train_and_evaluate_model(
            model=None,
            data=conf_data,
            control=conf_control,
            metrics=conf_metrics,
            display=conf_display,
            naming=conf_naming,
            confidence=conf_confidence,
            predictions=conf_predictions,
        )

    return (internal_ensemble_method, next_ens_results, conf_results)


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
    # uniform tree-model suite (CB / XGB / LGB × 2 weight schemas) on
    # the 2026-04-24 prod log — turning the filter into a no-op + 36
    # noisy WARN lines per ensemble. Defaults flipped to relative
    # (``2.5×median``); pass non-zero ``max_mae`` / ``max_std`` to keep
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

    if (
        level_models_and_predictions[0].val_probs is not None
        or level_models_and_predictions[0].test_probs is not None
        or level_models_and_predictions[0].train_probs is not None
    ):
        is_regression = False
    else:
        is_regression = True
        ensure_prob_limits = False

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
        )

        if len(ensembling_methods) > 1 and effective_n_jobs > 1:
            # loky pickles kwargs across worker boundaries; closure-captured metrics/lambdas
            # blow up in workers. Pre-check so we can fall back to sequential with a clear warning.
            try:
                import pickle
                pickle.dumps((custom_ice_metric, custom_rice_metric, kwargs))
            except (pickle.PicklingError, AttributeError, TypeError) as exc:
                logger.warning(
                    "ensembling: falling back to sequential — one of "
                    "custom_ice_metric / custom_rice_metric / kwargs is not picklable: %s",
                    exc,
                )
                effective_n_jobs = 1

        if len(ensembling_methods) > 1 and effective_n_jobs > 1:
            # Parallel processing — loky + tiny max_nbytes keeps arrays in-memory (no spill) per pre-existing tuning
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
    return res


def compare_ensembles(
    ensembles: dict,
    sort_metric: str = "test.1.integral_error",
    show_plot: bool = True,
    figsize: tuple = (15, 3),
) -> pd.DataFrame:
    items = []
    for ens_name, ens_perf in ensembles.items():
        perf = copy.deepcopy(ens_perf.metrics)
        for set_name, set_perf in perf.items():
            if set_perf:
                for col in "feature_importances fairness_report robustness_report".split():
                    if col in set_perf:
                        del set_perf[col]
        ser = pd.json_normalize(perf).iloc[0, :]
        ser.name = ens_name
        items.append(ser)

    res = pd.DataFrame(items)
    if sort_metric in res:
        res = res.sort_values(sort_metric)

        if show_plot:
            if "test." in sort_metric:
                val_metric = sort_metric.replace("test.", "val.")
                if val_metric in res:
                    blank_metric = sort_metric.replace("test.", "")
                    ax = res.set_index(val_metric).sort_index()[sort_metric].plot(title=f"Ensembles {blank_metric}, val vs test", figsize=figsize)
                    ax.set_ylabel(sort_metric)
    return res
