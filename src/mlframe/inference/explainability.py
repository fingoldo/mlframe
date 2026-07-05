"""Everything related to explaining why a model made exactly that decisions and predictions."""

from __future__ import annotations

# ****************************************************************************************************************************
# Imports
# ****************************************************************************************************************************


# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Tuple

import numpy as np, pandas as pd
from pyutilz.system import tqdmu
from mlframe.metrics.core import show_calibration_plot as show_custom_calibration_plot
from mlframe.metrics.core import format_classification_report


def init_model_instance(model_class: object, params: dict) -> object:
    from imblearn.pipeline import Pipeline  # pylint: disable=import-outside-toplevel
    if isinstance(model_class, Pipeline):
        modified_steps = []
        for step in model_class.steps:
            if step[0] == "est":
                initialized_est = step[1](**params)
                modified_steps.append((step[0], initialized_est))
                return Pipeline(modified_steps)
            else:
                modified_steps.append(step)
    else:
        return model_class(**params)


def compute_shap_on_cv(
    X: object,
    y: object,
    model_class: object,
    model_params: dict,
    cv: object,
    groups=None,
    catboost_native_feature_importance: bool = False,
    show_oos_metrics: bool = True,
    show_classification_report: bool = False,
    oos_ts_max_size: int = None,
    display_labels: dict = None,
    gen_params: object = None,
    plot: bool = True,
    shap_oof: bool = False,
) -> Tuple[np.ndarray]:
    """Also computes oos Performance.

    ``shap_oof`` controls the SHAP-explain row set. Default ``False`` keeps the legacy contract: each fold explains the ENTIRE X under that fold's
    model, so ``values`` is stacked per fold with shape ``(n_folds, n_rows, n_features)``. This is ~k x redundant SHAP work for a k-fold cv.
    With ``shap_oof=True`` each fold explains only its own ``X_test`` and the results are assembled into a single out-of-fold matrix of shape
    ``(n_rows, n_features)`` (every row explained once, by the model of the fold that held it out). Because the active explainer is built without a
    background ``data=`` set, the SHAP value of a row is independent of the other rows passed to ``explainer(...)`` (verified bit-identical in
    ``_benchmarks/bench_shap_oof_per_fold.py``), so the OOF matrix is the bit-identical, ~k x cheaper test-row deliverable - but it is a DIFFERENT
    object from the legacy per-fold stack (train rows are not present), hence opt-in rather than the default. Ignored when
    ``catboost_native_feature_importance=True`` (that path uses CatBoost's native full-X importance, which is not OOF-decomposable here).
    """
    if display_labels is None:
        display_labels = {}
    import shap  # pylint: disable=import-outside-toplevel
    from catboost import EFstrType, Pool  # pylint: disable=import-outside-toplevel
    from imblearn.pipeline import Pipeline  # pylint: disable=import-outside-toplevel

    values, base_values, interaction_values, interaction_base_values, predictions, expected_values = [], [], [], [], [], []
    _X = Pool(X, cat_features=model_params.get("cat_features"))

    # OOF SHAP assembly (shap_oof=True): each fold contributes only its held-out test rows; we slot them back by original row position.
    oof_shap = [None] * len(X) if shap_oof else None
    oof_base = [None] * len(X) if shap_oof else None

    do_ts_oos = False
    if show_oos_metrics:
        if "TimeSeries" in type(cv).__name__:
            # We have free OOS data each time. Let's compute metrics on it.
            do_ts_oos = True
            all_probs = []
            all_true_values = []
            L = len(X)
    for train_ind, test_ind in tqdmu(cv.split(X, groups=groups)):
        X_train, X_test = X.iloc[train_ind, :], X.iloc[test_ind, :]
        y_train, y_test = y.iloc[train_ind], y.iloc[test_ind]

        # test is used purely for early stopping
        if gen_params:
            generated_params = gen_params()
            generated_params.update(model_params)
            logger.info("using %s", generated_params)
            model_instance = init_model_instance(model_class, generated_params)
        else:
            model_instance = init_model_instance(model_class, model_params)
            generated_params = model_params

        logger.info("Fitting...")

        if isinstance(model_instance, Pipeline):
            if "CatBoost" in type(model_instance.named_steps["est"]).__name__:
                if "eval_fraction" in generated_params:
                    model_instance.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), est__plot=plot)
                else:
                    model_instance.fit(X_train, y_train, est__eval_set=(X_test, y_test), est__plot=plot)
            else:
                if "eval_fraction" in generated_params:
                    model_instance.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
                else:
                    model_instance.fit(X_train, y_train)
        else:
            if "CatBoost" in type(model_instance).__name__:
                model_instance.fit(X_train, y_train, eval_set=(X_test, y_test), plot=plot)
            else:
                model_instance.fit(X_train, y_train)
        logger.info("Fitted. Explaining...")

        # make oos predictions
        if do_ts_oos:
            max_test_ind = np.array(test_ind).max() + 1
            if max_test_ind < L:
                pred_ind = np.arange(max_test_ind, L)

                if oos_ts_max_size:
                    pred_ind = pred_ind[: oos_ts_max_size + 1]

                probs = model_instance.predict_proba(X.iloc[pred_ind])
                all_probs.append(probs)
                all_true_values.append(y.iloc[pred_ind])
                nclasses = probs.shape[1]

        # Use SHAP to explain predictions
        if isinstance(model_instance, Pipeline):
            model_stub = model_instance.steps[-1][1]
        else:
            model_stub = model_instance

        if not catboost_native_feature_importance:
            explainer = shap.Explainer(
                model_stub, links=shap.links.logit
            )  # shap.TreeExplainer(model=model_stub, data=X, model_output="probability", feature_perturbation="interventional")  #
            expected_values.append(explainer.expected_value)
            logger.info("Getting shap values...")
            if shap_oof:
                shap_values = explainer(X_test)
                fold_vals = shap_values.values
                fold_base = shap_values.base_values
                for j, ridx in enumerate(test_ind):
                    oof_shap[ridx] = fold_vals[j]
                    oof_base[ridx] = fold_base[j] if np.ndim(fold_base) >= 1 and len(fold_base) == len(test_ind) else fold_base
            else:
                shap_values = explainer(X)
                values.append(shap_values.values)
                base_values.append(shap_values.base_values)
            logger.info("Got shap values.")

            # shap_interaction_values=explainer.shap_interaction_values(X)
            # interaction_values.append(shap_interaction_values.values)
            # interaction_base_values.append(shap_interaction_values.base_values)

        else:

            shap_values = model_stub.get_feature_importance(_X, type=EFstrType.ShapValues, verbose=0)
            shap_interaction_values = model_stub.get_feature_importance(_X, type=EFstrType.ShapInteractionValues, verbose=0)

            values.append(shap_values)
            interaction_values.append(shap_interaction_values)

        _proba = np.asarray(model_instance.predict_proba(X))
        # Binary -> positive-class column (1d); multiclass -> full (n, n_classes) matrix.
        y_pred = _proba[:, 1] if _proba.ndim == 2 and _proba.shape[1] == 2 else _proba
        predictions.append(y_pred)

    if do_ts_oos:
        all_true_values = np.hstack(all_true_values)
        probs = np.vstack(all_probs)
        # Derive nclasses from the stacked probs so it is defined even when the final TS fold
        # contributed no OOS rows (the per-fold ``nclasses = probs.shape[1]`` assignment is skipped then).
        nclasses = probs.shape[1]
        if show_classification_report:
            # Binary: threshold the positive-class column at 0.5; multiclass: take the argmax class.
            if nclasses == 2:
                hard_pred = (probs[:, 1] > 0.5).astype(np.int8)
            else:
                hard_pred = np.argmax(probs, axis=1).astype(np.int8)
            classification_report_text = format_classification_report(all_true_values, hard_pred, nclasses=nclasses, target_names=list(display_labels.values()))
            print(classification_report_text)
        show_custom_calibration_plot(
            y=all_true_values,
            probs=probs,
            nclasses=nclasses,
            display_labels=display_labels,
        )
    if shap_oof and not catboost_native_feature_importance:
        out_values = np.array(oof_shap)
        out_base_values = np.array(oof_base)
    else:
        out_values = np.array(values)
        out_base_values = np.array(base_values)
    return (
        out_values,
        out_base_values,
        np.array(interaction_values),
        np.array(interaction_base_values),
        np.array(predictions),
        np.array(expected_values),
    )
