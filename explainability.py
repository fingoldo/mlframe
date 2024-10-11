"""Everything related to explaining why a model made exactly that decisions and predictions.
"""
# ****************************************************************************************************************************
# Imports
# ****************************************************************************************************************************


# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.pythonlib import ensure_installed

ensure_installed("shap numpy")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import shap
import numpy as np, pandas as pd
from pyutilz.system import tqdmu
from catboost import EFstrType, Pool
from .evaluation import show_custom_calibration_plot
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline


def init_model_instance(model_class: object, params: dict) -> object:
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
    display_labels: dict = {},
    gen_params: object = None,
    plot: bool = True,
) -> Tuple[np.ndarray]:
    """Also computes oos Performance"""
    values, base_values, interaction_values, interaction_base_values, predictions, expected_values = [], [], [], [], [], []
    _X = Pool(X, cat_features=model_params.get("cat_features"))

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
            shap_values = explainer(X)
            logger.info("Got shap values.")
            values.append(shap_values.values)
            base_values.append(shap_values.base_values)

            # shap_interaction_values=explainer.shap_interaction_values(X)
            # interaction_values.append(shap_interaction_values.values)
            # interaction_base_values.append(shap_interaction_values.base_values)

        else:

            shap_values = model_stub.get_feature_importance(_X, type=EFstrType.ShapValues, verbose=0)
            shap_interaction_values = model_stub.get_feature_importance(_X, type=EFstrType.ShapInteractionValues, verbose=0)

            values.append(shap_values)
            interaction_values.append(shap_interaction_values)

        y_pred = model_instance.predict_proba(X)[:, 1]
        predictions.append(y_pred)

    if do_ts_oos:
        all_true_values = np.hstack(all_true_values)
        probs = np.vstack(all_probs)
        if show_classification_report:
            classification_report_text = classification_report(all_true_values, (probs[:, 1] > 0.5).astype(np.int8), target_names=display_labels.values())
            print(classification_report_text)
        show_custom_calibration_plot(
            y=all_true_values,
            probs=probs,
            nclasses=nclasses,
            display_labels=display_labels,
        )
    return (
        np.array(values),
        np.array(base_values),
        np.array(interaction_values),
        np.array(interaction_base_values),
        np.array(predictions),
        np.array(expected_values),
    )
