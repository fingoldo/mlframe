"""Business value: the comparison picks the run that is genuinely better, on real fits.

Every other test in this area feeds the comparison hand-written metric numbers, which proves the
arithmetic and nothing about whether the verdict is useful. Here two runs are FITTED -- run B gets an
informative feature the others lack -- and the metrics come from sklearn on held-out rows. The claim being
tested is the one a user actually relies on: pointing this at two suite runs that differ only in their
features names the better one.

Thresholds are set below the measured margin so the test pins the value, not the exact numbers.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from mlframe.training.targets_performance import compare_targets_performance

N_ROWS = 4_000
SEED = 0


def _entry(name: str, **metrics):
    """A model entry in the shape the suite produces: ``.metrics["test"][metric_name]``."""
    return types.SimpleNamespace(model_name=name, metrics={"test": dict(metrics)})


def _fit_binary(x_train, y_train, x_test, y_test):
    """``(ROC_AUC, log_loss)`` from a real logistic fit on held-out rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score

    model = LogisticRegression(max_iter=200).fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, proba)), float(log_loss(y_test, proba))


def _fit_regression(x_train, y_train, x_test, y_test):
    """``(RMSE, MAE)`` from a real ridge fit on held-out rows."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    pred = Ridge().fit(x_train, y_train).predict(x_test)
    return float(np.sqrt(mean_squared_error(y_test, pred))), float(mean_absolute_error(y_test, pred))


@pytest.fixture(scope="module")
def fitted_runs():
    """Two runs over the same two targets: ``rich`` sees the informative feature, ``poor`` does not."""
    rng = np.random.default_rng(SEED)
    noise = rng.normal(size=(N_ROWS, 3))
    signal = rng.normal(size=(N_ROWS, 1))

    y_regression = 2.5 * signal[:, 0] + rng.normal(scale=0.5, size=N_ROWS)
    y_binary = (signal[:, 0] + rng.normal(scale=0.5, size=N_ROWS) > 0).astype(int)

    cut = N_ROWS // 2
    poor_x, rich_x = noise, np.hstack([noise, signal])

    def build(features):
        """Fit both targets on ``features`` and pack the metrics the way the suite would."""
        auc, ll = _fit_binary(features[:cut], y_binary[:cut], features[cut:], y_binary[cut:])
        rmse, mae = _fit_regression(features[:cut], y_regression[:cut], features[cut:], y_regression[cut:])
        models = {
            "binary_classification": {"conversion": [_entry("logreg", ROC_AUC=auc, log_loss=ll)]},
            "regression": {"revenue": [_entry("ridge", RMSE=rmse, MAE=mae)]},
        }
        return models, {}

    return {"poor": build(poor_x), "rich": build(rich_x)}


def test_biz_val_targets_performance_features_informative_feature_wins(fitted_runs):
    """The run that can actually see the signal must be named the winner."""
    result = compare_targets_performance(fitted_runs)
    assert result.winner == "rich", f"the informative run lost: {result.scores.to_dict('records')}"
    assert result.reason.startswith("best mean scaled score")


def test_biz_val_targets_performance_features_both_targets_agree(fitted_runs):
    """Value only counts if the verdict is driven by the data, not by one lucky target.

    The informative feature helps BOTH the classifier and the regression, so both per-target scores must
    favour the same run -- a verdict resting on one of two targets would be a coin flip dressed up.
    """
    result = compare_targets_performance(fitted_runs)
    per_target = result.frame.groupby(["run", "target_name"], as_index=False)["scaled"].mean()
    rich = per_target[per_target["run"] == "rich"].set_index("target_name")["scaled"]
    assert float(rich["conversion"]) == 1.0, "the classifier did not favour the informative run"
    assert float(rich["revenue"]) == 1.0, "the regression did not favour the informative run"


def test_biz_val_targets_performance_features_margin_is_real_not_noise(fitted_runs):
    """The underlying metric gaps must be large enough that the verdict is not a rounding artefact.

    Floors sit well below the measured margin (AUC about +0.28, RMSE about -1.7 for the informative run),
    so this pins that the fixture really does separate the runs rather than that the numbers stay put.
    """
    (poor_models, _), (rich_models, _) = fitted_runs["poor"], fitted_runs["rich"]
    poor_auc = poor_models["binary_classification"]["conversion"][0].metrics["test"]["ROC_AUC"]
    rich_auc = rich_models["binary_classification"]["conversion"][0].metrics["test"]["ROC_AUC"]
    poor_rmse = poor_models["regression"]["revenue"][0].metrics["test"]["RMSE"]
    rich_rmse = rich_models["regression"]["revenue"][0].metrics["test"]["RMSE"]

    assert rich_auc - poor_auc > 0.15, f"AUC margin too small to be a business signal: {poor_auc:.3f} -> {rich_auc:.3f}"
    assert poor_rmse - rich_rmse > 1.0, f"RMSE margin too small to be a business signal: {poor_rmse:.3f} -> {rich_rmse:.3f}"


def test_biz_val_targets_performance_features_verdict_survives_either_normalisation(fitted_runs):
    """A verdict that flips with the scaling choice is not a verdict a user can act on."""
    for normalisation in ("rank", "minmax"):
        result = compare_targets_performance(fitted_runs, normalisation=normalisation)
        assert result.winner == "rich", f"{normalisation} picked {result.winner}"
