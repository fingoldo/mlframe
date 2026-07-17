"""Unit + biz_value tests for the composite-target model card.

``composite_model_card`` orchestrates provenance / attribution / conformal
state / leakage into one governance dict + markdown rendering.

biz_value: the card for a fitted ``linear_residual`` composite contains every
section non-empty -- identity, provenance formula, fitted params, training
summary, evaluation metrics, base-vs-residual attribution, the leakage check,
and the readiness flags -- and a no-data card still renders identity /
provenance / params.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.model_card import composite_model_card


def _make_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(10.0, 2.0, n)
    f1 = rng.normal(0.0, 1.0, n)
    y = 2.0 * base + 0.8 * f1 + rng.normal(0.0, 0.3, n)
    X = pd.DataFrame({"base": base, "f1": f1})
    return X, pd.Series(y)


def _fit_linres(X, y, base_column="base"):
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="linear_residual",
        base_column=base_column,
    )
    est.fit(X, y)
    return est


# ---------------------------------------------------------------------------
# biz_value: fitted card has EVERY section non-empty + leakage + readiness
# ---------------------------------------------------------------------------


def test_biz_val_model_card_linear_residual_all_sections_nonempty():
    X, y = _make_data()
    est = _fit_linres(X, y)
    # Calibrate conformal on a held-out slice so readiness + coverage populate.
    est.calibrate_conformal(X.iloc[:150], y.iloc[:150], alpha=0.1)

    card = composite_model_card(est, X, y)

    # Identity
    ident = card["identity"]
    assert ident["transform_name"] == "linear_residual"
    assert ident["base_columns"] == ["base"]
    assert ident["inner_estimator_type"] == "LinearRegression"
    assert ident["is_fitted"] is True

    # Provenance formula + stakeholder description, non-empty
    prov = card["provenance"]
    assert "base" in prov["forward_formula_human"]
    assert prov["inverse_formula_human"]
    assert "residual" in prov["stakeholder_description"]

    # Fitted params include the inversion coefficients
    params = card["fitted_params"]
    assert "alpha" in params and "beta" in params

    # Training summary
    train = card["training"]
    assert train["n_train"] > 0
    assert 0.0 < train["valid_domain_frac"] <= 1.0

    # Evaluation metrics: RMSE/MAE present + interval coverage (calibrated)
    ev = card["evaluation"]
    assert ev["rmse"] is not None and ev["rmse"] >= 0.0
    assert ev["mae"] is not None
    assert "interval_coverage" in ev
    cov = ev["interval_coverage"]["0.1"]
    assert cov["target_coverage"] == pytest.approx(0.9)
    assert 0.0 <= cov["empirical_coverage"] <= 1.0

    # Attribution: base dominates this 2*base target -> high base share
    attr = card["attribution"]
    assert attr["available"] is True
    assert attr["base_share"] > 0.7

    # Leakage check ran (base is a real predictor, not a re-encoding of y)
    leak = card["leakage"]
    assert leak["available"] is True
    assert leak["is_leaky"] is False
    assert leak["probed_base_column"] == "base"

    # Readiness flags
    rd = card["readiness"]
    assert rd["conformal_calibrated"] is True
    assert rd["online_refit_enabled"] is False
    assert rd["drift_monitor_present"] is True

    # Markdown renders every section
    md = card["markdown"]
    for header in (
        "## Identity",
        "## Provenance",
        "## Fitted parameters",
        "## Training summary",
        "## Evaluation",
        "## Base-vs-residual attribution",
        "## Leakage check",
        "## Deployment-readiness checklist",
    ):
        assert header in md


# ---------------------------------------------------------------------------
# Unit: no-data card still renders identity / provenance / params
# ---------------------------------------------------------------------------


def test_model_card_no_data_renders_identity_provenance_params():
    X, y = _make_data()
    est = _fit_linres(X, y)
    card = composite_model_card(est)  # no X, no y

    assert card["identity"]["transform_name"] == "linear_residual"
    assert card["provenance"]["forward_formula_human"]
    assert "alpha" in card["fitted_params"]
    assert card["training"]["n_train"] > 0
    # Data-dependent sections omitted
    assert "evaluation" not in card
    assert "attribution" not in card
    assert "leakage" not in card
    # Markdown still valid
    assert "## Identity" in card["markdown"]
    assert "## Deployment-readiness checklist" in card["markdown"]


def test_model_card_unfitted_estimator_renders_minimal():
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="linear_residual",
        base_column="base",
    )
    card = composite_model_card(est)
    assert card["identity"]["is_fitted"] is False
    assert card["provenance"]["forward_formula_human"]
    assert card["fitted_params"] == {}
    assert card["readiness"]["conformal_calibrated"] is False


def test_model_card_readiness_online_refit_flag():
    X, y = _make_data()
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="linear_residual",
        base_column="base",
        online_refit_enabled=True,
    )
    est.fit(X, y)
    card = composite_model_card(est)
    assert card["readiness"]["online_refit_enabled"] is True
    # No conformal calibration -> evaluation present but no interval coverage
    card2 = composite_model_card(est, X, y)
    assert "interval_coverage" not in card2["evaluation"]


def test_model_card_detects_leaky_base():
    # base ~= y itself -> a near-deterministic re-encoding of the CURRENT target.
    X, y = _make_data()
    leaky = y.to_numpy() + np.random.default_rng(3).normal(0, 1e-3, len(y))
    Xl = pd.DataFrame({"base": leaky, "f1": X["f1"].to_numpy()})
    est = _fit_linres(Xl, y)
    card = composite_model_card(est, Xl, y)
    leak = card["leakage"]
    assert leak["available"] is True
    assert leak["is_leaky"] is True


def test_model_card_attribution_unavailable_for_unary_transform():
    rng = np.random.default_rng(5)
    n = 300
    y = rng.lognormal(0.0, 1.0, n)
    X = pd.DataFrame({"f1": rng.normal(0, 1, n)})
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="log_y",
        base_column="",
    )
    est.fit(X, pd.Series(y))
    card = composite_model_card(est, X, pd.Series(y))
    assert card["attribution"]["available"] is False
    # leakage skipped too (no base column)
    assert card["leakage"] is None or card["leakage"].get("available") is False
