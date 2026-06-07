"""Regression coverage for the CTE + pre_pipeline base-column scaling bug.

Discovered by the TVT-2026-05-21 forensic audit. ``CompositeTargetEstimator``
(linear_residual / linear_residual_robust / etc.) fits ``alpha, beta`` on the
RAW base column during discovery. At predict time the suite default passes the
pre-pipeline-scaled X (e.g. ``StandardScaler`` z-scoring every numeric column
including the base column). The wrapper extracts the SCALED base, computes
``y = t_hat + alpha * base_scaled + beta`` instead of
``y = t_hat + alpha * base_raw + beta``; predictions collapse to residual
scale (mean ~ 0, std ~ residual_std). The fix routes raw X to
CompositeTargetEstimator-typed wrappers explicitly via ``_primary_for_model``
in ``predict.py``.
"""
from __future__ import annotations

import pathlib

import numpy as np

from mlframe.training.core import predict as predict_mod


def test_predict_module_has_cte_raw_x_dispatch():
    """Sanity: the predict dispatch site mentions the CTE class and the
    per-model selector. After the 2026-05-22 split the dispatch lives in
    ``_predict_main_from_models.py``; check the parent + all siblings.
    """
    _core = pathlib.Path(predict_mod.__file__).resolve().parent
    src = ""
    for _name in ("predict.py", "_predict_main.py", "_predict_main_from_models.py", "_predict_pre_pipeline.py"):
        _p = _core / _name
        if _p.exists():
            src += _p.read_text(encoding="utf-8")
            src += "\n"
    assert "CompositeTargetEstimator" in src, (
        "CTE-RAW-X dispatch missing from predict module -- the CTE+pre_pipeline "
        "base scaling fix would regress on the next prod run."
    )
    assert "_primary_for_model" in src, (
        "_primary_for_model selector missing -- the per-model raw-vs-scaled dispatch "
        "branch was removed; reinstate before shipping."
    )


def test_cte_wrapped_model_receives_raw_base_at_predict():
    """End-to-end behavioural test: build a synthetic linear-residual scenario,
    wrap inner in CompositeTargetEstimator, and verify wrapper.predict on RAW
    input returns y-scale predictions (mean ~ target mean, std ~ target std)
    while predict on SCALED input degenerates (the prod bug)."""
    from sklearn.compose import TransformedTargetRegressor  # noqa: F401
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    from mlframe.training.composite.estimator._estimator import CompositeTargetEstimator
    from mlframe.training.composite.transforms import (
        _linear_residual_fit, get_transform,
    )

    # Synthetic: y = 0.9 * base + signal + noise.
    rng = np.random.default_rng(0)
    n = 5000
    base = rng.normal(1000.0, 50.0, n).astype(np.float64)
    signal = rng.normal(0.0, 10.0, n)
    noise = rng.normal(0.0, 5.0, n)
    y = 0.9 * base + signal + noise

    # Fit linear_residual transform on RAW base.
    fitted_params = _linear_residual_fit(y, base)
    alpha = float(fitted_params["alpha"])
    transform = get_transform("linear_residual")

    # Build inner estimator trained on T = forward(y, base, params).
    t = transform.forward(y, base, fitted_params)
    X_df = __import__("pandas").DataFrame({"base": base, "noise": rng.normal(0, 1, n)})
    inner = Ridge().fit(X_df, t)

    # Wrap.
    wrapper = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner,
        transform_name="linear_residual",
        base_column="base",
        base_columns=None,
        transform_fitted_params=fitted_params,
        y_train=y,
    )

    # Predict on RAW X.
    pred_raw = np.asarray(wrapper.predict(X_df))
    assert abs(pred_raw.mean() - y.mean()) / max(abs(y.mean()), 1.0) < 0.05, (
        f"CTE.predict on RAW X: pred mean {pred_raw.mean():.1f} should ~ target mean {y.mean():.1f}"
    )
    # Predictions should track y to within ~2x residual std.
    assert abs(pred_raw.std() - y.std()) / max(abs(y.std()), 1.0) < 0.5, (
        f"CTE.predict on RAW X: pred std {pred_raw.std():.1f} should ~ target std {y.std():.1f}"
    )

    # Now demonstrate the bug: predict on z-scored X.
    scaler = StandardScaler().fit(X_df)
    X_scaled = __import__("pandas").DataFrame(
        scaler.transform(X_df), columns=X_df.columns,
    )
    pred_scaled = np.asarray(wrapper.predict(X_scaled))
    # Bug signature: pred mean is way off from y mean (the additive alpha*base term collapsed).
    assert abs(pred_scaled.mean() - y.mean()) > 100.0 * abs(alpha), (
        "Expected the CTE+scaled-X bug to be visible: pred mean should DIVERGE from "
        f"y mean when base is z-scored. Got pred_mean={pred_scaled.mean():.1f}, "
        f"y_mean={y.mean():.1f}, alpha={alpha:.3f}. If this assertion fails the "
        "regression test premise is invalidated -- re-check the bug repro."
    )
