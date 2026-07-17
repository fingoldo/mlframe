"""Regression tests for the audit-followup fixes to the TVT-MLP-collapse
class. Each test cements one of the audit findings landed today:

* ``additive_residual`` transform: roundtrip correctness + presence in
  the default transforms list (Agent C P0).
* ``generate_mlp`` Identity-on-multi-layer guard: warning fires
  (Agent A P0).
* Extended ``regression-collapse-sensor`` branches: linear-extrapolation
  + mean-shift trip even when std-collapse alone doesn't (Agent A P0 /
  Agent B P0).
* ``RANSACRegressor`` inner estimator is Ridge, not LinearRegression
  (Agent B P2).
* ``feature_drift_report`` Identity translator forces ``nlayers=1``
  (Agent A P0).
* ``CompositeTargetDiscoveryConfig.tiny_screening_models`` defaults
  to ``per_family`` covering ``("lightgbm", "linear")`` (Agent D P0).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest


class TestAdditiveResidualTransform:
    def test_registered_with_short_alias(self) -> None:
        from mlframe.training.composite.transforms import (
            TRANSFORM_NAME_SHORT,
            get_transform,
        )

        t = get_transform("additive_residual")
        assert t.name == "additive_residual"
        assert TRANSFORM_NAME_SHORT["additive_residual"] == "addres"

    def test_in_default_transforms_list(self) -> None:
        from mlframe.training.configs import CompositeTargetDiscoveryConfig

        cfg = CompositeTargetDiscoveryConfig()
        assert "additive_residual" in cfg.transforms

    def test_forward_inverse_roundtrip(self) -> None:
        """T = y - base - beta; inverse y = T + base + beta -- pure
        additive, no nonlinear branch. On AR(1) data with small noise
        the residual std should be << y std (MLP-friendly target)."""
        from mlframe.training.composite.transforms import get_transform

        rng = np.random.default_rng(0)
        n = 500
        y = np.zeros(n)
        y[0] = rng.normal(0, 5)
        for i in range(1, n):
            y[i] = 0.999 * y[i - 1] + rng.normal(scale=0.5)
        base = np.r_[y[0], y[:-1]]
        transform = get_transform("additive_residual")
        params = transform.fit(y, base)
        assert "beta" in params
        t = transform.forward(y, base, params)
        y_back = transform.inverse(t, base, params)
        np.testing.assert_allclose(y_back, y, atol=1e-10)
        # Residual std should be < y std (this is the MLP-saving invariant).
        assert np.std(t) < 0.5 * np.std(y)


class TestIdentityMLPGuard:
    def test_warns_on_identity_with_multilayer_regression(self, caplog) -> None:
        torch = pytest.importorskip("torch")
        from mlframe.training.neural.flat import generate_mlp

        caplog.set_level(logging.WARNING, logger="mlframe.training.neural.flat")
        generate_mlp(
            num_features=10,
            num_classes=1,
            nlayers=3,
            activation_function=torch.nn.Identity,
            use_layernorm=False,
            dropout_prob=0.0,
            inputs_dropout_prob=0.0,
            verbose=0,
        )
        assert any("COLLAPSE" in rec.message and "Identity" in rec.message for rec in caplog.records), [r.message for r in caplog.records]

    def test_no_warn_on_identity_with_single_layer(self, caplog) -> None:
        torch = pytest.importorskip("torch")
        from mlframe.training.neural.flat import generate_mlp

        caplog.set_level(logging.WARNING, logger="mlframe.training.neural.flat")
        generate_mlp(
            num_features=10,
            num_classes=1,
            nlayers=1,
            activation_function=torch.nn.Identity,
            use_layernorm=False,
            dropout_prob=0.0,
            inputs_dropout_prob=0.0,
            verbose=0,
        )
        assert not any("COLLAPSE" in rec.message for rec in caplog.records)

    def test_no_warn_on_relu_with_multilayer(self, caplog) -> None:
        torch = pytest.importorskip("torch")
        from mlframe.training.neural.flat import generate_mlp

        caplog.set_level(logging.WARNING, logger="mlframe.training.neural.flat")
        generate_mlp(
            num_features=10,
            num_classes=1,
            nlayers=3,
            activation_function=torch.nn.ReLU,
            use_layernorm=False,
            dropout_prob=0.0,
            inputs_dropout_prob=0.0,
            verbose=0,
        )
        assert not any("COLLAPSE" in rec.message for rec in caplog.records)


class TestRegressionCollapseSensorBranches:
    """The pre-fix sensor caught only std-collapse. After the fix, an
    additional ``group-ood-shift`` branch catches the patterns that bit
    production TVT: the old narrow ``linear-extrapolation`` (R^2 < -1 AND
    max|pred-y| > 5*y_std) and ``mean-shift`` (|pred_mean - y_mean| >
    3*y_std) branches were consolidated into a single ``group-ood-shift``
    label because in production the underlying cause is always the same
    (group-aware split with feature distribution shift between train and
    test groups). The unified label keeps the actionable mitigation set
    (composite-target / tree booster / group-split verification) intact."""

    def test_linear_extrapolation_branch_trips(self, caplog, monkeypatch) -> None:
        from mlframe.training.reporting import _reporting

        caplog.set_level(logging.WARNING, logger="mlframe.training.reporting._reporting")
        # Disable envelope-clip so the sensor sees the raw wildly-out-of-range
        # predictions instead of the clipped-to-envelope ones. With clipping
        # enabled the preds collapse to a single value and the std-collapse
        # branch trips before linear-extrapolation can. The env-var is the
        # documented opt-out path for callers that want to bench the raw
        # sensor behaviour.
        monkeypatch.setenv("MLFRAME_DISABLE_PREDICTION_ENVELOPE_CLIP", "1")
        # Targets centered around 11500 (TVT regime), preds wildly off.
        targets = 11500 + np.random.default_rng(0).normal(0, 100, 1000)
        preds = -50000 + np.random.default_rng(1).normal(0, 200, 1000)
        # Drive sensor through the public report function.
        _reporting.report_regression_model_perf(
            targets=targets,
            columns=[],
            model_name="test-extrap",
            model=None,
            preds=preds,
            print_report=False,
            show_perf_chart=False,
            verbose=False,
        )
        # group-ood-shift covers the old linear-extrapolation regime.
        assert any(
            "regression-collapse-sensor:group-ood-shift" in rec.message or "regression-collapse-sensor:linear-extrapolation" in rec.message
            for rec in caplog.records
        ), [r.message for r in caplog.records if "sensor" in r.message]

    def test_mean_shift_branch_trips(self, caplog) -> None:
        from mlframe.training.reporting import _reporting

        caplog.set_level(logging.WARNING, logger="mlframe.training.reporting._reporting")
        # Land in MEAN-SHIFT-only regime: pred mean shifted >3 sigma but
        # within <5 sigma envelope. group-ood-shift covers both the old
        # mean-shift and linear-extrapolation regimes since they're the
        # same prod cause; accept either label.
        rng_t = np.random.default_rng(0)
        rng_p = np.random.default_rng(1)
        targets = 11500 + rng_t.normal(0, 800, 1000)
        preds = 9000 + rng_p.normal(0, 300, 1000)
        _reporting.report_regression_model_perf(
            targets=targets,
            columns=[],
            model_name="test-shift",
            model=None,
            preds=preds,
            print_report=False,
            show_perf_chart=False,
            verbose=False,
        )
        assert any(
            "regression-collapse-sensor:" in rec.message
            and ("mean-shift" in rec.message or "linear-extrapolation" in rec.message or "group-ood-shift" in rec.message)
            for rec in caplog.records
        ), [r.message for r in caplog.records if "sensor" in r.message]


class TestRansacInnerIsRidge:
    def test_ransac_uses_ridge_not_linear_regression(self) -> None:
        from sklearn.linear_model import Ridge, LinearRegression
        from mlframe.training.configs import LinearModelConfig
        from mlframe.training.models import _build_ransac_regressor

        ransac = _build_ransac_regressor(LinearModelConfig())
        inner = ransac.estimator
        assert isinstance(inner, Ridge), (
            f"RANSAC inner estimator should be Ridge (L2-bounded), "
            f"got {type(inner).__name__}. Plain LinearRegression "
            "extrapolates unboundedly on group-aware test splits."
        )
        assert not isinstance(inner, LinearRegression) or isinstance(inner, Ridge)


class TestFeatureDriftIdentityTranslator:
    def test_identity_activation_forces_nlayers_1(self) -> None:
        pytest.importorskip("torch")
        from mlframe.training.feature_drift_report import (
            translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs,
        )

        out = translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs(
            {"activation": "identity", "hidden_layer_sizes": (32, 16)},
        )
        # nlayers MUST be forced to 1 so the network is an honest single
        # Linear -> Identity instead of a 3-layer stack that mathematically
        # collapses but extrapolates catastrophically.
        assert out["network_params"]["nlayers"] == 1


class TestTinyScreeningPerFamilyDefault:
    def test_default_is_per_family_lightgbm_plus_linear(self) -> None:
        from mlframe.training.configs import CompositeTargetDiscoveryConfig

        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.tiny_screening_models == "per_family"
        assert "lightgbm" in cfg.tiny_screening_families
        assert "linear" in cfg.tiny_screening_families
