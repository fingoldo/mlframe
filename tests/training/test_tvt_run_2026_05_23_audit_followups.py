"""Regression tests for the 2026-05-23 TVT-rerun audit followups.

After landing the 2026-05-22 audit-followup pack, a fresh TVT regression
run surfaced four more issues:

* P0-1: MLP auto-batch_size resolver re-probed CPU memory on every call
  -> 64x variance (65536 -> 1024) between consecutive MLP trainings on
  identical data shape, 38% wall-clock waste from 5 MLP timeouts.
* P0-2: Ridge tiny-screener crashed on NaN inputs (LGBM handles NaN
  natively, Ridge doesn't) -- tens of thousands of WARNING logs and 5
  extra minutes of discovery time.
* P0-3: ``TVT-logr-TVT_prev`` composite produced T with std=0.001 vs
  y std=644 (ratio 644000:1). Below noise floor; downstream models
  trained on essentially white noise AND inverse_transform amplified
  any tiny T-error into significant y-scale error.
* P0-4: ``regression-collapse-sensor:linear-extrapolation`` fired on
  Ridge / LinearRegression on group-aware split with feature
  distribution shift -- but the hint blamed Identity-MLP stacks
  instead of the actual cause (group-OOD).

Plus the C3 / C4 composite-transform additions from the disposition
table:

* C3: ``median_residual`` -- bin-conditional median residual with
  PURE additive inverse (MLP-friendly).
* C4: ``y_quantile_clip`` -- limit-damage unary transform.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pytest


class TestMLPBatchSizeProbeCache:
    """``_probe_available_memory_bytes`` caches its first successful read
    so consecutive auto-batch resolutions in one process share a stable
    memory budget."""

    def test_probe_cached_after_first_call(self) -> None:
        from mlframe.training import mlp_runtime_defaults as mrd
        # Reset the cache so the test runs from a known state.
        mrd._PROBE_MEM_CACHE = None
        first = mrd._probe_available_memory_bytes(cuda_available=False)
        # The cache is now populated; a second probe must return the SAME
        # value even if real free memory has fluctuated.
        cached = mrd._probe_available_memory_bytes(cuda_available=False)
        assert first is None or first == cached, (
            f"probe cache broken: first={first}, cached={cached}; should "
            "be identical to avoid 64x batch-size variance across MLPs"
        )

    def test_probe_reprobe_via_env(self, monkeypatch) -> None:
        from mlframe.training import mlp_runtime_defaults as mrd
        mrd._PROBE_MEM_CACHE = 12345
        # Without the env var, cached value is returned.
        assert mrd._probe_available_memory_bytes(cuda_available=False) == 12345
        # With MLFRAME_FORCE_REPROBE set, the cache is bypassed and a
        # fresh probe is performed.
        monkeypatch.setenv("MLFRAME_FORCE_REPROBE", "1")
        fresh = mrd._probe_available_memory_bytes(cuda_available=False)
        assert fresh != 12345, (
            "MLFRAME_FORCE_REPROBE should bypass the cache; "
            f"got the cached value {fresh}"
        )

    def test_train_batch_min_raised(self) -> None:
        """The min train batch was 32 (catastrophically small for
        tabular workloads). Raise to 4096 so even a worst-case memory
        probe never produces sub-1024 batches that bottleneck training."""
        from mlframe.training.mlp_runtime_defaults import _TRAIN_BATCH_MIN
        assert _TRAIN_BATCH_MIN >= 4096


class TestRidgeTinyScreenerNaNHandling:
    """``_build_tiny_model("linear", ...)`` wraps Ridge in a
    SimpleImputer pipeline so NaN inputs don't crash every fold."""

    def test_linear_family_handles_nan_inputs(self) -> None:
        from mlframe.training._composite_screening_tiny import _build_tiny_model
        model = _build_tiny_model(
            "linear", n_estimators=10, num_leaves=15,
            learning_rate=0.1, random_state=0,
        )
        # Construct a small dataset with NaN cells across rows AND cols.
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 5))
        y = X[:, 0] * 0.5 + rng.normal(scale=0.1, size=200)
        # Pepper NaN into 5% of cells.
        nan_mask = rng.random(X.shape) < 0.05
        X[nan_mask] = np.nan
        # Should fit without raising; the SimpleImputer absorbs NaN.
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (200,)
        assert np.isfinite(preds).all()


class TestCompositeResidualStdDegeneracyFilter:
    """A composite where T_std / y_std < 0.001 produces a residual
    below noise floor (downstream models train on white noise, inverse_
    transform amplifies tiny T-errors into y-scale errors). Discovery
    must reject such composites BEFORE per-target training."""

    def test_residual_std_check_in_fit_source(self) -> None:
        """The new check lives in _composite_discovery_fit.py.
        Verify the source guard is present so a refactor that
        accidentally drops it gets caught at test time."""
        from pathlib import Path
        import mlframe.training._composite_discovery_fit as mod
        src = Path(mod.__file__).read_text(encoding="utf-8")
        # The residual-std degeneracy check moved to the _composite_discovery_eval.py
        # sibling during the discovery-fit split; concat so the source guard matches.
        _sib = Path(mod.__file__).parent / "_composite_discovery_eval.py"
        if _sib.exists():
            src += "\n" + _sib.read_text(encoding="utf-8")
        assert "_residual_ratio < 0.001" in src, (
            "residual-std degeneracy check missing from "
            "_composite_discovery_fit.py -- the 0.001 threshold "
            "below which composites get rejected at fit time"
        )
        assert "below noise floor" in src, (
            "rejection reason string missing -- audit trail for "
            "the degeneracy filter"
        )

    def test_compute_residual_ratio_logic(self) -> None:
        """The check is: T_std / y_std < 0.001 -> reject. Verify the
        threshold lines up with what we documented in the run analysis
        (production TVT-logr-TVT_prev had T_std=0.001, y_std=644,
        ratio=1.5e-6 << 0.001)."""
        from mlframe.training.composite.transforms import get_transform
        rng = np.random.default_rng(0)
        n = 2000
        base = rng.uniform(100, 200, n).astype(np.float64)
        # y close enough to base that logratio is ~ base*1e-5 / base = 1e-5 scale.
        y = (base + rng.normal(scale=base * 1e-5, size=n)).astype(np.float64)
        transform = get_transform("logratio")
        params = transform.fit(y, base)
        t = transform.forward(y, base, params)
        ratio = float(np.std(t)) / max(float(np.std(y)), 1e-12)
        assert ratio < 0.001, (
            f"expected ratio < 0.001 for this synthetic regime; got {ratio:.2e}"
        )


class TestCollapseSensorGroupOODBranch:
    """When the failing model is NOT a neural stack (Ridge,
    LinearRegression, sklearn LGBM, etc.), the collapse sensor
    should tag the branch as ``group-ood-shift`` instead of
    ``linear-extrapolation``."""

    def test_ridge_with_extrapolation_signature_tagged_group_ood(
            self, caplog, monkeypatch) -> None:
        from mlframe.training import _reporting
        # Bypass the envelope-clip so the wildly-out-of-range preds reach the
        # sensor unclipped; otherwise pred_std collapses to ~0 and the
        # std-collapse branch trips before group-ood-shift can.
        monkeypatch.setenv("MLFRAME_DISABLE_PREDICTION_ENVELOPE_CLIP", "1")
        caplog.set_level(logging.WARNING, logger="mlframe.training._reporting")
        targets = 11500 + np.random.default_rng(0).normal(0, 100, 1000)
        preds = -50000 + np.random.default_rng(1).normal(0, 200, 1000)
        _reporting.report_regression_model_perf(
            targets=targets, columns=[], model_name="Ridge",
            model=None, preds=preds,
            print_report=False, show_perf_chart=False, verbose=False,
        )
        assert any(
            "regression-collapse-sensor:group-ood-shift" in rec.message
            for rec in caplog.records
        ), [r.message for r in caplog.records if "sensor" in r.message]

    def test_mlp_with_extrapolation_signature_still_tagged_linear_extrap(
            self, caplog, monkeypatch) -> None:
        from mlframe.training import _reporting
        monkeypatch.setenv("MLFRAME_DISABLE_PREDICTION_ENVELOPE_CLIP", "1")
        caplog.set_level(logging.WARNING, logger="mlframe.training._reporting")
        targets = 11500 + np.random.default_rng(0).normal(0, 100, 1000)
        preds = -50000 + np.random.default_rng(1).normal(0, 200, 1000)
        _reporting.report_regression_model_perf(
            targets=targets, columns=[],
            model_name="PytorchLightningRegressor",
            model=None, preds=preds,
            print_report=False, show_perf_chart=False, verbose=False,
        )
        assert any(
            "regression-collapse-sensor:linear-extrapolation" in rec.message
            for rec in caplog.records
        ), [r.message for r in caplog.records if "sensor" in r.message]


class TestMedianResidualTransform:
    """``median_residual`` is a bin-conditional non-parametric residual
    with PURE additive inverse: y = T + median_bin[base]. MLP-friendly
    because inverse is a constant-per-bin lookup, not a nonlinear
    function."""

    def test_registered_with_short_alias(self) -> None:
        from mlframe.training.composite.transforms import (
            TRANSFORM_NAME_SHORT, get_transform,
        )
        t = get_transform("median_residual")
        assert t.name == "median_residual"
        assert TRANSFORM_NAME_SHORT["median_residual"] == "medres"

    def test_in_default_transforms_list(self) -> None:
        from mlframe.training.configs import CompositeTargetDiscoveryConfig
        cfg = CompositeTargetDiscoveryConfig()
        assert "median_residual" in cfg.transforms

    def test_forward_inverse_roundtrip(self) -> None:
        from mlframe.training.composite.transforms import get_transform
        rng = np.random.default_rng(0)
        n = 1000
        base = rng.uniform(-1, 1, n)
        y = 2.0 * base + 3.0 * base * base + rng.normal(scale=0.5, size=n)
        transform = get_transform("median_residual")
        params = transform.fit(y, base)
        assert "bin_edges" in params
        assert "bin_medians" in params
        t = transform.forward(y, base, params)
        y_back = transform.inverse(t, base, params)
        np.testing.assert_allclose(y_back, y, atol=1e-9)

    def test_inverse_is_additive(self) -> None:
        """The MLP-saving invariant: inverse(T_hat) = T_hat + g(base);
        if MLP outputs T_hat=0 (degenerate), y_hat = g(base) (bin-median
        lookup), bounded by train-y range."""
        from mlframe.training.composite.transforms import get_transform
        rng = np.random.default_rng(0)
        n = 1000
        base = rng.uniform(-1, 1, n)
        y = 2.0 * base + rng.normal(scale=0.5, size=n)
        transform = get_transform("median_residual")
        params = transform.fit(y, base)
        # With T_hat=0 for all rows, y_hat must equal g(base) per-row.
        t_zero = np.zeros(n)
        y_hat = transform.inverse(t_zero, base, params)
        # y_hat should be bounded by train-y range (bin medians, not
        # extrapolation).
        assert y_hat.min() >= y.min() - 1e-6
        assert y_hat.max() <= y.max() + 1e-6


class TestYQuantileClipTransform:
    def test_registered_with_short_alias(self) -> None:
        from mlframe.training.composite.transforms import (
            TRANSFORM_NAME_SHORT, get_transform,
        )
        t = get_transform("y_quantile_clip")
        assert t.name == "y_quantile_clip"
        assert TRANSFORM_NAME_SHORT["y_quantile_clip"] == "yqclip"
        # Unary transform: no base column required.
        assert t.requires_base is False

    def test_in_default_transforms_list(self) -> None:
        from mlframe.training.configs import CompositeTargetDiscoveryConfig
        cfg = CompositeTargetDiscoveryConfig()
        assert "y_quantile_clip" in cfg.transforms

    def test_clips_to_train_quantiles(self) -> None:
        from mlframe.training.composite.transforms import get_transform
        rng = np.random.default_rng(0)
        y_train = rng.normal(0, 1, 10000)
        transform = get_transform("y_quantile_clip")
        params = transform.fit(y_train, np.zeros_like(y_train))
        # Fit quantiles should be ~ +/- 2.576 (0.5% / 99.5% of standard normal).
        assert -3.0 < params["q_lo"] < -2.0
        assert 2.0 < params["q_hi"] < 3.0
        # An extreme test value gets clipped.
        y_test = np.array([-100.0, 0.0, 100.0])
        t = transform.forward(y_test, np.zeros_like(y_test), params)
        assert t[0] == pytest.approx(params["q_lo"])
        assert -1.0 < t[1] < 1.0
        assert t[2] == pytest.approx(params["q_hi"])

    def test_inverse_also_clips(self) -> None:
        """Symmetric: a wild T_hat gets clipped to the same quantile
        range on inverse, so downstream MLP can't extrapolate past
        train-y bounds."""
        from mlframe.training.composite.transforms import get_transform
        rng = np.random.default_rng(0)
        y_train = rng.normal(0, 1, 10000)
        transform = get_transform("y_quantile_clip")
        params = transform.fit(y_train, np.zeros_like(y_train))
        t_hat = np.array([-100.0, 0.0, 100.0])
        y_hat = transform.inverse(t_hat, np.zeros_like(t_hat), params)
        assert y_hat[0] == pytest.approx(params["q_lo"])
        assert y_hat[1] == pytest.approx(0.0, abs=1.0)
        assert y_hat[2] == pytest.approx(params["q_hi"])


class TestForceInjectConfigKnob:
    """C1 force-inject config knob exists (full implementation deferred)."""

    def test_force_inject_default_disabled(self) -> None:
        from mlframe.training.configs import CompositeTargetDiscoveryConfig
        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.force_inject_diff_on_top_ablation_pct == 0.0
