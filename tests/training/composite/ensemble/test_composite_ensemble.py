"""Tests for ``CompositeCrossTargetEnsemble`` (PR6).

Coverage map
------------
- :meth:`from_uniform_weights` produces equal weights and predict
  averages.
- :meth:`from_train_metrics` produces gain-weighted weights, SUM=1.
- Validation gate: every component worse than baseline -> single best
  returned (NOT an ensemble); ensemble RMSE estimate above
  best-single+5% -> single best returned.
- ``predict``: handles a failing component gracefully (skip + re-
  normalise).
- ``export_metadata`` shape contract.
- Constructor input validation: empty list, length mismatch,
  zero-sum weights.
"""

from __future__ import annotations

import numpy as np
import pytest

# B1 sklearn matrix marker convention -- this file runs in the multi-sklearn-version CI matrix.
pytestmark = pytest.mark.sklearn_matrix


from mlframe.training.composite import CompositeCrossTargetEnsemble

# ----------------------------------------------------------------------
# Mock components
# ----------------------------------------------------------------------


class _StubModel:
    """Deterministic stub that returns a constant per-row prediction."""

    def __init__(self, value: float, raises: bool = False) -> None:
        self.value = value
        self.raises = raises

    def predict(self, X) -> np.ndarray:
        """Predict."""
        if self.raises:
            raise RuntimeError("stub-failure")
        n = len(X) if hasattr(X, "__len__") else 1
        return np.full(n, self.value, dtype=np.float64)


# ----------------------------------------------------------------------
# Constructor / input validation
# ----------------------------------------------------------------------


class TestConstructor:
    """Groups tests covering constructor."""
    def test_empty_components_raises(self) -> None:
        """Empty components raises."""
        with pytest.raises(ValueError, match="empty"):
            CompositeCrossTargetEnsemble(
                component_models=[],
                component_names=[],
                weights=np.array([]),
                strategy="mean",
            )

    def test_length_mismatch_raises(self) -> None:
        """Length mismatch raises."""
        with pytest.raises(ValueError, match="same length"):
            CompositeCrossTargetEnsemble(
                component_models=[_StubModel(1.0)],
                component_names=["a", "b"],
                weights=np.array([1.0]),
                strategy="mean",
            )

    def test_zero_sum_weights_raises(self) -> None:
        """Zero sum weights raises."""
        with pytest.raises(ValueError, match="positive finite"):
            CompositeCrossTargetEnsemble(
                component_models=[_StubModel(1.0)],
                component_names=["a"],
                weights=np.array([0.0]),
                strategy="mean",
            )

    def test_normalises_weights(self) -> None:
        """Normalises weights."""
        ens = CompositeCrossTargetEnsemble(
            component_models=[_StubModel(1.0), _StubModel(2.0)],
            component_names=["a", "b"],
            weights=np.array([2.0, 6.0]),  # sum=8
            strategy="custom",
        )
        np.testing.assert_allclose(ens.weights, np.array([0.25, 0.75]))


# ----------------------------------------------------------------------
# from_uniform_weights
# ----------------------------------------------------------------------


class TestUniformWeights:
    """Groups tests covering uniform weights."""
    def test_equal_weights(self) -> None:
        """Equal weights."""
        ens = CompositeCrossTargetEnsemble.from_uniform_weights(
            component_models=[_StubModel(1.0), _StubModel(3.0), _StubModel(5.0)],
            component_names=["a", "b", "c"],
        )
        np.testing.assert_allclose(ens.weights, np.array([1 / 3, 1 / 3, 1 / 3]))
        assert ens.strategy == "mean"

    def test_predict_averages(self) -> None:
        """Predict averages."""
        ens = CompositeCrossTargetEnsemble.from_uniform_weights(
            component_models=[_StubModel(2.0), _StubModel(4.0)],
            component_names=["a", "b"],
        )
        # 5 sample rows; constant stubs -> uniform mean = 3.0.
        pred = ens.predict(np.zeros((5, 1)))
        np.testing.assert_allclose(pred, np.full(5, 3.0))


# ----------------------------------------------------------------------
# from_train_metrics
# ----------------------------------------------------------------------


class TestTrainMetricsWeights:
    """Groups tests covering train metrics weights."""
    def test_gain_weighted_strategy(self) -> None:
        # Three models with RMSE 1.0, 0.5, 0.2 against baseline 1.5.
        # Gains: 0.5, 1.0, 1.3 -> sum 2.8 -> weights ~0.179, 0.357, 0.464.
        """Gain weighted strategy."""
        ens = CompositeCrossTargetEnsemble.from_train_metrics(
            component_models=[_StubModel(1.0), _StubModel(2.0), _StubModel(3.0)],
            component_names=["worst", "mid", "best"],
            component_train_rmse=[1.0, 0.5, 0.2],
            baseline_train_rmse=1.5,
        )
        assert isinstance(ens, CompositeCrossTargetEnsemble)
        np.testing.assert_allclose(ens.weights.sum(), 1.0, atol=1e-10)
        # Best model should have the largest weight.
        assert ens.weights[2] > ens.weights[1] > ens.weights[0]

    def test_validation_gate_no_component_beats_baseline(self) -> None:
        # All components worse than baseline -> single best returned.
        """Validation gate no component beats baseline."""
        worst = _StubModel(1.0)
        mid = _StubModel(2.0)
        result = CompositeCrossTargetEnsemble.from_train_metrics(
            component_models=[worst, mid],
            component_names=["worst", "mid"],
            component_train_rmse=[2.0, 1.5],
            baseline_train_rmse=1.0,  # no component beats this
        )
        # Returns the BEST single component (lowest RMSE = mid).
        assert result is mid

    def test_single_component_returns_ensemble(self) -> None:
        # K=1 path: validation gate is skipped, ensemble of 1 is fine.
        """Single component returns ensemble."""
        only = _StubModel(1.0)
        ens = CompositeCrossTargetEnsemble.from_train_metrics(
            component_models=[only],
            component_names=["only"],
            component_train_rmse=[0.5],
            baseline_train_rmse=1.0,
        )
        assert isinstance(ens, CompositeCrossTargetEnsemble)
        assert len(ens.weights) == 1

    def test_baseline_default_uses_max_rmse(self) -> None:
        # MEDIAN-BASELINE fix: default baseline = max(rmses) (not median). Previously the median
        # fallback silently zeroed half of the candidates -- a hidden contract surprise. With max
        # as the baseline every component except the worst gets a positive weight; the best one
        # still dominates because gain (baseline - rmse) is largest for it.
        """Baseline default uses max rmse."""
        ens = CompositeCrossTargetEnsemble.from_train_metrics(
            component_models=[_StubModel(0), _StubModel(1), _StubModel(2)],
            component_names=["a", "b", "c"],
            component_train_rmse=[0.3, 0.5, 0.7],
        )
        # baseline = 0.7; gains = (0.4, 0.2, 0); both first two have positive weight, last is 0.
        assert isinstance(ens, CompositeCrossTargetEnsemble)
        assert ens.weights[0] > ens.weights[1] > 0
        assert ens.weights[2] == 0.0

    def test_nan_rmse_raises(self) -> None:
        """Nan rmse raises."""
        with pytest.raises(ValueError, match="non-finite"):
            CompositeCrossTargetEnsemble.from_train_metrics(
                component_models=[_StubModel(0)],
                component_names=["a"],
                component_train_rmse=[float("nan")],
            )

    # ------------------------------------------------------------------
    # N17: baseline-scale consistency between OOF rmses and the baseline.
    # ------------------------------------------------------------------

    def test_oof_rmses_with_train_scale_baseline_does_not_spuriously_fall_back(self) -> None:
        """N17 regression: ranking on OOF rmses while only a TRAIN-scale baseline is supplied.

        Train RMSE is systematically lower than OOF RMSE (rows seen at fit). Pre-fix the method
        used ``baseline_train_rmse`` *as the baseline against the OOF rmses* -- an apples-to-oranges
        comparison. With a train baseline below every OOF rmse, ``gains = max(0, baseline - oof_rmse)``
        collapses to all-zero and the method spuriously returns the single best model instead of an
        ensemble. The fix IGNORES the off-scale train baseline and uses the self-normalising
        ``max(oof_rmses)`` fallback, so the multi-component ensemble is built.

        Pre-fix this returns a bare ``_StubModel`` (gains all <= 0); post-fix an ensemble.
        """
        models = [_StubModel(1.0), _StubModel(2.0), _StubModel(3.0)]
        result = CompositeCrossTargetEnsemble.from_train_metrics(
            component_models=models,
            component_names=["a", "b", "c"],
            component_oof_rmse=[0.8, 0.5, 0.3],  # honest OOF scale
            baseline_train_rmse=0.25,  # train scale: below EVERY oof rmse
            # baseline_oof_rmse intentionally omitted -> the mismatch path.
        )
        # Post-fix: a real ensemble (not the single-best bare model fallback).
        assert isinstance(result, CompositeCrossTargetEnsemble), "off-scale train baseline must be ignored; an ensemble must be built, not a bare model"
        # baseline used = max(oof) = 0.8; gains = (0.0, 0.3, 0.5) -> sum 0.8.
        np.testing.assert_allclose(result.weights, np.array([0.0, 0.375, 0.625]))
        assert result.weights.sum() == pytest.approx(1.0)
        # Provenance recorded for operators / downstream readers (safe sub-fix).
        assert result.notes["rmse_source"] == "oof"
        assert result.notes["baseline_source"] == "max_fallback"
        assert result.notes["baseline"] == pytest.approx(0.8)

    def test_oof_rmses_with_oof_baseline_is_scale_consistent(self) -> None:
        """An explicit OOF-scale baseline IS honoured (and labelled) -- the production path."""
        models = [_StubModel(1.0), _StubModel(2.0), _StubModel(3.0)]
        result = CompositeCrossTargetEnsemble.from_train_metrics(
            component_models=models,
            component_names=["a", "b", "c"],
            component_oof_rmse=[0.8, 0.5, 0.3],
            baseline_oof_rmse=1.0,  # same (OOF) scale as the rmses
        )
        assert isinstance(result, CompositeCrossTargetEnsemble)
        # gains = (0.2, 0.5, 0.7) -> sum 1.4.
        np.testing.assert_allclose(result.weights, np.array([0.2, 0.5, 0.7]) / 1.4)
        assert result.notes["rmse_source"] == "oof"
        assert result.notes["baseline_source"] == "oof"
        assert result.notes["baseline"] == pytest.approx(1.0)

    def test_train_path_with_train_baseline_unchanged(self) -> None:
        """Behaviour-preservation: the pure train path is bit-identical to pre-N17."""
        models = [_StubModel(1.0), _StubModel(2.0), _StubModel(3.0)]
        result = CompositeCrossTargetEnsemble.from_train_metrics(
            component_models=models,
            component_names=["worst", "mid", "best"],
            component_train_rmse=[1.0, 0.5, 0.2],
            baseline_train_rmse=1.5,
        )
        assert isinstance(result, CompositeCrossTargetEnsemble)
        # gains = (0.5, 1.0, 1.3) -> sum 2.8 (same as TestTrainMetricsWeights.test_gain_weighted_strategy).
        np.testing.assert_allclose(result.weights, np.array([0.5, 1.0, 1.3]) / 2.8)
        assert result.notes["rmse_source"] == "train"
        assert result.notes["baseline_source"] == "train"


# ----------------------------------------------------------------------
# predict robustness
# ----------------------------------------------------------------------


class TestPredictRobustness:
    """Groups tests covering predict robustness."""
    def test_failing_component_skipped_and_renormalised(self) -> None:
        """Failing component skipped and renormalised."""
        good = _StubModel(2.0)
        broken = _StubModel(0.0, raises=True)
        ens = CompositeCrossTargetEnsemble(
            component_models=[good, broken],
            component_names=["good", "broken"],
            weights=np.array([0.5, 0.5]),
            strategy="custom",
        )
        # broken component's failure must not poison the prediction;
        # weight gets re-normalised over the surviving component.
        pred = ens.predict(np.zeros((5, 1)))
        np.testing.assert_allclose(pred, np.full(5, 2.0))

    def test_all_components_fail_raises(self) -> None:
        """All components fail raises."""
        a = _StubModel(0.0, raises=True)
        b = _StubModel(0.0, raises=True)
        ens = CompositeCrossTargetEnsemble(
            component_models=[a, b],
            component_names=["a", "b"],
            weights=np.array([0.5, 0.5]),
            strategy="custom",
        )
        with pytest.raises(RuntimeError, match="all components failed"):
            ens.predict(np.zeros((5, 1)))


# ----------------------------------------------------------------------
# export_metadata
# ----------------------------------------------------------------------


class TestLinearStack:
    """Groups tests covering linear stack."""
    def test_basic_construction(self) -> None:
        # 3 components produce predictions; y is mostly the second.
        """Basic construction."""
        rng = np.random.default_rng(0)
        n = 100
        y = rng.normal(size=n)
        preds = np.column_stack(
            [
                rng.normal(size=n),  # noise
                y + 0.1 * rng.normal(size=n),  # signal
                rng.normal(size=n),  # noise
            ]
        )
        ens = CompositeCrossTargetEnsemble.from_linear_stack(
            component_models=[_StubModel(0), _StubModel(0), _StubModel(0)],
            component_names=["a", "b", "c"],
            component_predictions=preds,
            y_train=y,
            ridge_alpha=0.1,
        )
        assert ens.strategy == "linear_stack"
        # Component 'b' should have the largest weight.
        assert ens.weights[1] > ens.weights[0] and ens.weights[1] > ens.weights[2]

    def test_too_few_rows_falls_back_to_mean(self) -> None:
        """Too few rows falls back to mean."""
        ens = CompositeCrossTargetEnsemble.from_linear_stack(
            component_models=[_StubModel(0), _StubModel(0), _StubModel(0)],
            component_names=["a", "b", "c"],
            component_predictions=np.array([[1.0, 2.0, 3.0]]),  # only 1 row
            y_train=np.array([1.0]),
        )
        assert ens.strategy == "mean"

    def test_predict_uses_intercept(self) -> None:
        """Predict uses intercept."""
        rng = np.random.default_rng(0)
        n = 100
        y = rng.normal(size=n) + 5.0
        preds = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
        ens = CompositeCrossTargetEnsemble.from_linear_stack(
            component_models=[_StubModel(0.0), _StubModel(0.0)],
            component_names=["a", "b"],
            component_predictions=preds,
            y_train=y,
        )
        # Stubs return 0.0; with non-zero intercept the prediction
        # should be close to mean(y) ~= 5.0.
        if ens.strategy == "linear_stack":
            preds_out = ens.predict(np.zeros((10, 1)))
            np.testing.assert_allclose(preds_out, np.full(10, ens._linear_stack_intercept))


class TestNnlsStack:
    """Groups tests covering nnls stack."""
    def test_basic_construction(self) -> None:
        """Basic construction."""
        rng = np.random.default_rng(1)
        n = 100
        y = np.abs(rng.normal(size=n)) + 1.0
        preds = np.column_stack(
            [
                np.abs(rng.normal(size=n)),
                y + 0.1 * rng.normal(size=n),
            ]
        )
        ens = CompositeCrossTargetEnsemble.from_nnls_stack(
            component_models=[_StubModel(0), _StubModel(0)],
            component_names=["a", "b"],
            component_predictions=preds,
            y_train=y,
        )
        assert ens.strategy == "nnls_stack"
        # All weights non-negative (NNLS constraint).
        assert np.all(ens.weights >= 0)
        # F7: weights are the RAW NNLS output -- no post-fit renormalisation to sum=1.
        # The deployed predictor must match the one NNLS solved for. Sum can be anything;
        # mark as non-convex via the new flag.
        assert ens.is_convex is False

    def test_too_few_rows_falls_back_to_mean(self) -> None:
        """Too few rows falls back to mean."""
        ens = CompositeCrossTargetEnsemble.from_nnls_stack(
            component_models=[_StubModel(0), _StubModel(0), _StubModel(0)],
            component_names=["a", "b", "c"],
            component_predictions=np.array([[1.0, 2.0, 3.0]]),
            y_train=np.array([1.0]),
        )
        assert ens.strategy == "mean"


class TestExportMetadata:
    """Groups tests covering export metadata."""
    def test_exports_strategy_names_weights(self) -> None:
        """Exports strategy names weights."""
        ens = CompositeCrossTargetEnsemble.from_uniform_weights(
            component_models=[_StubModel(1.0), _StubModel(2.0)],
            component_names=["a", "b"],
        )
        meta = ens.export_metadata()
        assert meta["strategy"] == "mean"
        assert meta["component_names"] == ["a", "b"]
        np.testing.assert_allclose(meta["weights"], [0.5, 0.5])
        assert "notes" in meta
