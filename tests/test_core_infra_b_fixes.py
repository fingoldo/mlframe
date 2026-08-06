"""Regression tests for audits/full_audit_2026-07-21/core_infra_b.md findings B1-B3, B7-B10.

B4/B5/B6 are covered by tests/integrations/test_mlflow.py. B11 (inspection/interaction.py
test-coverage gap) is ALREADY satisfied by tests/inspection/test_interaction.py's existing
quantitative additive-vs-product-model comparisons, just not under the strict test_biz_val_*
filename convention -- no duplicate test added, matching this fix-wave's established precedent
for the same situation elsewhere. P6 (knn_shapley per-row loop perf) assessed and deferred: the
audit itself calls it unmeasured/likely-negligible, and no reported bug motivates it.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# B1 / B2: dtw_dispatch falls back to CPU when the chosen GPU backend raises at call time
# ---------------------------------------------------------------------------


def test_b1_b2_dtw_dispatch_falls_back_to_cpu_when_cupy_raises(monkeypatch):
    """A cupy backend raising at call time must fall back to CPU, not propagate."""
    pytest.importorskip("dtaidistance")
    from mlframe.signal import dtw

    monkeypatch.setattr(dtw, "_HAS_CUPY", True)
    monkeypatch.setattr(dtw._DTW_SPEC, "choose", lambda **kw: "cupy")

    def _raising_cupy(*args, **kwargs):
        """Simulate a GPU driver hiccup."""
        raise RuntimeError("simulated CUDA driver hiccup")

    monkeypatch.setattr(dtw, "dtw_cupy", _raising_cupy)

    x = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    y = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    dist, _path = dtw.dtw_dispatch(x, y)
    assert np.isfinite(dist), "B1/B2 REGRESSION: a raising cupy backend must fall back to CPU, not propagate"


def test_b1_b2_dtw_dispatch_falls_back_to_cpu_when_cuda_raises(monkeypatch):
    """A numba.cuda backend raising at call time must fall back to CPU, not propagate."""
    pytest.importorskip("dtaidistance")
    from mlframe.signal import dtw

    monkeypatch.setattr(dtw, "_HAS_NB_CUDA", True)
    monkeypatch.setattr(dtw._DTW_SPEC, "choose", lambda **kw: "cuda")

    def _raising_cuda(*args, **kwargs):
        """Simulate a GPU driver hiccup."""
        raise RuntimeError("simulated CUDA driver hiccup")

    monkeypatch.setattr(dtw, "dtw_cuda", _raising_cuda)

    x = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    y = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    dist, _path = dtw.dtw_dispatch(x, y)
    assert np.isfinite(dist)


def test_b1_b2_dtw_dispatch_cpu_choice_unaffected():
    """Baseline: the ordinary CPU-choice path is unaffected by the new try/except wrapping."""
    pytest.importorskip("dtaidistance")
    from mlframe.signal import dtw

    x = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    y = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    dist_direct, path_direct = dtw.dtw_cpu(x, y)
    dist_dispatch, path_dispatch = dtw.dtw_dispatch(x, y, backend="cpu")
    assert dist_direct == pytest.approx(dist_dispatch)
    assert path_direct == path_dispatch


# ---------------------------------------------------------------------------
# B3: knn_shapley now validates y_val's dtype too, not just y_train's
# ---------------------------------------------------------------------------


def test_b3_knn_shapley_continuous_y_val_raises_with_integer_y_train():
    """A continuous y_val must raise NotImplementedError even when y_train is integer."""
    from mlframe.data_valuation import knn_shapley

    rng = np.random.default_rng(0)
    n = 30
    X_train = rng.normal(size=(n, 3))
    y_train = rng.integers(0, 2, n).astype(np.int64)  # genuine integer labels
    X_val = rng.normal(size=(10, 3))
    y_val = rng.uniform(0, 1, 10)  # continuous -- e.g. a probability column passed by mistake

    with pytest.raises(NotImplementedError, match="continuous y_val"):
        knn_shapley(X_train, y_train, X_val, y_val)


def test_b3_knn_shapley_integer_y_val_still_works():
    """Baseline: integer y_val still runs normally."""
    from mlframe.data_valuation import knn_shapley

    rng = np.random.default_rng(1)
    n = 30
    X_train = rng.normal(size=(n, 3))
    y_train = rng.integers(0, 2, n).astype(np.int64)
    X_val = rng.normal(size=(10, 3))
    y_val = rng.integers(0, 2, 10).astype(np.int64)

    values = knn_shapley(X_train, y_train, X_val, y_val, k=3)
    assert values.shape == (n,)
    assert np.all(np.isfinite(values))


# ---------------------------------------------------------------------------
# B7: recursive_forecast's growth_ratio no longer silently NaN/inf on a zero step-1 MSE
# ---------------------------------------------------------------------------


def test_b7_growth_ratio_zero_step1_mse_all_zero_gives_ones():
    """A zero step-1 MSE with all-zero later steps must report growth_ratio=1.0, not NaN."""
    from mlframe.inference.recursive_forecast import diagnose_error_accumulation

    class _PerfectModel:
        """Predicts the true target exactly at every step (recursive_mse always 0)."""

        def predict(self, X):
            """Return the true target column verbatim."""
            return np.asarray(X["y"])

    n_steps = 4
    n_rows = 5
    true_targets = np.tile(np.arange(n_rows, dtype=np.float64), (n_steps, 1))

    import pandas as pd

    initial_features = pd.DataFrame({"y": true_targets[0], "lag": true_targets[0]})

    def _update(features, pred, step):
        """Advance features to the next step's true target."""
        features = features.copy(deep=False)
        features["y"] = true_targets[step] if step < n_steps else true_targets[-1]
        features["lag"] = features["y"]
        return features

    out = diagnose_error_accumulation(
        _PerfectModel(), initial_features, n_steps, "lag", _update, true_targets,
    )
    assert np.all(out["recursive_mse"] == 0.0), "test setup: model must be exact at every step"
    assert np.all(out["growth_ratio"] == 1.0), "B7 REGRESSION: a zero step-1 MSE with all-zero later steps must report growth_ratio=1.0, not NaN"


def test_b7_growth_ratio_zero_step1_mse_nonzero_later_gives_inf():
    """A nonzero later step against a perfect step-1 baseline must report +inf, not NaN."""
    from mlframe.inference.recursive_forecast import diagnose_error_accumulation

    class _ExactThenDriftingModel:
        """Exact at step 0, then drifts -- forces recursive_mse[0]=0 with recursive_mse[k>0] > 0."""

        def __init__(self):
            self.call = 0

        def predict(self, X):
            """Return the true target on the first call, drifted by +5 afterwards."""
            self.call += 1
            base = np.asarray(X["y"])
            return base if self.call == 1 else base + 5.0

    n_steps = 3
    n_rows = 4
    true_targets = np.tile(np.arange(n_rows, dtype=np.float64), (n_steps, 1))
    import pandas as pd

    initial_features = pd.DataFrame({"y": true_targets[0], "lag": true_targets[0]})

    def _update(features, pred, step):
        """Advance features' lag column to the model's own prior prediction."""
        features = features.copy(deep=False)
        features["lag"] = pred
        return features

    out = diagnose_error_accumulation(
        _ExactThenDriftingModel(), initial_features, n_steps, "lag", _update, true_targets,
    )
    assert out["recursive_mse"][0] == 0.0, "test setup: step 0 must be exact"
    assert out["growth_ratio"][0] == 1.0
    assert np.isinf(out["growth_ratio"][1]), "B7 REGRESSION: a nonzero later step against a perfect step-1 baseline must report +inf honestly, not NaN"


def test_b7_growth_ratio_normal_nonzero_baseline_unaffected():
    """Baseline: a genuinely nonzero step-1 MSE keeps growth_ratio's ordinary division behavior."""
    from mlframe.inference.recursive_forecast import diagnose_error_accumulation

    class _NoisyModel:
        """Predicts the true target plus Gaussian noise."""

        def __init__(self, rng):
            self.rng = rng

        def predict(self, X):
            """Return the true target column plus noise."""
            base = np.asarray(X["y"])
            return base + self.rng.normal(scale=0.5, size=base.shape)

    rng = np.random.default_rng(3)
    n_steps = 3
    n_rows = 20
    true_targets = np.tile(np.arange(n_rows, dtype=np.float64), (n_steps, 1))
    import pandas as pd

    initial_features = pd.DataFrame({"y": true_targets[0] + 10.0, "lag": true_targets[0] + 10.0})  # offset so step-1 MSE > 0

    def _update(features, pred, step):
        """Advance features' lag column to the model's own prior prediction."""
        features = features.copy(deep=False)
        features["lag"] = pred
        return features

    out = diagnose_error_accumulation(_NoisyModel(rng), initial_features, n_steps, "lag", _update, true_targets)
    assert out["recursive_mse"][0] > 0.0
    assert np.all(np.isfinite(out["growth_ratio"]))
    assert out["growth_ratio"][0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# B8: apply_group_zero_sum_constraint rejects negative weights
# ---------------------------------------------------------------------------


def test_b8_group_zero_sum_constraint_rejects_negative_weights():
    """A negative weight must raise ValueError, not silently corrupt the projection."""
    from mlframe.inference.group_zero_sum_constraint import apply_group_zero_sum_constraint

    preds = np.array([1.0, 2.0, 3.0, 4.0])
    group_ids = np.array([0, 0, 1, 1])
    weights = np.array([1.0, -0.5, 1.0, 1.0])  # a signed weight by mistake

    with pytest.raises(ValueError, match="non-negative"):
        apply_group_zero_sum_constraint(preds, group_ids, target_sum=0.0, weights=weights)


def test_b8_group_zero_sum_constraint_rejects_negative_extra_constraint_coefs():
    """A negative extra-constraint coefficient must also raise ValueError."""
    from mlframe.inference.group_zero_sum_constraint import apply_group_zero_sum_constraint

    preds = np.array([1.0, 2.0, 3.0, 4.0])
    group_ids = np.array([0, 0, 1, 1])
    bad_coefs = np.array([1.0, -1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="non-negative"):
        apply_group_zero_sum_constraint(
            preds, group_ids, target_sum=0.0,
            extra_constraint_coefs=[bad_coefs], extra_constraint_targets=[0.0],
        )


def test_b8_group_zero_sum_constraint_nonnegative_weights_still_work():
    """Baseline: non-negative weights still project correctly."""
    from mlframe.inference.group_zero_sum_constraint import apply_group_zero_sum_constraint

    preds = np.array([1.0, 2.0, 3.0, 4.0])
    group_ids = np.array([0, 0, 1, 1])
    weights = np.array([1.0, 2.0, 1.0, 1.0])
    out = apply_group_zero_sum_constraint(preds, group_ids, target_sum=0.0, weights=weights)
    assert out.shape == preds.shape
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# B9: sample_random_variable raises a clear ValueError on an invalid kind
# ---------------------------------------------------------------------------


def test_b9_sample_random_variable_invalid_kind_raises_value_error():
    """An unknown kind must raise a clear ValueError, not NameError on the unbound source."""
    from mlframe.data.synthetic import sample_random_variable

    with pytest.raises(ValueError, match="kind"):
        sample_random_variable(kind="not_a_real_kind")


def test_b9_sample_random_variable_valid_kinds_still_work():
    """Baseline: every documented kind still samples normally."""
    from mlframe.data.synthetic import sample_random_variable

    for kind in ("cont", "cat", "mixed"):
        dist_name, data = sample_random_variable(kind=kind, size=50, random_state=0)
        assert isinstance(dist_name, str) and dist_name
        assert data is not None and len(data) == 50


# ---------------------------------------------------------------------------
# B10: the numba.cuda full-matrix DTW kernel is documented as intentional A/B reference
# ---------------------------------------------------------------------------


def test_b10_numba_cuda_diagonal_step_documented_as_reference_not_dead_code():
    """The numba.cuda kernel's docstring must self-document as an intentional A/B reference."""
    from mlframe.signal import dtw

    if not dtw._HAS_NB_CUDA:
        pytest.skip("numba.cuda not available on this host")
    doc = dtw._numba_cuda_diagonal_step.__doc__ or ""
    assert "reference" in doc.lower() or "REJECTED" in doc
