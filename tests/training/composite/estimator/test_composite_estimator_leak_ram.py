"""Regression sensors for the 2026-06-10 composite estimator/shim audit.

- E2: the per-model composite wrap built the y-clip envelope from FULL y
  (train+val+test); since the end-of-target pass skips already-wrapped
  entries, that envelope persisted and leaked the test range into reported
  metrics. Fixed by slicing to train_idx.
- E4: fit() copied the whole X even when every row passed domain_check
  (the common case) -- a RAM doubling forbidden on 100+ GB frames.
- E5: PrePipelinePredictShim._transform swallowed ALL exceptions and fed
  untransformed X to the inner (silent garbage). Now re-raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.post_shim import PrePipelinePredictShim


class _RecordingInner(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.fit_X_id = None

    def fit(self, X, y, **kw):
        self.fit_X_id = id(X)
        self.n_features_in_ = X.shape[1]
        self._t = float(np.mean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self._t, dtype=np.float64)


class TestFitNoCopyOnAllValid:
    def test_inner_receives_same_frame_object_when_all_rows_valid(self) -> None:
        """E4: with no domain violations, fit() must pass X through to the
        inner WITHOUT materialising a row-subset copy."""
        rng = np.random.default_rng(0)
        n = 500
        base = rng.normal(5.0, 1.0, size=n)
        feat = rng.normal(0.0, 1.0, size=n)
        y = base + 0.3 * feat + rng.normal(0.0, 0.1, size=n)
        X = pd.DataFrame({"b": base, "feat": feat})
        inner = _RecordingInner()
        est = CompositeTargetEstimator(
            base_estimator=inner,
            transform_name="diff",
            base_column="b",
        )
        est.fit(X, y)
        # diff domain is all-finite -> all rows valid -> no copy -> the (cloned)
        # inner must have been fit on the very same DataFrame object.
        assert est.estimator_.fit_X_id == id(X), "fit() copied X even though every row passed domain_check"


class _RaisingPipeline(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise ValueError("simulated scaler/frame-type failure")


class TestShimReraises:
    def test_transform_failure_raises_not_silently_passthrough(self) -> None:
        """E5: a failing pre_pipeline.transform must raise (loud), never feed
        the untransformed X to the inner."""
        shim = PrePipelinePredictShim(
            model=_RecordingInner(),
            pre_pipeline=_RaisingPipeline(),
            name="c0",
        )
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(RuntimeError, match="refusing to feed untransformed"):
            shim._transform(X)

    def test_no_pipeline_is_passthrough(self) -> None:
        shim = PrePipelinePredictShim(model=_RecordingInner(), pre_pipeline=None)
        X = pd.DataFrame({"a": [1.0, 2.0]})
        assert shim._transform(X) is X


class TestPerModelEnvelopeTrainOnly:
    def _run_hook(self, train_idx):
        from mlframe.training.core._phase_composite_wrapping import (
            emit_per_model_composite_y_scale_test,
        )

        n = 100
        rng = np.random.default_rng(1)
        feat = rng.normal(0.0, 1.0, size=n)
        y_full = np.empty(n, dtype=np.float64)
        y_full[:80] = rng.uniform(0.0, 10.0, size=80)  # train range
        y_full[80:] = 1000.0  # extreme test rows
        test_idx = np.arange(80, n)
        X = pd.DataFrame({"feat": feat})
        test_df = X.iloc[test_idx].reset_index(drop=True)

        class _Entry:
            pass

        entry = _Entry()
        entry.model = _RecordingInner().fit(X.iloc[:80], y_full[:80])
        spec = {
            "transform_name": "cbrt_y",  # unary, no base column needed
            "base_column": "",
            "fitted_params": {},
            "extra_base_columns": (),
        }
        emit_per_model_composite_y_scale_test(
            entry=entry,
            composite_spec=spec,
            orig_target_name="y",
            composite_name="y-cbrt",
            target_name="y-cbrt",
            y_full=y_full,
            test_idx=test_idx,
            test_df_pd=test_df,
            train_idx=train_idx,
        )
        return entry.model

    def test_train_idx_excludes_test_extremes_from_clip(self) -> None:
        """E2: with train_idx supplied, the y-clip envelope must reflect the
        train range (~[0,10] -> high ~100), NOT the 1000-valued test rows."""
        wrapper = self._run_hook(train_idx=np.arange(80))
        assert isinstance(wrapper, CompositeTargetEstimator)
        high = float(wrapper.fitted_params_["y_clip_high"])
        assert high < 200.0, f"y_clip_high={high} leaked the extreme test rows into the clip envelope (expected ~100 from the train range)"

    def test_full_y_would_leak_demonstrates_bug(self) -> None:
        """Control: without train_idx (legacy full-y path) the envelope is
        polluted by the 1000-valued test rows -- this is exactly the leak the
        fix removes."""
        wrapper = self._run_hook(train_idx=None)
        high = float(wrapper.fitted_params_["y_clip_high"])
        assert high > 500.0, "full-y control should show the leaked wide clip"
