"""Regression tests verifying that ``FeatureSelectionConfig.use_sample_weights_in_fs`` is wired
end-to-end through ``_build_pre_pipelines`` -> ``_passthrough_cols_fit_transform`` -> ``_call_fit``.

Pre-fix:
- ``_phase_train_one_target`` never forwarded the flag into ``_build_pre_pipelines``, so the
  marker ``_mlframe_use_sample_weights_in_fs_`` was always stamped False.
- ``_passthrough_cols_fit_transform`` ignored ``sample_weight`` and only forwarded ``groups``.
- The pre-pipeline LRU cache key omitted ``sample_weight``, so weight-aware fits collided
  with uniform-weight fits across weight schemas.

Post-fix the marker is honoured by ``_call_fit`` and the cache key folds sample_weight when
the marker is True.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.mark.fast
def test_passthrough_forwards_sample_weight_when_marker_set():
    """A selector marked with ``_mlframe_use_sample_weights_in_fs_=True`` MUST receive
    ``sample_weight`` in its fit kwargs when one is supplied."""
    from mlframe.training._pipeline_helpers import _passthrough_cols_fit_transform

    captured: dict = {}

    class _DummySelector:
        # Marker: the suite stamps this on MRMR / RFECV when the config flag is True.
        _mlframe_use_sample_weights_in_fs_ = True

        def fit_transform(self, X, y=None, **kwargs):
            captured["kwargs"] = dict(kwargs)
            return X

    selector = _DummySelector()
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    target = np.array([0, 1, 0])
    sample_weight = np.array([0.1, 0.7, 0.2])

    _passthrough_cols_fit_transform(
        selector.fit_transform,
        df,
        passthrough_cols=None,
        fit=True,
        target=target,
        sample_weight=sample_weight,
    )

    assert "sample_weight" in captured["kwargs"], "sample_weight must reach the selector fit"
    np.testing.assert_array_equal(captured["kwargs"]["sample_weight"], sample_weight)


@pytest.mark.fast
def test_passthrough_skips_sample_weight_when_marker_unset():
    """Default-OFF: when the selector is NOT marked weight-aware, sample_weight is NOT forwarded
    so the FS cache stays valid across weight schemas."""
    from mlframe.training._pipeline_helpers import _passthrough_cols_fit_transform

    captured: dict = {}

    class _PlainSelector:
        # No marker attribute -- default getattr returns False.
        def fit_transform(self, X, y=None, **kwargs):
            captured["kwargs"] = dict(kwargs)
            return X

    selector = _PlainSelector()
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    sample_weight = np.array([0.5, 0.5])

    _passthrough_cols_fit_transform(
        selector.fit_transform,
        df,
        passthrough_cols=None,
        fit=True,
        target=np.array([0, 1]),
        sample_weight=sample_weight,
    )

    assert "sample_weight" not in captured["kwargs"], (
        "Unmarked selector must NOT receive sample_weight (keeps FS cache valid across weight schemas)"
    )


@pytest.mark.fast
def test_build_pre_pipelines_stamps_marker_when_flag_true():
    """``_build_pre_pipelines(use_sample_weights_in_fs=True)`` stamps the marker on the MRMR
    instance so the downstream fit driver knows to forward weights."""
    pytest.importorskip("mlframe.feature_selection.filters")

    from mlframe.training.core._setup_helpers import _build_pre_pipelines

    pipelines, names = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=True,
        mrmr_kwargs={"verbose": 0},
        use_sample_weights_in_fs=True,
    )
    assert len(pipelines) >= 1
    mrmr_instance = next(p for p, n in zip(pipelines, names) if n.strip() == "MRMR")
    assert getattr(mrmr_instance, "_mlframe_use_sample_weights_in_fs_", None) is True


@pytest.mark.fast
def test_pre_pipeline_cache_key_folds_sample_weight_only_when_marker_set():
    """Cache key MUST diverge across weight schemas when the pipeline is weight-aware,
    and stay invariant otherwise (so default-OFF uniform-weight FS shares cache slots)."""
    from mlframe.training._pipeline_helpers import _pre_pipeline_cache_key

    class _MarkedPipeline:
        _mlframe_use_sample_weights_in_fs_ = True
        steps = []  # sklearn-style iterable, empty so signature is stable

    class _PlainPipeline:
        _mlframe_use_sample_weights_in_fs_ = False
        steps = []

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    target = pd.Series([0, 1, 0])
    w1 = np.array([0.1, 0.8, 0.1])
    w2 = np.array([0.5, 0.4, 0.1])

    marked = _MarkedPipeline()
    plain = _PlainPipeline()

    # Weight-aware pipeline: different weights -> different keys.
    assert _pre_pipeline_cache_key(df, None, marked, target, "y", sample_weight=w1) != \
        _pre_pipeline_cache_key(df, None, marked, target, "y", sample_weight=w2)

    # Weight-agnostic pipeline: different weights -> SAME key (cache reuse preserved).
    assert _pre_pipeline_cache_key(df, None, plain, target, "y", sample_weight=w1) == \
        _pre_pipeline_cache_key(df, None, plain, target, "y", sample_weight=w2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-s", "--no-cov", "--tb=short"]))
