"""Regression: ``apply_preprocessing_extensions`` must preserve the
human-readable column names emitted by each named transformer
(scaler, kbins, poly, dim_reducer, ...). Pre-fix every output column
was renamed to ``ext_<i>`` which destroyed interpretability.

Fix uses ``pipe.get_feature_names_out()`` when available (sklearn>=1.3),
otherwise falls back to ``ext_<step_name>_<i>``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _toy_frame(n=80, p=4, seed=0):
    """Toy frame."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])


def test_extension_columns_use_descriptive_names():
    """Scaler + KBins pipeline output columns should mention the source feature
    OR the step name - never bare ``ext_<i>`` indices."""
    from mlframe.training.pipeline import apply_preprocessing_extensions
    from mlframe.training.configs import PreprocessingExtensionsConfig

    cfg = PreprocessingExtensionsConfig(scaler="StandardScaler", kbins=3)
    train = _toy_frame()
    val = _toy_frame(seed=1)
    test = _toy_frame(seed=2)
    out_train, _out_val, _out_test, pipe = apply_preprocessing_extensions(
        train,
        val,
        test,
        cfg,
        verbose=0,
    )
    assert pipe is not None
    cols = list(out_train.columns)
    # Reject the pre-fix naming pattern: cols must NOT be ['ext_0', 'ext_1', ...]
    pre_fix_pattern = [f"ext_{i}" for i in range(len(cols))]
    assert cols != pre_fix_pattern, f"columns still use pre-fix bare-index naming: {cols}"
    # Each col name should encode the source feature (sklearn get_feature_names_out)
    # OR the step name (fallback). We accept either signal.
    src_features = {f"f{i}" for i in range(4)}
    step_names = {"scaler", "kbins", "imputer"}
    informative = sum(1 for c in cols if any(s in c for s in src_features) or any(s in c for s in step_names))
    assert informative == len(cols), f"some output columns lack a recognizable source-feature or step-name substring: {cols}"


def test_extension_columns_consistent_across_splits():
    """Train, val, test output must share the same column ordering."""
    from mlframe.training.pipeline import apply_preprocessing_extensions
    from mlframe.training.configs import PreprocessingExtensionsConfig

    cfg = PreprocessingExtensionsConfig(scaler="MinMaxScaler")
    train = _toy_frame()
    val = _toy_frame(seed=1)
    test = _toy_frame(seed=2)
    out_train, out_val, out_test, _ = apply_preprocessing_extensions(
        train,
        val,
        test,
        cfg,
        verbose=0,
    )
    assert list(out_train.columns) == list(out_val.columns) == list(out_test.columns)
