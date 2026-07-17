"""Regression tests for SER1 (MLPRanker pickle) + MEM1/MEM4 (custom estimator copies)."""

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

import numpy as np
import pandas as pd
import pytest

from mlframe.estimators.custom import PdOrdinalEncoder, PdKBinsDiscretizer, MyDecorrelator


def test_mlpranker_getstate_nulls_live_trainer():
    """SER1: a fitted-shape MLPRanker holding a live lightning.Trainer must pickle without
    error and come back with trainer_=None (the live Trainer references a WarningCache the
    save_load SafeUnpickler blocks). Stub the live attrs to avoid a full lightning fit."""
    L = pytest.importorskip("lightning")
    from mlframe.training.neural.ranker import MLPRanker

    est = MLPRanker(n_estimators=1)
    # Simulate post-fit state: a live Trainer + a torch module (the unpicklable-via-safe-loader combo).
    est.trainer_ = L.Trainer(logger=False, enable_checkpointing=False, enable_progress_bar=False)
    import torch

    est.module_ = torch.nn.Linear(3, 1)
    est.n_features_in_ = 3

    blob = pickle.dumps(est)
    restored = pickle.loads(blob)  # nosec B301 -- round-trip of a locally-created, trusted object

    assert restored.trainer_ is None
    # Module params survive on CPU and stay usable.
    assert restored.module_ is not None
    assert next(restored.module_.parameters()).device.type == "cpu"


def test_pd_ordinal_encoder_no_full_copy_of_int32_array():
    """MEM1: transform on an already-int32 ndarray must not broadcast-copy through a DataFrame."""
    enc = PdOrdinalEncoder()
    X = np.array([[0, 1], [1, 0], [2, 2]], dtype=np.int32)
    enc.fit(X)
    out = enc.transform(X)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int32
    # super().transform returns float; the cast still happens, but the result must equal the
    # ground-truth codes and must NOT be wrapped in a DataFrame for the ndarray-input path.
    expected = enc.fit_transform(X).astype(np.int32)
    assert np.array_equal(out, expected)


def test_pd_ordinal_encoder_no_extra_copy_when_already_int32():
    """MEM1: when the post-encode buffer is already int32, astype(copy=False) reuses it."""
    enc = PdOrdinalEncoder()
    X = np.array([[0, 1], [1, 0]], dtype=np.int32)
    enc.fit(X)
    # Patch super().transform to return an already-int32 array so we can assert zero-copy.
    arr_int32 = np.array([[0, 1], [1, 0]], dtype=np.int32)
    from sklearn.preprocessing import OrdinalEncoder

    orig = OrdinalEncoder.transform
    try:
        OrdinalEncoder.transform = lambda self, X: arr_int32
        out = enc.transform(X)
    finally:
        OrdinalEncoder.transform = orig
    assert np.shares_memory(out, arr_int32), "int32 buffer should be reused, not copied"


def test_kbins_discretizer_ndarray_path_no_dataframe_wrap():
    """MEM1: KBins transform on ndarray input returns a plain int32 ndarray."""
    disc = PdKBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform")
    rng = np.random.default_rng(0)
    X = rng.random((30, 2))
    disc.fit(X)
    out = disc.transform(X)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int32


def test_mydecorrelator_dataframe_no_full_wrap():
    """MEM4: a DataFrame input drops correlated columns directly (no fresh-frame wrap)."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0], "c": [4.0, 1.0, 3.0, 2.0]})
    dec = MyDecorrelator(threshold=0.99)
    dec.fit(df)
    out = dec.transform(df)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == set(df.columns) - dec.correlated_features_
    # a and b are perfectly correlated; exactly one is dropped.
    assert len(out.columns) == 2


def test_mydecorrelator_ndarray_path_index_drop():
    """MEM4: an ndarray input drops correlated columns by index without a DataFrame wrap."""
    X = np.array([[1.0, 1.0, 4.0], [2.0, 2.0, 1.0], [3.0, 3.0, 3.0], [4.0, 4.0, 2.0]])
    dec = MyDecorrelator(threshold=0.99)
    dec.fit(X)
    out = dec.transform(X)
    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 2)
