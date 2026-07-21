"""mrmr_audit_2026-07-20 edge_cases.md #171: partial_fit resumption across mismatched dtypes. A
float32 batch followed by a float64 (or int64) batch on the SAME column must not silently corrupt
or crash -- pandas.concat's automatic dtype-upcast on the accumulated buffer means every subsequent
fit() re-bins the WHOLE (now-homogeneous) buffer from scratch, so there is no cross-batch binning-
edge drift; this test pins that the values survive the upcast exactly and the fit does not crash."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import MRMR


def _kw(**overrides):
    """Fast-fitting default MRMR constructor kwargs, overridable per test."""
    base = dict(random_seed=42, verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2, fe_max_steps=0, skip_retraining_on_same_content=False, partial_fit_min_recompute=1)
    base.update(overrides)
    return base


def test_float32_then_float64_batch_upcasts_without_value_corruption():
    """A float32 first batch followed by a float64 second batch on the same column must not crash,
    and the accumulated buffer's values must survive the upcast bit-for-bit (as widened doubles)."""
    rng = np.random.default_rng(0)
    n1, n2 = 150, 150
    a32 = rng.standard_normal(n1).astype(np.float32)
    y1 = pd.Series((a32 > 0).astype(int))
    X1 = pd.DataFrame({"a": a32})

    m = MRMR(**_kw())
    m.partial_fit(X1, y1)
    assert m._partial_fit_X_buffer_["a"].dtype == np.dtype("float32")

    a64 = rng.standard_normal(n2).astype(np.float64)
    y2 = pd.Series((a64 > 0).astype(int))
    X2 = pd.DataFrame({"a": a64})
    m.partial_fit(X2, y2)

    buf = m._partial_fit_X_buffer_["a"].to_numpy()
    assert buf.dtype == np.float64, "pandas.concat must upcast the mixed-dtype buffer to float64"
    np.testing.assert_array_equal(buf[:n1], a32.astype(np.float64))
    np.testing.assert_array_equal(buf[n1:], a64)
    assert hasattr(m, "support_")


def test_float32_then_int64_batch_upcasts_without_value_corruption():
    """Same contract for a float32 -> int64 dtype transition across batches."""
    rng = np.random.default_rng(1)
    n1, n2 = 150, 150
    a32 = rng.standard_normal(n1).astype(np.float32)
    y1 = pd.Series((a32 > 0).astype(int))
    X1 = pd.DataFrame({"a": a32})

    m = MRMR(**_kw())
    m.partial_fit(X1, y1)

    a_int = rng.integers(-5, 5, n2).astype(np.int64)
    y2 = pd.Series((a_int > 0).astype(int))
    X2 = pd.DataFrame({"a": a_int})
    m.partial_fit(X2, y2)

    buf = m._partial_fit_X_buffer_["a"].to_numpy()
    assert buf.dtype == np.float64, "mixed float32/int64 must upcast to float64, not silently truncate"
    np.testing.assert_array_equal(buf[:n1], a32.astype(np.float64))
    np.testing.assert_array_equal(buf[n1:], a_int.astype(np.float64))
    assert hasattr(m, "support_")
