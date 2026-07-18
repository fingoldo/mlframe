"""Regression: CatBoost Pool label coercion must not silently lose float32 precision.

Large-magnitude regression targets (int counts/IDs > 2**24, high-precision floats)
collapse adjacent values under an unconditional ``astype(np.float32)``, biasing the
fit. The guard keeps float64 when float32 would be lossy (CatBoost accepts float64).
"""

import numpy as np

from mlframe.training.cb._cb_pool import _coerce_label_for_cb_pool


def test_large_int_target_not_truncated_to_float32():
    # 2**24 + 1 = 16_777_217 is the first int not exactly representable in float32
    # (rounds to 16_777_216). Two adjacent labels must stay distinct.
    """Large int target not truncated to float32."""
    target = np.array([16_777_216, 16_777_217, 16_777_219], dtype=np.int64)
    out = _coerce_label_for_cb_pool(target)
    # Pre-fix: float32 -> [16777216, 16777216, 16777220] (first two collapse).
    assert len(np.unique(out)) == 3, "adjacent large-int labels collapsed under float32"
    assert np.array_equal(out.astype(np.int64), target)


def test_high_precision_float_target_preserved():
    """High precision float target preserved."""
    target = np.array([1234567.89, 1234567.90, 0.123456789012345], dtype=np.float64)
    out = _coerce_label_for_cb_pool(target)
    # float32 cannot distinguish 1234567.89 vs 1234567.90; must stay float64.
    assert out.dtype == np.float64
    assert out[0] != out[1]


def test_small_classification_labels_still_downcast_to_float32():
    # Benign case: small int class labels remain float32 (lossless, memory-friendly).
    """Small classification labels still downcast to float32."""
    target = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    out = _coerce_label_for_cb_pool(target)
    assert out.dtype == np.float32
    assert np.array_equal(out, target.astype(np.float32))


def test_lossless_float64_downcasts():
    """Lossless float64 downcasts."""
    target = np.array([0.5, 1.0, -2.25], dtype=np.float64)  # all exact in float32
    out = _coerce_label_for_cb_pool(target)
    assert out.dtype == np.float32
