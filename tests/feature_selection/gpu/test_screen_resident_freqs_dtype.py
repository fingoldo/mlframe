"""The resident class-frequency vector the screen hands to ``mi_direct_gpu`` has the element type of the kernel pointer it is bound to.

``compute_mi_from_classes_cuda`` declares ``const double *freqs_y``. The screen uploaded the vector in the criterion dtype (float32 by default), so the kernel read a float32 buffer as doubles;
the launch-time audit (``kernel_arg_dtype_audit``) reported it as ``expects float64*, got a float32 array``.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:  # pragma: no cover - no device
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters._screen_predictors import _resident_screen_targets
from mlframe.feature_selection.filters.gpu import mi_direct_gpu


def _data(seed=0, n=3000):
    """A 2-column code matrix whose second column carries the target, with the class codes and frequencies of the target."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n).astype(np.int32)
    x = np.where(rng.random(n) < 0.7, y, rng.integers(0, 3, n)).astype(np.int32)
    data = np.column_stack([x, y]).astype(np.int32)
    freqs_y = np.bincount(y, minlength=3) / n
    return data, y, freqs_y


def test_the_resident_class_frequencies_are_float64():
    """The device frequency vector is float64 whatever the criterion dtype, and the label copy stays int32."""
    _, y, freqs_y = _data()
    classes_y_safe, freqs_y_safe = _resident_screen_targets(True, y, freqs_y, None)
    assert freqs_y_safe.dtype == cp.float64
    assert classes_y_safe.dtype == cp.int32


def test_a_prewarmed_frequency_buffer_gives_the_same_mi_as_none():
    """``mi_direct_gpu`` with the screen's pre-warmed device buffers agrees with the call that builds its own."""
    data, y, freqs_y = _data(1)
    nbins = np.array([3, 3], dtype=np.int32)
    common = {"x": (0,), "y": (1,), "factors_nbins": nbins, "npermutations": 40, "classes_y": y, "freqs_y": freqs_y, "return_null_mean": True, "base_seed": 7}
    base = mi_direct_gpu(data, **common)
    classes_y_safe, freqs_y_safe = _resident_screen_targets(True, y, freqs_y, None)
    warm = mi_direct_gpu(data, **common, classes_y_safe=classes_y_safe, freqs_y_safe=freqs_y_safe)
    assert base[0] > 0.1
    # (mi, confidence, null mean, p-value): the null mean comes from the per-permutation kernel launches that read the frequency buffer
    assert warm[2] == pytest.approx(base[2], rel=1e-9)
    assert warm[1] == pytest.approx(base[1], abs=1e-12)
