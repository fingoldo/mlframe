"""The device |corr(y)| used for leader usability on the deferred-float GPU path agrees with the host njit statistic and moves only scalars."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_abs_corr_gpu import abs_corr_finite_gpu, abs_corr_or_none
from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core_steps import _abs_corr_finite_njit
from mlframe.feature_selection.filters._gpu_strict_fe import residency_audit


def _case(n: int, seed: int, nan_frac: float, offset: float = 0.0):
    """A column correlated with y, a target with some non-finite rows and a column with some non-finite entries."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    a = 0.7 * y + rng.standard_normal(n) + offset
    y[rng.random(n) < nan_frac] = np.nan
    a[rng.random(n) < nan_frac] = np.inf
    return a, y, np.isfinite(y)


@pytest.mark.parametrize("n, seed, nan_frac, offset", [(500, 0, 0.0, 0.0), (5000, 1, 0.05, 0.0), (5000, 2, 0.0, 1e7), (64, 3, 0.4, 0.0)])
def test_device_abs_corr_matches_the_host_kernel(n, seed, nan_frac, offset):
    """Same statistic as ``_abs_corr_finite_njit`` to FP-reorder precision, including offset data and non-finite rows."""
    a, y, fin = _case(n, seed, nan_frac, offset)
    host = float(_abs_corr_finite_njit(a, y, fin, 8))
    dev = abs_corr_finite_gpu(cp.asarray(a), y, fin, 8)
    assert dev == pytest.approx(host, rel=1e-9, abs=1e-12)


def test_device_abs_corr_degenerate_inputs_return_zero_like_the_host():
    """A constant column, too few joint-finite rows and a missing target all give 0.0."""
    y = np.random.default_rng(0).standard_normal(200)
    fin = np.isfinite(y)
    assert abs_corr_finite_gpu(cp.full(200, 3.0), y, fin, 8) == 0.0
    assert abs_corr_finite_gpu(cp.asarray(y[:5]), y[:5], fin[:5], 8) == 0.0
    assert abs_corr_or_none(cp.asarray(y), None, None) == 0.0
    assert abs_corr_or_none(cp.asarray(y[:10]), y, fin) == 0.0


def test_device_abs_corr_only_moves_scalars_to_the_host():
    """No bulk device->host copy: the column stays resident and only the reduced statistics return."""
    a, y, fin = _case(200_000, 4, 0.01)
    a_dev = cp.asarray(a)
    abs_corr_finite_gpu(a_dev, y, fin, 8)  # warm: uploads the target once
    with residency_audit() as rep:
        abs_corr_finite_gpu(a_dev, y, fin, 8)
    assert not rep.bulk_d2h
    assert rep.scalar_d2h_bytes < 1024


@pytest.mark.parametrize("n, seed, nan_frac", [(500, 0, 0.0), (5000, 1, 0.05), (64, 3, 0.4)])
def test_device_zerofill_corr_matches_the_host_kernel(n, seed, nan_frac):
    """The zero-fill statistic used by the degenerate-pair veto agrees with ``_abs_corr_zerofill_njit``, including columns with non-finite entries."""
    from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_abs_corr_gpu import abs_corr_zerofill_gpu
    from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core import _abs_corr_zerofill_njit

    a, b, _ = _case(n, seed, nan_frac)
    host = float(_abs_corr_zerofill_njit(a, b))
    assert abs_corr_zerofill_gpu(cp.asarray(a), b) == pytest.approx(host, rel=1e-9, abs=1e-12)
    assert abs_corr_zerofill_gpu(cp.full(n, 2.0), b) == 0.0
