"""The cheap constancy check of the extra-basis legs compares with the threshold exactly as ``np.std`` does."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe_generate import _std_vs

_THR = 1e-12


def _cases():
    """Constant (including the float32 rounding-residue case), tiny-spread, non-finite, empty and ordinary columns of both float widths."""
    rng = np.random.default_rng(0)
    out = [
        np.zeros(5),
        np.full(100, 3.3, dtype=np.float32),  # np.std is 9.5e-7 here, not 0
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([-np.inf, np.inf]),
        np.array([0.0, 1e-13] * 50),
        np.array([0.0, 4e-12] * 50),
        np.array([0.0, 3e-12] * 5000, dtype=np.float32),
        np.empty(0),
    ]
    out += [rng.standard_normal(n).astype(dt) * sc for n in (2, 10, 1000, 30000) for dt in (np.float32, np.float64) for sc in (1.0, 1e-11, 1e-13, 1e-14)]
    return out


@pytest.mark.parametrize("v", _cases(), ids=lambda v: f"{v.dtype}-{v.size}")
def test_same_verdict_as_numpy_std(v):
    """``> thr`` and ``<= thr`` give the same answers as on ``float(np.std(v))``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        want = float(np.std(v))
        got = _std_vs(v, _THR)
    assert (got > _THR) == (want > _THR)
    assert (got <= _THR) == (want <= _THR)
