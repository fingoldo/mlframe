"""The one-pass composite residuals equal the vectorised chain they replace, bit for bit, with NaN rows, constant groups, near-zero means and unseen groups."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _composite_group_agg_fe as C


def _old_chain(x, keys, uniq, inverse, lookup_mean, lookup_std, gm, gs):
    """The original per-row vectors followed by the elementwise z / ratio formulas."""
    pm = C._broadcast_lookup(keys, lookup_mean, gm, uniq=uniq, inverse=inverse)
    ps = C._broadcast_lookup(keys, lookup_std, gs, uniq=uniq, inverse=inverse)
    ps = np.where(ps > 0.0, ps, 1.0)
    z = np.nan_to_num((x - pm) / ps, nan=0.0, posinf=0.0, neginf=0.0).astype(x.dtype, copy=False)
    denom = C._ratio_denominator(pm, gm, gs)
    ratio = np.nan_to_num(x / denom, nan=1.0, posinf=1.0, neginf=1.0).astype(x.dtype, copy=False)
    return z, ratio


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", [31, 4097])
def test_kernel_equals_the_chain(dtype, n):
    """Same z and ratio columns, element for element."""
    rng = np.random.default_rng(0)
    keys = rng.choice(np.array(["a", "b", "c", "d", "e"], dtype=object), size=n)
    x = rng.normal(size=n).astype(dtype)
    x[::17] = np.nan
    x[5] = np.inf
    uniq, inverse = C._unique_inverse(keys)
    lookup_mean = {"a": 0.5, "b": 0.0, "c": 1e-15, "d": float("nan"), "e": 2.0}
    lookup_std = {"a": 1.5, "b": 0.0, "c": float("nan"), "d": 2.0, "e": 0.3}
    gm, gs = 0.1, 1.2
    want = _old_chain(x, keys, uniq, inverse, lookup_mean, lookup_std, gm, gs)
    got = C._composite_residuals(x, uniq, inverse, lookup_mean, lookup_std, gm, gs)
    for g, w in zip(got, want):
        assert g.dtype == w.dtype
        np.testing.assert_array_equal(g, w)
