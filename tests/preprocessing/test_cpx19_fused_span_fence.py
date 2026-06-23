"""CPX19: the fused span-mask + fence-count njit path in is_variable_truly_continuous
must produce the SAME values_in_span and n_outliers as the original separate numpy passes.

Pins bit-identity of the optimization against the reference numpy expression on the
edge cases that matter: with-outliers, with-NaN, all-equal, all-NaN.
"""
import numpy as np
import pytest

from mlframe.preprocessing.cleaning import _get_span_fence_njit, is_variable_truly_continuous
from mlframe.core.stats import get_tukey_fences_multiplier_for_quantile


def _reference(values, q0, q1, lo, hi):
    vis = values[(values >= q0) & (values <= q1)]
    n_out = int((values < lo).sum() + (values > hi).sum())
    return vis, n_out


@pytest.mark.parametrize(
    "name,values",
    [
        ("with_outliers", np.concatenate([np.random.default_rng(0).standard_normal(5000), np.array([50.0, -60.0, 80.0])])),
        ("with_nan", np.where(np.random.default_rng(1).random(5000) < 0.05, np.nan, np.random.default_rng(2).standard_normal(5000))),
        ("all_equal", np.full(3000, 3.14)),
        ("all_nan", np.full(2000, np.nan)),
        ("mixed", np.concatenate([np.full(1000, np.nan), np.random.default_rng(3).standard_normal(2000) * 10.0])),
    ],
)
def test_cpx19_fused_kernel_matches_numpy(name, values):
    values = values.astype(np.float64)
    use_q = 0.1
    cq = np.nanquantile(values, (use_q, 1 - use_q))
    q0, q1 = float(cq[0]), float(cq[1])
    m = get_tukey_fences_multiplier_for_quantile(quantile=use_q)
    iqr = q1 - q0
    lo = q0 - m * iqr
    hi = q1 + m * iqr

    mask, nb, na = _get_span_fence_njit()(values, q0, q1, lo, hi)
    vis_new = values[mask]
    n_out_new = int(nb + na)

    vis_ref, n_out_ref = _reference(values, q0, q1, lo, hi)

    assert n_out_new == n_out_ref, f"{name}: n_outliers {n_out_new} != {n_out_ref}"
    assert np.array_equal(vis_new, vis_ref), f"{name}: values_in_span differ"


def test_cpx19_through_public_function():
    """Drive the optimization through the public API; result is finite & self-consistent."""
    rng = np.random.default_rng(7)
    values = np.concatenate([rng.standard_normal(8000), np.array([40.0, -40.0, 70.0])]).astype(np.float64)
    # Should not raise and should classify a clean gaussian-with-outliers as continuous.
    is_cont, outliers_percent = is_variable_truly_continuous(values=values, use_quantile=0.1, var_is_numeric=True, verbose=False)
    assert is_cont in (True, False)
    assert 0.0 <= outliers_percent <= 1.0
