"""The pair-prewarp ALS target is winsorized only when its variance is carried by a thin tail."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_setup import winsorize_heavy_tailed_target
from mlframe.feature_selection.filters.hermite_fe._hermite_prewarp import apply_operand_prewarp, fit_pair_prewarp_als


def _case2(n=30000):
    """Two operands and a heavy-tailed target containing a log(c) * sin(d) factor."""
    rng = np.random.default_rng(0)
    a, b, c, d, f = (rng.random(n) for _ in range(5))
    return c, d, 0.2 * a**2 / b + f / 5.0 + np.log(c * 2) * np.sin(d / 3)


def test_light_tailed_target_is_returned_unchanged():
    """Includes smoothly heavy-tailed targets (lognormal, Student-t(3)) whose tail is part of the signal."""
    rng = np.random.default_rng(1)
    for y in (rng.standard_normal(5000), rng.lognormal(0.0, 1.0, 5000), rng.standard_t(3, 5000)):
        assert winsorize_heavy_tailed_target(y) is y


def test_heavy_tailed_target_is_clipped_to_its_1_99_quantiles():
    """A heavy-tailed target is clipped exactly to its 1st and 99th percentiles."""
    _, _, y = _case2()
    out = winsorize_heavy_tailed_target(y)
    lo, hi = np.quantile(y, [0.01, 0.99])
    assert out.min() == lo and out.max() == hi


def test_prewarp_recovers_log_factor_under_heavy_tailed_target():
    """``y = 0.2 a**2/b + f/5 + log(2c) sin(d/3)``: on the raw target the ALS warp of ``c`` chases the a**2/b tail."""
    c, d, y = _case2()
    raw_a, _ = fit_pair_prewarp_als(c, d, y)
    win_a, _ = fit_pair_prewarp_als(c, d, winsorize_heavy_tailed_target(y))
    truth = np.log(2 * c)
    raw_corr = abs(np.corrcoef(apply_operand_prewarp(c, raw_a), truth)[0, 1])
    win_corr = abs(np.corrcoef(apply_operand_prewarp(c, win_a), truth)[0, 1])
    assert win_corr > raw_corr + 0.03 and win_corr > 0.95, (raw_corr, win_corr)


def test_shared_var_warp_binds_to_its_synergistic_pair_not_the_first_pairing():
    """``c`` is shared by (a, c) and (c, d); (a, c) ranks first but only (c, d) interacts. The c-warp fit against ``a`` gives
    ``prewarp(c)*prewarp(d)`` binned MI 0.23 about y (below raw c's 0.245); c must bind to the (c, d) joint warp instead."""
    from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_setup import _fit_prewarp_and_gate_med

    rng = np.random.default_rng(0)
    n = 30000
    a, b, c, d, f = (rng.random(n) for _ in range(5))
    y = 0.2 * a**2 / b + f / 5.0 + np.log(c * 2) * np.sin(d / 3)
    cols = {0: a, 2: c, 3: d}
    _, specs, _, _ = _fit_prewarp_and_gate_med(
        prospective_pairs={((0, 2), 0.56): 1.0, ((2, 3), 0.35): 1.0},
        prewarp_enable=True, prewarp_y=y, prewarp_y_continuous=y, prewarp_basis="chebyshev", prewarp_max_degree=4,
        prewarp_min_val_corr=0.08, fe_gate_med_enable=False, original_cols=list(cols), _use_subsample=False,
        _full_n_rows=n, _sample_idx=None, _extval_raw_col=cols.get,
    )
    sc, sd = fit_pair_prewarp_als(c, d, winsorize_heavy_tailed_target(y))
    np.testing.assert_allclose(apply_operand_prewarp(c, specs[2]), apply_operand_prewarp(c, sc))
    np.testing.assert_allclose(apply_operand_prewarp(d, specs[3]), apply_operand_prewarp(d, sd))
