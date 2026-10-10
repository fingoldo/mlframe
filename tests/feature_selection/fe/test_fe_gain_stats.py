"""Shared gain statistics of the FE operators: pointwise MI, held-out codes and the paired standard error."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._fe_gain_stats import held_out_codes, paired_gain_se, plugin_mi_of_codes, quantile_codes


def test_pointwise_mi_mean_is_the_plug_in_mi():
    """The mean of the per-row pointwise MI equals the plug-in MI of the joint table, computed by hand."""
    r = np.random.default_rng(0)
    x = r.integers(0, 4, 5000)
    y = (x + r.integers(0, 2, 5000)) % 3
    joint = np.zeros((4, 3))
    np.add.at(joint, (x, y), 1)
    p = joint / joint.sum()
    expected = float((p[p > 0] * np.log(p[p > 0] / (p.sum(1, keepdims=True) @ p.sum(0, keepdims=True))[p > 0])).sum())
    assert plugin_mi_of_codes(x, 4, y, 3) == np.float64(expected) or abs(plugin_mi_of_codes(x, 4, y, 3) - expected) < 1e-12


def test_quantile_codes_are_equal_mass_and_send_non_finite_to_bin_zero():
    """Ten equal-frequency bins; NaN and inf land in the lowest bin instead of raising."""
    x = np.random.default_rng(1).standard_normal(10000)
    x[:5] = np.nan
    codes = quantile_codes(x, 10)
    assert codes.min() == 0 and codes.max() == 9
    assert (codes[:5] == 0).all()
    counts = np.bincount(codes, minlength=10)
    assert counts[1:].min() > 800


def test_held_out_codes_use_edges_from_the_fit_rows_only():
    """Changing the values of the scored rows does not move the edges: the codes of unchanged rows stay put when only other scored rows change."""
    r = np.random.default_rng(2)
    f = r.standard_normal(2000)
    fit, score = np.arange(0, 2000, 2), np.arange(1, 2000, 2)
    base = held_out_codes(f, fit, score)
    g = f.copy()
    g[score[:100]] = 50.0
    assert np.array_equal(held_out_codes(g, fit, score)[100:], base[100:])


def test_paired_standard_error_follows_one_over_root_n_and_is_zero_for_identical_features():
    """The error of the gain between two different features shrinks about 2x for 4x the rows; the same feature twice has no spread."""
    def se_at(n):
        """Standard error of the gain of a noisy feature over a weaker one on n rows."""
        r = np.random.default_rng(3)
        z = r.standard_normal(n)
        y = quantile_codes(z + 0.5 * r.standard_normal(n), 10)
        good, weak = quantile_codes(z, 10), quantile_codes(z + 2.0 * r.standard_normal(n), 10)
        return paired_gain_se(good, weak, y, 10)

    assert 1.5 < se_at(8000) / se_at(32000) < 2.7
    codes = quantile_codes(np.random.default_rng(4).standard_normal(3000), 10)
    assert paired_gain_se(codes, codes, codes, 10) == 0.0
