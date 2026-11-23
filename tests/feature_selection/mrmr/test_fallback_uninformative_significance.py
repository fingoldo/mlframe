"""The never-empty fallback reports ``uninformative`` by a permutation test, not by the sign of the MI (AUD23-71).

The selection itself does not change: the fallback still keeps one column (the never-empty guarantee). What changes is the metadata and the warning: a kept column whose permutation p-value does not pass the significance level is ``uninformative``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl._fallback_maxt import FAMILY_ALPHA
from mlframe.feature_selection.filters._mrmr_fit_impl._finalise import _commit_fallback_support, _top_column_pvalue
from mlframe.feature_selection.filters.mrmr import MRMR


def _noise(n: int = 3000, p: int = 12, seed: int = 0):
    """Pure noise: 12 standard-normal columns and a random binary target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"n{i}" for i in range(p)])
    y = (rng.standard_normal(n) > 0).astype(int)
    return X, y


def test_pure_noise_keeps_one_column_but_reports_it_uninformative():
    """The never-empty fallback still keeps a column on noise (the selection is unchanged), and its metadata says it carries no signal."""
    X, y = _noise()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(verbose=0, random_seed=0, fe_max_steps=0).fit(X, y)
    assert list(m.get_feature_names_out()) == ["n0"]
    meta = m.fallback_metadata_
    assert meta["fallback_used"] is True
    assert meta["uninformative"] is True
    assert meta["top_p_value"] is not None and meta["top_p_value"] > FAMILY_ALPHA


def _codes(x: np.ndarray, bins: int) -> np.ndarray:
    """Equal-frequency integer codes of a column."""
    edges = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    return np.searchsorted(edges, x, side="right").astype(np.int8)


def test_the_p_value_of_a_real_signal_passes_the_level():
    """A column that drives the target has a permutation p-value at the level or below; the same test on noise does not pass (both through the function the fallback calls)."""
    rng = np.random.default_rng(1)
    n = 3000
    s = rng.standard_normal(n)
    y = (s + 0.5 * rng.standard_normal(n) > 0).astype(np.int8)
    noise = rng.standard_normal(n)
    data = np.column_stack([_codes(s, 5), _codes(noise, 5), y.astype(np.int8)])
    nbins = np.array([5, 5, 2])
    p_signal = _top_column_pvalue(data, nbins, 2, 0)
    p_noise = _top_column_pvalue(data, nbins, 2, 1)
    assert p_signal <= FAMILY_ALPHA, f"the signal column's p-value {p_signal} does not pass {FAMILY_ALPHA}"
    assert p_noise > FAMILY_ALPHA, f"a noise column's p-value {p_noise} passes {FAMILY_ALPHA}"


def test_uninformative_needs_a_significant_p_value_not_only_positive_mi():
    """A positive MI with a p-value above the level is uninformative; a significant one with positive MI is not."""
    class _Est:
        """The few attributes ``_commit_fallback_support`` reads and writes on an estimator."""

        feature_names_in_ = np.array(["a"], dtype=object)
        n_features_in_ = 1
        n_features_ = 0
        support_ = None
        fallback_used_ = False
        fallback_metadata_ = None

    est = _Est()
    raw_mi = [(0, 0.003, 0)]
    _commit_fallback_support(est, [0], raw_mi, 0, 1, p_value=0.5)
    assert est.fallback_metadata_["uninformative"] is True
    est2 = _Est()
    _commit_fallback_support(est2, [0], raw_mi, 0, 1, p_value=0.001)
    assert est2.fallback_metadata_["uninformative"] is False
    est3 = _Est()
    _commit_fallback_support(est3, [0], raw_mi, 0, 1, p_value=float("nan"))
    assert est3.fallback_metadata_["uninformative"] is True, "a p-value that cannot be computed must never read as significant"
