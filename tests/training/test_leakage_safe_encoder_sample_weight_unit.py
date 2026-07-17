"""Regression sentry: LeakageSafeEncoder.fit / fit_transform with sample_weight=None match legacy path.

When sample_weight is omitted, explicitly None, or uniform-constant, output must equal the unweighted output
byte-for-byte. Non-uniform weights must change the per-category means and the OOF encoding.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


def _toy_categorical(n=200, k=4, seed=0):
    """Toy categorical."""
    rng = np.random.default_rng(seed)
    cats = rng.choice([f"c{i}" for i in range(k)], size=n)
    # y depends on category index.
    cat_to_idx = {f"c{i}": i for i in range(k)}
    y = np.array([cat_to_idx[c] / float(k - 1) + 0.05 * rng.normal() for c in cats], dtype=np.float64)
    return cats, y


def test_fit_sample_weight_none_matches_omitted():
    """Fit sample weight none matches omitted."""
    from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder

    cats, y = _toy_categorical()
    a = LeakageSafeEncoder(method="target_mean", cv=3, random_state=7).fit(cats, y)
    b = LeakageSafeEncoder(method="target_mean", cv=3, random_state=7).fit(cats, y, sample_weight=None)
    assert a._global_prior == b._global_prior
    assert a._category_means == b._category_means
    assert a._category_counts == b._category_counts


def test_fit_uniform_sample_weight_matches_unweighted():
    """Fit uniform sample weight matches unweighted."""
    from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder

    cats, y = _toy_categorical()
    sw = np.full(len(y), 3.7)
    a = LeakageSafeEncoder(method="target_mean", cv=3, random_state=7).fit(cats, y)
    b = LeakageSafeEncoder(method="target_mean", cv=3, random_state=7).fit(cats, y, sample_weight=sw)
    assert a._category_means == b._category_means


def test_fit_nonuniform_sample_weight_changes_per_category_means():
    """Fit nonuniform sample weight changes per category means."""
    from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder

    cats, y = _toy_categorical()
    # Up-weight rows where y is high; down-weight low-y rows. Per-cat means should shift upward.
    sw = 1.0 + 4.0 * (y > np.median(y)).astype(np.float64)
    unweighted = LeakageSafeEncoder(method="target_mean", cv=3, random_state=7).fit(cats, y)
    weighted = LeakageSafeEncoder(method="target_mean", cv=3, random_state=7).fit(cats, y, sample_weight=sw)
    # At least one category mean differs.
    diffs = [abs(unweighted._category_means[c] - weighted._category_means[c]) for c in unweighted._category_means]
    assert max(diffs) > 1e-6, f"weighted vs unweighted means agree everywhere: {diffs}"


def test_fit_validates_sample_weight():
    """Fit validates sample weight."""
    from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder

    cats, y = _toy_categorical()
    enc = LeakageSafeEncoder(method="target_mean", cv=3, random_state=7)
    with pytest.raises(ValueError, match="length"):
        enc.fit(cats, y, sample_weight=np.ones(len(y) - 1))
    with pytest.raises(ValueError, match="finite and non-negative"):
        sw = np.ones(len(y))
        sw[0] = -1.0
        enc.fit(cats, y, sample_weight=sw)
    with pytest.raises(ValueError, match="sums to zero"):
        enc.fit(cats, y, sample_weight=np.zeros(len(y)))


def test_woe_weighted_pos_neg_mass():
    """WoE positive / negative mass per category must equal weighted sums when sample_weight is non-uniform."""
    from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder

    rng = np.random.default_rng(1)
    n = 400
    cats = rng.choice(["a", "b"], size=n)
    y = (rng.uniform(size=n) > 0.5).astype(np.float64)
    sw = rng.uniform(0.5, 3.0, size=n)
    enc = LeakageSafeEncoder(method="woe", cv=3, random_state=11).fit(cats, y, sample_weight=sw)
    # Independently compute expected weighted positive / negative mass for category "a".
    a_mask = cats == "a"
    n_pos_a = float(sw[a_mask & (y == 1.0)].sum())
    n_neg_a = float(sw[a_mask & (y == 0.0)].sum())
    total_pos = float(sw[y == 1.0].sum())
    total_neg = float(sw[y == 0.0].sum())
    a = enc.woe_smoothing  # WoE Laplace alpha is its own knob (default 0.5), not the mean smoothing
    expected_p = (n_pos_a + a) / (total_pos + a)
    expected_q = (n_neg_a + a) / (total_neg + a)
    np.testing.assert_allclose(enc._woe_pos["a"], expected_p, rtol=1e-10)
    np.testing.assert_allclose(enc._woe_neg["a"], expected_q, rtol=1e-10)
