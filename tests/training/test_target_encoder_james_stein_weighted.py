"""James-Stein variance parameters must treat sample_weight as frequency weights, not mix a weighted mass with the raw row count."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder


def _data(n: int = 400):
    """Three categories with distinct means and unit noise."""
    rng = np.random.default_rng(4)
    cats = rng.choice(np.array(["a", "b", "c"]), size=n)
    shift = {"a": 0.0, "b": 1.0, "c": 2.5}
    y = np.array([shift[c] for c in cats]) + rng.normal(size=n)
    return cats, y


def test_weighted_fit_matches_replicated_rows() -> None:
    """Integer weights give the same (sigma2, tau2) as repeating each row that many times (before the fix the grand mean mixed mass and row count)."""
    cats, y = _data()
    w = np.random.default_rng(2).integers(1, 5, size=len(y))
    weighted = LeakageSafeEncoder(method="target_james_stein", smoothing=5.0, cv=5, random_state=0)
    weighted.fit(cats, y, sample_weight=w.astype(np.float64))
    replicated = LeakageSafeEncoder(method="target_james_stein", smoothing=5.0, cv=5, random_state=0)
    replicated.fit(np.repeat(cats, w), np.repeat(y, w))
    assert weighted._js_sigma2 == pytest.approx(replicated._js_sigma2, rel=1e-9)
    assert weighted._js_tau2 == pytest.approx(replicated._js_tau2, rel=1e-9)


def test_unit_weights_equal_unweighted() -> None:
    """All-ones weights reproduce the unweighted variance parameters."""
    cats, y = _data()
    plain = LeakageSafeEncoder(method="target_james_stein", smoothing=5.0, cv=5, random_state=0).fit(cats, y)
    ones = LeakageSafeEncoder(method="target_james_stein", smoothing=5.0, cv=5, random_state=0).fit(cats, y, sample_weight=np.ones(len(y)))
    assert plain._js_sigma2 == pytest.approx(ones._js_sigma2, rel=1e-12)
    assert plain._js_tau2 == pytest.approx(ones._js_tau2, rel=1e-12)


def test_weights_concentrated_on_one_category_change_sigma2() -> None:
    """Pooled within-category variance is weighted: up-weighting a noisy category raises sigma2."""
    rng = np.random.default_rng(8)
    cats = np.repeat(np.array(["lo", "hi"]), 200)
    y = np.concatenate([rng.normal(0.0, 0.1, 200), rng.normal(5.0, 3.0, 200)])
    plain = LeakageSafeEncoder(method="target_james_stein", smoothing=5.0, cv=5, random_state=0).fit(cats, y)
    w = np.concatenate([np.ones(200), np.full(200, 20.0)])
    heavy = LeakageSafeEncoder(method="target_james_stein", smoothing=5.0, cv=5, random_state=0).fit(cats, y, sample_weight=w)
    assert heavy._js_sigma2 > plain._js_sigma2 * 1.5
