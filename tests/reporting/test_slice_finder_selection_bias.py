"""The worst slice is a MAXIMUM over thousands of candidates, so a marginal interval is the wrong yardstick.

Measured before the fix: on data where the predictions are independent of the target, the top slice's marginal 95%
interval came back [1.015, 1.104] -- excluding 1.0, i.e. announcing a real weakness where none exists. The interval
has to hold simultaneously over the candidates that were screened, or the chart's headline is an artefact of the
search rather than a property of the model.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.slice_finder import _norm_ppf_one_sided, find_weak_slices


@pytest.fixture
def features():
    """A feature matrix shared by both directions of the test."""
    return np.random.default_rng(0).normal(size=(20_000, 8))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_null_data_top_slice_interval_covers_one(features, seed):
    """With predictions independent of the target, no slice is genuinely worse than the model overall."""
    rng = np.random.default_rng(seed)
    res = find_weak_slices(features, rng.normal(size=len(features)), rng.normal(size=len(features)), task="regression", max_arity=2)
    top = res.table.iloc[0]
    assert top["error_ratio_lo"] <= 1.0 <= top["error_ratio_hi"], (
        f"null data still declares a weak slice: ratio {top['error_ratio']:.3f} " f"CI [{top['error_ratio_lo']:.3f}, {top['error_ratio_hi']:.3f}]"
    )


def test_a_real_weak_region_is_still_found_and_significant(features):
    """The correction must not cost the detection it exists to qualify."""
    rng = np.random.default_rng(0)
    n = len(features)
    y = rng.normal(size=n)
    y_pred = y + rng.normal(0.0, 0.3, n)
    weak = features[:, 3] > 1.2
    y_pred[weak] = y[weak] + rng.normal(0.0, 3.0, int(weak.sum()))

    res = find_weak_slices(features, y, y_pred, task="regression", max_arity=2)
    top = res.table.iloc[0]
    assert "f3" in top["bounds"], f"the injected weak region was not surfaced: {top['bounds']}"
    assert top["error_ratio_lo"] > 1.0, "a 2.5x weak region must remain significant after the correction"


def test_the_title_states_the_candidate_count_and_the_correction():
    """A reader cannot discount a max-over-search headline without being told a search happened."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4_000, 5))
    res = find_weak_slices(X, rng.normal(size=4_000), rng.normal(size=4_000), task="regression", max_arity=2)
    title = next(p for row in res.figure.panels for p in row if p).title
    assert "candidate slices screened" in title and "SIMULTANEOUS" in title


class TestNormalQuantile:
    """The Sidak level needs a normal quantile, and this module carries its own (numpy-only dependency set)."""

    @pytest.mark.parametrize("p", [0.5, 0.75, 0.975, 0.999, 0.9999])
    def test_matches_scipy(self, p):
        """Acklam's approximation is documented as |error| < 1.15e-9; assert against scipy where available."""
        scipy_stats = pytest.importorskip("scipy.stats")
        assert _norm_ppf_one_sided(p) == pytest.approx(scipy_stats.norm.ppf(p), abs=1e-8)
