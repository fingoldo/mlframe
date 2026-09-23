"""The tree rescue fits on the columns the user allowed, and says what it rescued each feature on.

``factors_to_use`` was applied to the ranking but not to the fit, so excluded columns still consumed the GBM's split budget and shifted the
importances of the allowed ones: the exclusion was a display filter for this path rather than a restriction. The rescue also logged only a
count and a list of names, so nothing recorded whether a rescued feature was worth anything, and the importance it ranks on is in-screen (the
GBM sees the rows MRMR already used), which the log did not say either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

lgb = pytest.importorskip("lightgbm")


@pytest.fixture
def frame():
    """A frame with two informative columns and several that are not."""
    rng = np.random.default_rng(0)
    n = 1200
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    # The rescue only fires on a pool wider than ``tree_rescue_min_p``, which is the regime it exists for.
    X = pd.DataFrame({"a": a, "b": b, **{f"n{i}": rng.normal(size=n) for i in range(10)}})
    y = ((a + 0.8 * b) > 0).astype(np.int64)
    return X, y


def _rescued_estimator(frame, allowed, monkeypatch):
    """Fit the tree-rescued estimator, recording the design width every LGBM fit saw."""
    from mlframe.feature_selection.filters import MRMRTreeRescued

    widths: list = []
    real_fit = lgb.LGBMClassifier.fit

    def recording_fit(self, X, y, *a, **kw):
        """Record the number of columns the rescue GBM was actually given."""
        widths.append(np.asarray(X).shape[1])
        return real_fit(self, X, y, *a, **kw)

    monkeypatch.setattr(lgb.LGBMClassifier, "fit", recording_fit)

    X, y = frame
    MRMRTreeRescued._FIT_CACHE.clear()
    est = MRMRTreeRescued(
        random_state=0,
        verbose=0,
        fe_max_steps=0,
        full_npermutations=3,
        baseline_npermutations=2,
        tree_rescue=True,
        tree_rescue_top_k=3,
        tree_rescue_min_p=4,
        factors_to_use=allowed,
    ).fit(X, y)
    return est, widths


def test_the_rescue_fit_sees_only_the_allowed_columns(frame, monkeypatch):
    """With three columns allowed, the rescue GBM's design must be three wide, not the full frame."""
    allowed = [0, 1, 2]
    _est, widths = _rescued_estimator(frame, allowed, monkeypatch)
    # Asserted: this fixture and configuration exist to make the rescue fire, so an empty record means it silently stopped running and
    # the width contract below was never examined.
    assert widths, "the rescue GBM never fit, so no design width was observed"
    assert all(w == len(allowed) for w in widths), f"the rescue fit on {widths} columns, expected {len(allowed)}"


def test_rescued_features_come_with_a_recorded_importance(frame, monkeypatch):
    """Each rescued feature's importance share is recorded, so the fit itself carries the evidence."""
    est, widths = _rescued_estimator(frame, [0, 1, 2, 3], monkeypatch)
    assert widths, "the rescue GBM never fit, so nothing could have been rescued"
    shares = getattr(est, "tree_rescue_importances_", None)
    # Both asserted rather than skipped: an empty mapping would make the ``all(...)`` below true for vacuous reasons, which is exactly how
    # a rescue that records nothing keeps a green test.
    assert shares, f"the rescue fired but recorded no importance share: {shares!r}"
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in shares.values()), shares


def test_no_rescued_feature_falls_outside_the_allowed_set(frame, monkeypatch):
    """Whatever the rescue adds must come from the columns the user allowed."""
    allowed = [0, 1, 2, 3]
    est, widths = _rescued_estimator(frame, allowed, monkeypatch)
    assert widths, "the rescue GBM never fit, so the containment claim below is vacuous"
    assert set(np.asarray(est.support_).tolist()) <= set(allowed), f"support {sorted(np.asarray(est.support_).tolist())} escaped {allowed}"
