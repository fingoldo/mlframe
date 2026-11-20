"""Stability sub-fits must run under the suite's temporal / grouped split policy, restricted to their own rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

import mlframe.feature_selection.boruta_shap as bsmod
from mlframe.feature_selection.cv_policy import POLICY_ATTR, CVPolicy


def test_stability_subfits_receive_the_row_restricted_cv_policy(monkeypatch):
    """Every sub-fit that builds its held-out split sees a temporal policy aligned to its subsample, not None."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = (X["a"] > 0).astype(int)
    policy = CVPolicy("temporal", "test", timestamps=np.arange(n))
    seen = []
    real = bsmod.holdout_indices

    def _spy(pol, n_rows, test_size, **kw):
        """Record the policy and row count of each holdout request."""
        seen.append((pol, n_rows))
        return real(pol, n_rows, test_size, **kw)

    monkeypatch.setattr(bsmod, "holdout_indices", _spy)
    bs = bsmod.BorutaShap(n_trials=2, verbose=False, train_or_test="test", stability_subsamples=2)
    setattr(bs, POLICY_ATTR, policy)
    bs.fit(X, y)
    assert seen
    assert all(p is not None and p.kind == "temporal" and len(p.timestamps) == n_rows for p, n_rows in seen)
