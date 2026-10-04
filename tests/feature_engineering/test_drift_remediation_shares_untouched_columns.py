"""remediate_drifting_features must materialise only the columns it rewrites, not copy both frames."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_engineering.drift_remediation import remediate_drifting_features


def _frames():
    """Train/test frames where ``drift`` separates them and ``stable`` does not."""
    rng = np.random.default_rng(0)
    n = 600
    g = np.repeat(np.arange(6), n // 6)
    train = pd.DataFrame({"g": g, "drift": rng.normal(0, 1, n), "stable": rng.normal(0, 1, n), "stable2": rng.normal(0, 1, n)})
    test = pd.DataFrame({"g": g, "drift": rng.normal(5, 1, n), "stable": rng.normal(0, 1, n), "stable2": rng.normal(0, 1, n)})
    return train, test


def test_untouched_columns_share_buffers_and_inputs_are_unchanged():
    """Unflagged columns share memory with the inputs, rewritten ones do not, and the caller's frames are bit-identical afterwards."""
    train, test = _frames()
    snap_tr, snap_te = train.copy(deep=True), test.copy(deep=True)
    tr_out, te_out, report = remediate_drifting_features(train, test, group_col="g", n_std=0.5)
    flagged = set(report.loc[report["flagged"], "feature"])
    assert flagged, "the drifting column must be flagged for this test to exercise the rewrite path"
    untouched = [c for c in ("stable", "stable2") if c not in flagged]
    assert untouched
    for c in untouched:
        assert np.shares_memory(tr_out[c].to_numpy(), train[c].to_numpy())
        assert np.shares_memory(te_out[c].to_numpy(), test[c].to_numpy())
    for c in flagged:
        assert not np.shares_memory(tr_out[c].to_numpy(), train[c].to_numpy())
    pd.testing.assert_frame_equal(train, snap_tr)
    pd.testing.assert_frame_equal(test, snap_te)
