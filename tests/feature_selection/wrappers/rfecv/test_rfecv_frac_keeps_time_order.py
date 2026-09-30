"""RFECV's per-fold ``frac`` subsampling must keep the fold's own row order, or a time-ordered CV loses its chronology."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.cv_policy import TimestampOrderedSplit
from mlframe.feature_selection.wrappers.rfecv import RFECV


class _RecordingLogit(LogisticRegression):
    seen: list = []

    def fit(self, X, y, **kw):
        type(self).seen.append(np.asarray(X)[:, 0].copy())
        return super().fit(X, y, **kw)


def test_rfecv_frac_subsample_keeps_chronological_train_order():
    n = 300
    rng = np.random.default_rng(0)
    ts = rng.permutation(n)
    # every column carries the row's timestamp (plus tiny noise) so whichever column a fit sees first reveals the row order it was given
    X = pd.DataFrame(ts[:, None] + 0.001 * rng.normal(size=(n, 4)), columns=list("abcd"))
    y = (rng.normal(size=n) > 0).astype(int)
    _RecordingLogit.seen = []
    rfecv = RFECV(estimator=_RecordingLogit(), cv=TimestampOrderedSplit(n_splits=3, timestamps=ts), cv_shuffle=False, frac=0.5, max_runtime_mins=1.0, verbose=0)
    rfecv.fit(X, y)
    train_fits = [s for s in _RecordingLogit.seen if 20 < len(s) < 0.6 * n]
    assert train_fits, "no fold fit was recorded"
    for rows in train_fits:
        assert np.all(np.diff(rows) > 0), "subsampled fold train rows are no longer in chronological order"
