"""The honest selection score scores each fold on a contiguous block of time when the rows are time-ordered.

A shuffled row split lets a test row sit beside its own training neighbours in time, so the honest score of an autocorrelated target is
optimistic. With ``time_ordering`` the test folds are contiguous periods in time, and no fold mixes a period's rows with another's.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.training.composite import hpo_ensembling


def test_honest_fold_test_rows_are_contiguous_blocks_in_time(monkeypatch):
    """With ``time_ordering`` each held-out fold is a run of consecutive time stamps, not a random scatter."""
    n, folds = 60, 3
    rng = np.random.default_rng(0)
    time_stamp = rng.permutation(n)
    row_id = np.arange(n, dtype=np.float64)
    stacked = np.vstack([row_id, row_id * 0.5])

    monkeypatch.setattr(hpo_ensembling, "stepwise_ensemble_selection", lambda *a, **k: SimpleNamespace(kept=[0]))
    seen_test_rows = []

    def record_metric(y_true, y_pred):
        """Record the rows of each held-out fold."""
        seen_test_rows.append(np.asarray(y_true).astype(int))
        return 0.0

    hpo_ensembling._nested_selection_score(
        stacked, row_id, record_metric, folds, 0, {}, time_ordering=time_stamp,
    )

    assert len(seen_test_rows) == folds
    for rows in seen_test_rows:
        times = np.sort(time_stamp[rows])
        assert np.array_equal(times, np.arange(times[0], times[0] + times.size))
