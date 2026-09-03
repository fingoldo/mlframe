"""FS_BORUTA_ROOT-4: the "Undecided features" progress log was dedented out of the trial loop.

It ran exactly once, after the LAST trial, instead of every 5 trials as its own message implies. Pins
that the log now fires more than once across a multi-trial fit that does not resolve/early-stop quickly.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest


def test_progress_log_fires_more_than_once_across_trials(caplog):
    """The "Undecided features" progress log must fire on more than one trial, not only once at the end."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier

    from mlframe.feature_selection.boruta_shap import BorutaShap

    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({f"noise_{i}": rng.standard_normal(n) for i in range(15)})
    y = pd.Series(rng.integers(0, 2, n), name="target")

    b = BorutaShap(
        model=RandomForestClassifier(n_estimators=20, n_jobs=2, random_state=0),
        importance_measure="gini",
        classification=True,
        n_trials=12,
        verbose=False,
        random_state=0,
    )
    with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.boruta_shap._fit_explain"):
        b.fit(X, y)

    hits = [r for r in caplog.records if "Undecided features" in r.getMessage()]
    assert len(hits) > 1, f"expected the periodic progress log to fire more than once, got {len(hits)}: {[r.getMessage() for r in hits]}"
