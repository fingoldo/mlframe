"""The default BorutaShap surrogate forest runs single-threaded on small frames and parallel on large ones.

Parallel tree fitting pays joblib dispatch on every trial. On a 300-row frame that made each trial 1.8x slower, so a
fuzz combo spent minutes in feature selection. The gate keeps the parallel default where it measures faster.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.boruta_shap import BorutaShap


def _frame(n_rows: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Random binary-classification frame with a handful of informative columns."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n_rows, 6)), columns=[f"f{i}" for i in range(6)])
    y = (X["f0"] + rng.normal(scale=0.5, size=n_rows) > 0).astype(int).to_numpy()
    return X, y


def test_small_frame_uses_single_threaded_surrogate():
    """A 300-row frame gets an ``n_jobs=1`` surrogate forest."""
    X, y = _frame(300)
    sel = BorutaShap(classification=True, n_trials=2, random_state=0, verbose=False)

    sel.fit(X, y)

    assert sel.model_.n_jobs == 1


def test_large_frame_keeps_parallel_surrogate():
    """A frame above the crossover keeps the parallel default ``n_jobs=-1``."""
    X, y = _frame(1500)
    sel = BorutaShap(classification=True, n_trials=2, random_state=0, verbose=False)

    sel.fit(X, y)

    assert sel.model_.n_jobs == -1
