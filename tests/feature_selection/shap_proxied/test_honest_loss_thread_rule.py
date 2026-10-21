"""Pin the size-keyed xgboost thread rule in ``_parallel_honest_losses``.

Measured by interleaved A/B on a box at about 90% load: a 1500x6 fit is about 16x faster with ``n_jobs=1`` than
``n_jobs=-1``, while a 10000x400 fit is about 1.5x faster with ``n_jobs=-1``. The rule keeps one thread per fit below
``_HONEST_INNER_THREADS_MIN_CELLS`` cells and lets xgboost use all threads at or above it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _shap_proxy_loss as loss_mod


def _capture_inner(monkeypatch, n_rows, n_cols):
    """Run ``_parallel_honest_losses`` with ``_honest_loss`` stubbed and return the ``inner_n_jobs`` each task received."""
    seen: list[int] = []

    def fake_honest_loss(model_template, X_tr, y_tr, X_ev, y_ev, idx, classification, metric, seed=None, inner_n_jobs=None, **kw):
        """Record the requested inner thread count and return a zero loss."""
        seen.append(inner_n_jobs)
        return 0.0

    monkeypatch.setattr(loss_mod, "_honest_loss", fake_honest_loss)
    X = np.zeros((n_rows, n_cols))
    y = np.zeros(n_rows)
    loss_mod._parallel_honest_losses([((0,), 0)], None, X, y, X, y, True, "brier", 1)
    return seen


def test_small_fit_runs_single_threaded(monkeypatch):
    """1500x6 (9000 cells) is far below the cutoff, so the fit must use one thread."""
    assert _capture_inner(monkeypatch, 1500, 6) == [1]


def test_large_fit_keeps_xgboost_all_threads(monkeypatch):
    """10000x400 (4e6 cells) is above the cutoff, so xgboost keeps its own thread pool (n_jobs=-1)."""
    assert _capture_inner(monkeypatch, 10000, 400) == [-1]


@pytest.mark.parametrize("cells_rows, cols", [(1000, 1000), (2000, 500)])
def test_cutoff_is_inclusive_at_threshold(monkeypatch, cells_rows, cols):
    """Exactly ``_HONEST_INNER_THREADS_MIN_CELLS`` cells selects all threads (``>=``)."""
    assert cells_rows * cols == loss_mod._HONEST_INNER_THREADS_MIN_CELLS
    assert _capture_inner(monkeypatch, cells_rows, cols) == [-1]
