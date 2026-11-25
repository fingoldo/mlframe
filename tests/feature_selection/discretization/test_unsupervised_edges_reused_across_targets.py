"""Unsupervised ``binning_scheme`` edges depend on the column alone, so a second target (a second MRMR fit of a multi-target suite, a refit on another label) must not compute them again.

The edge memo is content-keyed and leaves the target out of the key for a method that does not use it; these tests pin that, for every unsupervised strategy and for the whole path
(``per_feature_edges``), and pin the contrast: a SUPERVISED strategy does depend on the target and recomputes.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _adaptive_nbins_columns as cols
from mlframe.feature_selection.filters._adaptive_nbins import per_feature_edges
from tests.feature_selection.binning_cases import edges_call

UNSUPERVISED = ["rice_capped", "uniform", "freedman_diaconis", "sturges"]


@pytest.fixture()
def frame():
    """A float frame with a few columns and two different binary targets."""
    rng = np.random.default_rng(11)
    n = 4000
    X = rng.standard_normal((n, 5))
    y1 = (X[:, 0] + rng.standard_normal(n) > 0).astype(np.int8)
    y2 = (X[:, 1] * X[:, 2] + rng.standard_normal(n) > 0).astype(np.int8)
    cols.clear_edge_memo()
    return X, y1, y2


def _count_uncached(monkeypatch):
    """Count the columns whose edges are really computed (a memo hit does not reach the uncached function)."""
    calls = []
    real = cols._compute_col_edges_uncached

    def counting(col, *args, **kwargs):
        """Record one real computation and delegate."""
        calls.append(1)
        return real(col, *args, **kwargs)

    monkeypatch.setattr(cols, "_compute_col_edges_uncached", counting)
    return calls


@pytest.mark.parametrize("method", UNSUPERVISED)
def test_a_second_target_reuses_the_edges_of_an_unsupervised_strategy(frame, monkeypatch, method):
    """The first call computes every column; the second, with another target, computes none and returns equal edges."""
    X, y1, y2 = frame
    calls = _count_uncached(monkeypatch)
    first = per_feature_edges(X, y1, **edges_call(method), n_jobs=1)
    assert len(calls) == X.shape[1]
    second = per_feature_edges(X, y2, **edges_call(method), n_jobs=1)
    assert len(calls) == X.shape[1], "the edges were computed again for a second target"
    for a, b in zip(first, second):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_a_supervised_strategy_recomputes_for_another_target(frame, monkeypatch):
    """Contrast: mdlp's edges depend on the target, so the second target is computed again (the memo must not serve it from the first)."""
    X, y1, y2 = frame
    calls = _count_uncached(monkeypatch)
    per_feature_edges(X, y1, method="mdlp", n_jobs=1)
    done = len(calls)
    per_feature_edges(X, y2, method="mdlp", n_jobs=1)
    assert len(calls) == 2 * done
