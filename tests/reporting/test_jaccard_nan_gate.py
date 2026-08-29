"""A fastmath kernel may not decide a threshold comparison on a NaN.

CHARTS_A-35: ``_jaccard_rows_numba`` is compiled with ``fastmath=True``, which lets the compiler assume no NaN is
present -- so ``y_proba[i, k] >= 0.5`` on a NaN is unspecified and free to disagree with the numpy reference the
module documents as equivalent. The dispatcher gates on finiteness rather than dropping fastmath.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts._jaccard_kernel import _jaccard_rows_numpy, jaccard_rows


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_probabilities_match_the_numpy_reference(bad):
    """The gate's whole job: a non-finite probability must not reach the fastmath kernel."""
    rng = np.random.default_rng(0)
    y_true = (rng.random((200, 6)) < 0.4).astype(np.int8)
    y_proba = rng.random((200, 6)).astype(np.float32)
    y_proba[rng.random(y_proba.shape) < 0.05] = bad
    assert np.array_equal(jaccard_rows(y_true, y_proba), _jaccard_rows_numpy(y_true, y_proba))


def test_finite_input_still_takes_the_fast_path_and_agrees():
    """The gate must not change the answer on the input the kernel was written for."""
    rng = np.random.default_rng(1)
    y_true = (rng.random((500, 8)) < 0.3).astype(np.int8)
    y_proba = rng.random((500, 8)).astype(np.float32)
    assert np.array_equal(jaccard_rows(y_true, y_proba), _jaccard_rows_numpy(y_true, y_proba))


def test_all_empty_rows_keep_the_vacuous_match_convention():
    """Both paths score an empty-vs-empty row as 1.0, matching sklearn's jaccard_score."""
    y_true = np.zeros((3, 4), dtype=np.int8)
    y_proba = np.zeros((3, 4), dtype=np.float32)
    assert np.array_equal(jaccard_rows(y_true, y_proba), np.ones(3))
