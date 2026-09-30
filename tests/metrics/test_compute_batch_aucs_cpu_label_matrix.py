"""``compute_batch_aucs`` on the CPU backend must accept the (N, M) one-vs-rest label matrix its GPU twin accepts.

The per-class report builds an (N, K) label matrix for multiclass / multilabel targets and an (N, 1) one for binary. The CPU loop handed the
whole matrix to ``fast_aucs`` for every column, which takes one 1-D label vector, so every call raised (a numba TypingError when compiled, a
ValueError interpreted). The report swallowed it at debug level and quietly used the slower per-class path, and with JIT disabled that path
returned NaN for every class.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from mlframe.metrics.core import compute_batch_aucs


def _one_vs_rest(n: int = 600, k: int = 4, seed: int = 0):
    """A separable-ish K-class problem as an (N, K) one-vs-rest label matrix and an (N, K) score matrix."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, size=n)
    scores = rng.random((n, k)) + 0.8 * np.eye(k)[y]
    return np.column_stack([(y == c).astype(np.int8) for c in range(k)]), scores


def test_a_label_matrix_is_scored_column_by_column_on_the_cpu_backend():
    """Each column's AUCs equal scikit-learn's on that column's own labels."""
    labels, scores = _one_vs_rest()
    roc, pr = compute_batch_aucs(labels, scores, force_backend="cpu")
    for j in range(labels.shape[1]):
        assert roc[j] == pytest.approx(roc_auc_score(labels[:, j], scores[:, j]), abs=1e-12)
        assert pr[j] == pytest.approx(average_precision_score(labels[:, j], scores[:, j]), abs=1e-9)


def test_a_single_column_label_matrix_matches_the_one_dimensional_labels():
    """The binary report passes (N, 1) labels; they must score the same as the flat vector."""
    labels, scores = _one_vs_rest(k=2)
    col = labels[:, [1]]
    roc_2d, pr_2d = compute_batch_aucs(col, scores[:, [1]], force_backend="cpu")
    roc_1d, pr_1d = compute_batch_aucs(col[:, 0], scores[:, [1]], force_backend="cpu")
    np.testing.assert_array_equal(roc_2d, roc_1d)
    np.testing.assert_array_equal(pr_2d, pr_1d)
