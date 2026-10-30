"""``_triple_mi_kernel`` returns, in lexicographic order, the plug-in MI of the joint code of every triple of columns with y."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_triplet_fe import _triple_mi_kernel


def _mi(a: np.ndarray, b: np.ndarray, c: np.ndarray, y: np.ndarray, base: int) -> float:
    """Plug-in MI of the triple's joint code with y, from a dense contingency table."""
    cell = (a * base + b) * base + c
    n_y = int(y.max()) + 1
    table = np.zeros((base**3, n_y))
    np.add.at(table, (cell, y), 1)
    n = y.size
    px = table.sum(1, keepdims=True)
    py = table.sum(0, keepdims=True)
    nz = table > 0
    return float(((table / n)[nz] * np.log((table * n)[nz] / (px @ py)[nz])).sum())


@pytest.mark.parametrize("n_y", [2, 3])
def test_triple_mi_matches_the_dense_reference(n_y):
    """Every triple, in lexicographic order, with the reference MI."""
    rng = np.random.default_rng(0)
    base, p, n = 5, 6, 700
    codes = rng.integers(0, base, (p, n)).astype(np.int64)
    y = rng.integers(0, n_y, n).astype(np.int64)
    y = np.where(codes[1] == 2, 0, y)
    oi, oj, ok, mi = _triple_mi_kernel(codes, y, base, n_y)
    triples = list(itertools.combinations(range(p), 3))
    assert list(zip(oi, oj, ok)) == triples
    for t, (i, j, k) in enumerate(triples):
        assert mi[t] == pytest.approx(_mi(codes[i], codes[j], codes[k], y, base), abs=1e-12)
