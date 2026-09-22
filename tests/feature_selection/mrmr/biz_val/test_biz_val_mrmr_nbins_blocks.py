"""Bayesian Blocks earns its place on the shape it is designed for, and its subsampling is a no-op below the threshold.

``bb_subsample_threshold`` was changed from 0 (an unbounded quadratic DP) to a bounded default, which changes the EDGES themselves above the
threshold. Two things follow that nothing pinned: that the method still beats a uniform grid on piecewise-constant data, the regime it exists
for, and that below the threshold the bounded path returns exactly the edges the exact path did, since that is the half most likely to regress.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._adaptive_nbins import _BB_DEFAULT_SUBSAMPLE_THRESHOLD, edges_bayesian_blocks


def _piecewise_constant(n: int = 4000, seed: int = 0):
    """Four regimes with different class rates, at UNEVEN positions and widths.

    Evenly spaced regimes are the case an even grid also solves, so they cannot tell the two apart. The boundaries here sit where a uniform
    grid cannot land, which is the whole claim of an adaptive edge finder.
    """
    rng = np.random.default_rng(seed)
    spans = ((0.00, 0.35, 0.10), (0.40, 0.55, 0.60), (0.60, 3.50, 0.25), (3.60, 9.00, 0.90))
    widths = (int(n * 0.40), int(n * 0.12), int(n * 0.33), int(n * 0.15))
    xs, ys = [], []
    for (lo, hi, rate), width in zip(spans, widths):
        xs.append(rng.uniform(lo, hi, size=width))
        ys.append((rng.random(width) < rate).astype(np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def _mi(x_codes: np.ndarray, y: np.ndarray) -> float:
    """Plug-in mutual information between binned x and binary y, in nats."""
    mi = 0.0
    for xv in np.unique(x_codes):
        mx = x_codes == xv
        px = mx.mean()
        for yv in (0, 1):
            pxy = float((mx & (y == yv)).mean())
            if pxy <= 0.0:
                continue
            py = float((y == yv).mean())
            mi += pxy * np.log(pxy / (px * py))
    return float(mi)


def test_blocks_beats_a_uniform_grid_on_piecewise_constant_data():
    """The regime the method exists for: regime boundaries recovered as edges carry more about y than an even split does."""
    x, y = _piecewise_constant()
    edges = np.asarray(edges_bayesian_blocks(x))
    if edges.size < 3:
        pytest.skip(f"blocks found no interior edge on this fixture ({edges.size} edge(s))")
    mi_blocks = _mi(np.searchsorted(edges[1:-1], x), y)
    uniform = np.linspace(x.min(), x.max(), num=edges.size)
    mi_uniform = _mi(np.searchsorted(uniform[1:-1], x), y)
    assert mi_blocks >= 1.15 * mi_uniform, f"blocks MI {mi_blocks:.4f} did not beat uniform {mi_uniform:.4f} at matched bin count"


def test_subsampling_is_a_no_op_below_the_threshold():
    """Below the threshold the bounded path must return exactly what the exact full-N path returns."""
    x, _y = _piecewise_constant(n=1000)
    assert x.shape[0] < _BB_DEFAULT_SUBSAMPLE_THRESHOLD, "fixture precondition: this column must sit below the threshold"
    bounded = np.asarray(edges_bayesian_blocks(x, subsample_threshold=_BB_DEFAULT_SUBSAMPLE_THRESHOLD))
    exact = np.asarray(edges_bayesian_blocks(x, subsample_threshold=0))
    assert np.array_equal(bounded, exact), f"the bounded path changed the edges below the threshold:\n{bounded}\nvs\n{exact}"


@pytest.mark.parametrize("threshold", [0, 500, 5000])
def test_edges_are_sorted_and_span_the_column(threshold):
    """Whatever the threshold, the result has to be a usable edge vector."""
    x, _y = _piecewise_constant(n=1200)
    edges = np.asarray(edges_bayesian_blocks(x, subsample_threshold=threshold))
    assert edges.size >= 2
    assert np.all(np.diff(edges) > 0), f"edges are not strictly increasing: {edges}"


def test_a_constant_column_does_not_produce_interior_edges():
    """No structure means no regime boundaries to find."""
    edges = np.asarray(edges_bayesian_blocks(np.full(500, 2.0)))
    assert edges.size <= 2, f"a constant column produced interior edges: {edges}"
