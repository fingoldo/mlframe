"""Regression test: the fused njit Knuth posterior search (``_knuth_best_M``) picks the SAME
optimal M -- and ``_knuth_bin_edges`` returns the SAME edges -- as the prior object-mode
``for M: np.histogram(...) -> _knuth_log_posterior`` scan.

Pins the perf optimisation in discretization/_benchmarks/bench_knuth_posterior_fused.py
(6-47x faster) to its bit-identity guarantee, so a future change to the kernel cannot silently
shift bin counts (and hence MI-plugin selection) downstream.
"""

from __future__ import annotations


import numpy as np
import pytest

from mlframe.feature_selection.filters.discretization._discretization_edges import (
    _knuth_bin_edges,
    _knuth_best_M,
    _knuth_log_posterior,
)


def _best_M_reference(a: np.ndarray, m_max_cap: int) -> int:
    """The exact prior object-mode search, kept here as the identity oracle."""
    a = np.asarray(a, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    n = a.size
    a_min, a_max = float(a.min()), float(a.max())
    M_max = int(min(max(4, int(np.sqrt(n) * 4)), int(m_max_cap)))
    best_M, best_logp = 2, -1e300
    for M in range(2, M_max + 1):
        edges = np.linspace(a_min, a_max, M + 1)
        counts, _ = np.histogram(a, bins=edges)
        logp = _knuth_log_posterior(M, n, counts.astype(np.int64))
        if logp > best_logp:
            best_logp = logp
            best_M = M
    return best_M


def _cols(n, rng):
    """Helper that cols."""
    return {
        "uniform": rng.uniform(0, 1, n),
        "normal": rng.normal(0, 1, n),
        "heavy_tail": rng.standard_t(2.0, n),
        "skewed": rng.exponential(1.0, n),
        "tie_heavy": rng.integers(0, 7, n).astype(np.float64),
        "lognormal": rng.lognormal(0, 1.0, n),
    }


@pytest.mark.parametrize("n", [500, 2000, 10000])
@pytest.mark.parametrize("cap", [64, 500])
def test_knuth_best_M_bit_identical_to_reference(n, cap):
    """Knuth best M bit identical to reference."""
    rng = np.random.default_rng(20260623 + n + cap)
    for name, col in _cols(n, rng).items():
        a = col[np.isfinite(col)].astype(np.float64)
        a_min, a_max = float(a.min()), float(a.max())
        M_max = int(min(max(4, int(np.sqrt(a.size) * 4)), int(cap)))
        fused = _knuth_best_M(np.sort(a), a_min, a_max, M_max)
        ref = _best_M_reference(col, cap)
        assert fused == ref, f"best_M mismatch col={name} n={n} cap={cap}: fused={fused} ref={ref}"


def test_knuth_bin_edges_identical_uniform_and_quantile():
    """Knuth bin edges identical uniform and quantile."""
    rng = np.random.default_rng(99)
    for n in (500, 4000):
        for col in _cols(n, rng).values():
            for edge_type in ("uniform", "quantile"):
                edges = _knuth_bin_edges(col, edge_type=edge_type, m_max_cap=64)
                # reconstruct expected edges from the reference best_M
                a = np.asarray(col, dtype=np.float64).ravel()
                a = a[np.isfinite(a)]
                best_M = _best_M_reference(col, 64)
                if edge_type == "quantile":
                    expected = np.nanpercentile(a, np.linspace(0.0, 100.0, best_M + 1))
                else:
                    expected = np.linspace(float(a.min()), float(a.max()), best_M + 1)
                np.testing.assert_array_equal(edges, expected)
