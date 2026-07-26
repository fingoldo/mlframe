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
    """The exact prior object-mode search, kept here as the identity oracle.

    The search RANGE carries the kernel's distinct-value ceiling, because that ceiling is a documented
    pre-filter on the range rather than a change to the posterior itself. Knuth's criterion assumes
    continuous data: on a tie-heavy column every bin past the number of distinct values is empty and adds
    only a constant lgamma(0.5), which ``n*log(M)`` keeps outweighing, so the raw scan never turns around
    and runs away to M_max (measured: logp 24.8 at M=5 climbing to 994.7 at M=64 on a 7-value column).
    Leaving the ceiling out here would make this oracle assert the runaway is correct. What the identity
    claim is actually about - that the fused lgamma accumulation reproduces
    ``np.histogram`` + ``_knuth_log_posterior`` for every M in range - is unaffected and still fully tested.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    n = a.size
    a_min, a_max = float(a.min()), float(a.max())
    M_max = int(min(max(4, int(np.sqrt(n) * 4)), int(m_max_cap)))
    n_distinct = int(np.unique(a).size)
    if n_distinct >= 2:
        M_max = min(M_max, n_distinct)
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


@pytest.mark.parametrize("n_distinct", [3, 7, 12])
def test_knuth_best_M_never_exceeds_the_distinct_value_count(n_distinct):
    """A tie-heavy column must not run away to M_max - bins past the distinct-value count are all empty.

    Pinned separately from the identity test because the oracle there carries the same ceiling: if the
    kernel silently dropped it, both sides would run away together and the identity test would still pass.
    Here the bound is asserted against the data, so dropping the ceiling fails outright. The runaway is not
    cosmetic - M_max bins on a 7-value column feeds the MI plug-in a contingency table that is almost all
    empty cells, which is exactly the regime where its finite-sample bias is worst.
    """
    rng = np.random.default_rng(4242 + n_distinct)
    n = 4000
    a = rng.integers(0, n_distinct, n).astype(np.float64)
    a_sorted = np.sort(a)
    M_max = int(min(max(4, int(np.sqrt(n) * 4)), 500))
    assert M_max > n_distinct, "test is vacuous unless the raw ceiling is above the distinct-value count"

    fused = _knuth_best_M(a_sorted, float(a.min()), float(a.max()), M_max)
    assert 2 <= fused <= n_distinct, f"n_distinct={n_distinct}: best_M={fused} exceeds the distinct-value ceiling (M_max={M_max})"


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
