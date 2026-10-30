"""The packed (64 labels per word) root test returns the verdict of ``root_decision_binary`` on the same labels: the deduplicated stream, the forbidden boundaries and the verdict agree for noise, planted steps and tied values."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._numba_utils import column_tie_runs, root_decision_binary
from mlframe.feature_selection.filters._numba_utils_cross_module import (
    dedup_label_bits,
    pack_sorted_labels,
    root_decision_packed,
)

_MIN_SPLIT, _ALPHA = 5, 0.05


def _both(xs: np.ndarray, ys: np.ndarray, analytic_min_n: int = 30) -> tuple[int, int, int, int]:
    """``(old verdict, packed verdict, old stream length, packed stream length)`` for sorted values ``xs`` with sorted-order labels ``ys``."""
    starts, ends = column_tie_runs(xs)
    n = xs.shape[0]
    yd = np.empty(n, dtype=np.int8)
    forb = np.empty(starts.shape[0] + 1, dtype=np.int64)
    old = int(root_decision_binary(ys.astype(np.int8), starts, ends, yd, forb, _MIN_SPLIT, _ALPHA, analytic_min_n, 5.0))
    words = (n + 63) >> 6
    raw = np.zeros(words, dtype=np.uint64)
    pack_sorted_labels(ys.astype(np.int8), np.arange(n, dtype=np.int64), raw)
    out = np.zeros(words + 1, dtype=np.uint64)
    forb2 = np.empty(starts.shape[0] + 1, dtype=np.int64)
    m, nf = dedup_label_bits(raw, n, starts, ends, out, forb2)
    new = int(root_decision_packed(out, m, forb2, nf, _MIN_SPLIT, _ALPHA, analytic_min_n, 5.0))
    return old, new, int(forb.shape[0]), m


@pytest.mark.parametrize("n", [37, 64, 65, 200, 5000, 70001])
@pytest.mark.parametrize("balance", [0.5, 0.15])
@pytest.mark.parametrize("signal", [0.0, 0.04, 0.3])
@pytest.mark.parametrize("ties", ["none", "few", "many"])
def test_packed_verdict_equals_the_unpacked_verdict(n, balance, signal, ties):
    """Same verdict on every combination, including planted steps that are accepted."""
    rng = np.random.default_rng(n * 7 + int(signal * 100))
    for rep in range(4):
        x = rng.standard_normal(n)
        if ties == "few":
            x = np.where(rng.random(n) < 0.03, np.round(x, 1), x)
        elif ties == "many":
            x = np.round(x, 1)
        order = np.argsort(x, kind="stable")
        xs = x[order]
        ys = (rng.random(n) < balance).astype(np.int8)
        cut = int(n * rng.uniform(0.3, 0.7))
        ys[cut:] = np.where(rng.random(n - cut) < balance + signal, 1, ys[cut:])
        old, new, _, _ = _both(xs, ys)
        assert old == new, (n, balance, signal, ties, rep, old, new)


def test_accepted_roots_occur_and_are_matched():
    """The planted-step cases reach ACCEPTED in both implementations (the comparison is not all REJECTED)."""
    rng = np.random.default_rng(0)
    n = 20000
    x = np.sort(rng.standard_normal(n))
    ys = (rng.random(n) < 0.5).astype(np.int8)
    ys[n // 2 :] = (rng.random(n - n // 2) < 0.8).astype(np.int8)
    old, new, _, _ = _both(x, ys)
    assert old == new == 1


def test_joint_counts_from_packed_labels_equal_the_unpacked_pass():
    """The per-bin counts taken as popcounts over ranges equal the single pass over the sorted values and labels, with edges inside, between and outside the data."""
    from mlframe.feature_selection.filters._mdlp_honest_bycol_njit import _joint_from_bits, _joint_from_sorted

    rng = np.random.default_rng(5)
    for n in (70, 64, 129, 5000):
        x = np.sort(np.round(rng.standard_normal(n), 2)).astype(np.float32)
        ys = (rng.random(n) < 0.4).astype(np.int8)
        raw = np.zeros((n + 63) >> 6, dtype=np.uint64)
        pack_sorted_labels(ys, np.arange(n, dtype=np.int64), raw)
        for edges in (np.array([]), np.array([0.0]), np.array([-0.5, 0.1, 0.1 + 1e-9, 3.0, 9.0]), np.array([float(x[0]), float(x[-1])])):
            edges = np.asarray(edges, dtype=np.float64)
            n_x = edges.shape[0] + 1
            want = np.zeros((n_x, 2), dtype=np.int64)
            got = np.zeros((n_x, 2), dtype=np.int64)
            _joint_from_sorted(x, ys.astype(np.int64), edges, n_x, 2, want)
            _joint_from_bits(x, raw, edges, n_x, 2, got)
            assert np.array_equal(want, got), (n, edges)
