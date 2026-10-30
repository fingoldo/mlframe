"""The int8 root precheck of the honest MDLP null decides the root split exactly as the general tree search does.

``root_decision_binary`` replaces, for binary labels, the deduplicated copy and full candidate scan of ``_mdlp_cuts_presorted`` with a scan on an int8 label stream that evaluates the exact gain only
where a rigorous bound (``gain <= ln(1 + phi^2)``) reaches the acceptance threshold. The decision must not change: a tree has at least one cut exactly when its root is accepted.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._numba_utils import column_tie_runs, root_decision_binary
from mlframe.feature_selection.filters._numba_utils_cross_module import ACCEPTED, REJECTED, UNDECIDED
from mlframe.feature_selection.filters._numba_utils_cross_module import _mdlp_cuts_presorted

_MIN_SPLIT, _ALPHA, _MAX_DEPTH, _N_PERM, _ANALYTIC_MIN_N, _MIN_CELL = 5, 0.05, 8, 15, 15000, 5.0


def _general(xs: np.ndarray, ys: np.ndarray) -> int:
    """The general route's root verdict: any cut means the root was accepted."""
    cuts = _mdlp_cuts_presorted(xs, ys.astype(np.int64), 2, _MIN_SPLIT, _MAX_DEPTH, _ALPHA, _N_PERM, 0, _ANALYTIC_MIN_N, _MIN_CELL)
    return ACCEPTED if cuts.shape[0] > 0 else REJECTED


def _precheck(xs: np.ndarray, ys: np.ndarray, min_n: int = _ANALYTIC_MIN_N) -> int:
    """The precheck's verdict for labels ``ys`` in the sorted order of ``xs``."""
    starts, ends = column_tie_runs(xs)
    yd = np.empty(xs.shape[0], dtype=np.int8)
    forb = np.empty(starts.shape[0] + 1, dtype=np.int64)
    return int(root_decision_binary(ys.astype(np.int8), starts, ends, yd, forb, _MIN_SPLIT, _ALPHA, min_n, _MIN_CELL))


def _column(rng: np.random.Generator, n: int, ties: str) -> np.ndarray:
    """A sorted column: continuous, float32-rounded (a few ties), or coarse (many ties)."""
    x = rng.standard_normal(n)
    if ties == "float32":
        x = x.astype(np.float32).astype(np.float64)
    elif ties == "coarse":
        x = np.round(x, 1)
    return np.sort(x)


@pytest.mark.parametrize("ties", ["none", "float32", "coarse"])
def test_precheck_agrees_with_the_general_route_on_noise_and_planted_signal(ties):
    """Pure noise (rejected) and a signal swept from absent to strong (accepted) give the verdicts of the general tree search."""
    rng = np.random.default_rng(7)
    n = 20_000
    verdicts = []
    for trial in range(60):
        xs = _column(rng, n, ties)
        strength = 0.0 if trial % 3 == 0 else float(rng.uniform(0.0, 0.05))
        p = 0.5 + strength * np.sign(xs - np.median(xs))  # labels drift with x: a real split near the median
        ys = (rng.random(n) < np.clip(p, 0.0, 1.0)).astype(np.int8)
        got, want = _precheck(xs, ys), _general(xs, ys)
        if got == UNDECIDED:
            assert ties == "coarse", "only a column whose deduplicated size falls below analytic_min_n may be left to the general route"
            continue
        assert got == want, f"trial {trial} strength {strength:.4f}: precheck {got}, general route {want}"
        verdicts.append(got)
    if ties == "coarse":
        assert not verdicts, "~100 distinct (x, y) points: every root is judged by the permutation test"
    else:
        assert ACCEPTED in verdicts and REJECTED in verdicts, "the sweep must exercise both outcomes"


def test_precheck_agrees_on_skewed_class_priors():
    """Rare positive class: the bound uses n_1 * n_0, which is small there."""
    rng = np.random.default_rng(11)
    n = 20_000
    for prior in (0.01, 0.05, 0.3):
        for _ in range(15):
            xs = _column(rng, n, "float32")
            ys = (rng.random(n) < prior).astype(np.int8)
            ys[: int(n * 0.1)] |= (rng.random(int(n * 0.1)) < 0.02).astype(np.int8)
            assert _precheck(xs, ys) == _general(xs, ys)


def test_precheck_declines_when_the_root_is_not_judged_analytically():
    """Below ``analytic_min_n`` the root takes a permutation test; the precheck leaves that to the general route."""
    rng = np.random.default_rng(3)
    xs = _column(rng, 2000, "none")
    ys = (rng.random(2000) < 0.5).astype(np.int8)
    assert _precheck(xs, ys, min_n=15000) == UNDECIDED


def test_tie_runs_are_the_maximal_runs_of_equal_values():
    """Runs are reported as half-open ranges, singletons are not runs."""
    starts, ends = column_tie_runs(np.array([1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 4.0]))
    assert starts.tolist() == [1, 5] and ends.tolist() == [4, 7]


def _bycol_args(n: int, k: int, rows: str, seed: int = 5):
    """Arguments of ``honest_null_columns_bycol``: ``k`` continuous columns (float32-rounded, so with ties), binary labels, ``rows`` = ``"all"`` (every row evaluated, the labels of the tree are the screen's
    codes: the joint-from-sorted path) or ``"subset"`` (a third of the rows evaluated)."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, k)).astype(np.float32).astype(np.float64)
    y = (rng.random(n) < 0.5).astype(np.int8)
    n_perm = 12
    perms = np.stack([y[rng.permutation(n)] for _ in range(n_perm)])
    edges = np.stack([np.quantile(raw[:, j], np.linspace(0.1, 0.9, 9)) for j in range(k)])
    eval_rows = np.arange(n, dtype=np.int64) if rows == "all" else np.sort(rng.choice(n, n // 3, replace=False)).astype(np.int64)
    return (
        raw, np.ones(k, dtype=np.bool_), np.zeros((n, 1), dtype=np.uint8), np.zeros(k, dtype=np.int64), perms, perms.copy(), 2, _MIN_SPLIT, _MAX_DEPTH, _ALPHA, _N_PERM, 0,
        _ANALYTIC_MIN_N, _MIN_CELL, edges, np.full(k, 9, dtype=np.int64), np.zeros((k, 1)), np.zeros(k, dtype=np.int64), eval_rows, np.arange(k, dtype=np.int64),
        np.zeros((n, k), dtype=np.bool_),
    )


@pytest.mark.parametrize("rows", ["all", "subset"])
def test_null_columns_are_the_same_with_and_without_the_precheck(rows):
    """The whole column-parallel null (both the joint-from-sorted path of an all-rows evaluation and the code-table path) is bit-identical with the precheck off and on.

    A rejected root leaves the sorted int64 labels unfilled; the all-rows path must read the int8 stream instead (it read the unfilled buffer once, out of bounds)."""
    from mlframe.feature_selection.filters._mdlp_honest_bycol_njit import honest_null_columns_bycol

    args = _bycol_args(20_000, 4, rows)
    off = honest_null_columns_bycol(*args, False)
    on = honest_null_columns_bycol(*args, True)
    assert np.array_equal(off[0], on[0]) and np.array_equal(off[1], on[1])
