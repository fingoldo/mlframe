"""The held-out protection gates factorise their base design once instead of refitting it per candidate.

Every protection gate asks the same question of many candidates against one fixed base: does adding this column lift a held-out linear fit.
The raw floor-drop gate was rewritten to QR-factorise the base once and extend it by one column per candidate, measured 20x at production
width. Its siblings kept re-running ``np.column_stack`` over the full design and a fresh SVD ``lstsq`` per candidate, which is O(n*p) plus
O(n*p^2) repeated per candidate, and at n=2M with a hundred selected columns each of those column_stacks is gigabytes.

Scoring has to stay the same, so these tests pin the answers against the plain per-candidate least-squares fit as well as the call counts.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._mrmr_fit_impl._friend_graph_and_redundancy._heldout_gate import build_heldout_incr_probe
from mlframe.feature_selection.filters._mrmr_fit_impl._friend_graph_and_redundancy._raw_protect_r2 import heldout_r2_scorer


@pytest.fixture
def design():
    """A base design of several columns, a target they partly explain, and a train/validation split."""
    rng = np.random.default_rng(0)
    n, p = 900, 6
    base = [np.ones(n)] + [rng.normal(size=n) for _ in range(p)]
    y = 2.0 * base[1] - 1.5 * base[2] + 0.3 * rng.normal(size=n)
    va = np.zeros(n, dtype=bool)
    va[rng.permutation(n)[: n // 3]] = True
    return base, y, ~va, va


def _reference_r2(design_cols, y, tr, va) -> float:
    """The plain per-candidate fit the gates used to do: stack the whole design, lstsq on train, score on validation."""
    A = np.column_stack(design_cols)
    coef = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
    yv = y[va]
    return 1.0 - float(np.sum((yv - A[va] @ coef) ** 2)) / float(np.sum((yv - yv.mean()) ** 2))


def test_the_incremental_scorer_matches_a_per_candidate_least_squares_fit(design):
    """Same answers as the form it replaces, for the base alone and for several candidates."""
    base, y, tr, va = design
    scorer = heldout_r2_scorer(base, y, tr, va)
    assert scorer() == pytest.approx(_reference_r2(base, y, tr, va), abs=1e-9)
    rng = np.random.default_rng(1)
    for _ in range(5):
        cand = rng.normal(size=y.shape[0])
        assert scorer(cand) == pytest.approx(_reference_r2([*base, cand], y, tr, va), abs=1e-9)


def test_the_scorer_accepts_a_block_of_columns_that_only_mean_something_together(design):
    """A sin/cos leg pair is inserted as one block, and must score as the design that holds both."""
    base, y, tr, va = design
    scorer = heldout_r2_scorer(base, y, tr, va)
    rng = np.random.default_rng(2)
    pair = np.column_stack([rng.normal(size=y.shape[0]), rng.normal(size=y.shape[0])])
    assert scorer(pair) == pytest.approx(_reference_r2([*base, pair[:, 0], pair[:, 1]], y, tr, va), abs=1e-9)


def test_a_candidate_collinear_with_the_base_is_still_scored_correctly(design):
    """An unpivoted QR cannot see rank deficiency, so the scorer must fall back rather than return noise."""
    base, y, tr, va = design
    scorer = heldout_r2_scorer(base, y, tr, va)
    duplicate = base[1].copy()
    got = scorer(duplicate)
    assert np.isfinite(got)
    assert got == pytest.approx(_reference_r2([*base, duplicate], y, tr, va), abs=1e-6)


def test_the_probe_does_not_rebuild_the_split_or_the_design_per_candidate(design, monkeypatch):
    """The split and the base factorisation depend only on the seed and the selected set, so they are done once."""
    base, y, _tr, _va = design
    calls = {"rng": 0, "stack": 0}
    real_rng, real_stack = np.random.default_rng, np.column_stack

    def counting_rng(*a, **k):
        """Count generator constructions, then defer to the real one."""
        calls["rng"] += 1
        return real_rng(*a, **k)

    def counting_stack(*a, **k):
        """Count full-design materialisations, then defer to the real one."""
        calls["stack"] += 1
        return real_stack(*a, **k)

    monkeypatch.setattr(np.random, "default_rng", counting_rng)
    monkeypatch.setattr(np, "column_stack", counting_stack)
    probe = build_heldout_incr_probe(y_gate=y, sel_value_cols=base[1:], random_seed=0)
    calls["rng"] = 0
    after_build = calls["stack"]
    rng = real_rng(3)
    for _ in range(12):
        probe(rng.normal(size=y.shape[0]))
    assert calls["rng"] == 0, "the split was redrawn per candidate, though it depends only on the seed and n"
    per_candidate = (calls["stack"] - after_build) / 12.0
    assert per_candidate <= 2.0, f"{per_candidate} column_stack calls per candidate: the base design is being rebuilt each time"


def test_the_probe_still_separates_a_subsumed_candidate_from_a_useful_one(design):
    """The rewrite is for cost, not for verdicts: the decisions it feeds must not move."""
    base, y, _tr, _va = design
    probe = build_heldout_incr_probe(y_gate=y, sel_value_cols=base[1:], random_seed=0)
    assert probe(2.0 * base[1] + 1.0) < 0.003, "a rescaled copy of a selected column must not clear the floor"
    rng = np.random.default_rng(4)
    extra = rng.normal(size=y.shape[0])
    lifted = build_heldout_incr_probe(y_gate=y + 3.0 * extra, sel_value_cols=base[1:], random_seed=0)
    assert lifted(extra) >= 0.003
