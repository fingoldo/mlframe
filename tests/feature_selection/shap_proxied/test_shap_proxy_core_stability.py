"""Unit tests for gt_02's least-core LP (``_shap_proxy_core_stability.py``): textbook cooperative games
with KNOWN analytic cores, plus infeasibility / determinism / dummy-player correctness.

These are the scientific-correctness proof for the LP construction -- v(C), the sampled-vs-exhaustive
coalition set, the eps-slack objective -- independent of any ShapProxiedFS integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_core_stability import (
    core_refine, least_core_allocation,
)


class _GameEvaluator:
    """Minimal stand-in for ``_Evaluator``: wraps an explicit characteristic function v(C) as a loss oracle.

    ``loss(idx) = -v(idx)`` so ``least_core_allocation``'s ``L_ref - loss`` construction recovers v
    directly (``L_ref`` = the worst singleton's loss = ``-min(v({i}))`` = 0 whenever every singleton
    has v=0, which holds for every textbook game below).
    """

    def __init__(self, v: dict):
        self.v = v

    def loss(self, idx) -> float:
        """Return -v(subset), the characteristic value negated into loss-game convention."""
        key = tuple(sorted(idx))
        return -float(self.v[key])


def _gloves_game() -> _GameEvaluator:
    """Classic 3-player glove game: player 0 owns 1 LEFT glove, players 1/2 each own a RIGHT glove.

    v(C) = 1 iff C contains player 0 AND at least one of {1, 2}, else 0. The known analytic core is
    the single point x = (1, 0, 0): left gloves are scarce (only 1 vs 2 right gloves), so the entire
    surplus accrues to the scarce side and eps* = 0 (the core is nonempty for this convex-like game).
    """
    v = {}
    for c in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]:
        v[c] = 1.0 if (0 in c and (1 in c or 2 in c)) else 0.0
    return _GameEvaluator(v)


def _dummy_player_game() -> _GameEvaluator:
    """v(C) = 1 iff player 0 in C, else 0 -- players 1 and 2 are pure dummies (add nothing to any coalition)."""
    v = {}
    for c in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]:
        v[c] = 1.0 if 0 in c else 0.0
    return _GameEvaluator(v)


def test_gloves_game_core_recovers_analytic_solution_exhaustive():
    """The known-analytic 3-player glove game core (all credit to the scarce left-glove owner) is
    recovered within 1e-6 using EXHAUSTIVE coalition enumeration."""
    ev = _gloves_game()
    rng = np.random.default_rng(0)
    x, eps_star, info = least_core_allocation(ev, (0, 1, 2), rng=rng, exhaustive=True)
    np.testing.assert_allclose(x, [1.0, 0.0, 0.0], atol=1e-6)
    assert abs(eps_star) < 1e-6, f"convex game must have eps*=0 (nonempty core), got {eps_star}"
    assert info["lp_status"] == "optimal"


def test_gloves_game_eps_star_zero_for_convex_game():
    """eps* == 0 confirms the sampled-coalition LP found a point IN the (nonempty) core, not merely
    an approximate least-core relaxation."""
    ev = _gloves_game()
    rng = np.random.default_rng(0)
    _, eps_star, _ = least_core_allocation(ev, (0, 1, 2), rng=rng, exhaustive=True)
    assert eps_star == pytest.approx(0.0, abs=1e-6)


def test_dummy_player_gets_zero_allocation():
    """Players 1 and 2 contribute nothing to any coalition (pure dummies) and must receive x_j = 0."""
    ev = _dummy_player_game()
    rng = np.random.default_rng(0)
    x, _eps, _info = least_core_allocation(ev, (0, 1, 2), rng=rng, exhaustive=True)
    np.testing.assert_allclose(x[1:], [0.0, 0.0], atol=1e-6)
    assert x[0] == pytest.approx(1.0, abs=1e-6)


def test_least_core_allocation_deterministic_under_fixed_rng():
    """Two calls with independently-seeded-but-identical rng streams must produce bit-for-bit-close results."""
    ev = _dummy_player_game()
    x1, eps1, _ = least_core_allocation(ev, (0, 1, 2), rng=np.random.default_rng(42), n_coalitions=4, exhaustive=False)
    x2, eps2, _ = least_core_allocation(ev, (0, 1, 2), rng=np.random.default_rng(42), n_coalitions=4, exhaustive=False)
    np.testing.assert_array_equal(x1, x2)
    assert eps1 == eps2


def test_least_core_allocation_single_player_trivial():
    """A single-player coalition has nothing to allocate away from: x = [0], eps* = 0, no LP solved."""
    ev = _GameEvaluator({(0,): 0.0})
    x, eps_star, info = least_core_allocation(ev, (0,), rng=np.random.default_rng(0))
    assert x.shape == (1,)
    assert x[0] == 0.0
    assert eps_star == 0.0
    assert info["lp_status"] == "trivial_k1"


def test_least_core_allocation_infeasible_lp_reports_status(monkeypatch):
    """LP-solver failure is handled gracefully, not a silent crash or a fabricated 'optimal' result.

    By construction ``v(C) = max(0, L_ref - loss(C)) >= 0`` and ``eps`` is unrestricted above, so the
    sampled-coalition LP as formulated is always feasible (any coalition deficit can be absorbed by a
    large-enough eps) -- genuine infeasibility only arises from numerical degeneracy inside HiGHS, not
    from the game's characteristic function. This test exercises the failure-handling BRANCH directly
    by monkeypatching ``linprog`` to report failure, proving ``least_core_allocation`` degrades to the
    documented equal-split-with-eps=+inf sentinel instead of raising or fabricating a false 'optimal'."""
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_core_stability as mod

    class _FailedResult:
        """Stand-in for scipy's OptimizeResult with success=False, mimicking an unsolvable HiGHS call."""

        success = False
        status = 2

    monkeypatch.setattr(mod, "linprog", lambda *a, **k: _FailedResult())
    ev = _gloves_game()
    x, eps_star, info = least_core_allocation(ev, (0, 1, 2), rng=np.random.default_rng(0), exhaustive=True)
    assert info["lp_status"].startswith("infeasible")
    assert eps_star == float("inf")
    np.testing.assert_allclose(x, [1.0 / 3, 1.0 / 3, 1.0 / 3])


def test_core_refine_drops_low_share_units_and_expands_to_members():
    """core_refine drops units below drop_threshold and expands survivors to member columns via unit_to_members."""
    ev = _gloves_game()
    unit_to_members = {0: [10, 11], 1: [20], 2: [30]}
    honest_calls = []

    def honest_gate(cols):
        """Record the proposed column set and always accept (isolates core_refine's drop/expand logic from the honest gate)."""
        honest_calls.append(sorted(cols))
        return True

    refined, info = core_refine(
        members=[10, 11, 20, 30], unit_players=(0, 1, 2), evaluator=ev, honest_loss_fn=honest_gate,
        drop_threshold=0.5, n_coalitions=64, rng=np.random.default_rng(0), unit_to_members=unit_to_members,
        legacy_refine_fn=lambda: [10, 11, 20, 30], legacy_refine_kwargs={},
    )
    assert refined == [10, 11], f"expected only unit 0's members to survive a drop_threshold=0.5 cut on x=(1,0,0), got {refined}"
    assert info["fallback"] is False
    assert set(info["dropped_by_core"]) == {1, 2}
    assert len(honest_calls) == 1


def test_core_refine_falls_back_to_legacy_on_honest_gate_rejection():
    """When the honest gate rejects the core proposal, core_refine returns the legacy fn's result verbatim and marks fallback=True."""
    ev = _gloves_game()
    unit_to_members = {0: [10], 1: [20], 2: [30]}

    refined, info = core_refine(
        members=[10, 20, 30], unit_players=(0, 1, 2), evaluator=ev, honest_loss_fn=lambda cols: False,
        drop_threshold=0.5, n_coalitions=64, rng=np.random.default_rng(0), unit_to_members=unit_to_members,
        legacy_refine_fn=lambda: [10, 20], legacy_refine_kwargs={},
    )
    assert refined == [10, 20]
    assert info["fallback"] is True
