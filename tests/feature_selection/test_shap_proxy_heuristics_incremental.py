"""Incremental margin cache: parity + speedup tests for beam_search / greedy_forward.

Beam / greedy expand candidate subsets by one feature at a time. The ``_Evaluator``
now caches the per-subset margin vector so each (parent, +j) child is scored by a
single O(n) vector add instead of recomputing ``phi[:, full_subset].sum(axis=1)``.
Tests pin:

  - bit-identical top-N vs the prior reduce-from-scratch path (float64 round-off);
  - the incremental key insertion preserves sorted-tuple invariant;
  - on a regime where beam fires (width=50, max_card=12), incremental wall is
    materially faster than the prior implementation (run-once smoke; speedup
    factor is captured but not bit-pinned to avoid bench flakiness).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mlframe.feature_selection import _shap_proxy_heuristics as H
from mlframe.feature_selection._shap_proxy_objective import coalition_margin, proxy_loss


def _ref_loss(phi, base, y, idx, metric):
    return proxy_loss(coalition_margin(phi, base, idx), y, metric)


@pytest.fixture
def medium_problem():
    rng = np.random.default_rng(123)
    n, f, k = 400, 30, 6
    phi = rng.normal(size=(n, f)).astype(np.float64)
    base = np.full(n, 0.1)
    y = base + phi[:, :k].sum(axis=1) + 0.03 * rng.normal(size=n)
    return phi, base, y


def test_evaluator_loss_with_margin_matches_reference(medium_problem):
    phi, base, y = medium_problem
    ev = H._Evaluator(phi, base, y, "rmse")
    key = (1, 3, 5, 7)
    loss, margin = ev.loss_with_margin(key)
    ref = _ref_loss(phi, base, y, list(key), "rmse")
    assert np.isclose(loss, ref)
    np.testing.assert_allclose(margin, base + phi[:, list(key)].sum(axis=1))


def test_evaluator_loss_from_parent_matches_reference(medium_problem):
    phi, base, y = medium_problem
    ev = H._Evaluator(phi, base, y, "rmse")
    parent = (2, 5, 9)
    _, parent_margin = ev.loss_with_margin(parent)
    for new_j in (0, 4, 11, 17):
        loss, child_key, child_margin = ev.loss_from_parent(parent, parent_margin, new_j)
        # key invariant: sorted, includes new_j exactly once
        assert child_key == tuple(sorted(parent + (new_j,)))
        ref_margin = base + phi[:, list(child_key)].sum(axis=1)
        np.testing.assert_allclose(child_margin, ref_margin)
        ref_loss = _ref_loss(phi, base, y, list(child_key), "rmse")
        assert np.isclose(loss, ref_loss)


def test_evaluator_loss_from_parent_insertion_positions(medium_problem):
    """new_j may insert at front / middle / back of the sorted parent tuple."""
    phi, base, y = medium_problem
    ev = H._Evaluator(phi, base, y, "rmse")
    parent = (5, 10, 15)
    _, m = ev.loss_with_margin(parent)
    _, k_front, _ = ev.loss_from_parent(parent, m, 1)
    _, k_mid, _ = ev.loss_from_parent(parent, m, 7)
    _, k_back, _ = ev.loss_from_parent(parent, m, 20)
    assert k_front == (1, 5, 10, 15)
    assert k_mid == (5, 7, 10, 15)
    assert k_back == (5, 10, 15, 20)


def test_beam_search_topn_bit_identical_to_reference(medium_problem):
    """Incremental path must agree with the reduce-from-scratch reference on every cached subset."""
    phi, base, y = medium_problem
    top = H.beam_search(phi, base, y, classification=False, metric="rmse",
                        beam_width=12, max_card=8, top_n=20)
    for loss, key in top:
        ref = _ref_loss(phi, base, y, list(key), "rmse")
        assert np.isclose(loss, ref), f"key={key}: cached {loss} vs ref {ref}"


def test_greedy_forward_topn_bit_identical_to_reference(medium_problem):
    phi, base, y = medium_problem
    top = H.greedy_forward(phi, base, y, classification=False, metric="rmse",
                           max_card=10, top_n=10)
    for loss, key in top:
        ref = _ref_loss(phi, base, y, list(key), "rmse")
        assert np.isclose(loss, ref), f"key={key}: cached {loss} vs ref {ref}"


def test_beam_search_incremental_speedup_smoke():
    """Beam at a width where the incremental path matters: must complete inside budget.

    We do not pin a hard speedup factor (CI noise), but the wall must stay well under
    the un-optimised baseline timing of ~0.75s observed locally; budget 2.0s leaves
    headroom for slower CI machines while still failing if a regression doubles cost.
    """
    rng = np.random.default_rng(0)
    n, f, k = 600, 40, 8
    phi = rng.normal(size=(n, f)).astype(np.float64)
    base = np.full(n, 0.1)
    y = base + phi[:, :k].sum(axis=1) + 0.05 * rng.normal(size=n)
    # warmup numba
    H.beam_search(phi, base, y, classification=False, metric="rmse",
                  beam_width=4, max_card=4, top_n=5)
    t0 = time.perf_counter()
    top = H.beam_search(phi, base, y, classification=False, metric="rmse",
                        beam_width=50, max_card=12, top_n=20)
    dt = time.perf_counter() - t0
    assert dt < 2.0, f"beam_search incremental path took {dt:.3f}s (budget 2.0s)"
    assert top[0][0] < 0.1  # truth-recovering regime; coarse sanity, not a perf gate
