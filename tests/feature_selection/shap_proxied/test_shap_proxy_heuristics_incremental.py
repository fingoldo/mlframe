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


import numpy as np
import pytest

from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_heuristics as H
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import coalition_margin, proxy_loss


def _ref_loss(phi, base, y, idx, metric):
    """Ref loss."""
    return proxy_loss(coalition_margin(phi, base, idx), y, metric)


@pytest.fixture
def medium_problem():
    """Medium problem."""
    rng = np.random.default_rng(123)
    n, f, k = 400, 30, 6
    phi = rng.normal(size=(n, f)).astype(np.float64)
    base = np.full(n, 0.1)
    y = base + phi[:, :k].sum(axis=1) + 0.03 * rng.normal(size=n)
    return phi, base, y


def test_evaluator_loss_with_margin_matches_reference(medium_problem):
    """Evaluator loss with margin matches reference."""
    phi, base, y = medium_problem
    ev = H._Evaluator(phi, base, y, "rmse")
    key = (1, 3, 5, 7)
    loss, margin = ev.loss_with_margin(key)
    ref = _ref_loss(phi, base, y, list(key), "rmse")
    assert np.isclose(loss, ref)
    np.testing.assert_allclose(margin, base + phi[:, list(key)].sum(axis=1))


def test_evaluator_loss_from_parent_matches_reference(medium_problem):
    """Evaluator loss from parent matches reference."""
    phi, base, y = medium_problem
    ev = H._Evaluator(phi, base, y, "rmse")
    parent = (2, 5, 9)
    _, parent_margin = ev.loss_with_margin(parent)
    for new_j in (0, 4, 11, 17):
        loss, child_key, child_margin = ev.loss_from_parent(parent, parent_margin, new_j)
        # key invariant: sorted, includes new_j exactly once
        assert child_key == tuple(sorted((*parent, new_j)))
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
    top = H.beam_search(phi, base, y, classification=False, metric="rmse", beam_width=12, max_card=8, top_n=20)
    for loss, key in top:
        ref = _ref_loss(phi, base, y, list(key), "rmse")
        assert np.isclose(loss, ref), f"key={key}: cached {loss} vs ref {ref}"


def test_greedy_forward_topn_bit_identical_to_reference(medium_problem):
    """Greedy forward topn bit identical to reference."""
    phi, base, y = medium_problem
    top = H.greedy_forward(phi, base, y, classification=False, metric="rmse", max_card=10, top_n=10)
    for loss, key in top:
        ref = _ref_loss(phi, base, y, list(key), "rmse")
        assert np.isclose(loss, ref), f"key={key}: cached {loss} vs ref {ref}"


_INCREMENTAL = ("loss_from_parent", "loss_from_parent_drop", "loss_from_parent_swap")


@pytest.fixture
def evaluation_counts(monkeypatch):
    """Counts of the evaluator's FULL recomputations (``loss`` / ``loss_with_margin``) and INCREMENTAL updates (one O(n) vector op each)."""
    counts = {"full": 0, "incremental": 0}

    def _wrap(name, bucket):
        """Replace ``H._Evaluator.<name>`` with a counting wrapper that adds to ``counts[bucket]``."""
        original = getattr(H._Evaluator, name)

        def counted(self, *args, **kwargs):
            """Count the call, then run the real method."""
            counts[bucket] += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(H._Evaluator, name, counted)

    for name in ("loss", "loss_with_margin"):
        _wrap(name, "full")
    for name in _INCREMENTAL:
        _wrap(name, "incremental")
    return counts


def _assert_incremental_dominates(counts, algorithm):
    """The speed-up is structural: nearly every candidate is scored from its parent's margin, not recomputed from scratch.

    A regression to the O(n * f * |current|^2) full path shows up as full recomputations matching the incremental count; unlike a wall-clock
    budget this does not move with CI load.
    """
    assert counts["incremental"] > 0, f"{algorithm} never used the incremental path"
    assert counts["incremental"] >= 20 * counts["full"], f"{algorithm} recomputed from scratch too often: {counts}"


def test_beam_search_incremental_speedup_smoke(evaluation_counts):
    """Beam at a width where the incremental path matters: candidates are scored from their parent's margin."""
    rng = np.random.default_rng(0)
    n, f, k = 600, 40, 8
    phi = rng.normal(size=(n, f)).astype(np.float64)
    base = np.full(n, 0.1)
    y = base + phi[:, :k].sum(axis=1) + 0.05 * rng.normal(size=n)
    top = H.beam_search(phi, base, y, classification=False, metric="rmse", beam_width=50, max_card=12, top_n=20)
    _assert_incremental_dominates(evaluation_counts, "beam_search")
    assert top[0][0] < 0.1  # truth-recovering regime; coarse sanity, not a perf gate


def test_evaluator_loss_from_parent_drop_matches_reference(medium_problem):
    """Drop variant: child margin = parent_margin - phi[:, drop_j]; key has drop_j removed."""
    phi, base, y = medium_problem
    ev = H._Evaluator(phi, base, y, "rmse")
    parent = (2, 5, 9, 13, 17)
    _, parent_margin = ev.loss_with_margin(parent)
    for drop_j in (2, 9, 17):  # front / mid / back of sorted parent
        loss, child_key, child_margin = ev.loss_from_parent_drop(parent, parent_margin, drop_j)
        assert child_key == tuple(x for x in parent if x != drop_j)
        ref_margin = base + phi[:, list(child_key)].sum(axis=1)
        np.testing.assert_allclose(child_margin, ref_margin)
        ref_loss = _ref_loss(phi, base, y, list(child_key), "rmse")
        assert np.isclose(loss, ref_loss)


def test_evaluator_loss_from_parent_drop_to_empty_is_inf(medium_problem):
    """Drop that empties the subset must yield inf loss (matches ``loss`` empty-subset semantics)."""
    phi, base, y = medium_problem
    ev = H._Evaluator(phi, base, y, "rmse")
    parent = (7,)
    _, parent_margin = ev.loss_with_margin(parent)
    loss, child_key, child_margin = ev.loss_from_parent_drop(parent, parent_margin, 7)
    assert child_key == ()
    assert loss == float("inf")
    np.testing.assert_allclose(child_margin, base)


def test_evaluator_loss_from_parent_swap_matches_reference(medium_problem):
    """Swap variant: one fused O(n) op == drop-then-add reference."""
    phi, base, y = medium_problem
    ev = H._Evaluator(phi, base, y, "rmse")
    parent = (3, 8, 14, 21)
    _, parent_margin = ev.loss_with_margin(parent)
    # cover out/in at different relative positions in the sorted key
    for out_j, in_j in [(3, 11), (14, 1), (21, 25), (8, 7), (3, 22)]:
        loss, child_key, child_margin = ev.loss_from_parent_swap(parent, parent_margin, out_j, in_j)
        ref_key = tuple(sorted([x for x in parent if x != out_j] + [in_j]))
        assert child_key == ref_key
        ref_margin = base + phi[:, list(child_key)].sum(axis=1)
        np.testing.assert_allclose(child_margin, ref_margin)
        ref_loss = _ref_loss(phi, base, y, list(child_key), "rmse")
        assert np.isclose(loss, ref_loss)


def test_multistart_local_topn_matches_reference(medium_problem):
    """Every cached subset reported by multistart_local must agree with the reduce-from-scratch ref."""
    phi, base, y = medium_problem
    top = H.multistart_local(phi, base, y, classification=False, metric="rmse", rng=np.random.default_rng(42), n_starts=6, max_card=8, top_n=15)
    for loss, key in top:
        ref = _ref_loss(phi, base, y, list(key), "rmse")
        assert np.isclose(loss, ref), f"key={key}: cached {loss} vs ref {ref}"


def test_simulated_annealing_topn_matches_reference(medium_problem):
    """Every cached subset reported by simulated_annealing must agree with the reduce-from-scratch ref."""
    phi, base, y = medium_problem
    top = H.simulated_annealing(phi, base, y, classification=False, metric="rmse", rng=np.random.default_rng(0), n_iter=400, top_n=15)
    for loss, key in top:
        ref = _ref_loss(phi, base, y, list(key), "rmse")
        assert np.isclose(loss, ref), f"key={key}: cached {loss} vs ref {ref}"


def test_multistart_local_speedup_smoke(evaluation_counts):
    """multistart_local at width=50, max_card=10: every swap/add/drop trial is one O(n) update of the parent's margin.

    Reducing from scratch per trial is roughly O(n * f * |current|^2) per hill-climb iteration.
    """
    rng = np.random.default_rng(0)
    n, f, k = 500, 50, 8
    phi = rng.normal(size=(n, f)).astype(np.float64)
    base = np.full(n, 0.1)
    y = base + phi[:, :k].sum(axis=1) + 0.05 * rng.normal(size=n)
    top = H.multistart_local(phi, base, y, classification=False, metric="rmse", rng=np.random.default_rng(1), n_starts=8, max_card=10, top_n=10)
    _assert_incremental_dominates(evaluation_counts, "multistart_local")
    assert top[0][0] < 0.2  # coarse truth-recovering sanity


def test_simulated_annealing_speedup_smoke(evaluation_counts):
    """simulated_annealing at width=40, n_iter=2000: each move is scored incrementally from the current state."""
    rng = np.random.default_rng(0)
    n, f, k = 500, 40, 8
    phi = rng.normal(size=(n, f)).astype(np.float64)
    base = np.full(n, 0.1)
    y = base + phi[:, :k].sum(axis=1) + 0.05 * rng.normal(size=n)
    top = H.simulated_annealing(phi, base, y, classification=False, metric="rmse", rng=np.random.default_rng(2), n_iter=2000, top_n=10)
    _assert_incremental_dominates(evaluation_counts, "simulated_annealing")
    assert top[0][0] < 0.5  # coarse sanity; SA wider tolerance vs greedy/beam
