"""Determinism coverage for ``heterogeneous_relevance_vote`` (hetero_vote) and ``HybridSelector``.

Closes coverage_asymmetry_wrappers-13: a same-seed/different-seed determinism guard existed for
MRMR-only; hetero_vote and the composed HybridSelector had no such sensor. A silently-ignored
``random_state`` (a member re-seeding itself, a shadow draw not keyed off the seed, an n_jobs pool
introducing nondeterministic reduction order) is a real reproducibility regression that none of the
existing behavioral tests would catch.

Three legs:
  (a) hetero_vote: same call twice (seed=0) -> identical ``accepted`` AND bit-equal ``info['vote_fraction']``;
      a different seed redraws the shadows -> ``vote_fraction`` differs on >=1 column (guards an ignored
      random_state). The diff leg uses ``percentile=50`` (a low shadow bar -> many borderline columns whose
      vote fraction moves with the shadow draw); at ``percentile=100`` the shadow bar is the column max and
      the integer signal/noise separation is seed-robust, so the diff is measured at the low bar.
  (b) HybridSelector(use_fe=False, use_tree_member=False, random_state=0): fit twice on a small linear frame
      -> identical ``raw_selected_``, ``member_selections_`` and ``fi_`` keys. Because every member runs with
      ``n_jobs=-1`` internally, this is also a de-facto n_jobs-stability sensor (a thread-order-dependent
      reduction would surface as a same-seed mismatch).
  (c) HybridSelector column order: fit on X vs X[reversed cols] (fresh seeded instances). The selection is
      NOT order-invariant (the composed members' positional tie-breaks / cluster-rep picks depend on column
      order); this is the documented reproducibility gap, mirrored from the contract suite's
      ``column_order_invariant=False`` spec -> xfail(strict=False), never a weakened assertion.

Seeds fixed everywhere. The HybridSelector fits run the real MRMR + ShapProxiedFS + BorutaShap members
(~10-20 s), so the heavy legs carry @pytest.mark.slow with a smaller fast representative kept via
MLFRAME_FAST=1 (the conftest fast-mode collection hook skips slow-marked tests). CPU-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.hetero_vote import heterogeneous_relevance_vote
from mlframe.feature_selection.hybrid_selector import HybridSelector

# ---------------------------------------------------------------------------
# Synthetic frames.
# ---------------------------------------------------------------------------

_HETERO_WEIGHTS = np.array([1.5, -1.2, 1.0, 0.9])
_HYBRID_WEIGHTS = np.array([1.6, -1.3, 1.1, 0.9])


def _clf_data(seed: int = 0, n: int = 900, p_sig: int = 4, p_noise: int = 12):
    """Binary target via the logistic of a 4-feature linear score + pure-noise columns."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, p_sig))
    logit = z @ _HETERO_WEIGHTS
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    cols = {f"sig_{i}": z[:, i] for i in range(p_sig)}
    for j in range(p_noise):
        cols[f"noise_{j}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), pd.Series(y, name="y")


def _linear_dataset(n: int = 1200, seed: int = 0):
    """Linear binary target with 4 informative + 3 near-duplicate redundant + 8 noise columns.

    Mirrors the ``_linear_dataset`` shape used by the HybridSelector behavioral suite: the redundant
    block exercises the corr-cluster de-dup path, so determinism here covers the clustering glue too.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 4))
    y = pd.Series((rng.random(n) < 1.0 / (1.0 + np.exp(-(z @ _HYBRID_WEIGHTS)))).astype(int))
    cols = {f"inf_{i}": z[:, i] for i in range(4)}
    for j in range(3):
        cols[f"red_{j}"] = z[:, 0] + 0.02 * rng.standard_normal(n)
    for k in range(8):
        cols[f"noise_{k}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), y


# ===========================================================================
# (a) hetero_vote determinism.
# ===========================================================================


@pytest.mark.timeout(900)
def test_hetero_vote_same_seed_identical_accepted_and_vote_fraction():
    """Two ``heterogeneous_relevance_vote`` calls with the same random_state must produce an identical
    accepted list AND a bit-equal ``info['vote_fraction']`` dict (the shadow rng is fully seeded).

    Each call fits a 3-member panel (n_jobs=-1 RandomForest + LR + kNN) across n_shadow_trials draws, so the
    two-call test runs ~18 estimator fits and carries the longer timeout (the global --timeout=60 is too tight
    for the real panel fits on a contended Windows box)."""
    X, y = _clf_data(seed=0)
    a1, i1 = heterogeneous_relevance_vote(X, y, classification=True, n_shadow_trials=3, random_state=0)
    a2, i2 = heterogeneous_relevance_vote(X, y, classification=True, n_shadow_trials=3, random_state=0)
    assert a1 == a2, f"same-seed accepted differs: {a1} vs {a2}"
    vf1, vf2 = i1["vote_fraction"], i2["vote_fraction"]
    assert vf1.keys() == vf2.keys()
    # bit-equality (not approx): the same seed must replay the exact float vote fractions.
    assert all(vf1[c] == vf2[c] for c in vf1), "same-seed vote_fraction is not bit-identical"
    assert vf1 == vf2


@pytest.mark.timeout(900)
def test_hetero_vote_different_seed_changes_vote_fraction():
    """A different random_state redraws the per-trial shadows, so at the low ``percentile=50`` shadow bar
    (many borderline columns) the vote fraction must differ on >=1 column -- guards an ignored random_state.

    Measured (seed 0 vs 7, n_shadow_trials=3, percentile=50): 6 columns change their vote fraction; the
    floor only requires >=1, so seed noise cannot trip it while a fully-ignored seed (0 diffs) fails it.
    """
    X, y = _clf_data(seed=0)
    _, i0 = heterogeneous_relevance_vote(X, y, classification=True, n_shadow_trials=3, percentile=50, vote_threshold=0.5, random_state=0)
    _, i7 = heterogeneous_relevance_vote(X, y, classification=True, n_shadow_trials=3, percentile=50, vote_threshold=0.5, random_state=7)
    vf0, vf7 = i0["vote_fraction"], i7["vote_fraction"]
    assert vf0.keys() == vf7.keys()
    diffs = [c for c in vf0 if vf0[c] != vf7[c]]
    assert diffs, "a different random_state must change vote_fraction on >=1 column (random_state ignored?)"


# ===========================================================================
# (b) HybridSelector determinism (also a de-facto n_jobs-stability sensor).
# ===========================================================================


def _assert_hybrid_deterministic(X, y):
    """Fit two fresh seeded HybridSelectors and assert identical raw_selected_, member_selections_, fi_ keys."""
    h1 = HybridSelector(use_fe=False, use_tree_member=False, random_state=0).fit(X, y)
    h2 = HybridSelector(use_fe=False, use_tree_member=False, random_state=0).fit(X, y)

    assert h1.raw_selected_ == h2.raw_selected_, f"same-seed raw_selected_ differs: {h1.raw_selected_} vs {h2.raw_selected_}"
    # member_selections_ is a dict name -> list of selected columns; full equality (keys + per-member lists).
    assert set(h1.member_selections_) == set(
        h2.member_selections_
    ), f"member_selections_ keys differ: {set(h1.member_selections_)} vs {set(h2.member_selections_)}"
    assert h1.member_selections_ == h2.member_selections_, "member_selections_ lists are not identical"
    # fi_ keys (the shared permutation-FI columns) must be identical; the n_jobs=-1 LightGBM/RF members make
    # this also a thread-order-stability check -- a nondeterministic reduction would change a key or a list.
    assert set(h1.fi_) == set(h2.fi_), f"fi_ keys differ: {set(h1.fi_) ^ set(h2.fi_)}"
    return h1, h2


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_hybrid_selector_same_seed_deterministic():
    """HybridSelector(use_fe=False, use_tree_member=False, random_state=0) fit twice on the same linear frame
    -> identical raw_selected_, member_selections_, and fi_ keys. The members each run n_jobs=-1 internally,
    so a stable result here is also evidence the parallel reductions are deterministic."""
    X, y = _linear_dataset(n=1200, seed=0)
    h1, _ = _assert_hybrid_deterministic(X, y)
    # Sanity: the raw-only path engineered nothing and the three real members all voted.
    assert h1.n_engineered_ == 0
    assert set(h1.member_selections_) == {"mrmr", "shap", "boruta"}


@pytest.mark.timeout(900)
def test_hybrid_selector_same_seed_deterministic_fast():
    """Fast representative (smaller n) so MLFRAME_FAST=1 still exercises the same-seed HybridSelector
    determinism contract on at least one frame. Still runs the real MRMR+ShapProxied+BorutaShap members
    (the determinism contract is about the composition), so it carries the longer timeout but no @slow."""
    X, y = _linear_dataset(n=600, seed=0)
    _assert_hybrid_deterministic(X, y)


# ===========================================================================
# (c) HybridSelector column-order: documented reproducibility gap -> xfail(strict=False).
# ===========================================================================


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.xfail(
    reason="HybridSelector selection depends on input column order: the composed members' positional "
    "tie-breaks and corr-cluster representative picks are order-sensitive (mirrors the contract suite's "
    "column_order_invariant=False spec for HybridSelector -- known reproducibility gap)",
    strict=False,
)
def test_hybrid_selector_column_order_invariant():
    """Fitting on X vs X[reversed columns] (fresh seeded instances) should select the same RAW feature set.

    It does NOT today -- the composition's order-sensitive tie-breaks change which equivalent columns survive
    (measured: forward keeps {inf_0, noise_1}; reversed keeps {red_2, noise_2}). This xfail documents the gap
    without weakening the contract; flip to strict (or delete the marker) once HybridSelector is made
    column-order invariant.
    """
    X, y = _linear_dataset(n=1200, seed=0)
    rev = list(X.columns)[::-1]
    h_fwd = HybridSelector(use_fe=False, use_tree_member=False, random_state=0).fit(X, y)
    h_rev = HybridSelector(use_fe=False, use_tree_member=False, random_state=0).fit(X[rev], y)
    assert set(h_fwd.raw_selected_) == set(
        h_rev.raw_selected_
    ), f"column reorder changed the selected set: fwd={sorted(h_fwd.raw_selected_)} rev={sorted(h_rev.raw_selected_)}"
