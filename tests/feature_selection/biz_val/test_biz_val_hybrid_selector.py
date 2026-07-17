"""biz_value coverage for HybridSelector decision-influencing capacity knobs.

The shared-FI / cluster-vote combine rule (``fi_guard``, ``vote``, ``expand_clusters``) is already pinned by the
injected-state unit tests in ``test_hybrid_selector.py``. This file covers the TREE-MEMBER capacity knob that those
state-injection tests cannot reach because it only matters during a real fit: ``tree_max_depth``.

``tree_max_depth`` controls the depth of the cheap LightGBM the tree member fits on raw X to propose co-occurrence
operand pairs (``_tree_signals``). A pair is detected only when the GBM branches on BOTH operands WITHIN one tree, so
a stump (depth=1) can never co-occur two features and proposes ZERO pairs -- the interaction product is never
engineered. depth>=2 lets the tree branch on the interaction operands together, recovering the products. On a pure
sign(a*b) interaction bed (operands with ~0 marginal signal, the regime MRMR's marginal-MI greedy structurally
misses) this is the difference between engineering the recovering product columns and engineering nothing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.hybrid_selector import HybridSelector


pytestmark = pytest.mark.timeout(60)  # untimed biz_val real-fit tier: surface a hang fast (global --timeout=600 is a coarse backstop)


def _interaction_bed(seed: int = 0, n: int = 1000, p_noise: int = 4):
    """y = Bernoulli(sigmoid(3 * a*b)): a pure two-way interaction. a, b each carry ~0 MARGINAL signal (their
    individual MI with y is near zero), so only a model that branches on both together -- or engineers their
    product -- recovers the signal. p_noise pure-noise columns share the frame."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-3.0 * (a * b)))).astype(int)
    cols = {"inf_a": a, "inf_b": b}
    for j in range(p_noise):
        cols[f"noise_{j}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), pd.Series(y, name="y")


@pytest.mark.slow
def test_biz_val_hybrid_tree_max_depth_recovers_interaction_products():
    """biz_value: ``tree_max_depth`` is the capacity that lets the tree member co-occur interaction operands and
    engineer their recovering product columns.

    On the pure sign(a*b) bed with use_fe=True:
      - tree_max_depth=1 (stumps): the GBM cannot branch on two features within one tree -> ZERO co-occurrence pairs
        -> ZERO engineered columns (the interaction is never recovered).
      - tree_max_depth=3: the GBM branches on inf_a + inf_b together -> co-occurrence pairs proposed -> the synergy
        gate admits the recovering products.
    Measured (seed=0, n=1000): depth=1 -> 0 pairs / 0 engineered; depth=3 -> 14 pairs / 11 engineered. Assert the
    stump engineers NOTHING and the deeper tree engineers a non-trivial set including a product on the true operands."""
    X, y = _interaction_bed(seed=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stump = HybridSelector(
            use_mrmr=True,
            use_fe=True,
            use_tree_member=True,
            tree_max_depth=1,
            tree_n_estimators=40,
            tree_cooccur_pairs=6,
            fe_max_steps=0,
            random_state=0,
        ).fit(X, y)
        deep = HybridSelector(
            use_mrmr=True,
            use_fe=True,
            use_tree_member=True,
            tree_max_depth=3,
            tree_n_estimators=40,
            tree_cooccur_pairs=6,
            fe_max_steps=0,
            random_state=0,
        ).fit(X, y)

    stump_pairs = getattr(stump, "_tree_prod_pairs_", []) or []
    deep_pairs = getattr(deep, "_tree_prod_pairs_", []) or []

    assert len(stump_pairs) == 0, f"a depth-1 stump cannot co-occur two features in one tree -> 0 co-occurrence pairs; got {len(stump_pairs)}"
    assert stump.n_engineered_ == 0, f"with no co-occurrence pairs the stump must engineer 0 product columns; got {stump.n_engineered_}"

    assert len(deep_pairs) >= 4, f"depth=3 must propose the interaction co-occurrence pairs (measured 14); got {len(deep_pairs)}"
    assert deep.n_engineered_ >= 4, f"depth=3 must engineer the recovering interaction products (measured 11); got {deep.n_engineered_}"
    # The true operand pair (inf_a, inf_b) must be among the proposed co-occurrence pairs at depth=3.
    deep_pair_sets = {frozenset(p) for p in deep_pairs}
    assert frozenset({"inf_a", "inf_b"}) in deep_pair_sets, f"depth=3 must co-occur the true interaction operands (inf_a, inf_b); got pairs {deep_pairs}"


def test_biz_val_hybrid_tree_max_depth_recovers_interaction_products_fast():
    """Fast representative (smaller n, fewer noise cols) so MLFRAME_FAST=1 still exercises the stump-vs-deep flip:
    depth=1 proposes ZERO tree co-occurrence products, depth=3 proposes the interaction products.

    The contract ``tree_max_depth`` controls is the TREE member's co-occurrence proposals (``_tree_prod_pairs_``):
    a stump cannot branch on two features within one tree, so it proposes none. ``n_engineered_`` is NOT the right
    sensor here -- it ALSO counts the MRMR member's synergy-bootstrap product, which fires on this pure-interaction
    bed independently of tree depth (even at ``fe_max_steps=0``, since the synergy bootstrap is not an FE step). So
    the stump's ``n_engineered_`` is 1 (the MRMR Hermite product), not 0. Mirror the slow sibling and assert on the
    tree-member quantity that ``tree_max_depth`` actually governs."""
    X, y = _interaction_bed(seed=0, n=600, p_noise=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stump = HybridSelector(
            use_mrmr=True,
            use_fe=True,
            use_tree_member=True,
            tree_max_depth=1,
            tree_n_estimators=30,
            tree_cooccur_pairs=4,
            fe_max_steps=0,
            random_state=0,
        ).fit(X, y)
        deep = HybridSelector(
            use_mrmr=True,
            use_fe=True,
            use_tree_member=True,
            tree_max_depth=3,
            tree_n_estimators=30,
            tree_cooccur_pairs=4,
            fe_max_steps=0,
            random_state=0,
        ).fit(X, y)
    stump_pairs = getattr(stump, "_tree_prod_pairs_", []) or []
    deep_pairs = getattr(deep, "_tree_prod_pairs_", []) or []
    assert len(stump_pairs) == 0, f"a depth-1 stump cannot co-occur two features -> 0 tree products; got {len(stump_pairs)}"
    assert len(deep_pairs) >= 1, f"depth=3 must propose >=1 tree co-occurrence interaction product; got {len(deep_pairs)}"
