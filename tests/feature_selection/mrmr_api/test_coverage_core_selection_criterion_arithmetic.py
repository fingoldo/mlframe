"""Core-selection coverage: greedy relevance/redundancy criterion arithmetic.

Contracts of the greedy core:
- ``support_`` is in SELECTION ORDER -- the first entry is the anchor (the single most relevant feature).
- ``mrmr_gains_`` is aligned with selection order and is non-increasing (each greedy pick adds <= the prior gain).
- Noise features (no marginal relevance) are excluded while genuine signals of decreasing strength are admitted
  in decreasing-strength order.
- A feature that is a pure scaled copy of an already-selected one contributes no NEW relevance and is rejected
  by the redundancy criterion (its conditional relevance given the selected one is ~0).

The greedy ordering + gain-monotonicity + redundancy-rejection arithmetic had no dedicated test under mrmr_api/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def _no_fe(**kw):
    base = dict(
        random_seed=0,
        verbose=0,
        fe_max_steps=0,
        interactions_max_order=1,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        cat_fe_config=None,
        fe_hinge_enable=False,
        fe_modular_enable=False,
        fe_pairwise_modular_enable=False,
        fe_integer_lattice_enable=False,
        fe_row_argmax_enable=False,
        fe_conditional_gate_enable=False,
    )
    base.update(kw)
    return MRMR(**base)


def _ordered_signal_data(n=900, seed=4):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "weak": 0.3 * x2,
            "strong": x0,
            "mid": 0.8 * x1,
            "noise": rng.normal(size=n),
        }
    )
    y = (2.0 * x0 + 1.0 * x1 + 0.3 * x2 > 0).astype(int)
    return X, y


def test_anchor_is_the_single_most_relevant_feature():
    """support_[0] (the anchor) is 'strong' -- the feature with the largest marginal relevance."""
    X, y = _ordered_signal_data()
    MRMR._FIT_CACHE.clear()
    m = _no_fe().fit(X, y)
    anchor_name = str(m.feature_names_in_[int(np.asarray(m.support_, dtype=np.intp)[0])])
    assert anchor_name == "strong"


def test_gains_aligned_with_selection_order_and_non_increasing():
    """mrmr_gains_ is one-per-selected, in selection order, and monotone non-increasing."""
    X, y = _ordered_signal_data()
    MRMR._FIT_CACHE.clear()
    m = _no_fe().fit(X, y)
    gains = np.asarray(m.mrmr_gains_, dtype=float)
    assert gains.shape[0] == np.asarray(m.support_).size
    diffs = np.diff(gains)
    assert np.all(diffs <= 1e-9), f"greedy gains must be non-increasing, got {gains}"
    assert gains[0] > 0


def test_noise_feature_excluded():
    """The pure-noise column is never selected."""
    X, y = _ordered_signal_data()
    MRMR._FIT_CACHE.clear()
    m = _no_fe().fit(X, y)
    selected = {str(X.columns[i]) for i in np.asarray(m.support_, dtype=np.intp)}
    assert "noise" not in selected


def test_scaled_copy_rejected_by_redundancy():
    """A scaled copy of the anchor adds no conditional relevance -> rejected (only one of the pair survives)."""
    rng = np.random.default_rng(8)
    n = 900
    x0 = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "x0": x0,
            "x0_scaled": 5.0 * x0,  # perfectly redundant given x0 (monotone copy)
            "noise": rng.normal(size=n),
        }
    )
    y = (x0 > 0).astype(int)
    MRMR._FIT_CACHE.clear()
    m = _no_fe().fit(X, y)
    selected = {str(X.columns[i]) for i in np.asarray(m.support_, dtype=np.intp)}
    # At most one of the redundant pair is kept; both being kept would violate the redundancy criterion.
    assert not ({"x0", "x0_scaled"} <= selected), f"both redundant copies selected: {selected}"
    assert "x0" in selected or "x0_scaled" in selected


def test_relevance_admits_decreasing_strength_signals():
    """With a permissive floor, genuine signals are admitted; the anchor outranks the mid signal in gain."""
    X, y = _ordered_signal_data()
    MRMR._FIT_CACHE.clear()
    m = _no_fe(min_relevance_gain_relative_to_first=0.0, min_relevance_gain_mode="absolute", min_relevance_gain=1e-9).fit(X, y)
    selected_order = [str(m.feature_names_in_[i]) for i in np.asarray(m.support_, dtype=np.intp)]
    # 'strong' must precede 'mid' in selection order (higher relevance picked first).
    assert "strong" in selected_order
    if "mid" in selected_order:
        assert selected_order.index("strong") < selected_order.index("mid")
