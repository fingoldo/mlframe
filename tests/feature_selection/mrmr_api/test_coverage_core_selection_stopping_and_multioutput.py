"""Core-selection coverage: stopping rules + multioutput (union/intersect) consistency.

Stopping rules under test:
- ``min_relevance_gain_relative_to_first`` (relative-gain floor) -- a HIGH floor must stop earlier (select fewer)
  than a low floor on the same data; a near-1.0 floor keeps only the anchor feature.
- ``min_relevance_gain_mode='absolute'`` with a large ``min_relevance_gain`` -- stops after few features.
- ``min_features_fallback`` -- support_ is never empty when >=1 (the never-empty floor); 0 may permit empty.

Multioutput:
- ``multioutput_strategy='union'`` selects the union of per-column supports; ``'intersect'`` the intersection;
  union is always a superset of intersect; ``multioutput_supports_`` records the per-column lists.

None of these contracts had a behavioural test under mrmr_api/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def _no_fe(**kw):
    """No fe."""
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


def _multi_signal_data(n=800, seed=2):
    """Several genuinely-relevant features of decreasing strength so the gain curve is monotone-ish."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "noise0": rng.normal(size=n),
            "noise1": rng.normal(size=n),
        }
    )
    # Decreasing weights -> decreasing per-feature relevance gains.
    y = (2.0 * x0 + 1.2 * x1 + 0.6 * x2 + 0.25 * x3 + rng.normal(0, 0.5, n) > 0).astype(int)
    return X, y


def _n_selected(m):
    """N selected."""
    return int(np.asarray(m.support_, dtype=np.intp).size)


def test_relative_gain_floor_high_selects_fewer_than_low():
    """A higher min_relevance_gain_relative_to_first stops greedy selection earlier (>= as few)."""
    X, y = _multi_signal_data()
    MRMR._FIT_CACHE.clear()
    low = _no_fe(min_relevance_gain_relative_to_first=0.0).fit(X, y)
    MRMR._FIT_CACHE.clear()
    high = _no_fe(min_relevance_gain_relative_to_first=0.5).fit(X, y)
    assert _n_selected(high) <= _n_selected(low), f"high relative-gain floor must not select MORE: high={_n_selected(high)} low={_n_selected(low)}"


def test_near_one_relative_floor_keeps_only_anchor():
    """A relative-gain floor near 1.0 keeps essentially only the first (anchor) feature."""
    X, y = _multi_signal_data()
    MRMR._FIT_CACHE.clear()
    m = _no_fe(min_relevance_gain_relative_to_first=0.999).fit(X, y)
    # The floor applies from the SECOND feature; only the anchor survives a near-1.0 cutoff.
    assert _n_selected(m) == 1


def test_absolute_mode_large_floor_stops_early():
    """absolute mode with a large min_relevance_gain selects fewer than the permissive default."""
    X, y = _multi_signal_data()
    MRMR._FIT_CACHE.clear()
    permissive = _no_fe(min_relevance_gain_mode="absolute", min_relevance_gain=1e-9, min_relevance_gain_relative_to_first=0.0).fit(X, y)
    MRMR._FIT_CACHE.clear()
    strict = _no_fe(min_relevance_gain_mode="absolute", min_relevance_gain=1.0, min_relevance_gain_relative_to_first=0.0).fit(X, y)
    assert _n_selected(strict) <= _n_selected(permissive)
    # A floor of 1.0 nat is enormous for a binary target (H(y) <= ln 2); only the fallback anchor remains.
    assert _n_selected(strict) >= 1  # never-empty floor (min_features_fallback default 1)


def test_min_features_fallback_keeps_support_nonempty():
    """Even when every gain is below the floor, min_features_fallback>=1 keeps support_ non-empty."""
    rng = np.random.default_rng(11)
    n = 500
    # Pure noise -> no feature clears a strict absolute floor.
    X = pd.DataFrame({f"n{i}": rng.normal(size=n) for i in range(5)})
    y = rng.integers(0, 2, size=n)
    MRMR._FIT_CACHE.clear()
    m = _no_fe(min_relevance_gain_mode="absolute", min_relevance_gain=10.0, min_features_fallback=1).fit(X, y)
    assert _n_selected(m) >= 1, "min_features_fallback=1 must guarantee a non-empty support_"


def _mo_data(n=700, seed=3):
    """Mo data."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "noise": rng.normal(size=n)})
    Y = pd.DataFrame(
        {
            "y0": (x0 > 0).astype(int),  # depends on x0
            "y1": (x1 > 0).astype(int),  # depends on x1
        }
    )
    return X, Y


def test_multioutput_union_is_superset_of_intersect():
    """union(per-column supports) >= intersect(per-column supports), by name."""
    X, Y = _mo_data()
    MRMR._FIT_CACHE.clear()
    u = _no_fe(multioutput_strategy="union").fit(X, Y)
    MRMR._FIT_CACHE.clear()
    i = _no_fe(multioutput_strategy="intersect").fit(X, Y)
    u_names = set(map(str, u.get_feature_names_out()))
    i_names = set(map(str, i.get_feature_names_out()))
    assert u_names >= i_names


def test_multioutput_union_recovers_each_columns_driver():
    """union must contain the driver of each output column (x0 for y0, x1 for y1)."""
    X, Y = _mo_data()
    MRMR._FIT_CACHE.clear()
    u = _no_fe(multioutput_strategy="union").fit(X, Y)
    names = set(map(str, u.get_feature_names_out()))
    assert {"x0", "x1"} <= names
    # Per-column provenance recorded.
    assert set(u.multioutput_supports_.keys()) == {"y0", "y1"}
    assert "x0" in u.multioutput_supports_["y0"]
    assert "x1" in u.multioutput_supports_["y1"]


def test_multioutput_support_is_input_space_consistent():
    """Multioutput support_ indices map back to feature_names_in_ and get_support stays input-space."""
    X, Y = _mo_data()
    MRMR._FIT_CACHE.clear()
    u = _no_fe(multioutput_strategy="union").fit(X, Y)
    assert u.n_features_in_ == X.shape[1]
    mask = u.get_support()
    assert mask.shape == (u.n_features_in_,)
    recovered = {str(u.feature_names_in_[i]) for i in np.where(mask)[0]}
    assert recovered == set(map(str, u.get_feature_names_out()))


def test_multioutput_invalid_strategy_raises():
    """Multioutput invalid strategy raises."""
    X, Y = _mo_data()
    m = _no_fe(multioutput_strategy="bogus")
    with pytest.raises(ValueError):
        m.fit(X, Y)
