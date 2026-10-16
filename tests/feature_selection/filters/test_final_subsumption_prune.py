"""End-of-fit subsumption pass: a column whose target information is already carried by a real compound is dropped, and only that.

The target is y = a**2/b + log(c)*sin(d) + 0.5*w with w independent of everything else. The compound ``u = a**2/b + log(c)*sin(d)`` is the
real compound; a column that re-encodes it is redundant, while ``w`` carries a private term the compound does not span.
"""

import numpy as np

from mlframe.feature_selection.filters._fe_final_subsumption import final_subsumption_prune

N = 4000


def _data(seed=0):
    """Synthetic frame with a compound target and its components."""
    rng = np.random.default_rng(seed)
    a, b, c, d, w = (rng.uniform(0.2, 1.2, N) for _ in range(5))
    u = a**2 / b + np.log(c) * np.sin(d)
    y = u + 0.5 * w + 0.05 * rng.standard_normal(N)
    y_binned = np.digitize(y, np.quantile(y, np.linspace(0.1, 0.9, 9)))
    return {"a": a, "b": b, "c": c, "d": d, "w": w, "u": u, "y": y}, y_binned, y


def test_column_that_reencodes_the_compound_is_dropped():
    """A column that re-encodes the compound term is pruned."""
    data, y_binned, y = _data()
    rng = np.random.default_rng(1)
    candidates = {
        "compound": data["u"] + 0.02 * rng.standard_normal(N),
        "compound_copy": data["u"] + 0.02 * rng.standard_normal(N),
    }
    dropped = final_subsumption_prune(candidates, anchors=["compound"], y_binned=y_binned, y_continuous=y, seed=0)
    assert dropped == ["compound_copy"]


def test_column_with_private_term_is_kept():
    """A column carrying a private term that the compound lacks is kept."""
    data, y_binned, y = _data()
    rng = np.random.default_rng(2)
    candidates = {
        "compound": data["u"] + 0.02 * rng.standard_normal(N),
        "w": data["w"].copy(),
    }
    dropped = final_subsumption_prune(candidates, anchors=["compound"], y_binned=y_binned, y_continuous=y, seed=0)
    assert "w" not in dropped


def test_no_anchor_drops_nothing():
    """With no anchor set, the prune drops no column."""
    data, y_binned, y = _data()
    candidates = {"u": data["u"], "w": data["w"]}
    assert final_subsumption_prune(candidates, anchors=[], y_binned=y_binned, y_continuous=y, seed=0) == []


def test_anchor_is_never_dropped():
    """An anchor column is never pruned, whatever the other columns say."""
    data, y_binned, y = _data()
    candidates = {"compound": data["u"], "other": data["w"]}
    dropped = final_subsumption_prune(candidates, anchors=["compound"], y_binned=y_binned, y_continuous=y, seed=0)
    assert "compound" not in dropped


def test_verdict_does_not_depend_on_candidate_order():
    """The prune verdict is invariant to the order of candidate columns."""
    data, y_binned, y = _data()
    rng = np.random.default_rng(3)
    candidates = {
        "compound": data["u"] + 0.02 * rng.standard_normal(N),
        "compound_copy": data["u"] + 0.02 * rng.standard_normal(N),
        "w": data["w"].copy(),
    }
    forward = final_subsumption_prune(candidates, ["compound"], y_binned, y, seed=0)
    reverse = final_subsumption_prune(dict(reversed(list(candidates.items()))), ["compound"], y_binned, y, seed=0)
    assert set(forward) == set(reverse)
