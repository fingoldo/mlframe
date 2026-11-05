"""A pure-form candidate must beat its better side under the retention's scorer: a noise operand mixed into a signal operand is dropped."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._fe_pure_form_sides import drop_forms_not_beating_their_sides
from mlframe.feature_selection.filters._usability_aware_selection import UsableCandidate
from mlframe.feature_selection.filters.engineered_recipes import build_unary_binary_recipe


def _cand(X, name, a, b, ua, ub, binary, fn):
    """A candidate built the way the usability pool builds one: a replayable ``unary_binary`` recipe plus its continuous values."""
    recipe = build_unary_binary_recipe(
        name=name, src_a_name=a, src_b_name=b, unary_a_name=ua, unary_b_name=ub, binary_name=binary,
        unary_preset="minimal", binary_preset="minimal", quantization_nbins=None, quantization_method=None, quantization_dtype=None,
    )
    return UsableCandidate(name=name, values=fn(X), mi=0.0, recipe=recipe, src=(a, b), ops=(ua, ub, binary))


def test_noise_mixed_into_a_signal_side_is_dropped_and_a_real_pair_kept():
    """On y = sign(x^2 - 1) ``max(abs(x), sin(noise))`` is no better than ``abs(x)`` alone and goes; ``abs(a) * abs(b)`` for y = sign(|a||b| - 1) stays."""
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({"x": rng.standard_normal(n), "noise": rng.standard_normal(n), "a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    y1 = (X["x"].to_numpy() ** 2 - 1.0 > 0).astype(int)
    noisy = _cand(X, "max(abs(x),sin(noise))", "x", "noise", "abs", "sin", "max", lambda d: np.maximum(np.abs(d["x"]), np.sin(d["noise"])))
    kept = drop_forms_not_beating_their_sides([noisy], X, y1, classification=True, seed=0)
    assert kept == []

    y2 = (np.abs(X["a"].to_numpy()) * np.abs(X["b"].to_numpy()) - 0.6 > 0).astype(int)
    real = _cand(X, "mul(abs(a),abs(b))", "a", "b", "abs", "abs", "mul", lambda d: np.abs(d["a"]) * np.abs(d["b"]))
    kept = drop_forms_not_beating_their_sides([real], X, y2, classification=True, seed=0)
    assert [c.name for c in kept] == ["mul(abs(a),abs(b))"]


def test_raw_passthrough_and_unreplayable_forms_are_kept():
    """A raw passthrough (no recipe) is never dropped."""
    raw = UsableCandidate(name="x", values=np.arange(10.0), mi=0.0)
    assert drop_forms_not_beating_their_sides([raw], pd.DataFrame({"x": np.arange(10.0)}), np.zeros(10, dtype=int), classification=True, seed=0) == [raw]
