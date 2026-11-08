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


def test_pool_and_single_thread_give_the_same_verdicts(monkeypatch):
    """The candidates are judged on the edge pool; the kept list equals the one-thread result, in the same order."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from mlframe.feature_selection.filters import _fe_pure_form_sides as sides

    rng = np.random.default_rng(0)
    n, k = 4000, 6
    X = pd.DataFrame({f"x{i}": rng.standard_normal(n) for i in range(k)})
    y = (X["x0"].to_numpy() * X["x1"].to_numpy() + 0.3 * rng.standard_normal(n) > 0).astype(int)
    cands = []
    for i in range(12):
        a, b = f"x{i % k}", f"x{(i * 5 + 1) % k}"
        rec = SimpleNamespace(kind="unary_binary", src_names=(a, b), unary_names=("identity", "identity"), unary_preset="all")
        cands.append(SimpleNamespace(name=f"c{i}", recipe=rec, values=X[a].to_numpy() * X[b].to_numpy()))
    monkeypatch.setattr(sides, "_side_values", lambda recipe, X_: [np.asarray(X_[s].to_numpy(), dtype=np.float64) for s in recipe.src_names])

    kept = {}
    for jobs in ("1", "4"):
        monkeypatch.setenv("MLFRAME_EDGE_JOBS", jobs)
        kept[jobs] = [c.name for c in sides.drop_forms_not_beating_their_sides(cands, X, y, classification=True, seed=0)]
    assert kept["1"] == kept["4"]
    assert 0 < len(kept["1"]) < len(cands), "the fixture must keep some candidates and drop others, or the order check proves nothing"


def test_a_side_shared_by_two_candidates_is_scored_once(monkeypatch):
    """Candidates built on the same source column and unary share a side: its CV loss is computed once, and the verdicts equal the uncached ones."""
    from types import SimpleNamespace

    from mlframe.feature_selection.filters import _fe_pure_form_sides as sides

    rng = np.random.default_rng(1)
    n = 2000
    X = pd.DataFrame({"s": rng.standard_normal(n), "u": rng.standard_normal(n), "v": rng.standard_normal(n)})
    y = (X["s"].to_numpy() * X["u"].to_numpy() > 0).astype(int)
    cands = []
    for other in ("u", "v"):
        rec = SimpleNamespace(kind="unary_binary", src_names=("s", other), unary_names=("identity", "identity"), unary_preset="all")
        cands.append(SimpleNamespace(name=f"s*{other}", recipe=rec, values=X["s"].to_numpy() * X[other].to_numpy()))
    monkeypatch.setattr(sides, "_side_values", lambda recipe, X_: [np.asarray(X_[s].to_numpy(), dtype=np.float64) for s in recipe.src_names])
    monkeypatch.setattr("mlframe.feature_selection.filters._edge_pool.map_on_edge_pool", lambda fn, items: [fn(i) for i in items])
    calls = []
    real = sides.cv_loss
    monkeypatch.setattr(sides, "cv_loss", lambda design, *a, **k: (calls.append(1), real(design, *a, **k))[1])
    kept = sides.drop_forms_not_beating_their_sides(cands, X, y, classification=True, seed=0)
    assert len(calls) == 2 + 3  # two forms and the three distinct sides s, u, v (not 2 + 4)
    assert all(c.name in {"s*u", "s*v"} for c in kept)
