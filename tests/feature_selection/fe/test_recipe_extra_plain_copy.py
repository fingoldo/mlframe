"""The recipe ``extra`` copy made at construction equals ``copy.deepcopy`` for the plain containers it holds and keeps the recipe independent of its source."""
import copy
from collections import OrderedDict

import numpy as np

from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe
from mlframe.feature_selection.filters.engineered_recipes._recipe_core import _copy_plain


def _same(a, b):
    """Structural equality that treats arrays by value and keeps container types."""
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, dict):
        return type(a) is type(b) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(p, q) for p, q in zip(a, b))
    return a == b


def _extra():
    """A nested extra of the kinds recipes carry: per-group edge lists, arrays, tuples, numpy scalars, an OrderedDict and a set."""
    rng = np.random.default_rng(0)
    return {
        "edges_by_group": {str(g): [float(x) for x in np.sort(rng.normal(size=9))] for g in range(40)},
        "table": np.arange(6, dtype=np.int64).reshape(2, 3),
        "pair": (1, [2, 3], {"k": np.float64(2.0)}),
        "ordered": OrderedDict(x=[1, 2]),
        "labels": {1, 2, 3},
        "name": "s",
        "none": None,
    }


def test_plain_copy_equals_deepcopy():
    """Every kind of value in the structure comes out equal to what ``copy.deepcopy`` returns, subclass and set included."""
    source = _extra()
    assert _same(_copy_plain(source, {}), copy.deepcopy(source))


def test_plain_copy_is_independent_and_keeps_aliasing():
    """Mutating the copy leaves the source alone; a list met twice is copied once."""
    shared = [1.0, 2.0]
    source = {"x": shared, "y": shared, "z": (shared,)}
    out = _copy_plain(source, {})
    out["x"][0] = -1.0
    assert shared == [1.0, 2.0]
    assert out["x"] is out["y"] and out["z"][0] is out["x"]


def test_recipe_does_not_follow_later_changes_to_its_source():
    """A recipe built from a dict keeps its values when the caller changes that dict afterwards, and its extra is read-only."""
    source = {"cuts": [1.0, 2.0, 3.0], "table": np.arange(4)}
    recipe = EngineeredRecipe(kind="grouped_quantile", name="r", src_names=("a",), extra=source)
    source["cuts"].append(99.0)
    source["table"][0] = 7
    assert recipe.extra["cuts"] == [1.0, 2.0, 3.0] and recipe.extra["table"][0] == 0
    try:
        recipe.extra["new"] = 1
    except TypeError:
        return
    raise AssertionError("the recipe extra accepted a write")
