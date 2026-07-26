"""Every declared ``EngineeredRecipe`` kind must have a replay branch in the dispatcher.

The dispatcher is a linear ``if recipe.kind == ...`` chain ending in ``raise ValueError("Unknown recipe
kind")``. Adding a new FE family means touching two places - the ``kind`` Literal and the chain - and
nothing tied them together, so six families shipped with a generator and a working ``_apply_*_recipe``
adapter that the chain never called: ``conditional_quantile_rank``, ``ordinal_pattern_te``,
``random_fourier``, ``sir_direction``, ``lof_score``, ``mahalanobis_density``.

The failure mode is the worst shape available: ``fit`` succeeds and reports the engineered columns, then
``transform`` on new data raises. Any caller that fits and predicts in one process on one frame never sees
it; a saved selector reloaded against fresh data dies.

Checked behaviourally rather than by grepping the chain for kind strings: a stub recipe of each kind is
pushed through the real dispatcher, and the only banned outcome is the unknown-kind ValueError. Every other
exception is fine and expected - the stub carries no payload, so a wired branch fails deep inside its own
replay helper, which is exactly the evidence that the branch exists.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from mlframe.feature_selection.filters.engineered_recipes._recipe_core import EngineeredRecipe
from mlframe.feature_selection.filters.engineered_recipes import _recipe_core


def _declared_kinds() -> list[str]:
    """Every string in the ``kind:`` Literal, read from the source of truth rather than duplicated here."""
    src = Path(_recipe_core.__file__).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"kind: Literal\[(.*?)\]", src, re.S)
    assert m, "could not locate the kind Literal in _recipe_core.py"
    kinds = re.findall(r'"([a-z_0-9]+)"', m.group(1))
    assert len(kinds) > 40, f"suspiciously few kinds parsed ({len(kinds)}) - the Literal format changed"
    return kinds


@pytest.mark.parametrize("kind", _declared_kinds())
def test_recipe_kind_reaches_a_dispatch_branch(kind):
    """The dispatcher must not reject this kind as unknown."""
    from mlframe.feature_selection.filters.engineered_recipes._recipe_dispatch import apply_recipe

    recipe = EngineeredRecipe(name=f"probe__{kind}", kind=kind, src_names=("a",))
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})

    try:
        apply_recipe(recipe, X)
    except ValueError as e:
        assert "Unknown recipe kind" not in str(e), (
            f"recipe kind {kind!r} is declared but has no branch in the dispatcher chain. fit() will emit it "
            "and transform() will raise on it, so a selector saved after fitting cannot be replayed on new "
            f"data. Wire the existing _apply_*_recipe adapter into _recipe_dispatch.py. Original: {e}"
        )
    except Exception:
        pass  # a wired branch failing on an empty stub payload is the expected outcome
