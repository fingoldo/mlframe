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

import typing
from typing import Optional

import pandas as pd
import pytest

from mlframe.feature_selection.filters.engineered_recipes._recipe_core import EngineeredRecipe


def _declared_kinds() -> list[str]:
    """Every member of the ``kind`` Literal, read from the dataclass annotation rather than duplicated here."""
    kinds = list(typing.get_args(typing.get_type_hints(EngineeredRecipe)["kind"]))
    assert len(kinds) > 40, f"suspiciously few kinds parsed ({len(kinds)}) - the Literal format changed"
    return kinds


def _dispatch_failure(kind: str) -> Optional[Exception]:
    """The exception the real dispatcher raises for a payload-less stub recipe of ``kind``, or None when it replays."""
    from mlframe.feature_selection.filters.engineered_recipes._recipe_dispatch import apply_recipe

    recipe = EngineeredRecipe(name=f"probe__{kind}", kind=kind, src_names=("a",))
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
    try:
        apply_recipe(recipe, X)
    except Exception as e:  # a wired branch failing on an empty stub payload is the expected outcome
        return e
    return None


def _is_unknown_kind_error(err: Optional[Exception]) -> bool:
    """True for the dispatcher's own unknown-kind rejection."""
    return isinstance(err, ValueError) and "Unknown recipe kind" in str(err)


def test_unknown_kind_detector_recognises_the_dispatcher_rejection():
    """A kind outside the Literal is rejected with the unknown-kind ValueError, so the per-kind check below can fail."""
    assert _is_unknown_kind_error(_dispatch_failure("no_such_recipe_kind_xyz"))
    assert not _is_unknown_kind_error(None)
    assert not _is_unknown_kind_error(KeyError("x"))


@pytest.mark.parametrize("kind", _declared_kinds())
def test_recipe_kind_reaches_a_dispatch_branch(kind):
    """The dispatcher must not reject this kind as unknown."""
    err = _dispatch_failure(kind)
    assert not _is_unknown_kind_error(err), (
        f"recipe kind {kind!r} is declared but has no branch in the dispatcher chain. fit() will emit it "
        "and transform() will raise on it, so a selector saved after fitting cannot be replayed on new "
        f"data. Wire the existing _apply_*_recipe adapter into _recipe_dispatch.py. Original: {err}"
    )
