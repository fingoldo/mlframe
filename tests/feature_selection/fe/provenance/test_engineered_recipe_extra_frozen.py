"""Wave 9.1 loop-iter-49 regression: ``EngineeredRecipe.extra`` must
be truly immutable, not just attribute-rebind-blocked.

Pre-fix at ``engineered_recipes.py:133`` the class was
``frozen=True, eq=False`` with ``extra: dict``. ``frozen=True``
blocks ``recipe.extra = {}`` rebind but NOT in-place mutation of the
dict itself. Four failure modes were possible:

H.1 - caller pops a required key from the source dict after passing
      it to the recipe; the recipe's ``extra`` lost the key too
      because it was the SAME dict.
H.2 - ``recipe.extra['cell_means'][:] = -999`` silently corrupted
      every subsequent ``apply_recipe`` replay.
H.3 - hash-eq invariant violated for any recipe in a set/dict-key:
      hash (``(kind, name)``) stays the same while ``__eq__`` flips
      with content after mutation.
H.4 - cache poisoning when recipe used as dict key.

Fix at engineered_recipes.py:135 (``__post_init__``):
- Deep-copy the input dict (severs caller-held reference).
- Freeze every ndarray inside it (``flags.writeable = False``).
- Wrap in ``MappingProxyType`` (read-only view; ``extra['x'] = v``
  raises TypeError).
- ``__getstate__`` / ``__setstate__`` for pickle round-trip
  (MappingProxyType isn't pickle-friendly directly).
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest


def test_extra_dict_assignment_raises():
    """``recipe.extra['new_key'] = 'phantom'`` must raise TypeError
    via MappingProxyType.
    """
    from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe

    r = EngineeredRecipe(
        name="x",
        kind="factorize",
        src_names=("a",),
        extra={"lookup": np.array([0.1, 0.5, 0.9])},
    )
    with pytest.raises(TypeError):
        r.extra["phantom"] = 1


def test_extra_ndarray_inplace_mutation_raises():
    """``recipe.extra['lookup'][:] = -999`` must raise ValueError
    because the ndarray is frozen.
    """
    from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe

    r = EngineeredRecipe(
        name="x",
        kind="factorize",
        src_names=("a",),
        extra={"lookup": np.array([0.1, 0.5, 0.9])},
    )
    with pytest.raises(ValueError, match="read-only"):
        r.extra["lookup"][:] = -999


def test_caller_mutating_source_dict_does_not_affect_recipe():
    """H.1 contract: the recipe deep-copies the input dict so caller
    pops on the source don't propagate.
    """
    from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe

    extra_payload = {"lookup": np.array([0.1, 0.5, 0.9])}
    r = EngineeredRecipe(
        name="x",
        kind="factorize",
        src_names=("a",),
        extra=extra_payload,
    )
    extra_payload.pop("lookup")
    assert "lookup" in r.extra


def test_pickle_round_trip_preserves_frozen_state():
    """After pickle round-trip the unpickled recipe must STILL be
    frozen - tests __getstate__/__setstate__ wiring.
    """
    from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe

    r = EngineeredRecipe(
        name="x",
        kind="factorize",
        src_names=("a",),
        extra={"lookup": np.array([0.1, 0.5, 0.9])},
    )
    r2 = pickle.loads(pickle.dumps(r))
    # Equality preserved.
    assert r == r2
    # Frozen state preserved.
    with pytest.raises(ValueError, match="read-only"):
        r2.extra["lookup"][:] = 0
    with pytest.raises(TypeError):
        r2.extra["new"] = 1


def test_deepcopy_round_trip_preserves_frozen_state():
    """Deepcopy must also re-freeze."""
    from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe

    r = EngineeredRecipe(
        name="x",
        kind="factorize",
        src_names=("a",),
        extra={"lookup": np.array([0.1, 0.5, 0.9])},
    )
    r2 = copy.deepcopy(r)
    assert r == r2
    with pytest.raises(ValueError, match="read-only"):
        r2.extra["lookup"][:] = 0


def test_clean_construction_still_works():
    """Negative control: building and using a recipe without mutation
    must work end-to-end.
    """
    from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe

    r = EngineeredRecipe(
        name="x",
        kind="factorize",
        src_names=("a", "b"),
        factorize_nbins=(3, 4),
        extra={"chain_lookups": [np.array([0, 1, 2]), np.array([0, 1])]},
    )
    # Read access still works.
    assert "chain_lookups" in r.extra
    assert len(r.extra["chain_lookups"]) == 2
