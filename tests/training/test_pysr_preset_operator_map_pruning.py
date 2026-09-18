"""A caller overriding the PySR operator lists must not inherit preset per-operator maps for absent operators.

PySR rejects ``complexity_of_operators`` / ``nested_constraints`` keyed on operators that are not in the operator
lists, and the in-suite fit swallowed that as best-effort, so ``pysr_params={"unary_operators": ["square"], ...}``
silently added no ``pysr__`` columns. Pure-python: needs no Julia runtime.
"""

from mlframe.feature_engineering.pysr_operators import get_preset_kwargs
from mlframe.training.pipeline._pipeline_extensions_pysr import _prune_preset_operator_maps


def test_override_operator_lists_prunes_preset_maps():
    """Overridden lists drop every preset map entry naming an absent operator."""
    params = dict(get_preset_kwargs("standard"))
    user = {"binary_operators": ["+", "-", "*"], "unary_operators": ["square"]}
    params.update(user)
    _prune_preset_operator_maps(params, user)
    assert params["complexity_of_operators"] == {"square": 1}
    assert params["nested_constraints"] is None


def test_default_preset_maps_untouched_and_explicit_maps_kept():
    """The unmodified preset keeps its maps; a map the caller passed explicitly is never pruned."""
    params = dict(get_preset_kwargs("standard"))
    before = (dict(params["complexity_of_operators"]), dict(params["nested_constraints"]))
    _prune_preset_operator_maps(params, {})
    assert (params["complexity_of_operators"], params["nested_constraints"]) == before

    explicit = {"unary_operators": ["square"], "complexity_of_operators": {"exp": 3}}
    params = dict(get_preset_kwargs("standard"))
    params.update(explicit)
    _prune_preset_operator_maps(params, explicit)
    assert params["complexity_of_operators"] == {"exp": 3}
