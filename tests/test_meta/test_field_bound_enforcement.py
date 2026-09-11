"""Every numeric bound and string ``Literal`` a training config declares is enforced at construction.

The failure this catches looks like a checked constraint and is not one: a field declares ``Field(ge=0)`` in a call
shape pydantic's runtime validator never reads, the schema and the IDE tooltip both say "must be >= 0", a caller
passes -1, construction accepts it, and things go wrong two functions deeper. For every field with a ``gt`` /
``ge`` / ``lt`` / ``le`` bound or a string ``Literal``, ``py_ci_shared.pydantic_field_bounds`` constructs the model
with a value that violates it and requires a ``ValidationError``.

The config classes live in sibling modules re-exported from ``mlframe.training.configs``, so the accepted homes are
``configs`` itself and each of those siblings.
"""

from __future__ import annotations

from py_ci_shared.pydantic_field_bounds import assert_field_bounds_enforced, iter_pydantic_models

from mlframe.training import configs as configs_module

_SIBLINGS = (
    "_preprocessing_configs",
    "_model_configs",
    "_training_runtime_configs",
    "_composite_target_discovery_config",
    "_reporting_configs",
    "_configs_base",
    "_feature_selection_config",
)
_ACCEPTED_MODULES = {configs_module.__name__, *(f"{configs_module.__package__}.{name}" for name in _SIBLINGS)}


def test_declared_bounds_and_literals_are_enforced() -> None:
    """Out-of-bounds numbers and out-of-set strings are rejected by every config model that declares them."""
    models = iter_pydantic_models([configs_module], accepted_modules=_ACCEPTED_MODULES)
    assert_field_bounds_enforced(models, require_numeric=True, require_literal=True)
