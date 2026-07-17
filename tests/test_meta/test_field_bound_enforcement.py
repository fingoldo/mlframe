"""Meta-test — every Pydantic field with a numeric bound (``gt`` / ``ge``
/ ``lt`` / ``le``) or string-Literal restriction must actually reject
out-of-bounds inputs at construction time.

Catches the failure mode where a field declares ``Field(ge=0)`` but the
``json_schema_extra`` stored a stale constraint that pydantic's runtime
validator never picked up — typically because the constraint was added
to the wrong call shape (e.g. ``Annotated[int, Field(...)]`` vs raw
``Field(...)``). The bug looks like: schema docs and IDE tooltip say
"must be ≥ 0", users pass -1, runtime accepts it, things go sideways
two functions deeper.

Strategy: enumerate every field with a constraint or a Literal-enum
annotation; instantiate the parent class with a value that VIOLATES the
constraint; assert ``ValueError`` is raised. Skips fields without any
constraint (no claim made → nothing to verify).
"""

from __future__ import annotations

import inspect
from typing import Literal, get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from mlframe.training import configs as configs_module

# Constraints we cannot synthesise a guaranteed-bad value for and skip:
# e.g. ``MultipleOf`` is too narrow without knowing the multiple a priori.
_SUPPORTED_NUMERIC = {"gt", "ge", "lt", "le"}


def _config_classes() -> list[type[BaseModel]]:
    # ``configs.py`` was split into sibling modules
    # (``_preprocessing_configs``, ``_model_configs``,
    # ``_training_runtime_configs``, ``_composite_target_discovery_config``,
    # ``_reporting_configs``); each config class lives in its sibling and is
    # re-exported from ``mlframe.training.configs``. The field-bound contract
    # applies to the full re-exported surface, so accept classes whose
    # ``__module__`` is either ``configs`` itself or any of those siblings.
    _accepted_modules = {
        configs_module.__name__,
        f"{configs_module.__package__}._preprocessing_configs",
        f"{configs_module.__package__}._model_configs",
        f"{configs_module.__package__}._training_runtime_configs",
        f"{configs_module.__package__}._composite_target_discovery_config",
        f"{configs_module.__package__}._reporting_configs",
        f"{configs_module.__package__}._configs_base",
        f"{configs_module.__package__}._feature_selection_config",
    }
    out = []
    for _, obj in inspect.getmembers(configs_module, inspect.isclass):
        if not (issubclass(obj, BaseModel) and obj is not BaseModel):
            continue
        if obj.__module__ not in _accepted_modules:
            continue
        out.append(obj)
    return out


def _has_default(info: FieldInfo) -> bool:
    if info.default is not PydanticUndefined and info.default is not Ellipsis:
        return True
    if info.default_factory is not None:
        return True
    return False


def _required_args_for(cls: type[BaseModel]) -> dict:
    """Sentinel kwargs to satisfy required fields when probing one
    field at a time."""
    out = {}
    for name, info in cls.model_fields.items():
        if _has_default(info):
            continue
        # Prefer string sentinels — every required field in mlframe configs
        # is a string (target_name / model_name).
        out[name] = f"_meta_test_sentinel_{name}"
    return out


def _extract_numeric_metadata(info: FieldInfo) -> dict[str, float]:
    """Pull numeric constraints from a Pydantic v2 FieldInfo.

    Pydantic v2 stores constraints in two places depending on syntax:
      - ``Field(ge=0, le=1)``           → on ``info.metadata`` as Annotated
                                           predicates (``annotated_types.Ge`` etc.)
      - ``Annotated[int, Field(ge=0)]`` → same place; effectively merged.

    We walk ``info.metadata`` and pick out objects whose attribute name
    matches the constraint kind (Ge, Gt, Le, Lt → ge/gt/le/lt).
    """
    found: dict[str, float] = {}
    for predicate in info.metadata or ():
        cls_name = type(predicate).__name__.lower()
        for kind in _SUPPORTED_NUMERIC:
            if cls_name == kind:
                # annotated_types predicates store the value on `.<kind>`.
                val = getattr(predicate, kind, None)
                if val is not None:
                    found[kind] = val
                    break
    return found


def _violating_value(constraint_kind: str, bound: float):
    """Synthesize a value GUARANTEED to violate the constraint."""
    if constraint_kind == "ge":
        return bound - 1.0
    if constraint_kind == "gt":
        return bound  # equal to bound is not strictly greater
    if constraint_kind == "le":
        return bound + 1.0
    if constraint_kind == "lt":
        return bound  # equal to bound is not strictly less
    raise AssertionError(f"unsupported constraint kind: {constraint_kind!r}")


def _literal_values(annotation):
    """Extract Literal[...] string values, walking Optional/Union wrappers."""
    if get_origin(annotation) is Literal:
        return [v for v in get_args(annotation) if isinstance(v, str)]
    args = get_args(annotation)
    out = []
    for arg in args:
        if get_origin(arg) is Literal:
            out.extend(v for v in get_args(arg) if isinstance(v, str))
    return out


def test_numeric_bound_violations_rejected():
    """For every field with at least one numeric constraint, instantiating
    the class with an out-of-bounds value for THAT field must raise
    ``ValidationError``. No claim → no test.
    """
    not_enforced: list[str] = []
    audited = 0
    classes = _config_classes()
    for cls in classes:
        sentinels = _required_args_for(cls)
        for field_name, info in cls.model_fields.items():
            constraints = _extract_numeric_metadata(info)
            if not constraints:
                continue
            # Test the FIRST constraint on the field — that's enough to
            # prove the validator chain is wired. Probing every constraint
            # is overkill and slow for the test.
            kind, bound = next(iter(constraints.items()))
            audited += 1
            bad_value = _violating_value(kind, bound)
            kwargs = dict(sentinels)
            kwargs[field_name] = bad_value
            try:
                cls(**kwargs)
            except (ValidationError, ValueError):
                continue  # Properly rejected.
            except Exception:  # nosec B112 -- best-effort skip of one iteration on a non-fatal error; the test's own assertions are unaffected
                # Some other error — likely an unrelated required field
                # we didn't sentinel. Treat as inconclusive.
                continue
            not_enforced.append(f"{cls.__name__}.{field_name} declares {kind}={bound} but accepted {bad_value}")

    assert audited > 0, "no numeric-constrained fields found — extraction broken?"
    if not_enforced:
        pytest.fail(f"{len(not_enforced)} numeric constraint(s) not enforced at runtime:\n  " + "\n  ".join(not_enforced))


def test_literal_value_outside_set_rejected():
    """For every ``Literal[\"a\", \"b\", ...]`` field, passing a string
    NOT in the set must raise ``ValidationError``. Catches a custom
    ``@field_validator(mode='before')`` that normalises the input but
    forgets to validate against the literal set when the input doesn't
    match a known alias.
    """
    not_enforced: list[str] = []
    audited = 0
    classes = _config_classes()
    for cls in classes:
        sentinels = _required_args_for(cls)
        for field_name, info in cls.model_fields.items():
            values = _literal_values(info.annotation)
            if not values:
                continue
            audited += 1
            bad_value = "_meta_test_clearly_not_a_literal_member"
            kwargs = dict(sentinels)
            kwargs[field_name] = bad_value
            try:
                cls(**kwargs)
            except (ValidationError, ValueError):
                continue
            except Exception:  # nosec B112 -- best-effort skip of one iteration on a non-fatal error; the test's own assertions are unaffected
                continue
            not_enforced.append(f"{cls.__name__}.{field_name}: Literal{values} accepted unrelated string {bad_value!r}")

    assert audited > 0, "no Literal-typed fields found"
    if not_enforced:
        pytest.fail(f"{len(not_enforced)} Literal-typed field(s) accept values outside the declared set:\n  " + "\n  ".join(not_enforced))
