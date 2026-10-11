"""Generic randomizer for config fields a named FuzzCombo axis does not already drive.

Hand-authoring a dedicated ``*_cfg`` axis for every one of the ~250 bool / bounded-numeric / Literal /
Enum fields across the training config surface does not scale (``CompositeTargetDiscoveryConfig`` alone
has 179 fields). Those simple scalar types have a value space fully derivable from the field's own
annotation and pydantic constraints, with no domain knowledge needed -- so :func:`randomize_scalar_fields`
flips/resamples every one of them that a combo axis left at its default, seeded off the combo so runs stay
reproducible.

Deliberately conservative about what it touches: a column name, file path, callable or nested sub-config
has a value space this module cannot know is even valid (a random string is not a real "lever" test of a
``time_column`` field -- it just trips an unrelated "column not found" error, testing nothing). Those stay
untouched here; see ``test_fuzz_config_field_coverage.py`` for the explicit, named list of fields this
leaves uncovered and why.
"""

from __future__ import annotations

import enum
import typing
from typing import Any

import numpy as np
from pydantic import BaseModel

__all__ = ["randomize_scalar_fields", "GENERICALLY_FUZZABLE_SHAPE"]


def _is_literal(ann: Any) -> bool:
    """True when ``ann`` is a ``typing.Literal[...]`` annotation."""
    return typing.get_origin(ann) is typing.Literal


def _unwrap_optional(ann: Any) -> tuple[Any, bool]:
    """``(inner_type, True)`` for ``Optional[X]`` / ``X | None``, else ``(ann, False)``."""
    if typing.get_origin(ann) is typing.Union:
        inner = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(inner) == 1:
            return inner[0], True
    return ann, False


def _union_choices(ann: Any) -> "list[Any] | None":
    """Flattened finite value set for a ``Union`` whose every member is ``bool`` or a ``Literal`` arm (e.g. ``bool | Literal['auto']``, the ``tune_decision_threshold`` / ``async_render`` shape) -- ``None`` when any member isn't one of those (a genuinely open type like ``bool | str``)."""
    if typing.get_origin(ann) is not typing.Union:
        return None
    choices: list[Any] = []
    for arg in typing.get_args(ann):
        if arg is type(None):
            continue
        if arg is bool:
            choices.extend([True, False])
        elif _is_literal(arg):
            choices.extend(typing.get_args(arg))
        else:
            return None
    return choices or None


def _numeric_bounds(metadata: list, kind: type, current: float) -> tuple[float, float]:
    """``(lo, hi)`` from pydantic ``Ge``/``Le``/``Gt``/``Lt`` constraints, falling back to a +-50% jitter band around ``current`` when unconstrained (never a fixed global range, since a probability-like 0..1 field and a row-count field need very different scales)."""
    lo, hi = None, None
    for m in metadata:
        if hasattr(m, "ge"):
            lo = m.ge
        elif hasattr(m, "gt"):
            lo = m.gt + (1 if kind is int else 1e-9)
        if hasattr(m, "le"):
            hi = m.le
        elif hasattr(m, "lt"):
            hi = m.lt - (1 if kind is int else 1e-9)
    if lo is None:
        lo = 0.0 if current >= 0 else current * 1.5
    if hi is None:
        hi = max(current * 1.5, current + 1, 1.0)
    if hi < lo:
        hi = lo
    return lo, hi


def GENERICALLY_FUZZABLE_SHAPE(annotation: Any) -> bool:
    """True when :func:`randomize_scalar_fields` knows how to vary a field of this annotation (used by the coverage meta-test to classify a gap as "closed generically" vs "needs its own axis / is a named, justified skip")."""
    ann, _opt = _unwrap_optional(annotation)
    if ann is bool:
        return True
    if _is_literal(ann):
        return True
    if isinstance(ann, type) and issubclass(ann, enum.Enum):
        return True
    if ann in (int, float):
        return True
    return _union_choices(ann) is not None


def randomize_scalar_fields(model: BaseModel, rng: np.random.Generator, skip: frozenset = frozenset()) -> BaseModel:
    """Return a copy of ``model`` with every bool / bounded-numeric / Literal / Enum field that is NOT already
    explicitly set (``model_fields_set`` -- i.e. not already driven by a real combo axis) and not in ``skip``
    resampled to a value seeded from ``rng``. Fields of any other shape (str, ``Any``, sequences, dicts,
    callables, nested models) pass through unchanged -- see the module docstring for why."""
    updates: dict[str, Any] = {}
    for name, info in type(model).model_fields.items():
        if name in skip or name in model.model_fields_set:
            continue
        ann, is_opt = _unwrap_optional(info.annotation)
        current = getattr(model, name)
        if ann is bool:
            new_val: Any = rng.choice([True, False, None]) if is_opt else (not current)
        elif _is_literal(ann):
            choices = [c for c in typing.get_args(ann) if c != current]
            if not choices:
                continue
            new_val = rng.choice(choices)
        elif isinstance(ann, type) and issubclass(ann, enum.Enum):
            choices = [m for m in ann if m != current]
            if not choices:
                continue
            new_val = rng.choice(choices)
        elif ann in (int, float):
            lo, hi = _numeric_bounds(info.metadata, ann, float(current) if current is not None else 0.0)
            new_val = int(rng.integers(int(lo), int(hi) + 1)) if ann is int else float(rng.uniform(lo, hi))
            if is_opt and rng.random() < 0.2:
                new_val = None
        elif (union_choices := _union_choices(ann)) is not None:
            choices = [c for c in union_choices if c != current]
            if not choices:
                continue
            new_val = rng.choice(choices)
        else:
            continue
        updates[name] = new_val
    return model.model_copy(update=updates) if updates else model
