"""Meta-test — for every ``Literal[...]`` / ``StrEnum`` field on a Pydantic
config in ``mlframe.training.configs``, every enumerable value must appear
at least once in the production corpus.

Catches the failure mode where a new enum variant is added to a config
field but the dispatch code that branches on it isn't updated — the
variant becomes an "accept-the-input-but-do-nothing" sink. Symptoms:
``cfg.strategy="stacking"`` silently falls into the auto path because
no branch matches "stacking".

Heuristic: collect every Literal/Enum-typed field, walk its allowed
values, grep production corpus (excluding configs.py + tests + docs).
A literal whose only mention is the field declaration itself flags.
"""

from __future__ import annotations

import inspect
import re
from enum import Enum
from pathlib import Path
from typing import Literal, get_args, get_origin

import pytest
from pydantic import BaseModel

import mlframe
from mlframe.training import configs as configs_module

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent

# Hard whitelist of (Class.field, value) pairs intentionally accepted-
# but-not-dispatched-on (e.g. listed for forward-compat with a planned
# future strategy).
_KNOWN_UNDISPATCHED: set[tuple[str, str]] = set()


def _consumer_corpus() -> str:
    chunks: list[str] = []
    for py in MLFRAME_DIR.rglob("*.py"):
        if py.resolve() == Path(configs_module.__file__).resolve():
            continue
        if "test" in py.parts or "__pycache__" in py.parts:
            continue
        try:
            chunks.append(py.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(chunks)


def _enumerable_values(annotation) -> list:
    """Extract ``Literal[...]`` member values, walking ``Optional[...]`` /
    ``Union[...]`` wrappers. Returns [] for non-enumerable annotations.
    """
    if get_origin(annotation) is Literal:
        return list(get_args(annotation))
    # Optional[Literal[...]] / Union[Literal[...], None] / etc.
    args = get_args(annotation)
    out: list = []
    for arg in args:
        if get_origin(arg) is Literal:
            out.extend(get_args(arg))
        elif inspect.isclass(arg) and issubclass(arg, Enum):
            out.extend(member.value for member in arg)
    if out:
        return out
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        return [member.value for member in annotation]
    return []


def _enum_literal_fields() -> list[tuple[type[BaseModel], str, list]]:
    """Every Literal/Enum-typed field across every config class."""
    out: list[tuple[type[BaseModel], str, list]] = []
    # ``configs.py`` was split into sibling modules (``_preprocessing_configs``,
    # ``_model_configs``, ``_training_runtime_configs``,
    # ``_composite_target_discovery_config``, ``_reporting_configs``); each
    # config class lives in its sibling and is re-exported from
    # ``mlframe.training.configs``. The exhaustiveness contract applies to the
    # full re-exported surface, so accept classes whose ``__module__`` is
    # either ``configs`` itself or any of those siblings.
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
    for _, obj in inspect.getmembers(configs_module, inspect.isclass):
        if not (issubclass(obj, BaseModel) and obj is not BaseModel):
            continue
        if obj.__module__ not in _accepted_modules:
            continue
        for field_name, info in obj.model_fields.items():
            values = _enumerable_values(info.annotation)
            # Only String-enum / Literal[str, ...] fields (we can't grep
            # for integer literals — too many false positives). Skip
            # numeric enums.
            string_values = [v for v in values if isinstance(v, str)]
            if string_values:
                out.append((obj, field_name, string_values))
    return out


def test_every_enum_value_is_dispatched_on():
    corpus = _consumer_corpus()
    fields = _enum_literal_fields()
    if not fields:
        pytest.skip("no Literal/Enum fields found in configs — nothing to police")

    undispatched: list[str] = []
    total_values = 0
    for cls, field_name, values in fields:
        for value in values:
            total_values += 1
            qualified = f"{cls.__name__}.{field_name}"
            if (qualified, value) in _KNOWN_UNDISPATCHED:
                continue
            # Look for the literal value as a string in the corpus.
            # Quoted ``"value"`` or ``'value'`` — exact match avoids
            # partial collisions (e.g. "auto" vs "automatic").
            quoted = (f'"{value}"', f"'{value}'")
            if any(q in corpus for q in quoted):
                continue
            undispatched.append(f"{qualified} = {value!r}")

    assert total_values >= 5, f"only {total_values} Literal/Enum string values found — extraction broken or no string-enum fields exist."
    if undispatched:
        pytest.fail(
            f"{len(undispatched)} Literal/Enum value(s) declared on a config "
            f"field but never appear as a quoted string in production code. "
            f'Either dispatch on them (``if cfg.strategy == "..."``), OR '
            f"remove the unused variant from the Literal[...], OR whitelist "
            f"in _KNOWN_UNDISPATCHED with reasoning:\n  " + "\n  ".join(undispatched)
        )
