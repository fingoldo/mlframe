"""Meta-test — fields whose docstring promises a normalisation
("Case-insensitive, normalized to lowercase") must actually normalise
mixed-case input at runtime.

Catches the failure mode where a docstring describes a behaviour that
never made it into code: users follow the docstring (pass mixed-case
input), the field accepts the value as-is, downstream code that branches
on ``cfg.task_type == "GPU"`` silently mismatches against ``"Gpu"``.

Heuristic — look for these phrases in per-field docstrings:
  * ``"Case-insensitive"`` or ``"case insensitive"``
  * ``"normalized to <something>"`` / ``"normalized to <something>case"``

For each match we construct the class with the field set to a mixed-case
sentinel and assert the normalised value on the instance matches the
declared case (lower / upper). Behavioural test — no source-grep.
"""

from __future__ import annotations

import inspect
import re
import typing

import pytest
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from mlframe.training import configs as configs_module

# Phrases that imply a normalising validator + the target case form.
_LOWER_PROMISES = (
    "case-insensitive",
    "case insensitive",
    "normalized to lowercase",
    "normalized to lower",
    "normalised to lowercase",
)
_UPPER_PROMISES = (
    "normalized to uppercase",
    "normalized to upper",
    "normalised to uppercase",
)


def _config_classes() -> list[type[BaseModel]]:
    """Every pydantic BaseModel subclass defined in the configs module's own submodules."""
    out = []
    for _, obj in inspect.getmembers(configs_module, inspect.isclass):
        if not (issubclass(obj, BaseModel) and obj is not BaseModel):
            continue
        if obj.__module__ not in {
            configs_module.__name__,
            f"{configs_module.__package__}._preprocessing_configs",
            f"{configs_module.__package__}._model_configs",
            f"{configs_module.__package__}._training_runtime_configs",
            f"{configs_module.__package__}._composite_target_discovery_config",
            f"{configs_module.__package__}._reporting_configs",
            f"{configs_module.__package__}._configs_base",
            f"{configs_module.__package__}._feature_selection_config",
        }:
            continue
        out.append(obj)
    return out


def _per_field_doc(cls: type[BaseModel]) -> dict[str, str]:
    """Best-effort: find the contiguous lines under a numpydoc
    ``field_name : type`` header and return them as the field's
    docstring snippet.
    """
    doc = cls.__doc__ or ""
    out: dict[str, str] = {}
    lines = doc.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^\s+([A-Za-z_]\w*)\s*:\s*\S", lines[i])
        if m:
            name = m.group(1)
            chunk = []
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if re.match(r"^\s+[A-Za-z_]\w*\s*:\s*\S", nxt) or nxt.strip() == "":
                    break
                chunk.append(nxt.strip())
                j += 1
            out[name] = " ".join(chunk).lower()
            i = j
        else:
            i += 1
    return out


def _field_accepts_str(info) -> bool:
    """True if the field's annotation accepts ``str`` (covers ``str``, ``Optional[str]``,
    ``Literal["a","b"]`` etc.)."""
    ann = info.annotation
    if ann is str:
        return True
    args = typing.get_args(ann)
    if not args:
        return False
    for a in args:
        if a is str or isinstance(a, str):  # Literal members are bare strings
            return True
    return False


def _required_kwargs(cls: type[BaseModel]) -> dict:
    """Sentinel values for any field without a default so we can construct the class."""
    out: dict = {}
    for name, info in cls.model_fields.items():
        if info.default is not PydanticUndefined or info.default_factory is not None:
            continue
        if _field_accepts_str(info):
            out[name] = "x"
        else:
            out[name] = 0
    return out


def test_normalisation_promises_actually_normalise():
    """Every field whose docstring promises normalization actually normalizes a non-canonical input."""
    failures: list[str] = []
    audited = 0
    classes = _config_classes()
    for cls in classes:
        per_field = _per_field_doc(cls)
        if not per_field:
            continue
        for field_name, doc in per_field.items():
            if field_name not in cls.model_fields:
                continue
            info = cls.model_fields[field_name]
            if not _field_accepts_str(info):
                continue
            wants_lower = any(p in doc for p in _LOWER_PROMISES)
            wants_upper = any(p in doc for p in _UPPER_PROMISES)
            if not (wants_lower or wants_upper):
                continue
            audited += 1

            sentinel_in = "MiXeDcAsE"
            base_kwargs = _required_kwargs(cls)
            base_kwargs[field_name] = sentinel_in
            try:
                instance = cls(**base_kwargs)
            except Exception:  # nosec B112 -- best-effort skip of one iteration on a non-fatal error; the test's own assertions are unaffected
                # Synth values couldn't satisfy other validators; can't audit this field.
                continue
            actual = getattr(instance, field_name, None)
            if not isinstance(actual, str):
                continue
            if wants_lower and actual != sentinel_in.lower():
                failures.append(f"{cls.__name__}.{field_name}: docstring promises lowercase normalisation; input {sentinel_in!r} kept as {actual!r}")
            elif wants_upper and actual != sentinel_in.upper():
                failures.append(f"{cls.__name__}.{field_name}: docstring promises uppercase normalisation; input {sentinel_in!r} kept as {actual!r}")

    if audited == 0:
        pytest.skip("no normalisation promises found in config docstrings with str-typed fields")
    if failures:
        pytest.fail(f"{len(failures)} normalisation promise(s) not actually enforced:\n  " + "\n  ".join(failures))
