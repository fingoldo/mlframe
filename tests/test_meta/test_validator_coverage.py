"""Meta-test — fields whose docstring promises a normalisation
("Case-insensitive, normalized to lowercase") must have a corresponding
``@field_validator(mode='before')`` actually applying that
normalisation.

Catches the failure mode where a docstring describes a behaviour that
never made it into code: users follow the docstring (pass mixed-case
input), the field accepts the value as-is, downstream code that branches
on ``cfg.task_type == "GPU"`` silently mismatches against ``"Gpu"``.

Heuristic — look for these phrases in per-field docstrings:
  * ``"Case-insensitive"`` or ``"case insensitive"``
  * ``"normalized to <something>"`` / ``"normalized to <something>case"``

For each match, search the class for any ``@field_validator`` whose
function source mentions the field name AND a normalising operation
(``.lower()`` / ``.upper()`` / ``.strip()``).

Limitation: heuristic matches a SHARED validator covering several
fields too. Per-field precision would require parsing the validator's
``@field_validator(*field_names)`` argument list — overkill for the
present scope.
"""

from __future__ import annotations

import inspect
import re

import pytest
from pydantic import BaseModel

from mlframe.training import configs as configs_module

# Phrases that imply a normalising validator.
_NORMALISATION_PROMISES = (
    "case-insensitive", "case insensitive",
    "normalized to lowercase", "normalized to uppercase",
    "normalized to upper", "normalized to lower",
    "normalised to lowercase", "normalised to uppercase",  # BE spelling
)

# Operations a normalising validator should perform.
_NORMALISATION_OPS = (".lower()", ".upper()", ".strip()", ".casefold()")


def _config_classes() -> list[type[BaseModel]]:
    out = []
    for _, obj in inspect.getmembers(configs_module, inspect.isclass):
        if not (issubclass(obj, BaseModel) and obj is not BaseModel):
            continue
        if obj.__module__ != configs_module.__name__:
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


def _validator_sources(cls: type[BaseModel]) -> list[str]:
    """Source code of every method on the class — we'll grep validator
    bodies. Keeping this loose (any method) avoids fighting Pydantic v2's
    decorator storage."""
    out: list[str] = []
    for name in dir(cls):
        attr = getattr(cls, name, None)
        if not callable(attr) or not hasattr(attr, "__module__"):
            continue
        if getattr(attr, "__module__", None) != cls.__module__:
            continue
        try:
            out.append(inspect.getsource(attr))
        except (OSError, TypeError):
            continue
    return out


def test_normalisation_promises_have_a_validator():
    failures: list[str] = []
    audited = 0
    classes = _config_classes()
    for cls in classes:
        per_field = _per_field_doc(cls)
        if not per_field:
            continue
        sources = _validator_sources(cls)
        joined = "\n".join(sources).lower()
        for field_name, doc in per_field.items():
            if not any(promise in doc for promise in _NORMALISATION_PROMISES):
                continue
            audited += 1
            # Must be at least one validator source mentioning the field
            # AND at least one normalising op anywhere in validator code.
            mentions_field = field_name.lower() in joined
            applies_op = any(op in joined for op in _NORMALISATION_OPS)
            if not (mentions_field and applies_op):
                failures.append(
                    f"{cls.__name__}.{field_name}: docstring promises "
                    f"normalisation but no validator on this class mentions "
                    f"the field AND a normalising op (.lower / .upper / "
                    f".strip)"
                )

    if audited == 0:
        pytest.skip("no normalisation promises found in config docstrings")
    if failures:
        pytest.fail(
            f"{len(failures)} normalisation promise(s) without a backing "
            f"validator:\n  " + "\n  ".join(failures)
        )
