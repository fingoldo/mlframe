"""Meta-test — every parameter listed in a config-class docstring's
``Parameters ----------`` (numpydoc) section must correspond to an actual
field on that class.

Catches the common drift pattern: a field is renamed / removed but the
docstring still describes it; new contributors then chase a name that
doesn't exist. Especially insidious for sub-config aggregators like
``TrainingConfig`` whose docstring lists 15+ fields by name — easy to miss
one rename when grepping.

Bidirectional check:
  (1) Every name in ``Parameters ----------`` must be in ``model_fields``
      (no stale documentation).
  (2) Every public ``model_fields`` key SHOULD have a docstring entry —
      this side is *warning-only* (logged via stderr, doesn't fail the
      test) because many small configs don't bother with full docstrings
      and we don't want to force boilerplate.
"""

from __future__ import annotations

import inspect
import re
import sys

import pytest
from pydantic import BaseModel

from mlframe.training import configs as configs_module

# Hard whitelist for parameter-line-but-no-field cases that are intentional
# (e.g. ``model_config`` is a pydantic-internal that some authors document
# anyway). Keep short — every entry is a documentation-honesty waiver.
_DOCSTRING_PSEUDO_PARAMS: dict[str, str] = {
    # Pydantic internals — listed in some docstrings but not user-facing
    # fields. Documented intentionally where they appear.
    # Format: "ClassName.param_name": "reason"
}

# Numpydoc ``Parameters ----------`` section markers.
# Conservative regex: a parameter line looks like ``    name : type``
# starting at any indentation level; our parsers handle the common
# ``    name : type`` and ``    name : type, optional`` shapes.
_PARAM_LINE_RE = re.compile(r"^\s+([A-Za-z_]\w*)\s*:\s*\S")


def _parse_docstring_params(docstring: str | None) -> list[str]:
    """Extract parameter names from a numpydoc-style docstring.

    Walks the ``Parameters`` section (between ``Parameters ----------``
    and the next blank-divided section header — ``Returns``, ``Raises``,
    ``Examples``, etc.). For each line matching ``    name : type``,
    yields ``name``.
    """
    if not docstring:
        return []
    lines = docstring.splitlines()
    in_params = False
    out: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "Parameters":
            # Next line should be ``----------``; tolerate either way.
            if i + 1 < len(lines) and set(lines[i + 1].strip()) <= {"-"}:
                in_params = True
            continue
        if not in_params:
            continue
        # Section break: another ``----------`` signals new section.
        if set(stripped) <= {"-"} and stripped:
            in_params = False
            continue
        # Recognised section headers that end Parameters.
        if stripped in {"Returns", "Raises", "Examples", "Notes",
                        "See Also", "Attributes", "Yields", "Warnings"}:
            in_params = False
            continue
        m = _PARAM_LINE_RE.match(line)
        if m:
            out.append(m.group(1))
    return out


def _config_classes() -> list[type[BaseModel]]:
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


def test_every_docstring_parameter_is_a_real_field():
    """Names in ``Parameters ----------`` must exist in ``model_fields`` —
    catches stale documentation after a rename / removal.
    """
    classes = _config_classes()
    assert classes, "no config classes found — class discovery broken?"

    stale: list[str] = []
    for cls in classes:
        params = _parse_docstring_params(cls.__doc__)
        if not params:
            continue
        actual_fields = set(cls.model_fields.keys())
        for param in params:
            qualified = f"{cls.__name__}.{param}"
            if qualified in _DOCSTRING_PSEUDO_PARAMS:
                continue
            if param not in actual_fields:
                stale.append(qualified)

    if stale:
        pytest.fail(
            f"{len(stale)} docstring parameter(s) reference fields that "
            f"don't exist on the class — stale documentation after a "
            f"rename / removal. Fix the docstring or add to "
            f"_DOCSTRING_PSEUDO_PARAMS with reasoning:\n  "
            + "\n  ".join(stale)
        )


def test_documented_field_count_consistency_warning():
    """Warning-only check: configs with a Parameters section should have
    fields covered roughly proportionally. A dramatic mismatch (e.g. 5
    docstring params vs 20 fields) signals incomplete documentation.

    Doesn't fail — too judgement-call to police strictly. Just emits a
    summary on stderr so it shows up in pytest output.
    """
    classes = _config_classes()
    incomplete: list[str] = []
    for cls in classes:
        params = _parse_docstring_params(cls.__doc__)
        if not params:
            continue  # No Parameters section — no claim made.
        actual = set(cls.model_fields.keys())
        # Keep it actionable: only flag when ≤ 50% of fields are documented.
        documented_pct = len(set(params) & actual) / max(len(actual), 1)
        if documented_pct < 0.5 and len(actual) >= 4:
            incomplete.append(
                f"{cls.__name__}: {len(set(params) & actual)}/{len(actual)} "
                f"fields documented ({documented_pct:.0%})"
            )
    if incomplete:
        # Print summary; don't fail the test.
        sys.stderr.write(
            "\n[test_documented_field_count_consistency_warning] "
            f"{len(incomplete)} config(s) have Parameters sections but "
            f"document <50% of fields:\n  " + "\n  ".join(incomplete) + "\n"
        )
