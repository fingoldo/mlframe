"""X_ARCHITECTURE_API_CONSISTENCY-1 (2026-08-05 audit): 3 of the 8 example imports in
``mlframe/__init__.py``'s "public API convention" docstring were broken ImportErrors against the
current tree (MRMR, expected_calibration_error, predict_from_models) -- a fresh reader following the
package's own documented convention would hit an error on the first try. Meta-test: extract every
``from ... import ...`` line from the module docstring and actually execute it, so any future drift
between the documented example and the real API surface fails CI instead of silently rotting.
"""

from __future__ import annotations

import importlib
import re

import mlframe

_IMPORT_LINE_RE = re.compile(r"^\s*from (mlframe\.\S+) import (.+)$")


def _parse_import_line(line: str) -> "tuple[str, list[str]] | None":
    """Parse a ``from mlframe.x.y import a, b`` line into (module_path, [names]); None if it doesn't match."""
    m = _IMPORT_LINE_RE.match(line)
    if m is None:
        return None
    module_path, names_raw = m.group(1), m.group(2)
    return module_path, [n.strip() for n in names_raw.split(",")]


def _import_failures(parsed: "list[tuple[str, list[str]]]") -> list[str]:
    """One message per ``(module_path, names)`` entry whose import or attribute lookup fails."""
    failures = []
    for module_path, names in parsed:
        try:
            module = importlib.import_module(module_path)
            for name in names:
                getattr(module, name)
        except Exception as exc:
            failures.append(f"from {module_path} import {', '.join(names)} -> {type(exc).__name__}: {exc}")
    return failures


def test_import_check_catches_a_missing_name_and_a_missing_module_and_passes_a_valid_import():
    """A bad attribute and a bad module are each reported; a resolvable import and the line parser behave as expected."""
    assert _parse_import_line("    from mlframe.training import a, b") == ("mlframe.training", ["a", "b"])
    assert _parse_import_line("import os") is None
    assert _import_failures([("os.path", ["join"])]) == []
    missing_name = _import_failures([("os.path", ["no_such_name_xyz"])])
    assert len(missing_name) == 1 and "AttributeError" in missing_name[0]
    missing_module = _import_failures([("no_such_module_xyz", ["a"])])
    assert len(missing_module) == 1 and "ModuleNotFoundError" in missing_module[0]


def test_root_docstring_import_examples_are_importable():
    """Every ``from mlframe.x import y`` example line in mlframe/__init__.py's module docstring must
    actually resolve, so the package's own documented public-API convention never silently rots."""
    doc = mlframe.__doc__ or ""
    parsed = [p for p in (_parse_import_line(line) for line in doc.splitlines()) if p is not None]
    assert len(parsed) >= 5, "expected the docstring's public-API-convention code block to list several example imports"

    failures = _import_failures(parsed)

    assert not failures, "mlframe/__init__.py's documented example imports are broken:\n" + "\n".join(failures)
