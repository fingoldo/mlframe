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


def test_root_docstring_import_examples_are_importable():
    """Every ``from mlframe.x import y`` example line in mlframe/__init__.py's module docstring must
    actually resolve, so the package's own documented public-API convention never silently rots."""
    doc = mlframe.__doc__ or ""
    parsed = [p for p in (_parse_import_line(line) for line in doc.splitlines()) if p is not None]
    assert len(parsed) >= 5, "expected the docstring's public-API-convention code block to list several example imports"

    failures = []
    for module_path, names in parsed:
        try:
            module = importlib.import_module(module_path)
            for name in names:
                getattr(module, name)
        except Exception as exc:
            failures.append(f"from {module_path} import {', '.join(names)} -> {type(exc).__name__}: {exc}")

    assert not failures, "mlframe/__init__.py's documented example imports are broken:\n" + "\n".join(failures)
