"""Meta-test enforcing the cross-module ``@njit`` invariant.

INVARIANT (per ``mlframe/feature_selection/filters/_numba_utils.py``):
    All @njit helpers called from > 1 filters submodule live in
    ``_numba_utils.py``. Single-module njit helpers stay in their owner.

This test fails if any filters submodule ``X.py`` (other than
``_numba_utils.py``) imports an ``@njit``-decorated symbol from a sibling
filters submodule. Such cross-module imports cause numba's dispatcher to
recompile against each importer's module path, producing silent cache
misses and Windows file-lock races during pytest-xdist runs.

The check is greppy on purpose -- a structural ``ast`` walk would be more
precise but pulls in pyutilz / mypy noise. False positives are vanishingly
rare given the current package shape.
"""

from __future__ import annotations

import re
from pathlib import Path

FILTERS_DIR = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "filters"
ALLOWED_HOST = "_numba_utils"
NJIT_NAMES_HINT = re.compile(r"^@njit", re.MULTILINE)
RELATIVE_IMPORT_RE = re.compile(
    r"^from\s+\.(\w+)\s+import\s+([^\n#]+)",
    re.MULTILINE,
)


def _enumerate_njit_names(text: str) -> set[str]:
    """Return the set of names defined directly under ``@njit`` in this source text."""
    names = set()
    for m in re.finditer(
        r"^@njit[^\n]*\n(?:[^\n]*\n)*?def\s+(\w+)\s*\(",
        text,
        re.MULTILINE,
    ):
        names.add(m.group(1))
    return names


def _violations(sources: dict[str, str]) -> list[str]:
    """Cross-module ``@njit`` imports among the ``{submodule stem: source text}`` mapping."""
    submodules = sorted(sources)
    njit_names_by_module = {sub: _enumerate_njit_names(sources[sub]) for sub in submodules}

    violations: list[str] = []
    for importer in submodules:
        if importer == ALLOWED_HOST:
            continue
        text = sources[importer]
        for m in RELATIVE_IMPORT_RE.finditer(text):
            source_module = m.group(1)
            if source_module in (ALLOWED_HOST, "_internals", "_legacy"):
                # Allowed to import @njit helpers from `_numba_utils`.
                # ``_legacy`` is the migration scaffold (etap 1-10) and is
                # exempted; the invariant only applies between the new
                # submodules.
                continue
            if source_module not in njit_names_by_module:
                continue
            imported = {n.strip() for n in m.group(2).split(",") if n.strip()}
            crossed = imported & njit_names_by_module[source_module]
            if crossed:
                violations.append(f"{importer}.py imports @njit symbol(s) {sorted(crossed)} from sibling {source_module}.py (should live in _numba_utils.py)")

    return violations


def test_cross_module_njit_detector_catches_a_sibling_import_and_passes_the_shared_host():
    """Importing an ``@njit`` helper from a sibling is reported; importing it from ``_numba_utils`` is not."""
    owner = "@njit(cache=True)\ndef kern(x):\n    return x\n"
    bad = _violations({"a": owner, "b": "from .a import kern\n"})
    assert bad == ["b.py imports @njit symbol(s) ['kern'] from sibling a.py (should live in _numba_utils.py)"]
    assert _violations({"_numba_utils": owner, "b": "from ._numba_utils import kern\n"}) == []


def test_no_cross_module_njit_imports():
    """No filters submodule imports an ``@njit`` helper from a sibling submodule."""
    assert FILTERS_DIR.is_dir(), f"filters package not found at {FILTERS_DIR}"
    sources = {p.stem: p.read_text(encoding="utf-8") for p in FILTERS_DIR.glob("*.py") if p.stem != "__init__"}
    assert sources, "no filters submodules found"
    violations = _violations(sources)
    assert not violations, "Cross-module @njit imports detected:\n  " + "\n  ".join(violations)
