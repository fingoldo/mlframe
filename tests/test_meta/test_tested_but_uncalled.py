"""A test must not certify a production function that nothing in production calls.

``refit_transform_on_fold`` sat in the uncalled-functions baseline while a biz_val test certified the leak fix it implemented,
so the suite reported a fix that no fit ran. This joins the two facts: a composite function listed as uncalled whose name a
test references must be recorded here with a reason (a public API kept for callers, a diagnostic run by hand), and a gate,
rerank, filter or per-group module in discovery must have a test that imports it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_UNCALLED = Path(__file__).resolve().parent / "_uncalled_functions_baseline.json"
_ACCEPTED = Path(__file__).resolve().parent / "_tested_but_uncalled_baseline.json"
_GATE_MODULE = re.compile(r"gate|rerank|_filter|_per_group")


def _test_text() -> str:
    """Every test module's source, joined."""
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in (_ROOT / "tests").rglob("test_*.py"))


def _tested_but_uncalled(blob: str) -> set[str]:
    """Composite functions in the uncalled baseline whose name appears in a test."""
    raw = json.loads(_UNCALLED.read_text(encoding="utf-8"))
    keys = list(raw) if isinstance(raw, (dict, list)) else []
    names = set(re.findall(r"\b[A-Za-z_]\w*\b", blob))
    return {k for k in keys if "/training/composite/" in k and k.split("::")[-1] in names}


def test_no_new_test_certifies_an_uncalled_composite_function():
    """Tested-but-uncalled composite functions are exactly the recorded ones; a fixed or wired entry must leave the list."""
    found = _tested_but_uncalled(_test_text())
    accepted = set(json.loads(_ACCEPTED.read_text(encoding="utf-8")))
    new, stale = sorted(found - accepted), sorted(accepted - found)
    assert not new, f"tests certify composite functions no production code calls; wire them in or record why: {new}"
    assert not stale, f"these entries are called now (or untested); remove them from {_ACCEPTED.name}: {stale}"


def test_every_gate_module_has_an_importing_test():
    """Each discovery gate / rerank / filter / per-group module is imported by at least one test."""
    blob = _test_text()
    mods = [p.stem for p in (_ROOT / "src/mlframe/training/composite/discovery").glob("*.py") if _GATE_MODULE.search(p.name)]
    assert len(mods) >= 5, f"only {len(mods)} gate modules matched; the pattern no longer fits the tree"
    missing = [m for m in mods if not re.search(rf"discovery[./]{m}\b|discovery import [^\n]*\b{m}\b", blob)]
    assert not missing, f"gate modules without an importing test: {missing}"
