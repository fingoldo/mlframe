"""CUDA.md rule 3 as a ratchet over every module with a CUDA kernel, not only the FE modules the MRMR layer imports.

``test_fe_gpu_streaming_contract`` checks the type-generic rule only for the modules listed as compliant (one today) and only for modules the MRMR layer imports, so a helper kernel module
(``_row_argmax_gpu``) declared ``const double*`` and was handed float32 operands (the crit dtype default): it read past the end of its input and aborted the process in a later thrust call.
A new module here must use ``{REAL}``; the baseline lists the modules that still declare bare ``double*`` / ``float*`` kernel pointers and only shrinks.
"""

from __future__ import annotations

import re
from pathlib import Path

import orjson

FILTERS = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "filters"
BASELINE = Path(__file__).with_name("_bare_kernel_types_baseline.json")
_SIG = re.compile(r"__global__\s+void\s+\w+\s*\(([^)]*)\)", re.S)
_BARE = re.compile(r"\b(?:const\s+)?(?:double|float)\s*\*")


def _modules_with_bare_pointer_types() -> set[str]:
    """Filters-relative paths of the modules whose CUDA source has a ``double*`` / ``float*`` kernel parameter."""
    found = set()
    for f in FILTERS.rglob("*.py"):
        text = f.read_text(encoding="utf-8", errors="ignore")
        if "__global__" in text and any(_BARE.search(m.group(1)) for m in _SIG.finditer(text)):
            found.add(f.relative_to(FILTERS).as_posix())
    return found


def test_no_new_module_declares_bare_float_pointers():
    """A module outside the baseline must use ``{REAL}`` placeholders in its kernel signatures."""
    baseline = set(orjson.loads(BASELINE.read_bytes())["modules"])
    new = sorted(_modules_with_bare_pointer_types() - baseline)
    assert not new, f"kernel modules with bare double*/float* parameters outside the baseline (use {{REAL}}, see CUDA.md rule 3): {new}"


def test_baseline_only_shrinks():
    """A module that no longer declares bare pointer types is removed from the baseline."""
    baseline = set(orjson.loads(BASELINE.read_bytes())["modules"])
    fixed = sorted(baseline - _modules_with_bare_pointer_types())
    assert not fixed, f"these modules are type-generic now; remove them from {BASELINE.name}: {fixed}"
