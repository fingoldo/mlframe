"""Every CUDA kernel that takes ``double*`` buffers is either dtype-templated or on the reviewed baseline.

The package keeps the precision-critical buffers in the critical dtype (float32 by default), while many RawKernel sources declare their pointers as ``double*``. A float32
buffer bound to a ``double*`` parameter is read past its end (a device-side illegal access that corrupts the CUDA context for the rest of the process), so the mismatch must not be
introduced silently. A kernel with ``double*`` parameters must either take its element type from a ``{REAL}`` template (compiled per dtype) or be listed in the baseline below until its
launch sites are reviewed; a new unlisted kernel fails this test.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "mlframe"
BASELINE = Path(__file__).with_name("_gpu_kernel_pointer_dtype_baseline.json")

_KERNEL_RE = re.compile(r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\((.*?)\)\s*\{', re.S)


def _double_pointer_kernels():
    """``{relative path: [kernel names with a double* parameter, not templated]}`` over the package source."""
    found: dict[str, list[str]] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "__global__" not in text:
            continue
        for m in _KERNEL_RE.finditer(text):
            name, params = m.group(1), m.group(2)
            if "{REAL}" in params or re.search(r"double\s*\*", params) is None:
                continue
            found.setdefault(str(path.relative_to(SRC_ROOT.parent.parent)).replace("\\", "/"), []).append(name)
    return found


def _baseline():
    """The reviewed ``{module path: [kernel names]}`` baseline of double* kernels."""
    return {k: set(v) for k, v in json.loads(BASELINE.read_text(encoding="utf-8")).items()}


def test_double_pointer_kernels_are_templated_or_baselined():
    """A ``double*`` kernel outside the reviewed baseline is a new dtype hazard and fails the test."""
    unlisted = {}
    base = _baseline()
    for rel, names in _double_pointer_kernels().items():
        new = sorted(set(names) - base.get(rel, set()))
        if new:
            unlisted[rel] = new
    assert not unlisted, (
        "CUDA kernels with double* parameters outside the reviewed baseline (template them on {REAL} or review their launch sites, "
        f"then add to the baseline): {unlisted}"
    )


def test_baseline_has_no_stale_entries():
    """A baseline entry whose kernel is gone or already templated is removed, so the baseline only shrinks."""
    current = _double_pointer_kernels()
    stale = {rel: sorted(set(names) - set(current.get(rel, []))) for rel, names in _baseline().items()}
    stale = {k: v for k, v in stale.items() if v}
    assert not stale, f"baseline entries no longer needed (remove them): {stale}"


@pytest.mark.parametrize("rel", sorted(_baseline()))
def test_baselined_module_exists(rel):
    """Every baseline path points at a real module, so a rename cannot hide a kernel."""
    assert (SRC_ROOT.parent.parent / rel).is_file(), rel
