"""Every CUDA kernel that takes ``double*`` buffers is either dtype-templated or on the reviewed baseline.

The package keeps the precision-critical buffers in the critical dtype (float32 by default), while many RawKernel sources declare their pointers as ``double*``. A float32
buffer bound to a ``double*`` parameter is read past its end (a device-side illegal access that corrupts the CUDA context for the rest of the process), so the mismatch must not be
introduced silently. A kernel with ``double*`` parameters must either take its element type from a ``{REAL}`` template (compiled per dtype) or be listed in the baseline below until its
launch sites are reviewed; a new unlisted kernel fails this test.

A templated kernel is only as safe as the type chosen at its call site. The type must follow the buffer that is actually launched, so the array whose ``dtype`` picks the
instantiation (``"float" if X.dtype == cp.float32 else "double"``) has to be passed to the launch by name; probing the dtype of an array that is cast to another dtype before the launch
compiles a ``double`` kernel for float32 buffers, which is the same out-of-bounds read the template exists to prevent.
"""

from __future__ import annotations

import ast
import orjson
import re
from pathlib import Path

import pytest

from tests.test_meta._scan_guard import assert_scanned_enough

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "mlframe"
BASELINE = Path(__file__).with_name("_gpu_kernel_pointer_dtype_baseline.json")

_KERNEL_RE = re.compile(r"__global__\s+void\s+(\w+)\s*\((.*?)\)\s*\{", re.S)  # with or without an ``extern "C"`` prefix: kernels inside an ``extern "C" { ... }`` block have none


def _double_pointer_kernels():
    """``{relative path: [kernel names with a double* parameter, not templated]}`` over the package source."""
    found: dict[str, list[str]] = {}
    files = sorted(SRC_ROOT.rglob("*.py"))
    assert_scanned_enough(len(files), str(SRC_ROOT))
    for path in files:
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
    return {k: set(v) for k, v in orjson.loads(BASELINE.read_text(encoding="utf-8")).items()}


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


def _launch_type_probe_violations(source: str) -> "list[str]":
    """Functions whose ``"float" if X.dtype == <float32> else "double"`` kernel-type selector names an array ``X`` that the module never passes to a launch tuple by name.

    The launch can sit in a different method than the probe (a class builds its table in ``__init__`` and launches it later), so the launch arguments are collected module-wide.
    """
    tree = ast.parse(source)
    named = {
        ast.unparse(el)
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        for arg in call.args
        if isinstance(arg, ast.Tuple)
        for el in arg.elts
        if isinstance(el, (ast.Name, ast.Attribute))
    }
    problems: list[str] = []
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        probes = {
            ast.unparse(node.test.left.value)
            for node in ast.walk(fn)
            if isinstance(node, ast.IfExp)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Attribute)
            and node.test.left.attr == "dtype"
            and isinstance(node.body, ast.Constant)
            and node.body.value == "float"
            and isinstance(node.orelse, ast.Constant)
            and node.orelse.value == "double"
        }
        problems.extend(f"{fn.name}: the kernel type follows {p}.dtype but {p} is not a launch argument" for p in sorted(probes - named))
    return problems


def test_launch_type_probe_checker_flags_a_cast_before_launch():
    """The checker reports a type probe on an array that is cast before the launch, and accepts one on the launched buffer."""
    bad = "def f(cp, mad, out):\n" "    k = get(cp, 'float' if mad.dtype == cp.float32 else 'double')\n" "    k((1,), (1,), (mad.astype(cp.float32), out))\n"
    good = (
        "def f(cp, mad, out):\n"
        "    mad_c = mad.astype(cp.float32)\n"
        "    k = get(cp, 'float' if mad_c.dtype == cp.float32 else 'double')\n"
        "    k((1,), (1,), (mad_c, out))\n"
    )
    assert _launch_type_probe_violations(bad)
    assert not _launch_type_probe_violations(good)


def test_kernel_type_follows_the_launched_buffer():
    """Everywhere in the package, the array that picks a kernel's element type is one of the arrays the launch receives."""
    found: dict[str, list[str]] = {}
    files = sorted(SRC_ROOT.rglob("*.py"))
    assert_scanned_enough(len(files), str(SRC_ROOT))
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "float32" not in text or "dtype" not in text:
            continue
        problems = _launch_type_probe_violations(text)
        if problems:
            found[str(path.relative_to(SRC_ROOT.parent.parent)).replace("\\", "/")] = problems
    assert not found, f"the element type of a kernel must follow the buffer actually launched: {found}"
