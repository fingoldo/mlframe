"""CUDA kernels do not stride a row-major matrix by the thread-derived row index, or the kernel is on the reviewed baseline.

In a kernel ``for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)`` neighbouring threads hold neighbouring ``i``. A read ``X[i * n_features + col]`` of a
row-major matrix then puts neighbouring threads ``n_features`` elements apart, so every warp load touches up to 32 cache lines instead of one and the kernel is memory-bound several times
over. The fix is a column-major copy made once (a transpose kernel or ``cp.ascontiguousarray(X.T)``) so the same read becomes ``X[col * n + i]``, where neighbouring threads read neighbouring
rows of one column; ``_pair_mi_perm_cuda.py`` does this.

Detector (text scan of every Python string that contains ``__global__``): inside a kernel body, find ``for`` loops whose initialiser depends on ``threadIdx.x`` (directly or through a
variable assigned from it), then report the kernel when the body reads (or atomically updates; a plain ``a[...] = v`` store is skipped) an array indexed with ``<loop var> * <non-constant name> + ...`` (casts such as ``(long long)`` allowed on either
factor). Loops driven only by ``blockIdx`` are ignored, since the threads of one block then share the row. Hits are keyed ``<module path>::<kernel name>``.

The baseline is a worklist, not a verdict: some hits are deliberate, for example a kernel that needs the row-major layout for another reason (a per-row sequential scan, a layout shared
with a host consumer, a matrix that is tiny enough to sit in cache) or one whose stride is 1 in practice. Review a hit with a profile before rewriting it. A new unlisted kernel fails the
test, and a baseline entry whose kernel is gone or fixed fails too, so the list only shrinks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.test_meta._scan_guard import assert_scanned_enough

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "mlframe"
BASELINE = Path(__file__).with_name("_gpu_uncoalesced_reads_baseline.json")

_KERNEL_RE = re.compile(r"__global__\s+void\s+(\w+)\s*\(")
_ASSIGN_RE = re.compile(r"(?:\b(?:unsigned|long|int|size_t|const)\b\s*)+(\w+)\s*=\s*([^;]+);")
_FOR_RE = re.compile(r"for\s*\(\s*(?:\b(?:unsigned|long|int|size_t|const)\b\s*)*(\w+)\s*=\s*([^;]*);")
_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_CAST = r"(?:\(\s*[A-Za-z_][\w ]*\)\s*)?"


def _kernel_bodies(text: str) -> "list[tuple[str, str]]":
    """``[(kernel name, body text)]`` for every ``__global__`` function in ``text`` (brace-matched)."""
    out: list[tuple[str, str]] = []
    for m in _KERNEL_RE.finditer(text):
        open_at = text.find("{", m.end())
        if open_at < 0:
            continue
        depth, i = 0, open_at
        while i < len(text):
            ch = text[i]
            depth += ch == "{"
            depth -= ch == "}"
            if depth == 0:
                break
            i += 1
        out.append((m.group(1), _COMMENT_RE.sub(" ", text[open_at : i + 1])))
    return out


def _thread_vars(body: str) -> "set[str]":
    """Variables whose value depends on ``threadIdx.x``, closed over simple assignments."""
    derived: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, expr in _ASSIGN_RE.findall(body):
            if name not in derived and ("threadIdx.x" in expr or any(re.search(rf"\b{re.escape(v)}\b", expr) for v in derived)):
                derived.add(name)
                changed = True
    return derived


def uncoalesced_kernels(text: str) -> "list[str]":
    """Names of kernels in ``text`` that index an array by ``<thread-strided loop var> * <non-constant> + ...``."""
    if "__global__" not in text:
        return []
    hits: list[str] = []
    for name, body in _kernel_bodies(text):
        derived = _thread_vars(body)
        for var, init in _FOR_RE.findall(body):
            if "threadIdx.x" not in init and not any(re.search(rf"\b{re.escape(v)}\b", init) for v in derived):
                continue
            row_major = re.compile(rf"\[[^\]\[]*?(?:\b{re.escape(var)}\s*\)?\s*\*\s*{_CAST}[A-Za-z_]\w*|[A-Za-z_]\w*\s*\*\s*{_CAST}\b{re.escape(var)}\b)[^\]\[]*?\](?!\s*=[^=])")
            if row_major.search(body):
                hits.append(name)
                break
    return hits


def _current() -> "set[str]":
    """``{"<module path>::<kernel>"}`` over the package source."""
    found: set[str] = set()
    files = sorted(SRC_ROOT.rglob("*.py"))
    assert_scanned_enough(len(files), str(SRC_ROOT))
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "__global__" not in text:
            continue
        rel = str(path.relative_to(SRC_ROOT.parent.parent)).replace("\\", "/")
        found.update(f"{rel}::{k}" for k in uncoalesced_kernels(text))
    return found


def _baseline() -> "set[str]":
    """The reviewed ``{"<module path>::<kernel>"}`` worklist."""
    return set(json.loads(BASELINE.read_text(encoding="utf-8")))


def test_no_new_uncoalesced_row_reads():
    """A kernel that strides a row-major matrix by its thread index, outside the baseline, fails the test; read a column-major copy instead."""
    new = sorted(_current() - _baseline())
    assert not new, f"kernels reading X[i * stride + col] with thread-strided i (use a column-major copy, or review and baseline): {new}"


def test_baseline_has_no_stale_entries():
    """A baseline entry whose kernel is gone or already coalesced is removed, so the baseline only shrinks."""
    stale = sorted(_baseline() - _current())
    assert not stale, f"baseline entries no longer needed (remove them): {stale}"


@pytest.mark.parametrize("rel", sorted({e.split("::")[0] for e in _baseline()}))
def test_baselined_module_exists(rel):
    """Every baseline path points at a real module, so a rename cannot hide a kernel."""
    assert (SRC_ROOT.parent.parent / rel).is_file(), rel


def test_checker_flags_row_major_stride_and_accepts_column_major_read():
    """``X[i * n_features + a]`` under a thread-strided loop is flagged; ``X[a * n + i]`` and a block-row loop are accepted."""
    bad = (
        'SRC = r"""\nextern "C" __global__ void k(const int* X, int n, int n_features, int a, int* out) {\n'
        "    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {\n"
        '        atomicAdd(&out[X[i * (long long)n_features + a]], 1);\n    }\n}\n"""'
    )
    good = (
        'SRC = r"""\nextern "C" __global__ void k(const int* X, int n, int n_features, int a, int* out) {\n'
        "    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {\n"
        "        atomicAdd(&out[X[a * (long long)n + i]], 1);\n    }\n}\n"
        'extern "C" __global__ void rows(const int* X, int n, int n_features, int* out) {\n'
        "    for (int r = blockIdx.x; r < n; r += gridDim.x) {\n"
        '        out[r] = X[r * n_features + threadIdx.x];\n    }\n}\n"""'
    )
    assert uncoalesced_kernels(bad) == ["k"]
    assert not uncoalesced_kernels(good)
