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

import orjson
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
        # Comments are stripped first: a comment in the parameter list can hold an unbalanced ")" such as "[0,Kx)".
        seg = _COMMENT_RE.sub(" ", text[m.end() :])
        # Skip the parameter list: templated kernels carry ``{ACC}``-style placeholders in it, which are not the body.
        depth, i = 1, 0
        while i < len(seg) and depth:
            depth += (seg[i] == "(") - (seg[i] == ")")
            i += 1
        open_at = seg.find("{", i)
        if open_at < 0:
            continue
        depth, j = 0, open_at
        while j < len(seg):
            ch = seg[j]
            depth += ch == "{"
            depth -= ch == "}"
            if depth == 0:
                break
            j += 1
        out.append((m.group(1), seg[open_at : j + 1]))
    return out


_GROUP_INDEX_RE = re.compile(r"threadIdx\.x\s*/\s*(?!1\b)[A-Za-z_]\w*")


def _thread_vars(body: str) -> "set[str]":
    """Variables whose value depends on ``threadIdx.x``, closed over simple assignments."""
    derived: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, expr in _ASSIGN_RE.findall(body):
            if _GROUP_INDEX_RE.fullmatch(expr.strip()):
                continue  # ``threadIdx.x / tile``: the ``tile`` neighbouring threads share the value, so they read adjacent columns of one row
            if name not in derived and ("threadIdx.x" in expr or any(re.search(rf"\b{re.escape(v)}\b", expr) for v in derived)):
                derived.add(name)
                changed = True
    return derived


# A single ``__shared__`` statement can declare several arrays at once (``__shared__ T a[N], b[M], c[K];``), a valid and common
# CUDA C++ form - the two regexes below first grab the WHOLE declaration up to its terminating ``;``, then pull every
# ``name[`` out of that one declaration, so none of its names is missed the way a one-name-per-keyword regex would.
_SHARED_DECL_RE = re.compile(r"__shared__[^;]*;")
# A statically sized ``__shared__`` buffer (``se[EDGE_CACHE]``) has a fixed footprint, so a fill of it from global memory is bounded
# by its declared size. A dynamic ``extern __shared__`` array (its size comes from the launch) has no such bound and is not exempt.
_STATIC_SHARED_DECL_RE = re.compile(r"(?<!extern )__shared__[^;]*;")
_SHARED_NAME_RE = re.compile(r"(\w+)\s*\[")
_ALIAS_RE = re.compile(r"\w+\s*\*\s*(\w+)\s*=\s*(\w+)\s*(?:\+[^;]*)?;")


def _without_shared_subscripts(body: str) -> str:
    """``body`` with every ``__shared__`` array access blanked: shared memory is not a global read.

    A store into a statically sized shared buffer is dropped as a whole statement, since its global read is bounded by the buffer size.
    """
    static_names = {n for decl in _STATIC_SHARED_DECL_RE.findall(body) for n in _SHARED_NAME_RE.findall(decl)}
    for name in static_names:
        body = re.sub(rf"\b{re.escape(name)}\s*\[[^\]]*\]\s*=(?!=)[^;]*;", ";", body)
    all_names = {n for decl in _SHARED_DECL_RE.findall(body) for n in _SHARED_NAME_RE.findall(decl)}
    for _ in range(3):  # ``int* ones = sh;`` / ``int* tot = sh + P * nbins;``: a pointer into a shared array is shared memory too
        all_names |= {a for a, base in _ALIAS_RE.findall(body) if base in all_names}
    for name in all_names:
        body = re.sub(rf"\b{re.escape(name)}\s*\[[^\]]*\]", "0", body)
    return body


def uncoalesced_kernels(text: str) -> "list[str]":
    """Names of kernels in ``text`` that index an array by ``<thread-strided loop var> * <non-constant> + ...``."""
    if "__global__" not in text:
        return []
    hits: list[str] = []
    for name, body in _kernel_bodies(text):
        derived = _thread_vars(body)
        body = _without_shared_subscripts(body)
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
    files = sorted(p for p in SRC_ROOT.rglob("*.py") if "_benchmarks" not in p.parts)
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
    return set(orjson.loads(BASELINE.read_text(encoding="utf-8")))


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


def test_row_group_index_of_a_column_tile_is_not_a_thread_stride():
    """A row-group index ``threadIdx.x / tile`` next to columns ``threadIdx.x % tile`` makes a warp read adjacent columns of one row, so its loop is not flagged."""
    tiled = (
        'SRC = r"""\nextern "C" __global__ void k(const signed char* X, int n, int K, int tile, int* out) {\n'
        "    const int cj = threadIdx.x % tile; const int rg = threadIdx.x / tile; const int ng = blockDim.x / tile;\n"
        "    for (long long i = rg; i < n; i += ng) { atomicAdd(&out[X[i * (long long)K + cj]], 1); }\n}\n"
        '"""'
    )
    assert not uncoalesced_kernels(tiled)


def test_pointer_alias_of_a_shared_array_is_shared_memory():
    """Reads through ``int* ones = sh;`` (``sh`` a dynamic shared array) are shared-memory reads and not flagged."""
    src = (
        'SRC = r"""' + chr(10) + 'extern "C" __global__ void k(int* out, int P, int nb) {' + chr(10)
        + '    extern __shared__ int sh[]; int* ones = sh;' + chr(10)
        + '    for (int p = threadIdx.x; p < P; p += blockDim.x) { out[p] = ones[p * nb + 1]; }' + chr(10) + '}' + chr(10) + '"""'
    )
    assert not uncoalesced_kernels(src)


def test_static_shared_fill_is_exempt_and_dynamic_shared_fill_is_flagged():
    """A fill of a statically sized shared buffer is bounded by its size, so it is not flagged; a dynamic ``extern __shared__`` fill is."""
    static_fill = """
extern "C" __global__ void fill_static(const double* __restrict__ edges, const int K, const int c, const int ne) {
    __shared__ double se[256];
    for (int s = threadIdx.x; s < ne; s += blockDim.x) se[s] = edges[(long long)s * K + c];
    __syncthreads();
    double acc = 0.0;
    for (int b = 0; b < 4; ++b) acc += se[b];
    (void)acc;
}
"""
    dynamic_fill = """
extern "C" __global__ void fill_dynamic(const double* __restrict__ yb, const long long words, const int B, const int b) {
    extern __shared__ unsigned int sy[];
    for (long long w = threadIdx.x; w < words; w += blockDim.x) sy[w] = yb[(long long)w * B + b];
    __syncthreads();
}
"""
    assert uncoalesced_kernels(static_fill) == []
    assert uncoalesced_kernels(dynamic_fill) == ["fill_dynamic"]
