"""Run a warm every-family fit with every cupy ``RawKernel`` / ``RawModule`` launch checked: the element type of each array argument against the pointer type the kernel source declares.

A kernel that declares ``const double*`` and is handed a float32 array reads twice the bytes the array holds (wrong values, and an illegal address when the buffer ends a mapped region); the
crit dtype is float32 by default, so this is the shape of bug a type-generic ``{REAL}`` kernel (CUDA.md) exists to prevent. Prints each distinct mismatch once with the nearest mlframe caller.

Usage: python scripts/kernel_dtype_audit.py [n] [p]     (n, p default 100000 100; run alone on the GPU)
"""

from __future__ import annotations

import os
import re
import sys
import threading
import warnings

warnings.simplefilter("ignore")
for _k, _v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(_k, _v)

import cupy as cp
import numpy as np

_CTYPE = {"double": np.dtype(np.float64), "float": np.dtype(np.float32), "int": np.dtype(np.int32), "long long": np.dtype(np.int64), "unsigned char": np.dtype(np.uint8),
          "char": np.dtype(np.int8), "short": np.dtype(np.int16), "unsigned int": np.dtype(np.uint32), "unsigned long long": np.dtype(np.uint64), "signed char": np.dtype(np.int8)}
_SIG = re.compile(r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)', re.S)
_LOCK = threading.Lock()
SEEN: dict = {}


def _params(code: str, name: str):
    """``[(ctype or None for a scalar, param name)]`` of kernel ``name`` in ``code``."""
    for m in _SIG.finditer(code):
        if m.group(1) != name:
            continue
        out = []
        for raw in m.group(2).split(","):
            raw = re.sub(r"__restrict__|\bconst\b|\bvolatile\b", " ", raw).strip()
            mm = re.match(r"(.+?)\s*(\*)\s*(\w+)$", raw)
            out.append((" ".join(mm.group(1).split()) if mm else None, (mm.group(3) if mm else raw.split()[-1]) if raw else ""))
        return out
    return None


def _caller() -> str:
    f = sys._getframe(3)
    while f is not None and "mlframe" not in f.f_code.co_filename:
        f = f.f_back
    return f"{os.path.basename(f.f_code.co_filename)}:{f.f_code.co_name}:{f.f_lineno}" if f else "?"


def _check(name: str, code: str, args: tuple) -> None:
    params = _params(code, name)
    if not params or len(params) != len(args):
        return
    for (ctype, pname), a in zip(params, args):
        if ctype is None or not isinstance(a, cp.ndarray):
            continue
        want = _CTYPE.get(ctype)
        if want is not None and a.dtype != want:
            key = (name, pname, ctype, str(a.dtype))
            with _LOCK:
                if key in SEEN:
                    continue
                SEEN[key] = _caller()
            sys.stdout.write(f"KERNEL-DTYPE-MISMATCH {name}({pname}: {ctype}*) got {a.dtype}  caller {SEEN[key]}\n")
            sys.stdout.flush()


_RealKernel, _RealModule = cp.RawKernel, cp.RawModule


class _Kernel:
    """A ``RawKernel`` whose launches are checked."""

    def __init__(self, code, name, *a, **k):
        self._k, self._code, self._name = _RealKernel(code, name, *a, **k), code, name

    def __call__(self, grid, block, args, *a, **k):
        _check(self._name, self._code, tuple(args))
        return self._k(grid, block, args, *a, **k)

    def __getattr__(self, item):
        return getattr(self._k, item)


class _Fn:
    """A kernel taken from a ``RawModule``, checked at launch."""

    def __init__(self, fn, code, name):
        self._fn, self._code, self._name = fn, code, name

    def __call__(self, grid, block, args, *a, **k):
        _check(self._name, self._code, tuple(args))
        return self._fn(grid, block, args, *a, **k)

    def __getattr__(self, item):
        return getattr(self._fn, item)


class _Module:
    """A ``RawModule`` whose functions are checked."""

    def __init__(self, *a, **k):
        self._m, self._code = _RealModule(*a, **k), k.get("code") or (a[0] if a else "")

    def get_function(self, name):
        return _Fn(self._m.get_function(name), self._code, name)

    def __getattr__(self, item):
        return getattr(self._m, item)


cp.RawKernel, cp.RawModule = _Kernel, _Module


def main() -> None:
    from mlframe.feature_selection._benchmarks.profiling import family_scale as F
    from mlframe.feature_selection.filters.mrmr import MRMR

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    kw = F.all_families_kwargs()
    kw["nbins_strategy"] = "mdlp"
    MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*F._make(20000, 30, 1))
    MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*F._make(n, p, 0))
    sys.stdout.write(f"DONE distinct mismatches: {len(SEEN)}\n")


if __name__ == "__main__":
    main()
