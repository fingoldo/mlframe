"""Launch-time audit of the CUDA kernel arguments: every array handed to a kernel must have the element type of the pointer it is bound to.

A float32 buffer bound to a ``double*`` parameter is read (or written) past its end; the device does not always fault at once, the stray write lands in a neighbouring block of the memory pool
and corrupts it, and the fault surfaces much later in an unrelated kernel. The audit wraps ``cupy.RawKernel`` and ``cupy.RawModule`` before the package compiles its kernels, parses each
kernel's signature, and compares the dtype of every device array (and the kind of every scalar) with the declared parameter type at each launch. It reports every distinct mismatch with the
call stack that produced it.

    python -m mlframe.feature_selection._benchmarks.kernel_arg_dtype_audit --small          # a quick strict-GPU fit, exit code 1 on a mismatch
    python -m mlframe.feature_selection._benchmarks.kernel_arg_dtype_audit --only conj,xor3  # the four-pillar cases (see bench_four_pillars)
"""

from __future__ import annotations

import argparse
import os
import re
import runpy
import sys
import traceback
from typing import Any, Optional

import numpy as np

_NUMPY_OF_C = {
    "double": np.float64,
    "float": np.float32,
    "int": np.int32,
    "unsigned int": np.uint32,
    "long long": np.int64,
    "unsigned long long": np.uint64,
    "signed char": np.int8,
    "char": np.int8,
    "unsigned char": np.uint8,
    "short": np.int16,
    "unsigned short": np.uint16,
    "long": np.int32,
    "bool": np.bool_,
}
_SIGNATURE = re.compile(r"__global__\s+void\s+(\w+)\s*\((.*?)\)\s*\{", re.S)
_FOUND: "dict[tuple, str]" = {}


def _numpy_type(ctype: str) -> Optional[type]:
    """The numpy scalar type of a C parameter type (qualifiers and ``*`` ignored), or ``None`` for a macro or typedef."""
    text = re.sub(r"/\*.*?\*/", " ", ctype, flags=re.S)
    text = re.sub(r"\b(const|__restrict__|restrict|volatile)\b", " ", text).replace("*", " ")
    text = " ".join(text.split()).replace("long long int", "long long")
    return _NUMPY_OF_C.get(text)


def kernel_parameters(source: str, name: str) -> "Optional[list[tuple[bool, Optional[type]]]]":
    """``[(is_pointer, numpy type or None)]`` for the parameters of the kernel ``name`` declared in ``source``, or ``None`` when it is not found."""
    for match in _SIGNATURE.finditer(source):
        if match.group(1) != name:
            continue
        params = []
        for raw in re.sub(r"//[^\n]*", "", match.group(2)).split(","):
            raw = raw.strip()
            if not raw:
                continue
            body = re.sub(r"\b\w+\s*$", "", raw.replace("*", " * ")).strip()
            params.append(("*" in raw, _numpy_type(body)))
        return params
    return None


def _record(name: str, params: "Optional[list]", args: Any) -> None:
    """Remember each distinct mismatch between the launch arguments and the parameter types, with the package call stack of the first occurrence."""
    import cupy as cp

    if params is None:
        return
    for position, (arg, (is_pointer, want)) in enumerate(zip(args, params)):
        if want is None:
            continue
        problem = None
        if is_pointer:
            if isinstance(arg, cp.ndarray):
                if arg.dtype != np.dtype(want):
                    problem = (f"expects {np.dtype(want)}*", f"got a {arg.dtype} array")
                elif not (arg.flags.c_contiguous or arg.flags.f_contiguous):
                    problem = ("expects a contiguous buffer", "got a strided view")
        elif isinstance(arg, np.generic):
            if np.dtype(want).kind != arg.dtype.kind and not (np.dtype(want).kind in "iu" and arg.dtype.kind in "iu"):
                problem = (f"scalar {np.dtype(want)}", f"got {arg.dtype}")
            elif arg.dtype.itemsize < np.dtype(want).itemsize:
                problem = (f"scalar {np.dtype(want)}", f"got the narrower {arg.dtype}")
        elif isinstance(arg, float) and np.dtype(want).kind != "f":
            problem = (f"scalar {np.dtype(want)}", "got a python float")
        key = (name, position, *problem) if problem else None
        if key is not None and key not in _FOUND:
            frames = [f for f in traceback.extract_stack(limit=12)[:-3] if "mlframe" in f.filename]
            _FOUND[key] = " <- ".join(f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in reversed(frames))[:600]


class _Launcher:
    """A kernel handle that records dtype mismatches before launching; every other attribute is the wrapped kernel's."""

    def __init__(self, fn: Any, name: str, params: "Optional[list]"):
        object.__setattr__(self, "_fn", fn)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_params", params)

    def __call__(self, grid: Any, block: Any, args: Any, **kwargs: Any) -> Any:
        _record(self._name, self._params, args)
        return self._fn(grid, block, args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._fn, item)

    def __setattr__(self, key: str, value: Any) -> None:
        setattr(self._fn, key, value)


def install() -> None:
    """Replace ``cupy.RawKernel`` and ``cupy.RawModule`` with auditing twins; call before the package compiles a kernel."""
    import cupy as cp

    real_kernel, real_module = cp.RawKernel, cp.RawModule

    class AuditedRawKernel(_Launcher):
        """Drop-in ``cupy.RawKernel``."""

        def __init__(self, source: str, name: str, *args: Any, **kwargs: Any):
            super().__init__(real_kernel(source, name, *args, **kwargs), name, kernel_parameters(source, name))

    class AuditedRawModule:
        """Drop-in ``cupy.RawModule``."""

        def __init__(self, *args: Any, **kwargs: Any):
            object.__setattr__(self, "_module", real_module(*args, **kwargs))
            object.__setattr__(self, "_code", kwargs.get("code", args[0] if args else ""))

        def get_function(self, name: str) -> Any:
            return _Launcher(self._module.get_function(name), name, kernel_parameters(self._code, name))

        def __getattr__(self, item: str) -> Any:
            return getattr(self._module, item)

        def __setattr__(self, key: str, value: Any) -> None:
            setattr(self._module, key, value)

    cp.RawKernel = AuditedRawKernel
    cp.RawModule = AuditedRawModule


def mismatches() -> "dict[tuple, str]":
    """``{(kernel, argument position, expected, got): call stack}`` of every distinct mismatch seen so far."""
    return dict(_FOUND)


def report() -> int:
    """Print every distinct mismatch and return their number."""
    print(f"ARG_DTYPE_MISMATCHES {len(_FOUND)}", flush=True)
    for (name, position, expected, got), stack in sorted(_FOUND.items()):
        print(f"MISMATCH kernel={name} arg#{position}: {expected}, {got}\n   at {stack}", flush=True)
    return len(_FOUND)


def _small_fit() -> None:
    """A quick strict-GPU fit that exercises the resident binning, the triplet and the pair stages on a binary conjunction frame with proxies and noise."""
    import pandas as pd

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 6000
    a, b, c = (rng.integers(0, 2, n) for _ in range(3))
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(-1.5 + 2.5 * a + 3.0 * (a & b & c))))).astype(np.int64)
    columns = {"A": a.astype(float), "B": b.astype(float), "C": c.astype(float)}
    for name, src in (("pa", a), ("pb", b), ("pc", c)):
        z = (src - src.mean()) / src.std()
        for k in range(5):
            columns[f"{name}{k}"] = 0.7 * z + np.sqrt(1 - 0.49) * rng.standard_normal(n)
    for k in range(10):
        columns[f"noise{k}"] = rng.standard_normal(n)
    MRMR(verbose=0).fit(pd.DataFrame(columns), pd.Series(y))


def main(argv: "Optional[list[str]]" = None) -> int:
    """Run the audit over a small fit (``--small``) or the bench_four_pillars cases (``--only``); exit code 1 when a mismatch was found."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--small", action="store_true", help="run the built-in quick fit instead of the pillar cases")
    parser.add_argument("--only", default="f2,noise,xor3,conj", help="pillar cases to run (bench_four_pillars --only)")
    args = parser.parse_args(argv)
    os.environ.setdefault("MLFRAME_FE_GPU_STRICT", "1")
    install()
    if args.small:
        _small_fit()
    else:
        bench = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_four_pillars.py")
        sys.argv = [bench, "--only", args.only]
        runpy.run_path(bench, run_name="__main__")
    return 1 if report() else 0


if __name__ == "__main__":
    raise SystemExit(main())
