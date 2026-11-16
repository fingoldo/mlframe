"""Every production ``cupy.ElementwiseKernel`` as compilable CUDA text, for the offline NVRTC/ptxas checks, built without a device.

The kernels in ``_offline_kernel_sources`` are ``__global__`` strings. An elementwise kernel is a body plus a parameter list, and cupy writes the
``__global__`` function (CArray / CIndexer wrappers, the grid-stride loop) when the kernel is first called. The same writer is reachable without a GPU:
``cupy.ElementwiseKernel(...)`` only parses the parameters, and ``_kernel._get_elementwise_kernel_code`` renders the source from argument descriptors, so
the text NVRTC gets here is the text production compiles (an ndarray per parameter; a scalar argument reads through the same ``const T &x`` reference).

The call sites are found by AST. A call whose four arguments are string literals is built as is; a call whose arguments are built from local names (a dtype,
a C type, a mode) is built by evaluating its own argument expressions under the values listed in ``_TEMPLATED``, so the kernel text lives only at the call site.
A site that is neither literal nor listed is reported by ``unresolved_sites`` and fails the coverage test: no site is skipped silently.
``ReductionKernel`` calls are listed by ``reduction_sites`` and covered by their own test until their code writer is wired in the same way.
"""

from __future__ import annotations

import ast
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "mlframe"
_EXCLUDED_PARTS = ("_benchmarks", "kernel_prototypes")

# "<path under mlframe>::<enclosing function>#<index of the call in that function>" -> the local-name environments its argument expressions are evaluated under.
_DTYPES = [{"key": "float32", "ctype": "float"}, {"key": "float64", "ctype": "double"}]
_TEMPLATED: dict = {
    "feature_selection/filters/_binned_numeric_agg_resident.py::_oof_gather_into_kernel#0": _DTYPES,
    "feature_selection/filters/_orthogonal_univariate_fe/_gpu_resident_cross_basis.py::_fused_leg_products#0": [{"dt": "float32", "ctype": "float"}, {"dt": "float64", "ctype": "double"}],
    "feature_selection/filters/_resident_candidate_mi.py::_tau_grid_kernel#0": [{"out_ctype": "float", "decl": "float32"}, {"out_ctype": "double", "decl": "float64"}],
    "feature_selection/filters/_resident_candidate_mi.py::_tau_grid_kernel#1": [{"out_ctype": "float", "decl": "float32"}, {"out_ctype": "double", "decl": "float64"}],
    "feature_selection/filters/_unary_stack_cuda.py::_kernel#0": [{}],
    "feature_selection/filters/_usability_greedy_clf_batched.py::_logistic_p_and_weight#0": [{"dt": "float32", "c": "float"}, {"dt": "float64", "c": "double"}],
    "feature_selection/filters/info_theory/_cmi_cuda.py::_get_xlogx_kernel#0": [{"real_name": "float32"}, {"real_name": "float64"}],
}
_GENERIC_DTYPES = ("float32", "float64")  # a generic parameter type (T, S, ...) is built once per value type


class Site:
    """One ``ElementwiseKernel(...)`` call: where it is, its argument nodes and the module's string constants."""

    def __init__(self, rel: str, func: str, index: int, args: list, constants: dict) -> None:
        """Record the file, the enclosing function, the call's index in it, its argument nodes and the module's string constants."""
        self.rel, self.func, self.index, self.args, self.constants = rel, func, index, args, constants
        self.key = f"{rel}::{func}#{index}" if func else f"{rel}::<module>#{index}"

    def is_literal(self) -> bool:
        """True when the in-params, out-params, operation and name arguments are all string literals."""
        return len(self.args) >= 4 and all(isinstance(a, ast.Constant) and isinstance(a.value, str) for a in self.args[:4])

    def evaluate(self, env: dict) -> tuple:
        """``(in_params, out_params, operation, name)`` with the argument expressions evaluated under ``env`` (and the module's string constants)."""
        scope = {**self.constants, **env}
        return tuple(_eval(a, scope) for a in self.args[:4])


def _eval(node: ast.AST, scope: dict) -> object:
    """Value of a call-site argument expression: string literals, names, ``+``, ``%`` formatting, tuples and f-strings, nothing else (an unsupported node raises, it is never guessed)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return scope[node.id]
    if isinstance(node, ast.Tuple):
        return tuple(_eval(e, scope) for e in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _eval(node.left, scope) + _eval(node.right, scope)  # type: ignore[operator]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return _eval(node.left, scope) % _eval(node.right, scope)  # type: ignore[operator]
    if isinstance(node, ast.JoinedStr):
        return "".join(str(_eval(v, scope)) for v in node.values)
    if isinstance(node, ast.FormattedValue):
        return _eval(node.value, scope)
    raise NotImplementedError(f"call-site expression {ast.dump(node)[:80]} is not supported by the offline elementwise builder")


def _call_name(node: ast.Call) -> str | None:
    """Name of the called function or method (``cp.ElementwiseKernel`` and ``ElementwiseKernel`` both give ``ElementwiseKernel``)."""
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)


def _string_constants(tree: ast.Module) -> dict:
    """``{name: value}`` of the module-level assignments of a string literal (kernel bodies kept in constants)."""
    return {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str) for t in n.targets if isinstance(t, ast.Name)}


def _collect(node: ast.AST, func: str | None, kind: str, rel: str, constants: dict, counters: dict, out: list) -> None:
    """Append a ``Site`` for every ``kind`` call below ``node``, numbering the calls of each enclosing function in source order."""
    for child in ast.iter_child_nodes(node):
        name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else func
        if isinstance(child, ast.Call) and _call_name(child) == kind:
            idx = counters.get(name, 0)
            counters[name] = idx + 1
            out.append(Site(rel, name or "", idx, list(child.args) + [kw.value for kw in child.keywords], constants))
        _collect(child, name, kind, rel, constants, counters, out)


def _sites(kind: str) -> list:
    """Every ``kind`` (``ElementwiseKernel`` / ``ReductionKernel``) call under ``src/mlframe`` outside the excluded trees."""
    out: list = []
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in _EXCLUDED_PARTS for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        _collect(tree, None, kind, path.relative_to(ROOT).as_posix(), _string_constants(tree), {}, out)
    return out


def elementwise_sites() -> list:
    """All elementwise call sites."""
    return _sites("ElementwiseKernel")


def reduction_sites() -> list:
    """All reduction call sites (not built here)."""
    return _sites("ReductionKernel")


def unresolved_sites() -> list:
    """Elementwise sites that are neither literal nor listed in ``_TEMPLATED``."""
    return [s.key for s in elementwise_sites() if not s.is_literal() and s.key not in _TEMPLATED]


def stale_templates() -> list:
    """``_TEMPLATED`` keys that match no call site (the code moved or was renamed)."""
    have = {s.key for s in elementwise_sites()}
    return [k for k in _TEMPLATED if k not in have]


def include_options() -> tuple:
    """``-I`` options for cupy's CArray header and the CUDA toolkit headers it includes (``cuda_fp16.h``)."""
    import cupy as cp

    opts = [f"-I{Path(cp.__file__).parent / '_core' / 'include'}"]
    cuda = os.environ.get("CUDA_PATH")
    if not cuda:
        exe = shutil.which("ptxas")
        cuda = str(Path(exe).parents[1]) if exe else ""
    if cuda and (Path(cuda) / "include").is_dir():
        opts.append(f"-I{Path(cuda) / 'include'}")
    return tuple(opts)


def _arg_descriptors(kernel: object, generics: dict) -> tuple:
    """``(arginfos, type_map)`` for cupy's code writer: one 1-d C-contiguous ndarray per parameter, then the indexer."""
    import cupy as cp
    import numpy as np
    from cupy._core import _kernel as K

    infos = []
    for p in kernel.params:  # type: ignore[attr-defined]
        if p.name == "_ind":
            infos.append(K._ArgInfo(2, cp.ndarray, None, 1, True, True))
            continue
        dtype = np.dtype(p.dtype) if p.dtype is not None else np.dtype(generics[p.ctype])
        infos.append(K._ArgInfo(1, cp.ndarray, dtype, 1, True, True))
    return tuple(infos), K._TypeMap(tuple((letter, np.dtype(dt).type) for letter, dt in generics.items()))


def _build(site: Site, env: dict, label: str) -> list:
    """The ``(label, source)`` builds of one site under one environment (a build per value type when the parameters are generic)."""
    import cupy as cp
    from cupy._core import _kernel as K

    in_params, out_params, operation, name = site.evaluate(env)
    kernel = cp.ElementwiseKernel(in_params, out_params, operation, name)
    letters = sorted({p.ctype for p in kernel.params if p.dtype is None and p.name != "_ind" and p.ctype})
    builds = []
    for dt in _GENERIC_DTYPES if letters else (None,):
        generics = {letter: dt for letter in letters}
        arginfos, type_map = _arg_descriptors(kernel, generics)
        code = K._get_elementwise_kernel_code(arginfos, type_map, kernel.params, kernel.operation, kernel.name, kernel.preamble)
        code = code[0] if isinstance(code, tuple) else code
        builds.append((f"{label}{'' if dt is None else ' ' + dt}", "#include <cupy/carray.cuh>\n" + code))
    return builds


def elementwise_builds() -> list:
    """``[(key, label, source)]`` for every literal or listed elementwise site, stable order."""
    out = []
    for site in elementwise_sites():
        if site.is_literal():
            envs = [({}, "ew")]
        elif site.key in _TEMPLATED:
            envs = [(e, "ew " + " ".join(f"{k}={v}" for k, v in e.items())) for e in _TEMPLATED[site.key]]
        else:
            continue
        for env, label in envs:
            for lab, src in _build(site, env, label):
                out.append((site.key, lab, src))
    return out
