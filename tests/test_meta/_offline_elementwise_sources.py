"""Every production ``cupy.ElementwiseKernel`` and ``cupy.ReductionKernel`` as compilable CUDA text, for the offline NVRTC/ptxas checks, built without a device.

The kernels in ``_offline_kernel_sources`` are ``__global__`` strings. An elementwise or reduction kernel is a body plus a parameter list, and cupy writes the
``__global__`` function (CArray / CIndexer wrappers, the grid-stride loop, the block reduction) when the kernel is first called. The same writers are reachable
without a GPU: constructing the kernel only parses its parameters, and ``_kernel._get_elementwise_kernel_code`` / ``_reduction._create_reduction_function_code``
render the source from argument descriptors, so the text NVRTC gets here is the text production compiles (an ndarray per parameter; a scalar argument reads
through the same ``const T &x`` reference).

The call sites are found by AST. A call whose code arguments are string literals is built as is; a call whose arguments are built from local names (a dtype,
a C type, a mode) is built by evaluating its own argument expressions under the values listed in ``_TEMPLATED``, so the kernel text lives only at the call site.
A site that is neither literal nor listed is reported by ``unresolved_sites`` and fails the coverage test: no site is skipped silently.
"""

from __future__ import annotations

import ast
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "mlframe"
_EXCLUDED_PARTS = ("_benchmarks", "kernel_prototypes")

#: Argument names in call order, the code-bearing ones (keyword or positional).
_ELEMENTWISE_ARGS = ("in_params", "out_params", "operation", "name")
_REDUCTION_ARGS = ("in_params", "out_params", "map_expr", "reduce_expr", "post_map_expr", "identity", "name")

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
_BLOCK_SIZE = 512  # block size cupy uses for a reduction kernel
_INDEX_OPTIONS = ("-DIndexT=int",)  # cupy's 32-bit indexing typedef for a reduction, which is passed to NVRTC as an option


class Site:
    """One kernel construction call: where it is, its argument nodes by parameter name and the module's string constants."""

    def __init__(self, rel: str, func: str, index: int, kind: str, args: dict, constants: dict) -> None:
        """Record the file, the enclosing function, the call's index in it, the kernel kind, its argument nodes by parameter name and the module's string constants."""
        self.rel, self.func, self.index, self.kind, self.args, self.constants = rel, func, index, kind, args, constants
        self.key = f"{rel}::{func or '<module>'}#{index}"

    @property
    def names(self) -> tuple:
        """The code-bearing parameter names of this kind of kernel."""
        return _ELEMENTWISE_ARGS if self.kind == "ElementwiseKernel" else _REDUCTION_ARGS

    def is_literal(self) -> bool:
        """True when every code-bearing argument is present and a string literal."""
        return all(isinstance(self.args.get(n), ast.Constant) and isinstance(self.args[n].value, str) for n in self.names)

    def evaluate(self, env: dict) -> dict:
        """``{parameter name: value}`` of the code-bearing arguments evaluated under ``env`` (and the module's string constants)."""
        scope = _ModuleScope({**self.constants, **env}, self.rel)
        return {n: _eval(self.args[n], scope) for n in self.names}


class _ModuleScope(dict):
    """The site's scope: literal constants first; a name that is not a literal (a body built at import time, e.g. ``_BODY = re.sub(...)``) is read from the imported module, so the test sees the text production compiles."""

    def __init__(self, base: dict, rel: str) -> None:
        """Keep the base names and the module path (``feature_selection/filters/x.py`` -> ``mlframe.feature_selection.filters.x``)."""
        super().__init__(base)
        self._module = "mlframe." + rel[: -len(".py")].replace("/", ".")

    def __missing__(self, name: str) -> object:
        """Import the module once and return the attribute; a name the module does not define still raises ``KeyError``."""
        import importlib

        module = importlib.import_module(self._module)
        if not hasattr(module, name):
            raise KeyError(name)
        return getattr(module, name)


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
    raise NotImplementedError(f"call-site expression {ast.dump(node)[:80]} is not supported by the offline kernel builder")


def _call_name(node: ast.Call) -> str | None:
    """Name of the called function or method (``cp.ElementwiseKernel`` and ``ElementwiseKernel`` both give ``ElementwiseKernel``)."""
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)


def _string_constants(tree: ast.Module) -> dict:
    """``{name: value}`` of the module-level assignments of a string literal (kernel bodies kept in constants)."""
    return {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str) for t in n.targets if isinstance(t, ast.Name)}


def _arguments(call: ast.Call, names: tuple) -> dict:
    """The call's argument nodes by parameter name: positional ones in the order of ``names``, keyword ones by their own name."""
    args = {n: a for n, a in zip(names, call.args)}
    args.update({kw.arg: kw.value for kw in call.keywords if kw.arg})
    return args


def _collect(node: ast.AST, func: str | None, kind: str, rel: str, constants: dict, counters: dict, out: list) -> None:
    """Append a ``Site`` for every ``kind`` call below ``node``, numbering the calls of each enclosing function in source order."""
    for child in ast.iter_child_nodes(node):
        name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else func
        if isinstance(child, ast.Call) and _call_name(child) == kind:
            idx = counters.get(name, 0)
            counters[name] = idx + 1
            names = _ELEMENTWISE_ARGS if kind == "ElementwiseKernel" else _REDUCTION_ARGS
            out.append(Site(rel, name or "", idx, kind, _arguments(child, names), constants))
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
    """All reduction call sites."""
    return _sites("ReductionKernel")


def unresolved_sites() -> list:
    """Sites (of either kind) that are neither literal nor listed in ``_TEMPLATED``."""
    return [s.key for s in elementwise_sites() + reduction_sites() if not s.is_literal() and s.key not in _TEMPLATED]


def stale_templates() -> list:
    """``_TEMPLATED`` keys that match no call site (the code moved or was renamed)."""
    have = {s.key for s in elementwise_sites() + reduction_sites()}
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


def _arg_descriptors(params: tuple, generics: dict) -> tuple:
    """``(arginfos, type_map)`` for cupy's code writers: one 1-d C-contiguous ndarray per data parameter, an indexer per ``_ind`` parameter, a scalar for the reduction's block stride."""
    import cupy as cp
    import numpy as np
    from cupy._core import _kernel as K

    infos = []
    for p in params:
        if p.name in ("_ind", "_in_ind", "_out_ind"):
            infos.append(K._ArgInfo(2, cp.ndarray, None, 1, True, True))
        elif p.name == "_block_stride":
            infos.append(K._ArgInfo(3, cp.ndarray, np.dtype("int32"), 0, True, True))
        else:
            dtype = np.dtype(p.dtype) if p.dtype is not None else np.dtype(generics[p.ctype])
            infos.append(K._ArgInfo(1, cp.ndarray, dtype, 1, True, True))
    return tuple(infos), K._TypeMap(tuple((letter, np.dtype(dt).type) for letter, dt in generics.items()))


def _generic_letters(params: tuple) -> list:
    """Generic type letters of the data parameters (``T``, ``S``, ...): those declared without a concrete dtype."""
    return sorted({p.ctype for p in params if p.dtype is None and p.ctype and not p.name.startswith("_")})


def _build_elementwise(values: dict) -> list:
    """``[(suffix, source)]`` of one elementwise kernel, once per float type when its parameters are generic."""
    import cupy as cp
    from cupy._core import _kernel as K

    kernel = cp.ElementwiseKernel(values["in_params"], values["out_params"], values["operation"], values["name"])
    letters = _generic_letters(kernel.params)
    out = []
    for dt in _GENERIC_DTYPES if letters else (None,):
        arginfos, type_map = _arg_descriptors(kernel.params, {letter: dt for letter in letters})
        code = K._get_elementwise_kernel_code(arginfos, type_map, kernel.params, kernel.operation, kernel.name, kernel.preamble)
        out.append(("" if dt is None else dt, "#include <cupy/carray.cuh>\n" + (code[0] if isinstance(code, tuple) else code)))
    return out


def _build_reduction(values: dict) -> list:
    """``[(suffix, source)]`` of one reduction kernel, once per float type when its parameters are generic."""
    import cupy as cp
    from cupy._core import _reduction as R

    kernel = cp.ReductionKernel(values["in_params"], values["out_params"], values["map_expr"], values["reduce_expr"], values["post_map_expr"], values["identity"], values["name"])
    params = kernel._params
    letters = _generic_letters(params)
    data = [p for p in params if not p.raw and not p.name.startswith("_")]
    input_expr = "\n".join(f"const {p.ctype} {p.name} = _raw_{p.name}[_in_ind.get()];" for p in data if p.is_const)
    output_expr = "\n".join(f"{p.ctype} &{p.name} = _raw_{p.name}[_out_ind.get()];" for p in data if not p.is_const)
    out = []
    for dt in _GENERIC_DTYPES if letters else (None,):
        arginfos, type_map = _arg_descriptors(params, {letter: dt for letter in letters})
        code = R._create_reduction_function_code(
            kernel.name, _BLOCK_SIZE, kernel.reduce_type, params, arginfos, kernel.identity, kernel.map_expr, kernel.reduce_expr, kernel.post_map_expr, type_map,
            input_expr, output_expr, kernel.preamble, (),
        )
        out.append(("" if dt is None else dt, "#include <cupy/carray.cuh>\n" + code))
    return out


def _environments(site: Site) -> list:
    """``[(env, label)]`` a site is built under; empty when it is neither literal nor listed."""
    if site.is_literal():
        return [({}, "")]
    if site.key in _TEMPLATED:
        return [(e, " ".join(f"{k}={v}" for k, v in e.items())) for e in _TEMPLATED[site.key]]
    return []


def kernel_builds() -> list:
    """``[(key, label, source, options)]`` for every literal or listed elementwise and reduction site, stable order. Labels start with ``ew`` or ``red``."""
    out = []
    for sites, prefix, build, options in (
        (elementwise_sites(), "ew", _build_elementwise, ()),
        (reduction_sites(), "red", _build_reduction, _INDEX_OPTIONS),
    ):
        for site in sites:
            for env, label in _environments(site):
                for suffix, source in build(site.evaluate(env)):
                    out.append((site.key, " ".join(x for x in (prefix, label, suffix) if x), source, options))
    return out
