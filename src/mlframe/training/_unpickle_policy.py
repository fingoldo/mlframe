"""Name-resolution policy for the restricted unpickler used by ``load_mlframe_model`` and ``safe_joblib_load``.

An allowlist of (module, name) references narrows what a tampered bundle can reach, but it cannot make untrusted pickle safe: the durable control is a
signature or HMAC over the bundle checked with a key only the loader holds. This policy keeps code-execution gadgets (code objects, ``eval``/``exec``,
IO entry points, nested unrestricted loaders) unreachable while still resolving everything a legitimate mlframe/sklearn/torch bundle references.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import dill  # nosec B403 - only the UnpicklingError type is used here

_BUILTIN_MODULES: frozenset[str] = frozenset({"builtins", "__builtin__"})

# Data-container and singleton types a legitimate bundle references from ``builtins``; everything else (eval, exec, type, memoryview, ...) is refused.
_SAFE_BUILTIN_NAMES: frozenset[str] = frozenset({
    "bool", "int", "float", "complex", "str", "bytes", "bytearray", "list", "dict", "set", "frozenset", "tuple", "slice", "range", "object",
    "NoneType", "ellipsis", "Ellipsis", "NotImplemented", "NotImplementedType",
})

# Name of an object a pickle never needs but that every payload wants: process/filesystem/eval/IO entry points. Checked on every dotted component.
_DENIED_NAMES: frozenset[str] = frozenset({
    "eval", "exec", "execfile", "compile", "__import__", "import_module", "system", "popen", "spawn", "fork", "call", "check_call", "check_output",
    "run", "Popen", "load", "loads", "dump", "dumps", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt", "fromfile", "fromregex",
    "tofile", "open", "memmap", "DataSource", "query", "runstring", "get_function", "hub", "cpp_extension",
})

# Name prefixes that denote IO entry points in the data libraries (pandas.read_csv, polars.scan_parquet, NDFrame.to_pickle, ...).
_DENIED_NAME_PREFIXES: tuple[str, ...] = ("read_", "scan_", "to_", "from_pickle")

# Module (sub)trees that expose code evaluation, IO, process or network capability.
_DENIED_MODULE_PREFIXES: tuple[str, ...] = (
    "numpy.testing", "numpy.f2py", "numpy.distutils", "numpy.ctypeslib", "numpy.lib.npyio", "numpy.lib._npyio_impl", "numpy.lib._datasource",
    "numpy.lib.format", "numpy.lib._format_impl", "numpy.lib._iotools", "numpy._pytesttester",
    "pandas.io", "pandas.core.computation", "pandas._testing", "pandas.util",
    "polars.io", "polars.testing",
    "scipy.io", "scipy._lib._testutils",
    "sklearn.datasets", "sklearn.utils._testing", "sklearn.externals",
    "torch.hub", "torch.jit", "torch.utils.cpp_extension", "torch.distributed", "torch.multiprocessing", "torch.serialization", "torch.package",
    "torch.testing", "torch._dynamo", "torch._inductor", "torch.compiler",
    "joblib.externals", "joblib.parallel", "joblib.memory", "joblib.disk",
    "lightning.fabric.utilities.cloud_io", "lightning.pytorch.utilities.cloud_io", "pytorch_lightning.utilities.cloud_io", "lightning_fabric.utilities.cloud_io",
    "pyarrow.fs", "pyarrow.parquet", "pyarrow.hdfs", "pyarrow.flight", "pyarrow.ipc",
)

# Library trees a bundle may reference. Value True means only classes resolve (plus the exact functions in ``_SAFE_FUNCTIONS``): those libraries hold
# first-party helpers whose functions can perform IO or load other bundles.
_SAFE_MODULE_PREFIXES: dict[str, bool] = {
    "numpy": False,
    "pandas": False,
    "polars": False,
    "polars_ds": False,
    "sklearn": False,
    "scipy": False,
    "catboost": False,
    "lightgbm": False,
    "xgboost": False,
    "category_encoders": False,
    "pyarrow": False,
    "joblib": False,
    "torch": True,
    "pytorch_lightning": True,
    "lightning": True,
    "lightning_fabric": True,
    "mlframe": True,
    "collections": True,
    "datetime": True,
    "dataclasses": True,
}

# Exact non-class references that legitimate bundles need from the class-only libraries.
_SAFE_FUNCTIONS: frozenset[tuple[str, str]] = frozenset({
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_tensor"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch._utils", "_rebuild_parameter_with_state"),
    ("torch._utils", "_rebuild_device_tensor_from_numpy"),
    ("torch._tensor", "_rebuild_from_type_v2"),
    ("torch._tensor", "_rebuild_from_type"),
    ("torch._utils", "_rebuild_qtensor"),
    ("torch._utils", "_rebuild_meta_tensor_no_storage"),
    ("torch._C", "_nn._parse_to"),
})

# Modules of pure metric/scoring callables that fitted estimators store as attributes (early-stopping / eval metric slots); any function there may resolve.
_SAFE_FUNCTION_MODULES: tuple[str, ...] = (
    "mlframe.metrics._ice_metric",
    "mlframe.training.neural._base_logging",
)

# Exact (module, name) pairs resolved without the prefix machinery.
_SAFE_SPECIFIC: frozenset[tuple[str, str]] = frozenset({
    ("types", "SimpleNamespace"),
    ("functools", "partial"),
    ("_functools", "partial"),
    ("dill._dill", "_create_array"),
    ("typing", "TypeAlias"),
    ("typing", "Any"),
    ("typing", "Optional"),
    ("typing", "Union"),
    ("typing", "List"),
    ("typing", "Dict"),
    ("typing", "Tuple"),
    ("typing", "Sequence"),
    ("typing", "Callable"),
    ("typing", "ClassVar"),
})

# Attribute names refused by the restricted ``getattr``/``setattr``: introspection escalation plus IO/eval methods reachable on allowlisted objects.
_DANGEROUS_GETATTR_ATTRS: frozenset[str] = frozenset({
    "__globals__", "__code__", "__closure__", "__func__", "__builtins__",
    "__subclasses__", "__bases__", "__mro__", "__dict__", "__getattribute__",
    "__reduce__", "__reduce_ex__", "__class__", "func_globals", "gi_frame",
    "cr_frame", "f_globals", "f_locals", "f_builtins",
})


def _attr_name_denied(name: str) -> bool:
    """True when an attribute/global name component is an IO, eval or process entry point."""
    return name in _DENIED_NAMES or name.startswith(_DENIED_NAME_PREFIXES)


def _safe_getattr(obj: Any, name: Any, *default: Any) -> Any:
    """Restricted ``getattr`` reconstructor: refuses introspection-escalation attribute names and IO/eval method names on allowlisted objects. CatBoost's
    ``__reduce__`` reaches ``_setattr`` through it, so plain and underscore-prefixed helpers stay reachable."""
    if not isinstance(name, str) or name in _DANGEROUS_GETATTR_ATTRS or _attr_name_denied(name):
        raise dill.UnpicklingError(f"Unsafe getattr blocked by _SafeUnpickler allowlist: getattr({type(obj).__name__}, {name!r})")
    return getattr(obj, name, *default)


def _safe_setattr(obj: Any, name: Any, value: Any) -> None:
    """Restricted ``setattr`` reconstructor (CatBoost restores state through it): dunder type-confusion names are refused."""
    if not isinstance(name, str) or name in _DANGEROUS_GETATTR_ATTRS:
        raise dill.UnpicklingError(f"Unsafe setattr blocked by _SafeUnpickler allowlist: setattr({type(obj).__name__}, {name!r})")
    setattr(obj, name, value)


def _safe_torch_load_from_bytes(b: bytes) -> Any:
    """Replacement for ``torch.storage._load_from_bytes``, which runs an unrestricted nested ``torch.load``; this one loads weights only."""
    import torch

    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)


def _under(module: str, prefix: str) -> bool:
    """True when ``module`` is ``prefix`` or a submodule of it."""
    return module == prefix or module.startswith(prefix + ".")


_TORCH_CONSTANT_TYPES: frozenset[str] = frozenset({"dtype", "layout", "memory_format"})


def _is_class_ref(obj: Any) -> bool:
    """True when a resolved reference is a class (instantiating a class named by a pickle is the data-reconstruction case) or a torch dtype/layout constant."""
    if isinstance(obj, type):
        return True
    cls = type(obj)
    return cls.__module__ == "torch" and cls.__name__ in _TORCH_CONSTANT_TYPES


def resolve_restricted_class(module: str, name: str, real_find_class: Callable[[str, str], Any]) -> Any:
    """Resolve a pickled (module, name) reference only when the allowlist policy permits it; otherwise raise ``UnpicklingError``.
    ``real_find_class`` is the underlying unpickler's own ``find_class``."""
    if module in _BUILTIN_MODULES:
        if name == "getattr":
            return _safe_getattr
        if name == "setattr":
            return _safe_setattr
        if name in _SAFE_BUILTIN_NAMES:
            return real_find_class(module, name)
        exc = real_find_class(module, name)
        if isinstance(exc, type) and issubclass(exc, BaseException):
            return exc
        raise dill.UnpicklingError(f"Unsafe builtin blocked by allowlist: {module}.{name}")
    if (module, name) in _SAFE_SPECIFIC:
        return real_find_class(module, name)
    if (module, name) == ("torch.storage", "_load_from_bytes"):
        return _safe_torch_load_from_bytes
    for prefix, class_only in _SAFE_MODULE_PREFIXES.items():
        if not _under(module, prefix):
            continue
        if any(_under(module, denied) for denied in _DENIED_MODULE_PREFIXES) or any(_attr_name_denied(part) for part in name.split(".")):
            break
        obj = real_find_class(module, name)
        if class_only and not _is_class_ref(obj) and (module, name) not in _SAFE_FUNCTIONS and module not in _SAFE_FUNCTION_MODULES:
            break
        return obj
    raise dill.UnpicklingError(f"Unsafe class blocked by _SafeUnpickler allowlist: {module}.{name}")

# Builtins refused by the joblib denylist loader, which accepts arbitrary custom classes and so cannot use the builtins allowlist above.
_UNSAFE_BUILTINS: frozenset[str] = frozenset({
    "eval", "exec", "execfile", "compile", "__import__", "import_module", "delattr", "globals", "locals", "vars", "open", "input", "breakpoint",
    "memoryview", "help", "type", "classmethod", "staticmethod", "property", "super",
})

# Exact references refused by the denylist loader: code/function constructors and attribute-walking callables.
_DENIED_SPECIFIC: frozenset[tuple[str, str]] = frozenset({
    ("operator", "methodcaller"), ("operator", "attrgetter"), ("functools", "reduce"),
})
_DENIED_TYPES_NAMES: frozenset[str] = frozenset({"CodeType", "FunctionType", "LambdaType", "MethodType", "ModuleType", "CellType", "FrameType", "new_class"})
