"""Shared AST helpers for the meta-tests on module-level mutable state.

The lock and eviction gates used to look only at names ending in ``_CACHE``. Shared state with any other name was
invisible to them. ``_DEFERRED_HOST_FILL`` in ``_gpu_resident_fe.py`` is one such registry: it was mutated from
several threads and dropped pending work on eviction, and neither lock gate ever looked at it. These helpers find
every module-level dict-like binding that some function mutates, whatever it is called, and follow a local alias
(``c = _REGISTRY; c[k] = v``), which is how that registry was written.
"""

from __future__ import annotations

import ast

MUTATING_METHODS = frozenset({"pop", "clear", "setdefault", "update", "popitem", "move_to_end"})


def is_dict_like_value(value: ast.AST) -> bool:
    """True if ``value`` is a fresh dict-like container literal or constructor call."""
    if isinstance(value, ast.Dict):
        return True
    if isinstance(value, ast.Call):
        func = value.func
        name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
        return name in ("dict", "OrderedDict", "defaultdict")
    return False


def module_has_lock_construction(tree: ast.Module) -> bool:
    """True if the module constructs a ``Lock()``/``RLock()`` anywhere."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
            if name in ("Lock", "RLock"):
                return True
    return False


def module_level_dicts(tree: ast.Module) -> dict[str, int]:
    """``{name: lineno}`` for every top-level binding of a dict-like value, whatever its name."""
    out: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not is_dict_like_value(value):
            continue
        for t in targets:
            if isinstance(t, ast.Name):
                out.setdefault(t.id, node.lineno)
    return out


def aliases_in(func: ast.AST, names: set[str]) -> dict[str, str]:
    """``{local: module_name}`` for plain local aliases ``local = MODULE_NAME`` inside ``func``."""
    out: dict[str, str] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id in names:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node.value.id
    return out


def base_name(node: ast.AST) -> str:
    """The ``C`` in ``C[k]`` / ``C[k][j]`` / ``C.pop(...)``, or "" when the target is not a plain name."""
    while isinstance(node, ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def mutations_of(func: ast.AST, names: set[str]) -> set[str]:
    """Module names from ``names`` mutated anywhere inside ``func``, directly or through a local alias.

    Nested functions are included: a closure mutating the dict is the same hazard, attributed to the enclosing
    definition a reviewer would read.
    """
    alias = aliases_in(func, names)

    def resolve(node: ast.AST) -> str:
        """Module name a mutation target refers to, or ""."""
        b = base_name(node)
        b = alias.get(b, b)
        return b if b in names else ""

    hit: set[str] = set()
    for node in ast.walk(func):
        targets: list = []
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Subscript)]
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Subscript):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = [t for t in node.targets if isinstance(t, ast.Subscript)]
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in MUTATING_METHODS:
                targets = [f.value]
        for t in targets:
            r = resolve(t)
            if r:
                hit.add(r)
    return hit


def function_defs(tree: ast.Module) -> list:
    """Every function definition in the module, nested ones included."""
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def mutable_module_dicts(tree: ast.Module) -> dict[str, int]:
    """``{name: lineno}`` for module-level dicts that at least one function mutates: shared mutable state."""
    dicts = module_level_dicts(tree)
    if not dicts:
        return {}
    names = set(dicts)
    mutated: set[str] = set()
    for fn in function_defs(tree):
        mutated |= mutations_of(fn, names)
    return {n: dicts[n] for n in mutated}


def module_qualname(rel_posix: str) -> str:
    """``training/cb/_cb_pool.py`` -> ``mlframe.training.cb._cb_pool``; a package ``__init__`` names the package."""
    parts = ["mlframe", *rel_posix[: -len(".py")].split("/")]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_from(node: ast.ImportFrom, qualname: str, is_package: bool) -> str:
    """Absolute module name an ``ImportFrom`` reads from."""
    if not node.level:
        return node.module or ""
    base = qualname.split(".")
    if not is_package:
        base = base[:-1]
    if node.level > 1:
        base = base[: len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def imported_module_dicts(tree: ast.Module, rel_posix: str, dict_index: dict[str, set[str]]) -> dict[str, str]:
    """``{local_name: "origin.module:NAME"}`` for names this module imports that are another module's module-level dict.

    A dict defined in one module and mutated or evicted in another is the same shared state; the per-module scans
    missed it (``_CB_VAL_POOL_CACHE`` is defined in ``_predict_guards`` and evicted in ``cb/_cb_pool``).
    """
    qual = module_qualname(rel_posix)
    is_pkg = rel_posix.endswith("__init__.py")
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            src = _resolve_from(node, qual, is_pkg)
            names = dict_index.get(src)
            if not names:
                continue
            for a in node.names:
                if a.name in names:
                    out[a.asname or a.name] = f"{src}:{a.name}"
    return out


def build_dict_index(modules: dict[str, ast.Module]) -> dict[str, set[str]]:
    """``{module_qualname: {module-level dict names}}`` for ``{rel_posix: tree}``."""
    return {module_qualname(rel): set(module_level_dicts(tree)) for rel, tree in modules.items()}
