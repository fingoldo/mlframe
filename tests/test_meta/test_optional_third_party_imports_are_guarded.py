"""A module-level import of a third-party package must be a hard dependency or be guarded.

``pip install mlframe`` installs only ``[project.dependencies]``; everything else lives in an extra. A module that does
``import catboost`` at the top of the file makes every importer of that module fail with a bare ``ModuleNotFoundError`` on a
core-only install, even when the code path in use never touches the extra.

The scan is AST based and name-agnostic. A module-level import is accepted when any of these holds:

* the top-level package maps to a distribution in ``[project.dependencies]``;
* it sits inside a ``try`` whose handlers catch ``ImportError`` (or a base class of it), including the re-raise-with-hint form;
* it sits under ``if TYPE_CHECKING:``, or under an ``if <flag>:`` where ``<flag>`` is a module-level name set inside such a ``try``;
* the line carries ``# optional-import-ok: <reason>``;
* the module lives inside a package that IS an extra's implementation namespace (``_EXTRA_NAMESPACES``).

Function-level imports are lazy by construction and are not scanned.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 / 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from tests.test_meta._scan_guard import assert_scanned_enough
from tests.test_meta._shared_ast_cache import parsed_ast, source_text

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "mlframe"
_SKIP_PARTS = {"_benchmarks", "_vendored", "__pycache__"}

# Packages that are the implementation namespace of one extra: every module under them may import that extra's libraries at module level,
# provided nothing outside the namespace imports them eagerly (the subprocess test below proves that for the main entry points).
_EXTRA_NAMESPACES = {
    "training/neural": {"torch", "lightning", "pytorch_lightning"},
}

# Import name -> distribution names for imports whose name differs from the distribution (llvmlite is numba's own hard requirement).
_IMPORT_TO_DIST = {
    "sklearn": {"scikit-learn"},
    "PIL": {"pillow"},
    "category_encoders": {"category-encoders"},
    "pyutilz": {"pyutilz"},
    "yaml": {"pyyaml"},
    "llvmlite": {"numba"},
}

_ALLOW_RE = re.compile(r"#\s*optional-import-ok:\s*\S")


def _norm(name: str) -> str:
    """PEP 503 normalised distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _hard_dependency_names() -> set:
    """Normalised distribution names listed in ``[project.dependencies]``."""
    data = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    names = set()
    for spec in data["project"]["dependencies"]:
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", spec)
        assert match, f"unparseable dependency spec {spec!r}"
        names.add(_norm(match.group(1)))
    return names


def _is_hard(top: str, hard: set) -> bool:
    """True when the import name ``top`` is provided by a hard dependency."""
    candidates = {_norm(top)} | {_norm(d) for d in _IMPORT_TO_DIST.get(top, ())}
    return bool(candidates & hard)


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    """True when the handler would catch ``ImportError`` (named, a base class, a bare except, or a tuple containing one)."""
    if handler.type is None:
        return True
    names = {n.id for n in ast.walk(handler.type) if isinstance(n, ast.Name)}
    return bool(names & {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"})


def _flag_names(tree: ast.Module) -> set:
    """Module-level names assigned inside a ``try`` that catches ImportError, i.e. availability flags like ``_HAS_XGBOOST``."""
    flags = set()
    for node in tree.body:
        if isinstance(node, ast.Try) and any(_catches_import_error(h) for h in node.handlers):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    flags.update(t.id for t in sub.targets if isinstance(t, ast.Name))
    return flags


def _is_guard_test(test: ast.expr, flags: set) -> bool:
    """True for ``TYPE_CHECKING`` or an availability-flag test (``if _HAS_X:``)."""
    names = {n.id for n in ast.walk(test) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(test) if isinstance(n, ast.Attribute)}
    return "TYPE_CHECKING" in names or bool(names & flags)


def _unguarded_imports(tree: ast.Module, lines: list) -> list:
    """``(lineno, top_level_name)`` for every module-level import not under a recognised guard or allow comment."""
    flags = _flag_names(tree)
    found: list = []

    def visit(body: list, guarded: bool) -> None:
        """Walk a statement list, propagating whether an enclosing construct guards the imports."""
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if guarded or (isinstance(node, ast.ImportFrom) and node.level):
                    continue
                if _ALLOW_RE.search(lines[node.lineno - 1]):
                    continue
                mods = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
                found.extend((node.lineno, m.split(".")[0]) for m in mods if m)
            elif isinstance(node, ast.Try):
                caught = any(_catches_import_error(h) for h in node.handlers)
                visit(node.body, guarded or caught)
                for handler in node.handlers:
                    visit(handler.body, guarded)
                visit(node.orelse, guarded)
                visit(node.finalbody, guarded)
            elif isinstance(node, ast.If):
                visit(node.body, guarded or _is_guard_test(node.test, flags))
                visit(node.orelse, guarded)
            elif isinstance(node, ast.With):
                visit(node.body, guarded)

    visit(tree.body, False)
    return found


def _violations() -> list:
    """Every unguarded module-level third-party import in ``src/mlframe`` that is not a hard dependency."""
    hard = _hard_dependency_names()
    stdlib = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()
    files = [p for p in sorted(_SRC.rglob("*.py")) if not (_SKIP_PARTS & set(p.parts))]
    assert_scanned_enough(len(files), str(_SRC))
    out = []
    for path in files:
        tree = parsed_ast(path)
        text = source_text(path)
        if tree is None or text is None:
            continue
        rel = path.relative_to(_SRC).as_posix()
        extra_ok = set().union(*(libs for prefix, libs in _EXTRA_NAMESPACES.items() if rel.startswith(prefix + "/")))
        for lineno, top in _unguarded_imports(tree, text.splitlines()):
            if top in stdlib or top == "mlframe" or top in extra_ok or _is_hard(top, hard):
                continue
            out.append(f"{rel}:{lineno} imports {top!r}")
    return out


def test_every_module_level_third_party_import_is_a_hard_dependency_or_guarded():
    """A module-level import of an extra-only package must be lazy or guarded so a core-only install can still import the module."""
    bad = _violations()
    assert not bad, (
        "module-level imports of packages outside [project.dependencies] without an ImportError guard (import lazily via "
        "mlframe._optional_imports.import_optional, guard with try/except ImportError, or mark `# optional-import-ok: <reason>`):\n" + "\n".join(bad)
    )


def test_the_detector_flags_an_unguarded_import_and_honours_each_guard_form():
    """Teeth-check: the scanner must flag the bare import and accept every documented guard, including the availability-flag form."""
    src = textwrap.dedent(
        """
        import bare_pkg
        import allowed_pkg  # optional-import-ok: leaf module
        try:
            import guarded_pkg
            _HAS_G = True
        except ImportError:
            _HAS_G = False
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            import typing_only_pkg
        if _HAS_G:
            import flag_guarded_pkg
        def f():
            import lazy_pkg
        """
    )
    found = {top for _, top in _unguarded_imports(ast.parse(src), src.splitlines())}
    assert found == {"bare_pkg", "typing"}, found


def test_core_only_install_can_import_the_public_subpackages():
    """With every optional-extra library made unimportable, the main public subpackages still import."""
    blocked = [
        "catboost", "lightgbm", "xgboost", "zstandard", "properscoring", "shap", "seaborn", "optbinning", "mlflow", "hypothesis", "torch", "lightning",
    ]
    modules = [
        "mlframe.training", "mlframe.training.core", "mlframe.inference.predict", "mlframe.feature_selection", "mlframe.feature_selection.filters",
        "mlframe.feature_selection.wrappers", "mlframe.feature_selection.boruta_shap", "mlframe.feature_selection.shap_proxied_fs",
        "mlframe.models", "mlframe.models.ensembling", "mlframe.evaluation.reports", "mlframe.calibration", "mlframe.training.composite",
        "mlframe.feature_engineering.numerical", "mlframe.votenrank", "mlframe.metrics",
    ]
    code = textwrap.dedent(
        f"""
        import importlib, sys
        BLOCK = set({blocked!r})
        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCK:
                    raise ModuleNotFoundError("blocked " + name)
                return None
        sys.meta_path.insert(0, Blocker())
        failed = []
        for m in {modules!r}:
            try:
                importlib.import_module(m)
            except Exception as exc:
                failed.append(m + ": " + repr(exc)[:160])
        print("\\n".join(failed))
        sys.exit(1 if failed else 0)
        """
    )
    env_path = str(_REPO / "src")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, cwd=str(_REPO), env={**os.environ, "PYTHONPATH": env_path})
    assert proc.returncode == 0, "imports that need an optional extra:\n" + proc.stdout + proc.stderr[-2000:]
