"""A module-level import of a third-party package must be a hard dependency or be guarded.

``pip install mlframe`` installs only ``[project.dependencies]``; everything else lives in an extra. A module that does
``import catboost`` at the top of the file makes every importer of that module fail with a bare ``ModuleNotFoundError`` on a
core-only install, even when the code path in use never touches the extra.

The static check is ``py_ci_shared.optional_imports_guarded`` over the modules ``import mlframe`` reaches; the dynamic check starts
the main public subpackages with every extra's libraries unimportable. A guarded import is a ``try`` catching ``ImportError``,
``TYPE_CHECKING``, an availability flag, a function-level import or a line marked ``# optional-import-ok: <reason>``.
"""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Set

from packaging.requirements import Requirement
from py_ci_shared.ci_install_covers_entry_imports import assert_entry_imports_without_extras
from py_ci_shared.optional_imports_guarded import assert_optional_imports_guarded

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 / 3.10
    import tomli as tomllib  # type: ignore[no-redef]

_REPO = Path(__file__).resolve().parents[2]

# Import name -> distribution for imports whose name differs from the distribution (llvmlite is numba's own hard requirement).
_NAME_MAP = {"llvmlite": "numba"}

# Packages that are the implementation namespace of one extra: modules under them may import that extra's libraries at module level.
_EXTRA_NAMESPACES = {"training/neural": ["torch", "lightning", "pytorch_lightning"]}


def _norm(name: str) -> str:
    """PEP 503 normalised distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _core_closure(core_specs: List[str]) -> Set[str]:
    """Distributions the core dependencies need, transitively: blocking one of them would break a core install, not test it."""
    seen: Set[str] = set()
    todo = [_norm(Requirement(spec).name) for spec in core_specs]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            requires = metadata.requires(name) or []
        except metadata.PackageNotFoundError:
            continue
        for spec in requires:
            req = Requirement(spec)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                todo.append(_norm(req.name))
    return seen


def _extra_dists(extra: str, optional: Dict[str, List[str]]) -> Set[str]:
    """Distributions an extra provides, following ``mlframe[...]`` self-references."""
    out: Set[str] = set()
    for spec in optional[extra]:
        req = Requirement(spec)
        if _norm(req.name) == "mlframe":
            for inner in req.extras:
                out |= _extra_dists(inner, optional)
        else:
            out.add(_norm(req.name))
    return out


def test_every_module_level_third_party_import_is_a_hard_dependency_or_guarded():
    """A module-level import of an extra-only package must be lazy or guarded so a core-only install can still import the module."""
    assert_optional_imports_guarded(_REPO, reachable_from=["mlframe"], name_map=_NAME_MAP, extra_namespaces=_EXTRA_NAMESPACES, min_files=1000)


def test_core_only_install_can_import_the_public_subpackages():
    """With the libraries of every extra the core does not itself need made unimportable, the main public subpackages still import."""
    project = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    optional = project["optional-dependencies"]
    closure = _core_closure(project["dependencies"])
    extras = sorted(extra for extra in optional if not _extra_dists(extra, optional) & closure)
    assert extras, "every extra overlaps the core closure; the test would block nothing"
    modules = [
        "mlframe.training",
        "mlframe.training.core",
        "mlframe.inference.predict",
        "mlframe.feature_selection",
        "mlframe.feature_selection.filters",
        "mlframe.feature_selection.wrappers",
        "mlframe.feature_selection.boruta_shap",
        "mlframe.feature_selection.shap_proxied_fs",
        "mlframe.models",
        "mlframe.models.ensembling",
        "mlframe.evaluation.reports",
        "mlframe.calibration",
        "mlframe.training.composite",
        "mlframe.feature_engineering.numerical",
        "mlframe.votenrank",
        "mlframe.metrics",
    ]
    assert_entry_imports_without_extras(_REPO, modules, blocked_extras=extras, name_map=_NAME_MAP)
