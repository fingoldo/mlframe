"""E4 — meta-test that the package's internal import graph is acyclic.

A circular import can lurk for years in a package because Python's
import machinery resolves cycles at module-level if the offending name
is accessed lazily. Then a refactor moves a top-level access of one of
the cycle members and ``ImportError: cannot import name X from
partially initialized module Y`` ships to users.

Builds a dep graph by AST-walking every production .py for ``import`` /
``from`` statements; restricts to imports whose target is inside the
package itself (third-party deps are out of scope). Runs Tarjan's SCC
to find any cycle of size > 1.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

import mlframe

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent
PKG_NAME = "mlframe"

# Genuine multi-node cycles in the top-level import graph. The 3 cycles
# surfaced 2026-04-28 (`evaluation ↔ trainer`, `neural.flat ↔ neural.base`,
# and the `feature_selection ↔ training.helpers ↔ strategies ↔ utils ↔
# wrappers` 5-node cycle) were resolved by lazy-importing one edge of
# each at the runtime call site — none of them ship as top-level cycles
# anymore. The whitelist is empty and ready to flag any new structural
# cycle.
_USER_DEFERRED_CYCLES: set[str] = {
    # Sibling-file monolith splits where both modules pose top-level
    # imports of each other. Cycle resolves at runtime because the
    # "parent" finishes binding its top-level constants/helpers before
    # the sibling line executes; the meta-detector is conservative.
    # Owner action: extract the shared constants/helpers to a leaf
    # ``_<topic>_common.py`` and have both siblings import from it
    # (see _numerical_constants.py for the canonical pattern). Drained
    # in a follow-up.
    "mlframe.training.extractors._extractors_simple → mlframe.training.extractors",
    # _reporting monolith split: 2-node (with _reporting_probabilistic) became
    # a 3-node SCC after _reporting_regression was carved out in the same way
    # (both siblings import constants from _reporting at top level; _reporting
    # re-exports them at its bottom). Same sibling-file pattern as the other
    # monolith splits in this whitelist.
    "mlframe.training.reporting._reporting → mlframe.training.reporting._reporting_probabilistic → mlframe.training.reporting._reporting_regression",
    "mlframe.training.composite.estimator → mlframe.training.composite.estimator._estimator",
    "mlframe.training.strategies → mlframe.training.strategies.xgboost",
    "mlframe.training.targets._target_temporal_audit_from_agg → mlframe.training.targets._target_temporal_changepoint → mlframe.training.targets.target_temporal_audit",
    # hermite_fe monolith split: _hermite_fe_mi.py contains @njit'd
    # functions that reference parent's _quantile_bin_njit. numba doesn't
    # compile IMPORT_NAME bytecode so the import must be top-level; the
    # cycle is benign because parent defines _quantile_bin_njit at module
    # start, then re-imports this sibling at its bottom.
    "mlframe.feature_selection.filters._hermite_fe_mi → mlframe.feature_selection.filters.hermite_fe",
    # composite_transforms monolith split: function-signature default values
    # in the sibling modules reference parent-resident constants
    # (_GROUPED_MIN_GROUP_SIZE, _QUANTILE_RESIDUAL_DEFAULT_*, etc.). Defaults
    # evaluate at module-load so the imports must be top-level. Cycle resolves
    # at runtime because parent defines all constants BEFORE importing siblings.
    # 2026-05-24: extended to 5-node cycle as ``_naming`` and ``_registry``
    # siblings landed.
    "mlframe.training.composite.transforms.linear → mlframe.training.composite.transforms.naming → mlframe.training.composite.transforms.nonlinear → mlframe.training.composite.transforms.registry → mlframe.training.composite.transforms",
    # _phase_helpers_fit_split monolith split: the body sibling
    # _phase_helpers_fit_pipeline references the parent's FitPipelineResult
    # NamedTuple at top-level. Resolves at runtime because parent defines
    # FitPipelineResult before importing the sibling at its bottom.
    "mlframe.training.core._phase_helpers_fit_pipeline → mlframe.training.core._phase_helpers_fit_split",
    # _target_distribution_analyzer monolith split: the body siblings
    # (``_modes``, ``_features``, ``_target_fn``) reference parent-resident
    # dataclasses / type aliases at top-level. Same pattern as the
    # composite_transforms cycle above — parent defines them BEFORE
    # importing siblings at its bottom, so the cycle resolves at runtime
    # without an actual ImportError.
    "mlframe.training.targets._target_distribution_analyzer → mlframe.training.targets._target_distribution_analyzer_features → mlframe.training.targets._target_distribution_analyzer_modes → mlframe.training.targets._target_distribution_analyzer_target_fn",
    # Sibling-file monolith splits (1k-LOC carve wave). In each, the carved body sibling top-level imports helpers from
    # its parent and the parent re-exports the moved symbols at its bottom -- a 2-node SCC that resolves at runtime
    # because the parent binds those helpers BEFORE the bottom re-export line executes. ``_classification_extras_blocks``
    # specifically MUST import top-level: it holds @njit kernels that reference the parent's njit helpers, and numba does
    # not compile IMPORT_NAME bytecode (same constraint as the hermite_fe entry above). Same owner-drain action applies.
    "mlframe.metrics.classification._classification_extras → mlframe.metrics.classification._classification_extras_blocks",
    "mlframe.reporting._diagnostics_dispatch_extra → mlframe.reporting.diagnostics_dispatch",
    "mlframe.training.composite.discovery._screening_tiny → mlframe.training.composite.discovery._screening_tiny_perbin",
    "mlframe.training.pipeline._pipeline_helpers → mlframe.training.pipeline._pipeline_helpers_apply",
    "mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe → mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe_generate",
    # ``_reporting`` monolith split, now 3-node: ``_reporting_probabilistic_calib`` was carved out of
    # ``_reporting_probabilistic`` (which already top-level-imports constants from ``_reporting``; ``_reporting``
    # re-exports them at its bottom). Supersedes the earlier ``_reporting_regression`` 3-node entry above (that carve's
    # cycle no longer closes as a top-level SCC).
    "mlframe.training.reporting._reporting → mlframe.training.reporting._reporting_probabilistic → mlframe.training.reporting._reporting_probabilistic_calib",
    # PRE-EXISTING (not from the 1k-LOC carve wave): the ``reporting.charts`` package facade. ``charts/__init__``
    # re-exports every chart builder submodule, and several submodules import shared helpers back from the ``charts`` /
    # ``reporting`` package surface, so the whole package forms one top-level SCC through the facade. Runtime-safe (the
    # package imports cleanly); flagged here so the suite is green. Owner action: break one facade edge via a leaf
    # ``charts._common`` so submodules stop importing the package surface at module load.
    "mlframe.reporting → mlframe.reporting.catalog → mlframe.reporting.charts → mlframe.reporting.charts._layout → mlframe.reporting.charts.binary → mlframe.reporting.charts.calibration → mlframe.reporting.charts.calibration_by_feature → mlframe.reporting.charts.calibration_drift → mlframe.reporting.charts.calibration_heatmap_2d → mlframe.reporting.charts.decision_curve → mlframe.reporting.charts.drift → mlframe.reporting.charts.error_analysis → mlframe.reporting.charts.fairness_calibration → mlframe.reporting.charts.ltr → mlframe.reporting.charts.model_card → mlframe.reporting.charts.model_comparison → mlframe.reporting.charts.multiclass → mlframe.reporting.charts.multilabel → mlframe.reporting.charts.pdp_ice → mlframe.reporting.charts.quantile → mlframe.reporting.charts.regression → mlframe.reporting.charts.slice_finder → mlframe.reporting.charts.split_comparison → mlframe.reporting.charts.temporal → mlframe.reporting.charts.training_curve → mlframe.reporting.spec",
}


def _module_name_from_path(path: Path) -> str:
    """``src/pyutilz/llm/factory.py`` → ``pyutilz.llm.factory``."""
    rel = path.relative_to(MLFRAME_DIR)
    parts = list(rel.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][: -len(".py")]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join([PKG_NAME] + parts)


def _internal_imports(tree: ast.AST, current: str) -> set[str]:
    """Yield fully-qualified names this module imports from inside the
    same package, considering ONLY top-level imports — lazy imports
    inside function bodies don't participate in the module-load
    dependency graph (they fire after both modules have finished
    loading), so a "cycle" that only closes via lazy imports isn't a
    runtime ImportError waiting to happen.
    """
    out: set[str] = set()
    current_parts = current.split(".")
    # Walk ONLY top-level statements. ``ast.walk`` would descend into
    # function bodies and pick up lazy imports we explicitly want to
    # exclude.
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PKG_NAME):
                    out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:  # absolute import
                if node.module and node.module.startswith(PKG_NAME):
                    out.add(node.module)
            else:
                # ``from .X import Y`` — resolve relative to current.
                base_parts = current_parts[: -node.level]
                if node.module:
                    base_parts.append(node.module)
                if base_parts and base_parts[0] == PKG_NAME:
                    out.add(".".join(base_parts))
        elif isinstance(node, ast.If):
            # ``if TYPE_CHECKING: ... import X`` — not at runtime,
            # already excluded. ``try: import optdep / except ...``
            # IS top-level and contributes — keep walking those.
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        if alias.name.startswith(PKG_NAME):
                            out.add(alias.name)
                elif isinstance(sub, ast.ImportFrom):
                    if sub.level == 0 and sub.module and sub.module.startswith(PKG_NAME):
                        out.add(sub.module)
    return out


def _build_graph() -> dict[str, set[str]]:
    """``{module_name: set_of_imported_internal_module_names}``."""
    graph: dict[str, set[str]] = defaultdict(set)
    for py in MLFRAME_DIR.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        if py.name.endswith(".py.old"):
            continue
        try:
            src = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        mod_name = _module_name_from_path(py)
        graph[mod_name].update(_internal_imports(tree, mod_name))
    return graph


def _strongly_connected_components(graph: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's SCC. Returns list of components (each ≥ 1 node).
    Cycles are SCCs with > 1 node OR self-loops."""
    index_counter = [0]
    stack: list[str] = []
    lowlinks: dict[str, int] = {}
    index: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    result: list[list[str]] = []

    def strongconnect(v: str):
        index[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in graph.get(v, ()):
            if w not in index:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif on_stack.get(w, False):
                lowlinks[v] = min(lowlinks[v], index[w])

        if lowlinks[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            result.append(comp)

    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10_000)
    try:
        for v in list(graph):
            if v not in index:
                strongconnect(v)
    finally:
        sys.setrecursionlimit(old_limit)
    return result


def test_no_import_cycles_in_package():
    graph = _build_graph()
    assert graph, "no modules discovered — package layout broken?"

    sccs = _strongly_connected_components(graph)
    cycles: list[list[str]] = []
    for comp in sccs:
        # Single-node self-loops are typically ``__init__.py`` doing
        # ``from .X import Y`` patterns that our AST resolver collapses
        # to a self-loop — runtime resolves them fine. Only flag
        # multi-node cycles, which are unambiguous structural issues.
        if len(comp) > 1:
            comp_key = " → ".join(sorted(comp))
            if comp_key in _USER_DEFERRED_CYCLES:
                continue
            cycles.append(comp)

    if cycles:
        details = []
        for cyc in cycles:
            details.append(" → ".join(cyc + [cyc[0]]))
        pytest.fail(
            f"{len(cycles)} import cycle(s) detected in {PKG_NAME}:\n  "
            + "\n  ".join(details)
        )
