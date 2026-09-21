"""A hand-kept table of transform names in ``src/`` must be a derived view of a ``Transform`` attribute, or carry a reason.

Name tables that encode a property drift out of sync with the registry: the soft-shrink set missed
``linear_residual_multi_robust`` and ``causal_anchor_residual``, the wrap-pass watchdog's additive set held
``quantile_residual`` (whose inverse is not additive in T) and missed most additive transforms, and the TST-03 tolerance
table lacked four names. Those are now read off ``Transform.linear_in_base`` / ``additive_in_t``. Every remaining literal
of at least five names, at least 80% of them registry names, is listed below with why it is a selection or an
implementation table rather than a property.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mlframe
from mlframe.training.composite.transforms import TRANSFORMS_REGISTRY

from tests.test_meta._scan_guard import assert_scanned_enough
from tests.test_meta._shared_ast_cache import parsed_ast

_SRC = Path(mlframe.__file__).resolve().parent

# relative path -> (number of such literals in the file, why they are not a property of the transform).
_ALLOWED: dict[str, tuple[int, str]] = {
    "training/_composite_target_discovery_config_base.py": (1, "the default discovery pool: a measured cost/benefit selection"),
    "training/composite/attribution.py": (1, "the neutral-T value of each multiplicative inverse: a per-transform constant table"),
    "training/composite/serving.py": (2, "the registry-free export table: a reimplementation of each exported inverse"),
    "training/composite/estimator/_smearing.py": (1, "y-only power transforms whose Jensen bias the smearing estimator corrects"),
    "training/composite/transforms/registry.py": (1, "the registry itself"),
    "training/composite/transforms/_registry_extended.py": (1, "the registry itself"),
    "training/_benchmarks/_profile_fuzz_1m.py": (1, "a benchmark's transform mode"),
}


def _name_literals(tree: ast.Module, names: set[str]) -> int:
    """Number of dict/set/list/tuple literals with at least 5 string keys, at least 80% of them registry names."""
    found = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            elts = node.keys
        elif isinstance(node, (ast.Set, ast.List, ast.Tuple)):
            elts = node.elts
        else:
            continue
        keys = [e.value for e in elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if len(keys) >= 5 and sum(k in names for k in keys) >= 0.8 * len(keys) and set(keys) != names:
            found += 1
    return found


def test_transform_name_tables_in_src_are_derived_or_explained():
    """No new hand-kept registry-name table in ``src/`` (benchmark folders excluded) without a reason above."""
    names = set(TRANSFORMS_REGISTRY)
    scanned = 0
    found: dict[str, int] = {}
    for path in sorted(_SRC.rglob("*.py")):
        rel = path.relative_to(_SRC).as_posix()
        if "/_benchmarks/" in f"/{rel}" and rel not in _ALLOWED:
            continue
        tree = parsed_ast(path)
        if tree is None:
            continue
        scanned += 1
        n = _name_literals(tree, names)
        if n:
            found[rel] = n
    assert_scanned_enough(scanned, "mlframe src modules")
    unexplained = {rel: n for rel, n in found.items() if n > _ALLOWED.get(rel, (0, ""))[0]}
    assert not unexplained, (
        f"hand-kept transform-name tables {unexplained}: derive them from a Transform attribute, or add the file to _ALLOWED with a reason"
    )
    stale = sorted(rel for rel in _ALLOWED if rel not in found)
    assert not stale, f"_ALLOWED entries whose literal is gone: {stale}"
