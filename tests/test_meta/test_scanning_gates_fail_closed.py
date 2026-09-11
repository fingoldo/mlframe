"""A meta-gate that scans zero files reports green forever.

Most gates in this directory locate the tree they check by walking up from ``__file__`` and globbing it.
That derivation fails silently: a renamed directory, an off-by-one in ``parents[N]``, or running outside a
source checkout yields an empty glob, no violations, and a pass. ``pytest.skip`` in that situation is no
better, since a skip is green in CI too -- ``test_no_file_over_1k_loc.py`` did exactly that until the
2026-09-08 review found it.

Fixing all fifty-odd in one mechanical pass would mean editing files whose scan shapes differ, without a
way to prove each edit preserved the gate. So this is a ratchet instead: every gate on the list below is a
known-unguarded one, the list may only shrink, and a NEW scanning gate must call
:func:`._scan_guard.assert_scanned_enough` (or carry its own explicit floor) from the start. Drain an entry
by adding the guard and deleting the name.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

META_DIR = Path(__file__).resolve().parent

# Every gate that globs a source tree without asserting the scan found anything. Measured 2026-09-08.
# Ratchet DOWN only: adding a name here requires a reason, removing one is always welcome.
UNGUARDED: set[str] = {
    "test_array_buffer_is_the_one_way_to_feed_a_hash.py",
    "test_baseline_registry_consistency.py",
    "test_config_field_consumption.py",
    "test_dead_helpers.py",
    "test_enum_exhaustiveness.py",
    "test_estimator_kwarg_parity.py",
    "test_fe_family_noop_no_copy.py",
    "test_filters_numba_invariant.py",
    "test_ktc_sweep_tolerances.py",
    "test_logger_lazy_formatting.py",
    "test_meta_meta.py",
    "test_module_cache_mutated_without_its_lock.py",
    "test_mrmr_research_doc_consistency.py",
    "test_no_audit_metadata_in_comments.py",
    "test_no_bare_except.py",
    "test_no_bare_pickle_load.py",
    "test_no_cast_in_compiled_functions.py",
    "test_no_fe_family_enabled_without_budget.py",
    "test_no_import_cycles.py",
    "test_no_inbound_edge_to_benchmarking.py",
    "test_no_inspect_getsource.py",
    "test_no_lazy_from_import_under_joblib_delayed.py",
    "test_no_module_level_env_mutation_in_tests.py",
    "test_no_module_level_logging_disable.py",
    "test_no_mutable_defaults.py",
    "test_no_njit_unsupported_numpy_reduction.py",
    "test_no_nondiscriminating_assert.py",
    "test_no_numba_config_env_restore_footgun.py",
    "test_no_single_shot_timing_assertion.py",
    "test_no_sklearn_metrics_in_production.py",
    "test_no_source_text_proxy.py",
    "test_no_stale_not_wired_docstrings.py",
    "test_no_stale_source_line_citations.py",
    "test_no_tick_isinstance_offset_check.py",
    "test_no_unlocked_module_cache.py",
    "test_no_unprotected_shap_treeexplainer.py",
    "test_orth_fe_recipes_freeze_preprocess_params.py",
    "test_public_annotations.py",
    "test_public_docstrings.py",
    "test_python_version_floor_respected.py",
    "test_readme_env_var_parity.py",
    "test_readme_modules_table_completeness.py",
    "test_readonly_to_numpy_mutation.py",
    "test_save_mlframe_model_lean_explicit.py",
    "test_sklearn_mixins_come_first.py",
    "test_subconfig_wiring_parity.py",
    "test_x_cicd_dependencies_fixes.py",
    "test_x_oss_hygiene_packaging_fixes.py",
}

_GUARD_PATTERNS = (
    "assert_scanned_enough",
    # A gate carrying its own explicit floor on how much it scanned counts as guarded.
    re.compile(r"assert\s+len\([a-z_]+\)\s*>=?\s*\d"),
    re.compile(r"assert\s+[a-z_]*(?:scanned|n_files|file_count)[a-z_]*\s*>=?\s*\d"),
)


def _calls_glob(tree: ast.AST) -> bool:
    """Whether this module actually CALLS glob/rglob, rather than merely mentioning it.

    An AST walk rather than a substring search: `"glob(" in text` is the source-text-proxy pattern the
    repo gates against, and it also matches the word inside a docstring or a comment, which is how a gate
    that scans nothing ends up on the list of gates that do.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in {"glob", "rglob", "iglob"}:
                return True
            # os.walk only. Bare `walk` would match ast.walk, which every AST gate in this directory calls
            # and which touches no filesystem at all -- including it reported three gates as scanners that
            # scan nothing.
            if name == "walk" and isinstance(func, ast.Attribute) and getattr(func.value, "id", None) == "os":
                return True
    return False


def _scanning_gates() -> set[str]:
    """Meta-test modules that glob a tree."""
    found = set()
    for p in META_DIR.glob("test_*.py"):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        if _calls_glob(tree):
            found.add(p.name)
    assert found, f"no scanning gates found in {META_DIR}; this gate cannot run and must not report green"
    return found


def _is_guarded(name: str) -> bool:
    """Whether a gate asserts its scan actually reached the tree."""
    text = (META_DIR / name).read_text(encoding="utf-8")
    return any(p in text if isinstance(p, str) else bool(p.search(text)) for p in _GUARD_PATTERNS)


def test_no_new_unguarded_scanning_gate():
    """A gate added without a scan guard would pass green while checking nothing."""
    new = sorted(name for name in _scanning_gates() if name not in UNGUARDED and not _is_guarded(name))
    assert not new, (
        f"{len(new)} scanning meta-gate(s) do not assert their scan found anything. Call "
        f"assert_scanned_enough(...) from tests/test_meta/_scan_guard.py right after the scan, so a broken "
        f"root-path derivation fails instead of reporting clean:\n" + "\n".join(f"  {n}" for n in new)
    )


def test_unguarded_list_has_no_stale_entries():
    """A drained or deleted entry must leave the list, or the ratchet stops meaning anything."""
    gates = _scanning_gates()
    gone = sorted(name for name in UNGUARDED if name not in gates)
    drained = sorted(name for name in UNGUARDED if name in gates and _is_guarded(name))
    assert not gone, "UNGUARDED names a gate that no longer exists (renamed or deleted): " + ", ".join(gone)
    assert not drained, f"{len(drained)} gate(s) now carry a scan guard. Remove them from UNGUARDED so a regression is " f"caught:\n" + "\n".join(
        f"  {n}" for n in drained
    )
