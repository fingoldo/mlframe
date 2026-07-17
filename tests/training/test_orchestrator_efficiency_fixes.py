"""Regression tests for orchestrator efficiency / dead-code fixes.

Each test maps 1:1 to a numbered fix in the batch and must FAIL pre-fix, PASS post-fix.
"""

from __future__ import annotations

import ast
import logging
import re
import sys
from pathlib import Path

import pytest


CORE = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "training" / "core"


def _read(name: str) -> str:
    """Read source by filename under training/core/.

    Monolith-split compat: when the requested file is a parent that had
    its body carved out (2026-05-21 ``_train_one_target``,
    2026-05-22 ``train_mlframe_models_suite``), concatenate the sibling
    so source-pattern sensors still match the relocated code.
    """
    primary = (CORE / name).read_text(encoding="utf-8")
    if name == "_phase_train_one_target.py":
        for _sib_name in (
            "_phase_train_one_target_body.py",
            "_phase_train_one_target_ensembling.py",
            "_phase_train_one_target_polars_fastpath.py",
            "_phase_train_one_target_pre_screen.py",
            "_phase_train_one_target_model_setup.py",
        ):
            _sib_path = CORE / _sib_name
            if _sib_path.exists():
                primary = primary + "\n" + _sib_path.read_text(encoding="utf-8")
    elif name == "main.py":
        sibling = CORE / "_main_train_suite.py"
        if sibling.exists():
            primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    return primary


# Fix 1: dead imports removed from main.py.
def test_main_dead_imports_removed():
    src = _read("main.py")
    tree = ast.parse(src)
    # Names known dead at audit time; post-fix, none should appear as bound import names.
    DEAD = {
        "sys",
        "timer",
        "glob",
        "deepcopy",
        "exists",
        "join",
        "TypeVar",
        "joblib",
        "psutil",
        "stats",
        "clone",
        "SimpleImputer",
        "StandardScaler",
        "ce",
        "BaselineDiagnostics",
        "format_baseline_diagnostics_report",
        "compute_label_distribution_drift",
        "format_drift_report",
        "load_mlframe_model",
        "LINEAR_MODEL_TYPES",
        "is_linear_model",
        "is_neural_model",
        "format_phase_summary",
        "make_train_test_split",
        "process_model",
        "select_target",
        "MRMR",
        "create_fairness_subgroups",
        "score_ensemble",
        "run_dummy_baselines",
        "run_per_target_diagnostics",
    }
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound = a.asname or a.name
                if bound in DEAD:
                    found.add(bound)
    assert not found, f"dead imports still present in main.py: {sorted(found)}"


def test_main_module_still_imports_cleanly():
    # `import mlframe.training.core.main` must not break after pruning.
    import importlib

    mod = importlib.import_module("mlframe.training.core.main")
    assert hasattr(mod, "train_mlframe_models_suite")


# Fix 2: dead `models = defaultdict(lambda: defaultdict(list))` removed.
def test_main_no_dead_models_defaultdict():
    src = _read("main.py")
    # Pattern was: `models = defaultdict(lambda: defaultdict(list))` immediately
    # overwritten on the next 20 lines by `models = ctx.models`.
    pat = re.compile(r"^\s*models\s*=\s*defaultdict\(lambda:\s*defaultdict\(list\)\)\s*$", re.MULTILINE)
    matches = pat.findall(src)
    assert not matches, "dead `models = defaultdict(...)` still present"


# Fix 3: migration-debt WHY rationale on the bulk setattr helper.
# Original test substring-matched the inline ``setattr(ctx, _k, locals()[_k])`` loop in
# main.py; Wave-7 routed every block through ``_bulk_setattr_to_ctx`` so the rationale
# now lives on the helper's docstring. Asserting on the helper is behavioural (docstring
# is part of the public contract, exposed via ``__doc__``), not source-inspection.
def test_main_setattr_block_has_why_comment():
    sys.path.insert(0, str(CORE.parents[3]))  # repo/src
    from mlframe.training.core._misc_helpers import _bulk_setattr_to_ctx

    doc = _bulk_setattr_to_ctx.__doc__
    assert doc, "_bulk_setattr_to_ctx must carry a docstring with the migration WHY"
    low = doc.lower()
    assert "migration" in low or "phase-extraction" in low or "ctx-form" in low or "phase->ctx" in low, (
        f"expected migration-debt WHY rationale in _bulk_setattr_to_ctx.__doc__; got: {doc!r}"
    )

    # Behavioural pin on the helper's fail-loud contract: a missing slot must raise rather
    # than silently degrade into an ``AttributeError: 'NoneType' has no attribute ...`` later.
    class _Bag:
        pass

    with pytest.raises(KeyError):
        _bulk_setattr_to_ctx(_Bag(), ("definitely_absent_slot",), {})


# Fix 4: strategies_for_check removed (or wired) -- it must NOT be a dead variable.
def test_phase_helpers_no_dead_strategies_for_check():
    src = _read("_phase_helpers.py")
    if "strategies_for_check" not in src:
        return  # cleanly removed
    # If still present, every name occurrence after the binding must reference it
    # (i.e. it must be read at least once outside its own assignment).
    pat_assign = re.compile(r"\bstrategies_for_check\s*=")
    pat_use = re.compile(r"\bstrategies_for_check\b")
    assignments = pat_assign.findall(src)
    uses = pat_use.findall(src)
    assert len(uses) > len(assignments), "strategies_for_check is bound but never read (kept as dead intermediate)"


# Fix 5: strategy_by_model hoisted out of per-target loop (or factored to helper).
def test_strategy_by_model_hoisted_out_of_inner_loop():
    src = _read("_phase_train_one_target.py")
    # The per-(pre_pipeline) loop starts around "for pre_pipeline, pre_pipeline_name in".
    # After fix: strategy_by_model must NOT appear AS AN ASSIGNMENT inside that loop body.
    m = re.search(r"for pre_pipeline, pre_pipeline_name in", src)
    assert m is not None, "outer pre_pipeline loop not found"
    body = src[m.start() :]
    # Find first assignment inside body
    assign_inside = re.search(r"^\s+strategy_by_model\s*=\s*\{id\(m\):", body, re.MULTILINE)
    assert assign_inside is None, "strategy_by_model is STILL recomputed inside the pre_pipeline loop; should be hoisted"


# Fix 6: len(list(sorted_models)) -> len(sorted_models).
def test_no_redundant_list_wrap_on_sorted():
    src = _read("_phase_train_one_target.py")
    assert "len(list(sorted_models))" not in src, "redundant `list()` wrap around already-list sorted_models still present"


# Fix 7: WHY comment on common_params.copy().
def test_common_params_copy_has_why_comment():
    src = _read("_phase_train_one_target.py")
    idx = src.find("current_common_params = common_params.copy()")
    assert idx >= 0, "expected per-iter common_params.copy() line"
    window = src[max(0, idx - 600) : idx]
    assert "isolation" in window.lower() or "bleed" in window.lower(), "expected WHY comment about isolation copy"


# Fix 8: WHY comment on per-iter psutil RSS probe.
def test_psutil_rss_sample_has_why_comment():
    src = _read("_phase_train_one_target.py")
    idx = src.find("memory_info().rss")
    assert idx >= 0, "expected per-iter psutil RSS sample"
    window = src[max(0, idx - 1200) : idx]
    assert "oom" in window.lower() or ("rss" in window.lower() and "intentional" in window.lower()), "expected WHY comment justifying per-iter RSS sample"


# Fix 9: dead try/except around _dropped_high_card_data.clear() removed.
def test_main_dropped_high_card_clear_no_dead_try_except():
    src = _read("main.py")
    # Either the entire ``try: _dropped_high_card_data.clear() except (NameError,
    # AttributeError): pass`` block is gone, or the except no longer lists those.
    bad_pat = re.compile(
        r"try:\s*_dropped_high_card_data\.clear\(\)\s*except\s*\(\s*NameError\s*,\s*AttributeError\s*\)\s*:\s*pass",
        re.DOTALL,
    )
    assert not bad_pat.search(src), "dead try/except around _dropped_high_card_data.clear() still present"


# Fix 11: _is_interactive_logp probe moved to module-import time.
def test_config_setup_interactive_probe_at_module_scope():
    src = _read("_phase_config_setup.py")
    # Module-level cache of the probe; should be a constant assignment at module
    # scope, not a re-probe inside setup_configuration.
    has_module_const = bool(re.search(r"^_MLFRAME_INTERACTIVE(_LOGP)?\s*=", src, re.MULTILINE))
    assert has_module_const, "interactive-mode probe should be cached at module-import time"


# Fix 12: _ensure_logging_visible early-returns if already configured.
def test_ensure_logging_visible_is_idempotent():
    src = _read("_misc_helpers.py")
    # After fix: function inspects root.handlers BEFORE mutating, returns early
    # when the asctime formatter is already in place.
    fn_match = re.search(r"def _ensure_logging_visible\([^)]*\)[^:]*:.*?(?=\ndef |\Z)", src, re.DOTALL)
    assert fn_match, "_ensure_logging_visible not found"
    body = fn_match.group(0)
    # Must contain an early-return guarded on asctime detection.
    assert re.search(r"return", body), "function lacks an early-return"
    assert "%(asctime)" in body, "function must reference asctime formatter literal"
    # Behavioural: a real second call must NOT add a handler.
    sys.path.insert(0, str(CORE.parents[3]))  # repo/src
    from mlframe.training.core._misc_helpers import _ensure_logging_visible

    root = logging.getLogger()
    _ensure_logging_visible()
    before = list(root.handlers)
    before_fmts = [getattr(h.formatter, "_fmt", None) for h in before]
    _ensure_logging_visible()
    after = list(root.handlers)
    after_fmts = [getattr(h.formatter, "_fmt", None) for h in after]
    assert len(after) == len(before), f"second call added handlers ({len(before)} -> {len(after)})"
    assert after_fmts == before_fmts, "second call mutated handler formatters"


# Fix 13: finalize_suite combines fairness + selected-features walks into one pass.
def test_finalize_suite_single_pass_walk():
    src = _read("_phase_finalize.py")
    tree = ast.parse(src)
    fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "finalize_suite")

    # Count top-level (depth=1 inside finalize_suite) ``for _ttype/_tt, _Y in ctx.models``-style walks.
    top_for_count = 0
    for stmt in fn.body:
        if isinstance(stmt, ast.For):
            # Pattern: iterating over ctx.models (.items()) or (ctx.models or {}).items()
            iter_src = ast.unparse(stmt.iter)
            if "ctx.models" in iter_src:
                top_for_count += 1
    assert top_for_count == 1, f"finalize_suite still has {top_for_count} top-level ctx.models walks; expected 1 after combine"


# Fix 14: WHY comment on `del df; ctx.df = None`.
def test_main_del_df_has_why_comment():
    src = _read("main.py")
    idx = src.find("del df\n    ctx.df = None")
    if idx < 0:
        idx = src.find("del df")
    assert idx >= 0, "del df line not found"
    window = src[max(0, idx - 400) : idx]
    assert "gc" in window.lower() or "decref" in window.lower() or "reclaim" in window.lower() or "free" in window.lower(), (
        "expected WHY comment on `del df; ctx.df = None`"
    )
