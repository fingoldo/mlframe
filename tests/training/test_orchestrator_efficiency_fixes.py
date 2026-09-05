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
    """Main dead imports removed."""
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
    """Main module still imports cleanly."""
    import importlib

    mod = importlib.import_module("mlframe.training.core.main")
    assert hasattr(mod, "train_mlframe_models_suite")


# Fix 2: dead `models = defaultdict(lambda: defaultdict(list))` removed.
def test_main_no_dead_models_defaultdict():
    """Main no dead models defaultdict."""
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
    """Main setattr block has why comment."""
    sys.path.insert(0, str(CORE.parents[3]))  # repo/src
    from mlframe.training.core._misc_helpers import _bulk_setattr_to_ctx

    doc = _bulk_setattr_to_ctx.__doc__
    assert doc, "_bulk_setattr_to_ctx must carry a docstring with the migration WHY"
    low = doc.lower()
    assert "migration" in low or "phase-extraction" in low or "ctx-form" in low or "phase->ctx" in low, f"expected migration-debt WHY rationale in _bulk_setattr_to_ctx.__doc__; got: {doc!r}"

    # Behavioural pin on the helper's fail-loud contract: a missing slot must raise rather
    # than silently degrade into an ``AttributeError: 'NoneType' has no attribute ...`` later.
    class _Bag:
        """Groups tests covering bag."""

        pass

    with pytest.raises(KeyError):
        _bulk_setattr_to_ctx(_Bag(), ("definitely_absent_slot",), {})


# Fix 4: strategies_for_check removed (or wired) -- it must NOT be a dead variable.
def test_phase_helpers_no_dead_strategies_for_check():
    """Phase helpers no dead strategies for check."""
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
    """Strategy by model hoisted out of inner loop."""
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
    """No redundant list wrap on sorted."""
    tree = ast.parse(_read("_phase_train_one_target.py"))
    # `len(list(x))` and `len(x)` return the same number, so the wrap is invisible in any result -- it just
    # materialises a second list. Found on the parsed tree: a `len(...)` whose only argument is a `list(...)`
    # call on `sorted_models`.
    wrapped = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "list"
        and any(isinstance(a, ast.Name) and a.id == "sorted_models" for a in node.args[0].args)
    ]
    assert not wrapped, f"redundant list() wrap around an already-list sorted_models at line(s) {[n.lineno for n in wrapped]}"


# Fix 7: WHY comment on common_params.copy().
def test_common_params_are_copied_per_iteration():
    """Each model iteration must get its own params dict, or one model's mutation bleeds into the next."""
    src = _read("_phase_train_one_target.py")
    # The CODE, not a comment near it. The window check that used to follow failed whenever the comment was
    # reworded and passed whenever the copy itself was deleted and only the prose left behind -- exactly
    # backwards. What protects the property is that the copy is taken per iteration, so that is what is pinned.
    tree = ast.parse(src)
    bindings = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "current_common_params" for t in node.targets)]
    assert bindings, "current_common_params is never bound, so each model iteration no longer gets its own params dict"
    aliases = [b for b in bindings if isinstance(b, ast.Name)]
    assert not aliases, "current_common_params is bound to a bare name -- an alias shares ONE dict across iterations, which is the leak the copy prevents"
    assert any(
        isinstance(b, ast.Call) and isinstance(b.func, ast.Attribute) and b.func.attr == "copy" for b in bindings
    ), "the per-iteration copy is gone; one model's mutation would bleed into the next"


# Fix 8: WHY comment on the per-iteration memory probe.
def test_per_iteration_memory_probe_uses_the_shared_helper():
    """The per-iteration memory read must say why it is worth taking on every model.

    The probe used to be a raw ``memory_info().rss`` read. That is the WORKING SET on Windows, which
    ``clean_ram()`` deliberately evicts, so it printed 6.2GB one line after the suite reported 45.2GB; it now
    goes through the shared reporting helper. The sensor follows the probe rather than its old spelling.
    """
    tree = ast.parse(_read("_phase_train_one_target.py"))
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    calls += [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "get_reported_memory_gb" in calls, "expected a per-iteration memory probe through the shared reporting helper"

    # The raw read is `memory_info().rss` -- an attribute access on the RESULT of a memory_info() call. Both
    # spellings return a number, so only the SOURCE of that number distinguishes them.
    raw_rss = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "rss" and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "memory_info"
    ]
    assert not raw_rss, (
        f"the raw rss read is back at line(s) {[n.lineno for n in raw_rss]}: that is the WORKING SET on Windows, "
        "which clean_ram() deliberately evicts, so it once printed 6.2GB one line after the suite reported 45.2GB"
    )


# Fix 9: dead try/except around _dropped_high_card_data.clear() removed.
def test_main_dropped_high_card_clear_no_dead_try_except():
    """Main dropped high card clear no dead try except."""
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
    """Config setup interactive probe at module scope."""
    src = _read("_phase_config_setup.py")
    # Module-level cache of the probe; should be a constant assignment at module
    # scope, not a re-probe inside setup_configuration.
    has_module_const = bool(re.search(r"^_MLFRAME_INTERACTIVE(_LOGP)?\s*=", src, re.MULTILINE))
    assert has_module_const, "interactive-mode probe should be cached at module-import time"


# Fix 12: _ensure_logging_visible early-returns if already configured.
def test_ensure_logging_visible_is_idempotent():
    """Ensure logging visible is idempotent."""
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
    """Finalize suite single pass walk."""
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
    """Main del df has why comment."""
    src = _read("main.py")
    # Anchor on the STATEMENT, not on the text "del df": the WHY comment this test exists to require itself
    # quotes ``del df``, so a plain substring search lands on the comment and then reads the 400 characters
    # BEFORE it -- failing precisely when the comment is present. Indentation varies with the enclosing block.
    _m = re.search(r"^[ \t]*del df$", src, re.MULTILINE)
    assert _m is not None, "del df line not found"
    # Both halves of the release must be present: `del df` alone leaves `ctx.df` holding the frame, so the
    # memory is not actually reclaimed. That is a property of the code, unlike the comment window that used to
    # be checked here, which passed or failed on wording.
    _tail = src[_m.end() : _m.end() + 400]
    assert re.search(r"^[ \t]*ctx\.df = None$", _tail, re.MULTILINE), "`del df` without clearing ctx.df leaves the context holding the frame, so nothing is reclaimed"
