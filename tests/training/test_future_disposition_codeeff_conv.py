"""Regression tests for FUTURE items from the Code-eff+Conversions disposition table.

Each test asserts a specific FUTURE item is resolved -- the canonical regression-test-for-bug-fix
pattern: the test must FAIL on pre-fix source and PASS on post-fix source.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest


def _read(rel: str) -> str:
    """Read a module source file from the mlframe src tree.

    Resolves relative to the repo root so tests are robust to where pytest is invoked from.

    Compat shim for the 2026-05-21 monolith split: when the requested module
    is ``_phase_train_one_target.py``, also append its body sibling so the
    source-pattern sensors that pre-date the split still match. The
    fingerprint / weight-loop / etc. code now lives in
    ``_phase_train_one_target_body.py``; the parent re-exports it.
    """
    here = Path(__file__).resolve()
    # tests/training/test_future_disposition_codeeff_conv.py -> repo root
    repo_root = here.parents[2]
    primary = (repo_root / "src" / "mlframe" / rel).read_text(encoding="utf-8")
    if rel.endswith("training/core/_phase_train_one_target.py"):
        _core_dir = repo_root / "src" / "mlframe" / "training" / "core"
        for _sib_name in (
            "_phase_train_one_target_body.py",
            "_phase_train_one_target_ensembling.py",
            "_phase_train_one_target_polars_fastpath.py",
            "_phase_train_one_target_pre_screen.py",
            "_phase_train_one_target_model_setup.py",
        ):
            _sib_path = _core_dir / _sib_name
            if _sib_path.exists():
                primary = primary + "\n" + _sib_path.read_text(encoding="utf-8")
    elif rel.endswith("training/core/main.py"):
        # 2026-05-22 split: ``train_mlframe_models_suite`` body moved to
        # ``_main_train_suite.py``; append so the source-pattern sensors
        # for the relocated kwargs / call-site code still match.
        sibling = repo_root / "src" / "mlframe" / "training" / "core" / "_main_train_suite.py"
        if sibling.exists():
            primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    return primary


# ---------- CODE-P1-4: run_temporal_audit_batch dead df param ----------

def test_codep14_run_temporal_audit_batch_has_no_df_param():
    from mlframe.training.core._phase_temporal_audit import run_temporal_audit_batch

    sig = inspect.signature(run_temporal_audit_batch)
    assert "df" not in sig.parameters, (
        "CODE-P1-4 regression: run_temporal_audit_batch should not declare a df param"
    )


def test_codep14_main_does_not_pass_df_to_temporal_audit():
    src = _read("training/core/main.py")
    # The call site must not pass df=...
    assert "run_temporal_audit_batch(" in src
    # crude but adequate: between the opening paren and closing paren of the call we should not see ``df=``
    call_idx = src.index("run_temporal_audit_batch(")
    closing = src.index(")", call_idx)
    call_block = src[call_idx:closing]
    assert "df=" not in call_block, "CODE-P1-4 regression: main.py still passes df= to run_temporal_audit_batch"


# ---------- CODE-P1-7: _prep_polars_df cycle-break -----------

def test_codep17_prep_polars_df_lives_in_misc_helpers():
    from mlframe.training.core import _misc_helpers as mh

    assert hasattr(mh, "_prep_polars_df"), "CODE-P1-7 regression: _prep_polars_df missing from _misc_helpers"


def test_codep17_no_local_main_import_in_train_one_target():
    src = _read("training/core/_phase_train_one_target.py")
    assert "from .main import _prep_polars_df" not in src, (
        "CODE-P1-7 regression: _train_one_target hot loop still locally imports _prep_polars_df from .main"
    )


# ---------- CODE-P1-10: fingerprint cache outside weight loop ----------

def test_codep110_fingerprint_cache_attr_on_ctx():
    from mlframe.training.core._training_context import TrainingContext

    ctx = TrainingContext()
    assert hasattr(ctx, "_model_input_fingerprint_cache"), (
        "CODE-P1-10 regression: TrainingContext must declare _model_input_fingerprint_cache"
    )
    assert ctx._model_input_fingerprint_cache == {}


def test_codep110_fingerprint_call_outside_weight_loop():
    """compute_model_input_fingerprint should be invoked at most once per (strategy, pre_pipeline),
    BEFORE the weight-schema iteration starts.

    We detect this by source-pattern: the call must appear before ``for weight_name, weight_values``.
    """
    src = _read("training/core/_phase_train_one_target.py")
    # Find the first occurrence of each
    fp_idx = src.find("compute_model_input_fingerprint(")
    weight_loop_idx = src.find("for weight_name, weight_values")
    assert fp_idx != -1 and weight_loop_idx != -1
    assert fp_idx < weight_loop_idx, (
        "CODE-P1-10 regression: compute_model_input_fingerprint is still called inside the weight loop"
    )
    # And only once in this module
    assert src.count("compute_model_input_fingerprint(") == 1, (
        "CODE-P1-10 regression: compute_model_input_fingerprint should be called exactly once"
    )


# ---------- CODE-P1-8: phase-runner namespace consolidation ----------

def test_codep18_phase_runners_namespace_present():
    """All 8 phase entry points must be importable from the consolidated namespace module."""
    from mlframe.training.core import _phase_runners as pr

    for name in (
        "apply_polars_categorical_fixes",
        "finalize_suite",
        "run_composite_post_processing",
        "run_composite_target_discovery",
        "run_temporal_audit_batch",
        "setup_configuration",
        "train_recurrent_models",
        "_train_one_target",
    ):
        assert hasattr(pr, name), f"_phase_runners missing {name}"


def test_codep18_main_uses_phase_runner_namespace():
    src = _read("training/core/main.py")
    # The consolidated import must be present.
    assert "from . import _phase_runners as pr" in src
    # And the calls should go through ``pr.``.
    for sym in (
        "pr.setup_configuration(",
        "pr.run_composite_target_discovery(",
        "pr.apply_polars_categorical_fixes(",
        "pr.run_temporal_audit_batch(",
        "pr._train_one_target(",
        "pr.train_recurrent_models(",
        "pr.finalize_suite(",
        "pr.run_composite_post_processing(",
    ):
        assert sym in src, f"main.py does not call through {sym}"


# ---------- CODE-P1-12: recurrent_models read from ctx ----------

def test_codep112_train_recurrent_models_reads_from_ctx():
    src = _read("training/core/main.py")
    # The call to train_recurrent_models must pass recurrent_models from ctx.recurrent_models
    # not from the closed-over function param. After the fix the call uses ctx.recurrent_models.
    assert "ctx.recurrent_models" in src
    # Quick proxy: between the train_recurrent_models call and its closing paren, the kwarg
    # recurrent_models= must be sourced from ctx.recurrent_models.
    call_idx = src.find("train_recurrent_models(")
    closing = src.index("\n    )", call_idx) if "\n    )" in src[call_idx:] else len(src)
    block = src[call_idx:closing]
    # heuristic: line containing recurrent_models= passes ctx.recurrent_models
    rec_line = [l for l in block.splitlines() if "recurrent_models=" in l and "ctx" not in l.split("=")[0]]
    if rec_line:
        # Either it reads from ctx or there is no recurrent_models= line at all.
        for l in rec_line:
            assert "ctx.recurrent_models" in l, (
                "CODE-P1-12 regression: train_recurrent_models() still receives the closed-over param "
                "instead of reading ctx.recurrent_models at call time"
            )


# ---------- CODE-P2-8: inspect import at module top ----------

def test_codep28_inspect_imported_at_module_top():
    """``inspect`` must be a module-level binding on _phase_train_one_target so it's available
    without any in-function ``import inspect`` (pre-fix the import happened inside hot paths,
    inflating per-call dispatch cost in tight loops)."""
    from mlframe.training.core import _phase_train_one_target as pt

    assert hasattr(pt, "inspect"), (
        "CODE-P2-8 regression: ``inspect`` not bound at module level of _phase_train_one_target; "
        "expected `import inspect` at module top."
    )
    import inspect as _inspect_canonical
    assert pt.inspect is _inspect_canonical, "module-level ``inspect`` is not the std-lib module"


# ---------- CODE-LOW-2: slug_to_original_target_name no-op write removed ----------

def test_codelow2_no_redundant_slug_assignment():
    """The single-line identity assignment slug_to_original_target_name[slugify(...)] = cur_target_name
    is the canonical write; any subsequent ``ctx.slug_to_original_target_name = local_dict`` would be a no-op."""
    src = _read("training/core/_phase_train_one_target.py")
    # There should be NO line that reassigns ctx.slug_to_original_target_name in this module.
    bad = [
        l for l in src.splitlines()
        if "ctx.slug_to_original_target_name =" in l
    ]
    assert not bad, (
        f"CODE-LOW-2 regression: redundant assignment to ctx.slug_to_original_target_name still present: {bad}"
    )


# ---------- CODE-LOW-3: models_dir read once ----------

def test_codelow3_models_dir_read_once():
    src = _read("training/core/_phase_train_one_target.py")
    # `models_dir = ctx.models_dir` must appear exactly once at the top of _train_one_target.
    count = src.count("models_dir = ctx.models_dir")
    assert count == 1, f"CODE-LOW-3 regression: models_dir = ctx.models_dir appears {count} times (expected 1)"


# ---------- CODE-LOW-7: dataset reuse cache helper ----------

def test_codelow7_dataset_reuse_cache_attrs_module_level():
    """The _DATASET_REUSE_CACHE_ATTRS tuple must be defined at module level so both
    forward-and-back transfer sites reference one canonical attribute list."""
    from mlframe.training.core import _phase_train_one_target as pt

    assert hasattr(pt, "_DATASET_REUSE_CACHE_ATTRS"), "CODE-LOW-7: tuple must be module-level"
    assert isinstance(pt._DATASET_REUSE_CACHE_ATTRS, tuple)
    assert "_cached_train_dmatrix" in pt._DATASET_REUSE_CACHE_ATTRS


def test_codelow7_dataset_reuse_helper_supports_bidirectional_transfer():
    """Both forward (template -> clone) and back (clone -> template, with skip_none) transfers
    must be expressible via the single shared helper. Pre-fix the back path was open-coded with
    a different attribute list. Behavioural surface: helper accepts skip_none kwarg and honours
    it (None values not stamped over existing destination values)."""
    from mlframe.training.core import _phase_train_one_target as pt

    class _Bag:
        pass

    src = _Bag()
    dst = _Bag()
    # Set a couple of attrs from the shared canonical list.
    attr = pt._DATASET_REUSE_CACHE_ATTRS[0]
    other = pt._DATASET_REUSE_CACHE_ATTRS[1] if len(pt._DATASET_REUSE_CACHE_ATTRS) > 1 else attr

    # Forward direction: src -> dst, including None values.
    setattr(src, attr, "fwd_value")
    setattr(src, other, None)
    setattr(dst, other, "preexisting")
    pt._forward_dataset_reuse_cache(src, dst)
    assert getattr(dst, attr) == "fwd_value", "forward transfer dropped non-None value"
    # Without skip_none None overwrites destination.
    assert getattr(dst, other) is None, "without skip_none, None must overwrite destination"

    # Back direction: dst -> src with skip_none=True; None on dst must NOT overwrite src.
    setattr(dst, attr, None)
    setattr(src, attr, "src_preexisting")
    pt._forward_dataset_reuse_cache(dst, src, skip_none=True)
    assert getattr(src, attr) == "src_preexisting", (
        "skip_none=True failed; None on source overwrote existing destination value"
    )


# ---------- CONV-MED-5: pandas view cache across strategies ----------

def test_convmed5_pandas_view_cache_attr_on_ctx():
    from mlframe.training.core._training_context import TrainingContext

    ctx = TrainingContext()
    assert hasattr(ctx, "_pandas_view_cache"), (
        "CONV-MED-5 regression: TrainingContext must declare _pandas_view_cache"
    )


def test_convmed5_cache_used_in_train_one_target():
    """The lazy pandas conversion site must hit ctx._pandas_view_cache before calling
    get_pandas_view_of_polars_df, so two non-Polars-native strategies converting the same
    source frame pay only one conversion total."""
    src = _read("training/core/_phase_train_one_target.py")
    assert "_pandas_view_cache" in src, (
        "CONV-MED-5 regression: lazy-conversion site does not consult ctx._pandas_view_cache"
    )


# ---------- CONV-LOW-15: np.isinf -> pl.Series.is_infinite ----------

def test_convlow15_preprocessing_uses_native_is_infinite():
    src = _read("training/preprocessing.py")
    # CONV-LOW-15: the polars _frame_contains_inf branch must use the native expression-side
    # ``df[name].is_infinite().any()`` rather than going through ``np.isinf(series.to_numpy())``.
    # We verify by source pattern: the polars-branch ``is_infinite()`` call must be present
    # and there must be NO ``np.isinf(`` invocation on a polars Series (only on the pandas
    # branch's ``.to_numpy()``, which already has a separate dtype guard).
    assert "is_infinite().any()" in src, (
        "CONV-LOW-15 regression: polars _frame_contains_inf branch must use ``is_infinite().any()``"
    )
    # Defensive: there must not be a polars-Series ``.is_infinite`` followed by ``.to_numpy()`` -
    # any np.isinf on .to_numpy() must be guarded behind a pandas dtype check, not a polars Series.
    # We scan: every ``np.isinf(...to_numpy()...)`` line must be reachable only after a pandas
    # ``select_dtypes`` upstream.
    for lineno, line in enumerate(src.splitlines(), start=1):
        if "np.isinf(" in line and ".to_numpy()" in line:
            # walk back to find the nearest enclosing block context
            prev = "\n".join(src.splitlines()[max(0, lineno - 30) : lineno])
            assert "select_dtypes" in prev, (
                f"CONV-LOW-15 regression: line {lineno} uses np.isinf(...to_numpy()) "
                f"outside a pandas select_dtypes block"
            )


# ---------- CODE-LOW-6: cProfile harness exists ----------

def test_codelow6_profile_harness_module_present():
    """The cProfile harness for train_mlframe_models_suite must live under tests/perf/."""
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    harness = repo_root / "tests" / "perf" / "profile_train_mlframe_models_suite.py"
    assert harness.is_file(), (
        f"CODE-LOW-6 regression: profile harness missing at {harness}"
    )
    txt = harness.read_text(encoding="utf-8")
    assert "cProfile" in txt and "tests/perf/results" in txt.replace("\\", "/"), (
        "CODE-LOW-6 regression: harness must invoke cProfile and write to tests/perf/results/"
    )


# ---------- CODE-P1-13: tqdmu_lazy_start audit ----------

def test_codep113_tqdmu_lazy_start_handles_single_item_internally():
    """``tqdmu_lazy_start`` already short-circuits to plain iteration when ``total < min_total``
    (default 2), so the per-iteration overhead on single-item loops is one ``len()`` + one
    comparison. All seven training-core call sites iterate over dynamic collections whose len
    cannot be statically pinned to 1, so we keep the wrapper -- the audit's "replace short-loops
    with plain iteration" is moot because the wrapper IS plain iteration on a 1-item input."""
    from pyutilz.system import tqdmu_lazy_start

    # Behavioural test: passing a 1-element iterable still yields the single element.
    out = list(tqdmu_lazy_start([42], desc="single-item-audit"))
    assert out == [42], "tqdmu_lazy_start must remain iter-equivalent on single-item input"
    # Two-element case still yields both items.
    out2 = list(tqdmu_lazy_start([1, 2], desc="two-item-audit"))
    assert out2 == [1, 2]


# ---------- CONV-HIGH-1: clone() gate documentation ----------

def test_convhigh1_clone_gate_documented():
    """The needs_polars_pre_clone clone()s must either be removed or carry an inline TODO/comment
    citing the destructive op that requires them. The audit verdict: keep if destructive, drop if not.

    Our post-fix state: the clone is gated behind ``needs_polars_pre_clone`` which is already a
    narrow predicate; we add a TODO comment pointing at the destructive site to document the gate.
    """
    src = _read("training/core/_phase_helpers.py")
    if "needs_polars_pre_clone" in src and ".clone()" in src:
        # If the gated clone still exists, a CONV-HIGH-1 marker must be present documenting why.
        assert "CONV-HIGH-1" in src, (
            "CONV-HIGH-1 regression: gated clone()s still present but no CONV-HIGH-1 TODO/marker "
            "explains the destructive operation that requires them"
        )
