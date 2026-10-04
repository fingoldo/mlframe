"""Regression tests for FUTURE items from the Code-eff+Conversions disposition table.

Each test asserts a specific FUTURE item is resolved -- the canonical regression-test-for-bug-fix
pattern: the test must FAIL on pre-fix source and PASS on post-fix source.
"""

from __future__ import annotations

import ast
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
            "_phase_train_one_target_steps.py",
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
        # ``_main_train_suite.py``. Subsequent splits carved the phase loop
        # into ``_main_train_suite_phases.py`` and the target-distribution
        # helpers into ``_main_train_suite_target_distribution.py``. Append
        # every sibling that exists so source-pattern sensors for relocated
        # call-sites still match.
        _core_dir = repo_root / "src" / "mlframe" / "training" / "core"
        for _sib_name in (
            "_main_train_suite.py",
            "_main_train_suite_phases.py",
            "_main_train_suite_target_distribution.py",
        ):
            _sib_path = _core_dir / _sib_name
            if _sib_path.exists():
                primary = primary + "\n" + _sib_path.read_text(encoding="utf-8")
    return primary


# ---------- CODE-P1-4: run_temporal_audit_batch dead df param ----------


def test_codep14_run_temporal_audit_batch_has_no_df_param():
    """Codep14 run temporal audit batch has no df param."""
    from mlframe.training.core._phase_temporal_audit import run_temporal_audit_batch

    sig = inspect.signature(run_temporal_audit_batch)
    assert "df" not in sig.parameters, "CODE-P1-4 regression: run_temporal_audit_batch should not declare a df param"


def test_codep14_main_does_not_pass_df_to_temporal_audit():
    """Every call of run_temporal_audit_batch reachable from main.py passes no ``df`` keyword."""
    tree = ast.parse(_read("training/core/main.py"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) == "run_temporal_audit_batch"]
    assert len(calls) == 1
    assert "df" not in {kw.arg for kw in calls[0].keywords}


# ---------- CODE-P1-7: _prep_polars_df cycle-break -----------


def test_codep17_prep_polars_df_lives_in_misc_helpers():
    """Codep17 prep polars df lives in misc helpers."""
    from mlframe.training.core import _misc_helpers as mh

    assert hasattr(mh, "_prep_polars_df"), "CODE-P1-7 regression: _prep_polars_df missing from _misc_helpers"


def test_codep17_no_local_main_import_in_train_one_target():
    """The _train_one_target hot loop does not import _prep_polars_df from .main at call time."""
    tree = ast.parse(_read("training/core/_phase_train_one_target.py"))
    imports_from_main = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module == "main"]
    assert not [a.name for n in imports_from_main for a in n.names if a.name == "_prep_polars_df"]


# ---------- CODE-P1-10: fingerprint cache outside weight loop ----------


def test_codep110_fingerprint_cache_attr_on_ctx():
    """Codep110 fingerprint cache attr on ctx."""
    from mlframe.training.core._training_context import TrainingContext

    ctx = TrainingContext()
    assert hasattr(ctx, "_model_input_fingerprint_cache"), "CODE-P1-10 regression: TrainingContext must declare _model_input_fingerprint_cache"
    assert ctx._model_input_fingerprint_cache == {}


def _fingerprint_calls_for(tmp_path, monkeypatch, weight_schemas) -> int:
    """Trains one small lgb suite with ``weight_schemas`` and counts input-fingerprint computations."""
    import numpy as np
    import pandas as pd

    pytest.importorskip("lightgbm")
    from mlframe.training import OutputConfig
    from mlframe.training.core import _phase_train_one_target_steps as body
    from mlframe.training.core import train_mlframe_models_suite
    from tests.training.shared import SimpleFeaturesAndTargetsExtractor

    calls = []
    real = body.compute_cached_model_input_fingerprint

    def _spy(**kwargs):
        """Counts the call, then fingerprints for real."""
        calls.append(kwargs["pre_pipeline_name"])
        return real(**kwargs)

    monkeypatch.setattr(body, "compute_cached_model_input_fingerprint", _spy)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    df["target"] = df["a"] + rng.normal(scale=0.1, size=300)
    name = "fp_" + "_".join(weight_schemas)
    train_mlframe_models_suite(
        df=df,
        target_name="t",
        model_name=name,
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True, weight_schemas=list(weight_schemas)),
        mlframe_models=["lgb"],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=str(tmp_path / name), models_dir="models"),
        verbose=0,
        hyperparams_config={"iterations": 10},
    )
    return len(calls)


def test_codep110_fingerprint_call_outside_weight_loop(tmp_path, monkeypatch):
    """The model-input fingerprint is computed once per (strategy, pre_pipeline), before the
    weight-schema loop: three weighting schemas cost exactly as many fingerprint calls as one."""
    one = _fingerprint_calls_for(tmp_path, monkeypatch, ["uniform"])
    three = _fingerprint_calls_for(tmp_path, monkeypatch, ["uniform", "recency", "flat"])
    assert one >= 1
    assert three == one, f"CODE-P1-10 regression: {three} fingerprint calls for 3 weight schemas vs {one} for 1"


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
    """main.py imports ``_phase_runners as pr`` and routes every phase entry point through ``pr`` / ``pr_module``."""
    # The per-target loop lives in its own module and receives the ``pr`` namespace as an argument; the post-loop tail lives in
    # _main_train_suite_phases and hands some phase functions on as ``pr_module.X`` callables rather than calling them inline.
    used = set()
    imports_pr = False
    for rel in ("training/core/main.py", "training/core/_main_train_suite_target_loop.py", "training/core/_main_train_suite_phases.py"):
        tree = ast.parse(_read(rel))
        used.update(n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in ("pr", "pr_module"))
        imports_pr = imports_pr or any(
            isinstance(n, ast.ImportFrom) and n.level == 1 and n.module is None and any(a.name == "_phase_runners" and a.asname == "pr" for a in n.names)
            for n in ast.walk(tree)
        )
    assert imports_pr
    expected = {
        "setup_configuration",
        "run_composite_target_discovery",
        "apply_polars_categorical_fixes",
        "run_temporal_audit_batch",
        "_train_one_target",
        "train_recurrent_models",
        "finalize_suite",
        "run_composite_post_processing",
    }
    assert expected - used == set()


# ---------- CODE-P1-12: recurrent_models read from ctx ----------


def test_codep112_train_recurrent_models_reads_from_ctx():
    """train_recurrent_models receives ctx.recurrent_models as it is at call time, not a value captured earlier."""
    import types

    from mlframe.training.core._main_train_suite_phases import run_recurrent_finalize_and_composite_post

    class _Stop(Exception):
        """Sentinel raised to abort once the kwargs are captured."""
        pass

    received = {}

    def _train_recurrent_models(**kwargs):
        """Capture the keyword arguments, then abort."""
        received.update(kwargs)
        raise _Stop

    ctx = types.SimpleNamespace(models={}, recurrent_models=["lstm"])
    ctx.recurrent_models = ["gru"]  # changed after construction: the call must see this one
    pr = types.SimpleNamespace(train_recurrent_models=_train_recurrent_models)
    names = run_recurrent_finalize_and_composite_post.__code__.co_varnames[: run_recurrent_finalize_and_composite_post.__code__.co_argcount]
    args = {name: None for name in names}
    args.update(ctx=ctx, pr_module=pr, model_name="m", target_name="t", verbose=0)
    with pytest.raises(_Stop):
        run_recurrent_finalize_and_composite_post(**args)
    assert received["recurrent_models"] == ["gru"] and received["ctx"] is ctx


# ---------- CODE-P2-8: inspect import at module top ----------


def test_codep28_inspect_imported_at_module_top():
    """``inspect`` must be a module-level binding on _phase_train_one_target so it's available
    without any in-function ``import inspect`` (pre-fix the import happened inside hot paths,
    inflating per-call dispatch cost in tight loops)."""
    from mlframe.training.core import _phase_train_one_target as pt

    assert hasattr(
        pt, "inspect"
    ), "CODE-P2-8 regression: ``inspect`` not bound at module level of _phase_train_one_target; expected `import inspect` at module top."
    import inspect as _inspect_canonical

    assert pt.inspect is _inspect_canonical, "module-level ``inspect`` is not the std-lib module"


# ---------- CODE-LOW-2: slug_to_original_target_name no-op write removed ----------


def test_codelow2_no_redundant_slug_assignment():
    """The single-line identity assignment slug_to_original_target_name[slugify(...)] = cur_target_name
    is the canonical write; any subsequent ``ctx.slug_to_original_target_name = local_dict`` would be a no-op."""
    src = _read("training/core/_phase_train_one_target.py")
    # There should be NO line that reassigns ctx.slug_to_original_target_name in this module.
    bad = [line for line in src.splitlines() if "ctx.slug_to_original_target_name =" in line]
    assert not bad, f"CODE-LOW-2 regression: redundant assignment to ctx.slug_to_original_target_name still present: {bad}"


# ---------- CODE-LOW-3: models_dir read once ----------


def test_codelow3_models_dir_read_once():
    """``models_dir = ctx.models_dir`` is bound exactly once across the _train_one_target modules."""
    tree = ast.parse(_read("training/core/_phase_train_one_target.py"))
    binds = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and [t.id for t in n.targets if isinstance(t, ast.Name)] == ["models_dir"]
        and isinstance(n.value, ast.Attribute)
        and n.value.attr == "models_dir"
        and isinstance(n.value.value, ast.Name)
        and n.value.value.id == "ctx"
    ]
    assert len(binds) == 1


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
        """Groups tests covering bag."""
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
    assert getattr(src, attr) == "src_preexisting", "skip_none=True failed; None on source overwrote existing destination value"


# ---------- CONV-MED-5: pandas view cache across strategies ----------


def test_convmed5_pandas_view_cache_attr_on_ctx():
    """Convmed5 pandas view cache attr on ctx."""
    from mlframe.training.core._training_context import TrainingContext

    ctx = TrainingContext()
    assert hasattr(ctx, "_pandas_view_cache"), "CONV-MED-5 regression: TrainingContext must declare _pandas_view_cache"


def test_convmed5_cache_used_in_train_one_target(monkeypatch):
    """Two lazy pandas conversions of the same polars frame cost one conversion: the second is a cache hit on ctx._pandas_view_cache."""
    import collections
    import types

    import polars as pl

    from mlframe.training.core import _phase_train_one_target_polars_fastpath as fp
    from mlframe.training.strategies import get_strategy

    conversions = []
    real = fp.get_pandas_view_of_polars_df

    def _spy(df):
        """Count the conversion, then convert for real."""
        conversions.append(id(df))
        return real(df)

    monkeypatch.setattr(fp, "get_pandas_view_of_polars_df", _spy)
    strategy = get_strategy("linear")
    assert not strategy.supports_polars
    frame = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    ctx = types.SimpleNamespace(_pandas_view_cache=collections.OrderedDict(), _cache_stats={})
    views = []
    for _ in range(2):
        common_params = {"train_df": frame, "val_df": None, "test_df": None}
        out = fp._prepare_strategy_inputs(
            polars_fastpath_active=False,
            mlframe_model_name="linear",
            strategy=strategy,
            cat_features=[],
            text_features=[],
            embedding_features=[],
            train_df_polars=frame,
            val_df_polars=None,
            test_df_polars=None,
            prepared_frames_cache={},
            tier_dfs_cache={},
            tier_enum_map_cache={},
            common_params=common_params,
            pre_pipeline_name="",
            ctx=ctx,
            verbose=False,
        )
        assert out["polars_fastpath_active"] is False
        views.append(common_params["train_df"])
    assert conversions == [id(frame)]
    assert views[0] is views[1]
    assert ctx._cache_stats["pandas_view_cache"] == {"hits": 1, "misses": 1}


# ---------- CONV-LOW-15: np.isinf -> pl.Series.is_infinite ----------


def test_convlow15_preprocessing_uses_native_is_infinite():
    """``_frame_contains_inf`` finds +/-inf in polars and pandas float columns (nullable included) and nothing in finite or integer frames."""
    import numpy as np
    import pandas as pd
    import polars as pl

    from mlframe.training.preprocessing import _frame_contains_inf

    assert _frame_contains_inf(pl.DataFrame({"a": [1.0, float("inf")], "b": [1, 2]})) is True
    assert _frame_contains_inf(pl.DataFrame({"a": [1.0, float("-inf")]})) is True
    assert _frame_contains_inf(pl.DataFrame({"a": [1.0, None, 3.0], "b": [1, 2, 3]})) is False
    assert _frame_contains_inf(pl.DataFrame({"s": ["x", "y"], "n": [1, 2]})) is False
    assert _frame_contains_inf(pd.DataFrame({"a": [1.0, np.inf], "b": [1, 2]})) is True
    assert _frame_contains_inf(pd.DataFrame({"a": [1.0, -np.inf]})) is True
    assert _frame_contains_inf(pd.DataFrame({"a": [1.0, np.nan, 3.0]})) is False
    assert _frame_contains_inf(pd.DataFrame({"a": pd.array([1.0, np.inf, None], dtype="Float64")})) is True
    assert _frame_contains_inf(pd.DataFrame({"a": pd.array([1.0, None, 2.0], dtype="Float64")})) is False


# ---------- CODE-LOW-6: cProfile harness exists ----------


def test_codelow6_profile_harness_module_present(tmp_path, monkeypatch):
    """The cProfile harness for train_mlframe_models_suite runs ``profile`` with its defaults and writes under tests/perf/results/."""
    import importlib.util

    repo_root = Path(__file__).resolve().parents[2]
    harness = repo_root / "tests" / "perf" / "profile_train_mlframe_models_suite.py"
    assert harness.is_file()
    spec = importlib.util.spec_from_file_location("p1_profile_harness", harness)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen = {}

    def _fake_profile(**kwargs):
        """Record the harness arguments instead of profiling a suite."""
        seen.update(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr(module, "profile", _fake_profile)
    monkeypatch.setattr("sys.argv", ["harness"])
    assert module.main() == 0
    assert seen["n_rows"] == 2000 and seen["top"] == 30
    assert Path(seen["output_path"]).parts[-3:] == ("perf", "results", "train_mlframe_models_suite.prof")
    custom = tmp_path / "custom.prof"
    monkeypatch.setattr("sys.argv", ["harness", "--n-rows", "123", "--output", str(custom)])
    assert module.main() == 0
    assert seen["n_rows"] == 123 and seen["output_path"] == custom


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
    """The pre-encoding polars clone is kept only when categorical encoding will actually mutate the polars frames."""
    from mlframe.training.core._main_train_suite_polars_gate import needs_polars_pre_clone

    assert needs_polars_pre_clone({"categorical_encoding": "ordinal"}, was_polars_input=True) is True
    assert needs_polars_pre_clone({"categorical_encoding": "ordinal"}, was_polars_input=False) is False
    assert needs_polars_pre_clone({"categorical_encoding": "ordinal", "skip_categorical_encoding": True}, was_polars_input=True) is False
    assert needs_polars_pre_clone({"categorical_encoding": None}, was_polars_input=True) is False
    assert needs_polars_pre_clone({}, was_polars_input=True) is False
    assert needs_polars_pre_clone(None, was_polars_input=True) is False
    cfg = type("Cfg", (), {"categorical_encoding": "target", "skip_categorical_encoding": False})()
    assert needs_polars_pre_clone(cfg, was_polars_input=True) is True
