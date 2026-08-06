"""Regression tests for audits/full_audit_2026-07-21/x_test_suite_architecture.md findings F1-F4, F7, F8.

F5 (mlflow.py has zero test coverage) and F6 (kernel_tuning_cache CLI has zero test coverage) are
real, scoped test-gaps worth closing but are large enough (new unit-test files, not a few-line fix)
to warrant their own dedicated pass rather than folding into this consolidated fixes file -- tracked
as a concrete follow-up, not silently dropped. F9 (499 filler docstrings from an earlier bulk
docstring-coverage campaign) and F10 (5 lower-confidence, individually-documented "stale build path"
skips) are assessed: F9 would require redoing a 369-file campaign with no reported bug (a doc-quality
nit, not a correctness gap) -- deferred; F10 is explicitly already visible/commented per-site with a
prior diagnosis (not a fresh bug) -- left as-is, no action needed beyond the audit's own visibility
flag. PR6/PR7 are positive-pattern/proposal notes with no reported bug -- no fix needed.

Every check below asserts on actual runtime behaviour (exception propagation, marker/collection
outcomes, real state transitions) rather than on ``inspect.getsource()`` string matching -- see
X_TEST_SUITE_ARCHITECTURE-7 (this file used to be the sole getsource-based exception in this
repo's own behavioral-tests convention; converted, no whitelist entry needed anymore).
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import types

import pytest

# ---------------------------------------------------------------------------
# F1/F2: the _try_import_suite skip-on-ImportError pattern no longer masks a genuine API break in
# train_mlframe_models_suite across the 3 files that had it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path",
    [
        "tests.training.test_biz_val_training_core",
        "tests.training.test_suite_api_ergonomics",
        "tests.training.test_precompute_bundle",
    ],
)
def test_f1_f2_try_import_suite_no_longer_skips_on_import_failure(module_path):
    """F1/F2 REGRESSION: _try_import_suite must be a plain import (no defensive except/pytest.skip),
    so a genuine break in the underlying symbol raises ImportError instead of being swallowed.

    Behavioural proof: replace the real ``mlframe.training.core`` module with a stub missing
    ``train_mlframe_models_suite`` and call ``_try_import_suite()`` -- it must raise ImportError,
    not catch it and skip.
    """
    mod = importlib.import_module(module_path)
    fake_core = types.ModuleType("mlframe.training.core")  # deliberately missing the symbol
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "mlframe.training.core", fake_core)
        with pytest.raises(ImportError):
            mod._try_import_suite()


@pytest.mark.parametrize(
    "module_path, test_func_name",
    [
        ("tests.training.test_biz_val_training_core", "test_biz_val_training_suite_regression_completes"),
        ("tests.training.test_suite_api_ergonomics", "test_default_extractor_regression_matches_explicit"),
    ],
)
def test_f1_f2_no_typeerror_importerror_skip_around_suite_call(module_path, test_func_name, tmp_path):
    """F1/F2 REGRESSION: the actual train_mlframe_models_suite() call sites must not wrap the call
    in except (TypeError, ImportError): pytest.skip(...).

    Behavioural proof: monkeypatch ``mlframe.training.core.train_mlframe_models_suite`` to raise
    TypeError immediately (before any real training happens) and run the real test function that
    calls it -- the TypeError must propagate out uncaught, not be swallowed into a skip.
    """
    import mlframe.training.core as core_mod

    mod = importlib.import_module(module_path)
    test_func = getattr(mod, test_func_name)

    def _raise_typeerror(*_args, **_kwargs):
        """Simulate the suite call raising a kwarg-contract-break TypeError."""
        raise TypeError("simulated kwarg-contract break")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(core_mod, "train_mlframe_models_suite", _raise_typeerror)
        with pytest.raises(TypeError, match="simulated kwarg-contract break"):
            test_func(tmp_path)


# ---------------------------------------------------------------------------
# F3: the module-level broad except-and-skip around the shortlist-adapter suite import is gone
# ---------------------------------------------------------------------------


def test_f3_shortlist_adapter_suite_import_failure_propagates_not_skipped(monkeypatch):
    """F3 REGRESSION: test_shortlist_transformer_adapter_suite.py must import its dependencies
    directly, not wrap them in a broad except Exception: pytest.skip(..., allow_module_level=True).

    Behavioural proof: break the underlying ``mlframe.training.core`` symbol and force a genuine
    re-import of the test module -- it must raise ImportError, not raise/produce a Skipped outcome.
    """
    mod_name = "tests.training.test_shortlist_transformer_adapter_suite"
    fake_core = types.ModuleType("mlframe.training.core")  # deliberately missing the symbol
    monkeypatch.setitem(sys.modules, "mlframe.training.core", fake_core)
    monkeypatch.delitem(sys.modules, mod_name, raising=False)
    with pytest.raises(ImportError):
        importlib.import_module(mod_name)
    # Force the next real import (by this test or a sibling) to re-resolve against the genuine module.
    monkeypatch.delitem(sys.modules, mod_name, raising=False)


# ---------------------------------------------------------------------------
# F4: PR2's cheap visibility guard for the empty/stale .test_durations file. Implemented as an
# honest xfail (not a hard failure) since the actual fix requires a multi-hour scheduled CI job
# this session cannot trigger or control -- matches this repo's own "genuine external limitation"
# xfail carve-out rather than either silently ignoring the gap or turning the suite permanently red.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="F4 (x_test_suite_architecture.md): .test_durations is empty -- the weekly "
    "update-test-durations.yml scheduled job has never completed successfully (cancelled "
    "after 5h33m per gh run history); pytest-split falls back to file-count sharding until "
    "it does. Flip this to a hard assertion once that workflow run succeeds.",
    strict=False,
)
def test_f4_test_durations_file_is_populated():
    """F4: .test_durations must eventually hold real per-test timing data, not stay the seeded {}."""
    from pathlib import Path

    import orjson

    repo_root = Path(__file__).resolve().parents[2]
    durations = orjson.loads((repo_root / ".test_durations").read_bytes())
    assert len(durations) > 0, ".test_durations is still the empty seed {} -- the scheduled refresh job has not completed"


# ---------------------------------------------------------------------------
# F7: the 4 stopfile-callback smoke tests exercise real stop-detection behaviour, not just
# "constructor didn't raise". Independent behavioural pin against the callback classes themselves
# (not a copy of test_evaluation_salvage.py's own assertions) so a future weakening of THOSE tests
# cannot silently pass this regression guard too.
# ---------------------------------------------------------------------------


def test_f7_catboost_stopfile_callback_real_state_transition(tmp_path):
    """F7 REGRESSION: CatBoostStopFileCallback must actually flip its return value once the
    stop-file is planted, not just construct without raising."""
    pytest.importorskip("catboost")
    from mlframe.training.callbacks import CatBoostStopFileCallback

    flag = tmp_path / "stop.flag"
    cb = CatBoostStopFileCallback(str(flag))
    before = cb.after_iteration(info=None)
    flag.write_text("x")
    after = cb.after_iteration(info=None)
    assert before is True, "must NOT signal stop before the stop-file exists"
    assert after is False, "must signal stop (return False) once the stop-file exists"
    assert before != after, "regression: callback ignored the planted stop-file (state did not transition)"


def test_f7_lightgbm_stopfile_callback_real_state_transition(tmp_path):
    """F7 REGRESSION: LightGBMStopFileCallback must actually raise EarlyStopException once the
    stop-file is planted, not just construct without raising."""
    lgb = pytest.importorskip("lightgbm")
    from mlframe.training.callbacks import LightGBMStopFileCallback

    flag = tmp_path / "stop.flag"
    cb = LightGBMStopFileCallback(str(flag))

    class _FakeEnv:
        """Minimal stand-in for lightgbm's callback env (iteration + evaluation_result_list)."""

        iteration = 3
        evaluation_result_list = []

    cb(_FakeEnv())  # before the stop-file exists: must not raise
    flag.write_text("x")
    with pytest.raises(lgb.callback.EarlyStopException):
        cb(_FakeEnv())


def test_f7_xgboost_stopfile_callback_real_state_transition(tmp_path):
    """F7 REGRESSION: XGBoostStopFileCallback must actually flip its return value once the
    stop-file is planted, not just construct without raising."""
    pytest.importorskip("xgboost")
    from mlframe.training.callbacks import XGBoostStopFileCallback

    flag = tmp_path / "stop.flag"
    cb = XGBoostStopFileCallback(str(flag))
    before = cb.after_iteration(model=None, epoch=0, evals_log={})
    flag.write_text("x")
    after = cb.after_iteration(model=None, epoch=1, evals_log={})
    assert before is False, "must NOT signal stop before the stop-file exists"
    assert after is True, "must signal stop (return True) once the stop-file exists"
    assert before != after, "regression: callback ignored the planted stop-file (state did not transition)"


def test_f7_lightning_stopfile_callback_real_state_transition(tmp_path):
    """F7 REGRESSION: LightningStopFileCallback must actually set trainer.should_stop once the
    stop-file is planted, not just construct without raising."""
    pytest.importorskip("pytorch_lightning")
    from mlframe.training.callbacks import LightningStopFileCallback

    class _FakeTrainer:
        """Minimal stand-in exposing only the attribute the callback mutates."""

        should_stop = False

    flag = tmp_path / "stop.flag"
    cb = LightningStopFileCallback(str(flag))
    trainer = _FakeTrainer()
    cb.on_train_epoch_end(trainer, pl_module=None)
    before = trainer.should_stop
    flag.write_text("x")
    cb.on_train_epoch_end(trainer, pl_module=None)
    after = trainer.should_stop
    assert before is False, "must NOT signal stop before the stop-file exists"
    assert after is True, "must signal stop (should_stop=True) once the stop-file exists"
    assert before != after, "regression: callback ignored the planted stop-file (state did not transition)"


# ---------------------------------------------------------------------------
# F8: the 4 permanently-inert AutoGluon/LAMA training tests now use the same collected-and-
# explicitly-deselected opt-in pattern (--run-heavy-automl) as this suite's fuzz/biz_transformer tests
# ---------------------------------------------------------------------------


def test_f8_automl_heavy_tests_use_opt_in_marker_not_bare_skip():
    """F8 REGRESSION: the 4 real-training AutoGluon/LAMA tests must carry the heavy_automl pytest
    marker (not a bare @pytest.mark.skip), checked via the actual marker objects attached to the
    test functions -- not by searching the source text for a decorator string."""
    import tests.training.test_automl as mod

    marked = []
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type):  # test class: walk its methods
            for meth_name in dir(obj):
                if not meth_name.startswith("test_"):
                    continue
                meth = getattr(obj, meth_name)
                marks = {m.name for m in getattr(meth, "pytestmark", [])}
                if "heavy_automl" in marks:
                    marked.append(f"{name}.{meth_name}")
                assert "skip" not in marks, f"{name}.{meth_name} REGRESSION: reverted to a bare @pytest.mark.skip"

    assert len(marked) == 4, f"expected exactly 4 heavy_automl-marked AutoGluon/LAMA training tests, found {len(marked)}: {marked}"


def test_f8_run_heavy_automl_flag_registered():
    """F8: --run-heavy-automl must actually gate the 4 heavy_automl-marked tests, checked by
    running pytest itself (not by grepping conftest.py's source for the flag string): without the
    flag they are runtime-SKIPPED with the documented reason; with the flag registered as a real
    CLI option, collection of those same tests succeeds instead of erroring on an unknown arg."""
    from pathlib import Path

    repo_root = str(Path(__file__).resolve().parents[2])
    node_filter = ["-k", "test_basic_training or test_training_with_test_df"]

    without_flag = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/training/test_automl.py", "-q", "--no-cov", *node_filter],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert "4 skipped" in without_flag.stdout, f"expected the 4 heavy_automl tests to be runtime-skipped without --run-heavy-automl; got:\n{without_flag.stdout}"
    assert "--run-heavy-automl" in without_flag.stdout, f"skip reason must name the opt-in flag; got:\n{without_flag.stdout}"

    with_flag = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/training/test_automl.py", "--collect-only", "-q", "--no-cov", "--run-heavy-automl", *node_filter],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert with_flag.returncode == 0, f"--run-heavy-automl must be a registered option, not rejected as unrecognized; stderr:\n{with_flag.stderr}"
    assert (
        "4/23 tests collected" in with_flag.stdout or "4 tests collected" in with_flag.stdout
    ), f"expected the 4 heavy_automl tests collected (no longer runtime-skipped) with --run-heavy-automl; got:\n{with_flag.stdout}"
