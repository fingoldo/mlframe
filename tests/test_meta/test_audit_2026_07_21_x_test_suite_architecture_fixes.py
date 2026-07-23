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
"""

from __future__ import annotations

import inspect

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
    """F1/F2 REGRESSION: _try_import_suite must be a plain import (no defensive
    except/pytest.skip) so a genuine API break fails loudly instead of silently skipping."""
    import importlib

    mod = importlib.import_module(module_path)
    src = inspect.getsource(mod._try_import_suite)
    assert "pytest.skip" not in src, f"{module_path}._try_import_suite must not silently skip on import failure"
    body = src.split('"""', 2)[-1]  # strip the docstring (which explains the fix and may say "except" in prose)
    assert "except" not in body, f"{module_path}._try_import_suite must not defensively catch the import"


@pytest.mark.parametrize(
    "module_path",
    [
        "tests.training.test_biz_val_training_core",
        "tests.training.test_suite_api_ergonomics",
        "tests.training.test_precompute_bundle",
    ],
)
def test_f1_f2_no_typeerror_importerror_skip_around_suite_call(module_path):
    """F1/F2 REGRESSION: the actual train_mlframe_models_suite() call sites must not wrap the call
    in except (TypeError, ImportError): pytest.skip(...) either."""
    import importlib

    mod = importlib.import_module(module_path)
    src = inspect.getsource(mod)
    assert "except (TypeError, ImportError)" not in src


# ---------------------------------------------------------------------------
# F3: the module-level broad except-and-skip around the shortlist-adapter suite import is gone
# ---------------------------------------------------------------------------


def test_f3_shortlist_adapter_suite_no_module_level_skip_on_import():
    """F3 REGRESSION: test_shortlist_transformer_adapter_suite.py must import its dependencies
    directly, not wrap them in a broad except Exception: pytest.skip(..., allow_module_level=True)."""
    import tests.training.test_shortlist_transformer_adapter_suite as mod

    src = inspect.getsource(mod)
    assert "allow_module_level=True" not in src
    assert "except Exception" not in src.split("class ")[0] if "class " in src else "except Exception" not in src


# ---------------------------------------------------------------------------
# F4: PR2's cheap visibility guard for the empty/stale .test_durations file. Implemented as an
# honest xfail (not a hard failure) since the actual fix requires a multi-hour scheduled CI job
# this session cannot trigger or control -- matches this repo's own "genuine external limitation"
# xfail carve-out rather than either silently ignoring the gap or turning the suite permanently red.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="F4 (x_test_suite_architecture.md): .test_durations is empty -- the weekly "
                    "update-test-durations.yml scheduled job has never completed successfully (cancelled "
                    "after 5h33m per gh run history); pytest-split falls back to file-count sharding until "
                    "it does. Flip this to a hard assertion once that workflow run succeeds.", strict=False)
def test_f4_test_durations_file_is_populated():
    """F4: .test_durations must eventually hold real per-test timing data, not stay the seeded {}."""
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    durations = json.loads((repo_root / ".test_durations").read_text(encoding="utf-8"))
    assert len(durations) > 0, ".test_durations is still the empty seed {} -- the scheduled refresh job has not completed"


# ---------------------------------------------------------------------------
# F7: the 4 stopfile-callback smoke tests now assert real stop-detection behaviour, not just
# "constructor didn't raise" -- covered by tests/evaluation/test_evaluation_salvage.py's own
# rewritten test_{catboost,lightgbm,xgboost,lightning}_stopfile_smoke; sanity-checked here that the
# weak "assert cb is not None" pattern is gone.
# ---------------------------------------------------------------------------


def test_f7_stopfile_smoke_tests_no_longer_weak_assertion_only():
    """F7 REGRESSION: the 4 stopfile-callback smoke tests must exercise real stop-detection
    behaviour, not just assert the constructor returned a non-None object."""
    import tests.evaluation.test_evaluation_salvage as mod

    for name in (
        "test_catboost_stopfile_smoke",
        "test_lightgbm_stopfile_smoke",
        "test_xgboost_stopfile_smoke",
        "test_lightning_stopfile_smoke",
    ):
        src = inspect.getsource(getattr(mod, name))
        assert "assert cb is not None" not in src, f"{name} REGRESSION: must not be reduced back to a tautological is-not-None check"
        assert "write_text" in src, f"{name} must actually plant the stop-file and assert the resulting state transition"


# ---------------------------------------------------------------------------
# F8: the 4 permanently-inert AutoGluon/LAMA training tests now use the same collected-and-
# explicitly-deselected opt-in pattern (--run-heavy-automl) as this suite's fuzz/biz_transformer tests
# ---------------------------------------------------------------------------


def test_f8_automl_heavy_tests_use_opt_in_marker_not_bare_skip():
    """F8 REGRESSION: the 4 real-training AutoGluon/LAMA tests must use the heavy_automl marker
    (collected + explicitly deselected, opt-in-able via --run-heavy-automl), not a bare
    @pytest.mark.skip that can never be turned on."""
    import tests.training.test_automl as mod

    src = inspect.getsource(mod)
    assert 'reason="AutoGluon heavy dependency - run manually if needed"' not in src
    assert 'reason="LAMA heavy dependency - run manually if needed"' not in src
    assert src.count("@pytest.mark.heavy_automl") == 4


def test_f8_run_heavy_automl_flag_registered():
    """F8: conftest.py must register the --run-heavy-automl CLI flag and the heavy_automl marker,
    matching the existing --run-fuzz / --run-biz-transformer pattern."""
    import tests.conftest as conftest_mod

    src = inspect.getsource(conftest_mod)
    assert "--run-heavy-automl" in src
    assert '"heavy_automl" in item.keywords' in src
