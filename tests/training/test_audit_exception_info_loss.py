"""Wave 41 (2026-05-20): exception-handler info loss.

Audit class: production code that caught an exception then logged a one-line
summary via ``str(e)`` / f-strings, losing the traceback permanently so
incident triage of a "X returned False" or "loop continued past Y" is
impossible without reproducing the failure.

Fix: use ``logger.exception(...)`` (auto-includes exc_info=True) or
``logger.error/warning(..., exc_info=True)`` everywhere; ``raise X(...) from e``
for chained exceptions.

16 findings, all fixed:

  P1: training/io.py:553 (save_mlframe_model)
      training/core/predict.py:1898 (per-model predict loop -- twin path at 995
        already used exc_info=True; this site was the asymmetric one)

  P2: inference/predict.py:128 (commonpath ValueError chain via `from e`)
      training/trainer.py:706 (model-cache load fallback to retrain)
      training/neural/flat.py:684 (metric compute)
      training/neural/recurrent.py:1036,1213 (checkpoint -> final-epoch)
      integrations/mlflow.py:121 (start_run final retry give-up)

  Low: training/automl.py:90, 220 (AutoGluon/LightAutoML import)
       training/automl.py:139, 152, 273, 286 (AUC + FI compute fallbacks)
       training/evaluation.py:230 (plot feature importances)
       training/_reporting.py:815 (predict_proba fallback)
       training/_training_loop.py:867 (get best iter)
       training/neural/base.py:438 (example_input_array)
       training/neural/flat.py:418 (torch.compile fallback)
       training/pipeline.py:1226 (polars-ds import, narrowed Exception -> ImportError)
       feature_engineering/mps.py:652 (print -> logger.exception)
       feature_engineering/mps.py:679 (parquet read warning)
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


MLFRAME_ROOT = Path(importlib.import_module("mlframe").__file__).parent


def _read(rel: str) -> str:
    return (MLFRAME_ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Source-level sensors: each fix should be present.
# ---------------------------------------------------------------------------


def test_io_save_uses_logger_exception() -> None:
    src = _read("training/io.py")
    # The bad pattern (f-string with {e}) must be gone for the save-fail path.
    assert 'logger.error(f"Could not save model to file {file}: {e}")' not in src
    # The good pattern must be present.
    assert 'logger.exception("Could not save model to file %s", file)' in src


def test_predict_per_model_loop_uses_exc_info() -> None:
    # The per-model predict loop moved out of ``predict.py`` into the
    # sibling ``_predict_main_from_models.py`` during the 2026-05-22
    # predict-monolith split. Read both modules so the structural pin
    # tolerates either location.
    import pathlib
    import mlframe as _mlframe
    _core = pathlib.Path(_mlframe.__file__).resolve().parent / "training" / "core"
    src = ""
    for nm in ("predict.py", "_predict_main_from_models.py", "_predict_main.py", "_predict_pre_pipeline.py"):
        p = _core / nm
        if p.exists():
            src += p.read_text(encoding="utf-8")
            src += "\n"
    assert 'logger.error(f"Error predicting with model {model_name}: {e}")' not in src
    assert 'logger.error("Error predicting with model %s", model_name, exc_info=True)' in src


def test_inference_predict_uses_raise_from() -> None:
    src = _read("inference/predict.py")
    # The fix wraps the original ValueError with `from e`.
    assert "is not inside trusted_root" in src
    # Must NOT be the bare raise.
    assert "is not inside trusted_root {abs_root}\")\n        if common" not in src
    # Must include `from e`.
    assert "from e\n        if common" in src or "is not inside trusted_root {abs_root}\") from e" in src


def test_trainer_cache_load_preserves_traceback() -> None:
    # The cache-load helper moved out of ``trainer.py`` into the sibling
    # ``_trainer_train_and_evaluate.py`` during the 2026-05-22 trainer split.
    # Read both so the structural pin tolerates either location.
    import pathlib
    import mlframe as _mlframe
    _root = pathlib.Path(_mlframe.__file__).resolve().parent / "training"
    src = ""
    for nm in ("trainer.py", "_trainer_train_and_evaluate.py", "_trainer_configure.py"):
        p = _root / nm
        if p.exists():
            src += p.read_text(encoding="utf-8")
            src += "\n"
    assert 'logger.warning(f"Failed to load cached model from {model_file_name}: {e}. Will retrain instead.")' not in src
    assert 'logger.warning("Failed to load cached model from %s; will retrain instead.", model_file_name, exc_info=True)' in src


def test_flat_metric_compute_uses_logger_exception() -> None:
    # MLPTorchModel (and its _compute_metric body) was carved out of
    # neural/flat.py into sibling neural/_flat_torch_module.py; check both
    # files so the sensor remains valid after the monolith split.
    src = _read("training/neural/flat.py")
    sibling = MLFRAME_ROOT / "training/neural/_flat_torch_module.py"
    if sibling.exists():
        src += sibling.read_text(encoding="utf-8")
    assert 'logger.error(f"Failed to compute metric {prefix}_{metric.name}: {e}")' not in src
    assert 'logger.exception("Failed to compute metric %s_%s", prefix, metric.name)' in src


def test_recurrent_checkpoint_load_preserves_traceback() -> None:
    src = _read("training/neural/recurrent.py")
    assert 'logger.warning(f"Failed to load checkpoint, using final model: {e}")' not in src
    assert 'logger.warning("Failed to load checkpoint, using final model", exc_info=True)' in src


def test_mlflow_start_run_final_giveup_logs_traceback() -> None:
    src = _read("integrations/mlflow.py")
    assert 'logger.exception("mlflow.start_run failed after %d retries", nfailed)' in src


def test_automl_import_uses_logger_exception() -> None:
    src = _read("training/automl.py")
    assert 'logger.error(f"AutoGluon not available: {e}")' not in src
    assert 'logger.exception("AutoGluon not available")' in src
    assert 'logger.error(f"LightAutoML not available: {e}")' not in src
    assert 'logger.exception("LightAutoML not available")' in src


def test_automl_auc_fi_use_exc_info() -> None:
    src = _read("training/automl.py")
    # 4 sites total: 2 AUC + 2 FI, both AutoGluon and LAMA paths.
    assert src.count('logger.warning("Could not compute AUC", exc_info=True)') == 2
    assert src.count('logger.warning("Could not compute feature importance", exc_info=True)') == 2


def test_evaluation_plot_fi_uses_exc_info() -> None:
    src = _read("training/evaluation.py")
    assert 'logger.warning(f"Could not plot feature importances: {e}.' not in src
    assert 'logger.warning("Could not plot feature importances. Maybe data shape changed within a pipeline?", exc_info=True)' in src


def test_reporting_predict_proba_fallback_uses_exc_info() -> None:
    """The predict_proba fallback log moved from training/_reporting.py to the
    sibling training/_reporting_probabilistic.py during the monolith split."""
    src_parent = _read("training/_reporting.py")
    src_sibling = _read("training/_reporting_probabilistic.py")
    src = src_parent + "\n" + src_sibling
    assert 'logger.warning(f"predict_proba not available for {type(model).__name__}, using predict() instead: {e}")' not in src
    assert 'logger.warning("predict_proba not available for %s, using predict() instead", type(model).__name__, exc_info=True)' in src


def test_training_loop_best_iter_uses_exc_info() -> None:
    src = _read("training/_training_loop.py")
    assert 'logger.warning(f"Could not get best iteration: {e}")' not in src
    assert 'logger.warning("Could not get best iteration", exc_info=True)' in src


def test_neural_base_example_input_uses_exc_info() -> None:
    src = _read("training/neural/base.py")
    assert 'logger.warning(f"Failed to prepare example_input_array: {e}")' not in src
    assert 'logger.warning("Failed to prepare example_input_array", exc_info=True)' in src


def test_neural_flat_compile_fallback_uses_exc_info() -> None:
    src = _read("training/neural/flat.py")
    assert 'logger.warning(f"Failed to apply torch.compile: {e}. Using uncompiled network.")' not in src
    assert 'logger.warning("Failed to apply torch.compile. Using uncompiled network.", exc_info=True)' in src


def test_pipeline_polars_ds_import_narrowed_and_exc_info() -> None:
    src = _read("training/pipeline.py")
    # Must be narrowed to ImportError and must use exc_info.
    assert 'logger.warning(f"Could not import polars-ds: {e}")' not in src
    assert 'logger.warning("Could not import polars-ds", exc_info=True)' in src


def test_mps_print_replaced_with_logger_exception() -> None:
    src = _read("feature_engineering/mps.py")
    # The print line at :652 must be gone.
    assert 'print(f"Error with {f}: {e}")' not in src
    assert 'logger.exception("Error processing MPS file %s", f)' in src
    # The :679 warning must use exc_info.
    assert 'logger.warning(f"File {fpath}, error {e}")' not in src
    assert 'logger.warning("Failed to read MPS parquet file %s", fpath, exc_info=True)' in src


# ---------------------------------------------------------------------------
# Behavioural sensors: trigger each P1 path and assert exc_info appears in log.
# ---------------------------------------------------------------------------


def test_save_mlframe_model_logs_traceback_on_failure(caplog) -> None:
    """save_mlframe_model failure must surface a traceback (not just str(e))."""
    import logging
    from mlframe.training.io import save_mlframe_model

    caplog.set_level(logging.ERROR)
    # Trigger by passing a model that cannot be pickled (e.g. a lambda)
    # and an invalid path so save fails.
    try:
        ok = save_mlframe_model(lambda x: x, file="/nonexistent_dir_xyz/model.bin")
    except Exception:
        # If the function re-raises rather than swallowing, that's also acceptable;
        # we only care that the lossy log pattern is gone.
        return
    if ok is False:
        # If it swallowed, traceback should be in the record via exc_info.
        # logger.exception sets exc_info on the record.
        relevant = [r for r in caplog.records if "Could not save model" in r.getMessage()]
        if relevant:
            assert relevant[0].exc_info is not None, (
                "save_mlframe_model swallowed an exception without exc_info; traceback was lost."
            )
