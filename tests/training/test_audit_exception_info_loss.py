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

import ast
import functools
import importlib
from pathlib import Path

MLFRAME_ROOT = Path(importlib.import_module("mlframe").__file__).parent


def _read(rel: str) -> str:
    """Read."""
    _path = MLFRAME_ROOT / rel
    if not _path.exists() and _path.suffix == ".py":
        # Monolith-split compat: the flat module became a subpackage
        # (``X.py`` -> ``X/__init__.py`` + submodules). Read the package
        # __init__ plus every submodule so structural source pins still match.
        _pkg = _path.with_suffix("")
        _init = _pkg / "__init__.py"
        if _init.exists():
            parts = [_init.read_text(encoding="utf-8")]
            for _sub in sorted(_pkg.glob("*.py")):
                if _sub.name != "__init__.py":
                    parts.append(_sub.read_text(encoding="utf-8"))
            return "\n".join(parts)
    return _path.read_text(encoding="utf-8")


@functools.lru_cache(maxsize=1)
def _corpus() -> "tuple[tuple[Path, str, ast.Module], ...]":
    """Every parseable module under ``mlframe/``, read and parsed once for the whole file.

    Eighteen call sites ask the same question of the same tree; without this each one re-read and
    re-parsed several hundred modules.
    """
    out = []
    for path in sorted(MLFRAME_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:  # pragma: no cover - a syntactically broken module is a different failure
            continue
        out.append((path, text, tree))
    return tuple(out)


def _logs_message_with_traceback(message: str) -> list[str]:
    """Every call site under ``mlframe/`` that logs ``message``, and whether it keeps the traceback.

    Returns one ``"<relpath>:<lineno> <ok|LOSES TRACEBACK>"`` entry per call site found.

    Matching the exact call text pins two things the contract does not care about: which file the
    handler currently lives in, and whether it reaches the logger directly or through a throttling
    wrapper. Both changed under the monolith splits and the log_throttle rollout while the traceback
    itself was preserved throughout, so this looks for the message anywhere in the tree and asks only
    whether that call carries exception info.
    """
    out: list[str] = []
    for path, text, tree in _corpus():
        if message not in text:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not any(isinstance(a, ast.Constant) and isinstance(a.value, str) and message in a.value for a in node.args):
                continue
            keeps = any(kw.arg == "exc_info" and not (isinstance(kw.value, ast.Constant) and not kw.value.value) for kw in node.keywords)
            if not keeps and isinstance(node.func, ast.Attribute):
                keeps = node.func.attr == "exception"  # logger.exception implies exc_info=True
            rel = path.relative_to(MLFRAME_ROOT).as_posix()
            out.append(f"{rel}:{node.lineno} {'ok' if keeps else 'LOSES TRACEBACK'}")
    return out


def _assert_traceback_preserved(message: str) -> None:
    """The message must be logged somewhere, and every site logging it must keep the traceback."""
    sites = _logs_message_with_traceback(message)
    assert sites, f"no call site logs {message!r} any more -- the handler was removed or the message reworded"
    bad = [s for s in sites if s.endswith("LOSES TRACEBACK")]
    assert not bad, f"call site(s) logging {message!r} discard the traceback: {bad}"


# ---------------------------------------------------------------------------
# Contract sensors: every handler that logs one of these messages keeps its traceback.
#
# These read the parse tree, not the text. Until 2026-09-03 each of them asserted a pair of exact
# call spellings -- the pre-fix f-string absent, the post-fix call present -- in a hand-maintained
# concatenation of the files the handler had lived in across three monolith splits. That is three
# things the contract does not care about (which file it is in, which logger method it reaches,
# how the arguments are spelled) and one it does: whether exception info survives. It also meant
# every split had to be chased through this file, and a handler deleted outright still passed the
# "old form absent" half.
#
# `_assert_traceback_preserved` asks the only question that matters, anywhere in the tree, and
# fails if the handler disappears entirely.
# ---------------------------------------------------------------------------


def test_io_save_uses_logger_exception() -> None:
    """A failed model save keeps its traceback."""
    _assert_traceback_preserved("Could not save model to file %s")


def test_predict_per_model_loop_uses_exc_info() -> None:
    """The per-model predict loop keeps its traceback; the twin path at 995 always did."""
    _assert_traceback_preserved("Error predicting with model %s")


def test_trainer_cache_load_preserves_traceback() -> None:
    """Falling back from a cached model to a retrain keeps its traceback."""
    _assert_traceback_preserved("Failed to load cached model from %s")


def test_flat_metric_compute_uses_logger_exception() -> None:
    """The per-metric compute failure keeps its traceback, wherever that handler lives."""
    _assert_traceback_preserved("Failed to compute metric %s_%s")


def test_recurrent_checkpoint_load_preserves_traceback() -> None:
    """Falling back from a checkpoint to the final model keeps its traceback."""
    _assert_traceback_preserved("Failed to load checkpoint, using final model")


def test_mlflow_start_run_final_giveup_logs_traceback() -> None:
    """Giving up on mlflow.start_run keeps its traceback (through log_throttle's passthrough)."""
    _assert_traceback_preserved("mlflow.start_run failed after %d retries")


def test_automl_import_uses_logger_exception() -> None:
    """Both optional-AutoML import failures keep their tracebacks."""
    _assert_traceback_preserved("AutoGluon not available")
    _assert_traceback_preserved("LightAutoML not available")


def test_automl_auc_fi_use_exc_info() -> None:
    """Four handlers -- AUC and feature importance, on both the AutoGluon and the LAMA path."""
    for message, expected in (("Could not compute AUC", 2), ("Could not compute feature importance", 2)):
        sites = _logs_message_with_traceback(message)
        assert len(sites) == expected, f"{message!r} is logged at {len(sites)} site(s), expected {expected}: {sites}"
        _assert_traceback_preserved(message)


def test_evaluation_plot_fi_uses_exc_info() -> None:
    """The feature-importance plot failure keeps its traceback."""
    _assert_traceback_preserved("Could not plot feature importances. Maybe data shape changed within a pipeline?")


def test_reporting_predict_proba_fallback_uses_exc_info() -> None:
    """Falling back from predict_proba to predict keeps its traceback."""
    _assert_traceback_preserved("predict_proba not available for %s, using predict() instead")


def test_training_loop_best_iter_uses_exc_info() -> None:
    """Failing to read the best iteration keeps its traceback."""
    _assert_traceback_preserved("Could not get best iteration")


def test_neural_base_example_input_uses_exc_info() -> None:
    """Failing to build example_input_array keeps its traceback."""
    _assert_traceback_preserved("Failed to prepare example_input_array")


def test_neural_flat_compile_fallback_uses_exc_info() -> None:
    """Falling back to the uncompiled network keeps its traceback."""
    _assert_traceback_preserved("Failed to apply torch.compile. Using uncompiled network.")


def test_pipeline_polars_ds_import_narrowed_and_exc_info() -> None:
    """The optional polars-ds import failure keeps its traceback."""
    _assert_traceback_preserved("Could not import polars-ds")


def test_mps_print_replaced_with_logger_exception() -> None:
    """Both MPS handlers log rather than print, and both keep their tracebacks."""
    _assert_traceback_preserved("Error processing MPS file %s")
    _assert_traceback_preserved("Failed to read MPS parquet file %s")


def test_trusted_path_rejection_chains_its_cause() -> None:
    """A cross-drive commonpath failure must not masquerade as a plain traversal rejection.

    Behavioural since 2026-09-03. This asserted that "is not inside trusted_root" appears in a
    concatenation of two files, that one exact two-line spelling does not, and that one of three
    `from`-clause spellings does -- none of which says the raised error actually carries a cause.
    `commonpath` raises ValueError for "paths don't have the same drive", and losing that cause
    turns a misconfigured trusted_root into what looks like an attempted escape.
    """
    import os

    import pytest

    from mlframe.core.helpers import validate_trusted_path

    def _cross_drive(_paths):
        """Stand in for the real commonpath, which raises this on a cross-drive comparison."""
        raise ValueError("Paths don't have the same drive")

    real = os.path.commonpath
    try:
        os.path.commonpath = _cross_drive
        with pytest.raises(ValueError) as excinfo:
            validate_trusted_path("D:/elsewhere/model.pkl", "C:/trusted")
    finally:
        os.path.commonpath = real

    assert "is not inside trusted_root" in str(excinfo.value)
    assert excinfo.value.__cause__ is not None, "the commonpath failure was swallowed; a cross-drive root reads as an escape attempt"
    assert "same drive" in str(excinfo.value.__cause__)


def test_a_path_outside_the_trusted_root_is_rejected(tmp_path) -> None:
    """The ordinary rejection, so the chaining test above cannot be the only thing holding it up."""
    import pytest

    from mlframe.core.helpers import validate_trusted_path

    trusted = tmp_path / "trusted"
    trusted.mkdir()
    outside = tmp_path / "elsewhere" / "model.pkl"

    with pytest.raises(ValueError, match="is not inside trusted_root"):
        validate_trusted_path(str(outside), str(trusted))


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
            assert relevant[0].exc_info is not None, "save_mlframe_model swallowed an exception without exc_info; traceback was lost."
