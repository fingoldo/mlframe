"""Audit 2026-11-20 finding 25-F26: exhausting the MLflow start-run retries raises, instead of returning a None run.

Only "already active" errors are retried; any other failure raises at once. The retry limit is therefore reached by a run that
stays active. The test makes start_run report an already-active run every time, with no handle to end it, and checks that the
sixth failure raises a scrubbed RuntimeError rather than returning None.
"""

import pytest

mlflow = pytest.importorskip("mlflow")

from mlframe.integrations import mlflow as mlframe_mlflow


def test_exhausted_retries_raise_a_scrubbed_error(monkeypatch):
    """Six 'already active' failures, none recoverable: the sixth raises RuntimeError with the retry count, not None."""
    attempts = []

    def already_active(**kwargs):
        """Report a run that stays active with no handle to end it."""
        attempts.append(1)
        raise Exception("Run with UUID abc is already active")

    monkeypatch.setattr(mlflow, "start_run", already_active)
    monkeypatch.setattr(mlflow, "active_run", lambda: None)
    with pytest.raises(RuntimeError, match="failed after 6 retries"):
        mlframe_mlflow._start_run_with_retries("run", "0", None)
    assert len(attempts) == 6
