"""The give-up path of get_or_create_mlflow_run never exposes the tracking URI's password (SEC-01).

After the retry limit the function used to log with exc_info=True, which wrote the raw exception and its traceback to the logs,
and it re-raised a scrubbed exception chained (``from e``) to the original, whose message still holds the password. It now logs
only the scrubbed message and raises the scrubbed exception without the chain.
"""

from __future__ import annotations

import logging
from unittest import mock

import pytest

SECRET = "supersecret"  # pragma: allowlist secret -- fake credential for the scrubbing test


@pytest.fixture
def failing_mlflow(monkeypatch):
    """Patch mlflow so every start_run raises an error that carries a credential-bearing tracking URI."""
    mlflow = pytest.importorskip("mlflow")
    from mlframe.integrations import mlflow as mod

    boom = RuntimeError(f"cannot reach https://user:{SECRET}@tracking.example.com/api")
    monkeypatch.setattr(mlflow, "start_run", mock.Mock(side_effect=boom))
    monkeypatch.setattr(mlflow, "search_runs", mock.Mock(return_value=[]))
    monkeypatch.setattr(mlflow, "set_experiment", mock.Mock())
    return mod


def test_giveup_error_does_not_carry_the_password(failing_mlflow, caplog):
    """After the retry limit the raised error and its log text are scrubbed, and no original cause is chained."""
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(RuntimeError) as info:
            failing_mlflow.get_or_create_mlflow_run("run-x")
    assert SECRET not in str(info.value)
    assert info.value.__cause__ is None
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert SECRET not in text
    assert not any(r.exc_info for r in caplog.records if r.name.startswith("mlframe"))
