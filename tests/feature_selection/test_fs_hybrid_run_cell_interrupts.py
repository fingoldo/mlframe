"""run_cell records ordinary cell failures and pytest skips, but lets an interrupt stop the run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid.run_experiment import CellSpec, run_cell

SPEC = CellSpec(scenario="s", arm="a", dataset_seed=0, cv_seed=0)


def _frames():
    """Small deterministic train/test frame pair."""
    x = pd.DataFrame({"a": np.arange(8.0), "b": np.arange(8.0) % 3})
    y = np.arange(8) % 2
    return x, y


def _raise(exc: BaseException):
    """Return a cell factory that raises ``exc`` when the cell builds its estimator."""

    def factory():
        """Raise the stored exception."""
        raise exc

    return factory


def test_keyboard_interrupt_during_a_cell_propagates():
    """A KeyboardInterrupt raised inside a cell stops the run instead of being recorded as the cell's status."""
    x, y = _frames()
    with pytest.raises(KeyboardInterrupt):
        run_cell(SPEC, _raise(KeyboardInterrupt()), x, y, x, y, {})


def test_ordinary_cell_error_is_recorded_with_its_status():
    """An ordinary exception inside a cell becomes a non-ok record that names the error."""
    x, y = _frames()
    record = run_cell(SPEC, _raise(RuntimeError("boom")), x, y, x, y, {})
    assert record["status"] != "ok"
    assert record["error"].startswith("RuntimeError: boom")


def test_platform_skip_inside_a_cell_is_recorded_with_the_skipped_prefix():
    """A pytest skip raised inside a cell is recorded with the ``Skipped: `` prefix so the test can skip itself."""
    x, y = _frames()
    record = run_cell(SPEC, _raise(pytest.skip.Exception("lightgbm unavailable")), x, y, x, y, {})
    assert record["status"] != "ok"
    assert record["error"].startswith("Skipped: ")
