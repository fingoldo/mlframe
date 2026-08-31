"""Two things a production log made unreadable: an unbudgeted diagnostic, and two memory numbers that disagreed.

The adversarial fold-selection diagnostic 5-fold cross-validates a train-vs-test classifier over the whole
train+test union. On a 2.4M-row fit that is five fits of a 2.05M-row frame -- about two minutes and 10.26M
materialised rows -- on a run whose model list was a single CatBoost. Subsampling would change what it returns
(it has to score EVERY train row to rank it by test-likeness), so the budget skips it and names the knob.

Separately, two adjacent log lines read ``Done. RAM usage: 45.2GB`` and ``process_model(cb) START -- RAM=6.2GB``.
Both were true of different quantities: ``get_own_memory_usage`` reports private commit on Windows, while a raw
``memory_info().rss`` reports the working set, which ``clean_ram()`` deliberately evicts. Every user-facing line
now goes through one helper, and the message says which quantity it is.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training._ram_helpers import get_reported_memory_gb, memory_measure_name
from mlframe.training.core._diagnostics_registry import (
    ADVERSARIAL_FOLD_MAX_UNION_ROWS,
    adapt_adversarial_fold_selection,
)


def _frames(n_train: int, n_test: int, n_cols: int = 3):
    """Numeric train/test frames of the requested size."""
    rng = np.random.default_rng(0)
    cols = [f"f{i}" for i in range(n_cols)]
    return (
        pd.DataFrame(rng.standard_normal((n_train, n_cols)), columns=cols),
        pd.DataFrame(rng.standard_normal((n_test, n_cols)), columns=cols),
    )


class TestTheAdversarialFoldBudget:
    """When the diagnostic runs and when it declines."""

    def test_an_oversized_union_is_skipped(self):
        """The production shape: a 2.05M-row union costing five fits of that size."""
        train, test = _frames(400, 400)
        out = adapt_adversarial_fold_selection(train, None, test, None, [], None, None, max_union_rows=100)
        assert out["status"] == "skipped"
        assert out["n_union_rows"] == 800

    def test_the_skip_names_the_knob(self, caplog):
        """A silent skip is a missing diagnostic; a named one is a decision the reader can reverse."""
        train, test = _frames(400, 400)
        with caplog.at_level(logging.WARNING, logger="mlframe.training.core._diagnostics_registry"):
            adapt_adversarial_fold_selection(train, None, test, None, [], None, None, max_union_rows=100)
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "max_union_rows" in text

    def test_a_small_union_still_runs(self):
        """The budget must not disable the diagnostic on the frames it was affordable on all along."""
        train, test = _frames(300, 200)
        out = adapt_adversarial_fold_selection(train, None, test, None, [], None, None)
        assert out.get("status") != "skipped"
        assert "n_selected" in out or "error" in out

    def test_zero_disables_the_budget(self):
        """The escape hatch for a caller who wants the diagnostic whatever it costs."""
        train, test = _frames(300, 200)
        out = adapt_adversarial_fold_selection(train, None, test, None, [], None, None, max_union_rows=0)
        assert out.get("status") != "skipped"

    def test_the_default_budget_would_have_caught_the_production_run(self):
        """2.05M union rows against the default: the case this exists for."""
        assert ADVERSARIAL_FOLD_MAX_UNION_ROWS < 2_052_546

    def test_the_default_budget_leaves_ordinary_frames_alone(self):
        """A few hundred thousand rows is a normal fit and should keep its diagnostic."""
        assert ADVERSARIAL_FOLD_MAX_UNION_ROWS >= 200_000


class TestTheMemoryMeasureIsNamed:
    """One quantity, said out loud."""

    def test_the_measure_has_a_name(self):
        """A bare "RAM" invites comparing two numbers that measure different things."""
        assert memory_measure_name() in {"private commit", "RSS"}

    def test_windows_reports_private_commit(self):
        """The working set is evicted by clean_ram, so on Windows it is not what a reader should be shown."""
        import sys

        if not sys.platform.startswith("win"):
            pytest.skip("platform-specific: this asserts the Windows choice")
        assert memory_measure_name() == "private commit"

    def test_the_reported_value_is_a_plausible_process_size(self):
        """Guards against the helper silently returning the 0.0 sentinel on every call."""
        assert get_reported_memory_gb() > 0.0

    def test_the_phase_table_labels_its_memory_column(self):
        """The column used to say "net RSS" on a platform where the number is private commit."""
        import time

        from mlframe.training.phases import format_phase_summary, phase, reset_phase_registry

        reset_phase_registry()
        with phase("p"):
            time.sleep(0.01)
        assert f"net {memory_measure_name()}" in format_phase_summary()

    def test_the_two_reporting_paths_agree(self):
        """The whole point: the suite's two "RAM usage" sources must return the same quantity."""
        from pyutilz.system import get_own_memory_usage

        assert get_reported_memory_gb() == pytest.approx(float(get_own_memory_usage()), rel=0.05)
