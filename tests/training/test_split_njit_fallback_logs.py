"""TRAINING_LOOSE_B-1 (2026-08-05 audit): _stratified_split_3way's bare `except Exception:` around the
njit multilabel-stratification kernel had zero logging and silently fell back to the ~50x slower
pure-Python iterstrat path on ANY failure, including a genuine bug in the njit kernel -- indistinguishable
from the expected/benign "numba absent" case. Fixed by logging a WARNING naming the exception type/message
before falling back.
"""

from __future__ import annotations

import numpy as np

import mlframe.training._split_helpers as sh


def test_njit_stratification_failure_logs_warning_before_fallback(monkeypatch, caplog):
    """A failure in the njit kernel must be logged at WARNING with the exception type/message, not
    silently swallowed."""

    def _boom(y_i8, r, seed_int):
        """Simulate a genuine bug in the njit kernel (not a signature/import mismatch)."""
        raise RuntimeError("simulated njit kernel bug")

    monkeypatch.setattr("mlframe.training._iterative_stratification_njit._iterative_stratification_njit", _boom)

    rng = np.random.default_rng(0)
    n = 200
    y = (rng.random((n, 3)) < 0.4).astype(bool)
    indices = np.arange(n)

    with caplog.at_level("WARNING", logger="mlframe.training._split_helpers"):
        train_idx, val_idx, test_idx = sh._stratified_split_3way(
            indices=indices,
            stratify_y=y,
            test_size=0.2,
            val_size=0.2,
            random_state=0,
        )

    assert len(train_idx) + len(val_idx) + len(test_idx) == n
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("simulated njit kernel bug" in r.message for r in warnings), f"expected a WARNING naming the njit failure, got: {[r.message for r in warnings]}"
    assert any("RuntimeError" in r.message for r in warnings)
