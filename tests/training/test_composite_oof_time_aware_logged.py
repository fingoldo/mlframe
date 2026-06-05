"""Time-aware OOF split is driven ONLY by the explicit ``time_ordering`` signal. The earlier behaviour probed base
columns and auto-switched to a trailing slice on ANY monotone base -- a false positive on sorted-but-non-temporal
bases (sorted ids, binned features) that silently changed the OOF leakage profile. These tests pin that a monotone
base column NO LONGER flips the split, while an explicit ``time_ordering`` still does.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlframe.training.composite_ensemble import compute_oof_holdout_predictions


def test_monotone_base_column_does_not_auto_switch(caplog) -> None:
    """A monotone base column must NOT trigger the trailing-slice path or any auto-detect log (no base-column probe)."""
    n = 200
    train_X = pd.DataFrame({"f": np.arange(n, dtype=np.float64)})
    y_train_full = np.linspace(0.0, 1.0, n)
    monotone_base = np.arange(n, dtype=np.float64)  # weakly monotone non-decreasing
    base_train_full_per_spec = {"monotone_ts": monotone_base}

    with caplog.at_level(logging.INFO, logger="mlframe.training.composite_ensemble"):
        compute_oof_holdout_predictions(
            component_models=[],
            component_names=[],
            component_specs=[],
            train_X=train_X,
            y_train_full=y_train_full,
            base_train_full_per_spec=base_train_full_per_spec,
            holdout_frac=0.2,
            random_state=0,
            time_ordering=None,
            kfold=1,
        )
    auto = [r.getMessage() for r in caplog.records if "auto-detected" in r.getMessage()]
    assert not auto, f"monotone base column must not auto-switch the split; got: {auto!r}"


def test_non_monotone_base_does_not_log(caplog) -> None:
    """Random-shuffled base must NOT trigger the auto-detect log - guards against the WARN/INFO firing on noise."""
    n = 200
    rng = np.random.default_rng(0)
    train_X = pd.DataFrame({"f": rng.normal(size=n)})
    y_train_full = rng.normal(size=n)
    # Shuffled base: clearly not monotone.
    non_monotone_base = rng.normal(size=n)
    base_train_full_per_spec = {"noise_col": non_monotone_base}

    with caplog.at_level(logging.INFO, logger="mlframe.training.composite_ensemble"):
        compute_oof_holdout_predictions(
            component_models=[],
            component_names=[],
            component_specs=[],
            train_X=train_X,
            y_train_full=y_train_full,
            base_train_full_per_spec=base_train_full_per_spec,
            holdout_frac=0.2,
            random_state=0,
            time_ordering=None,
            kfold=1,
        )
    log_messages = [r.getMessage() for r in caplog.records if "auto-detected" in r.getMessage()]
    assert not log_messages, f"unexpected auto-detect log on non-monotone base: {log_messages!r}"


def test_explicit_time_ordering_monotone_logs_distinct_line(caplog) -> None:
    """Explicit ``time_ordering`` (not auto-detected) uses a distinct INFO line so log-grepping can tell the two paths apart."""
    n = 200
    train_X = pd.DataFrame({"f": np.arange(n, dtype=np.float64)})
    y_train_full = np.linspace(0.0, 1.0, n)
    base_train_full_per_spec = {"some_base": np.arange(n, dtype=np.float64)}
    time_ordering = np.arange(n, dtype=np.float64)

    with caplog.at_level(logging.INFO, logger="mlframe.training.composite_ensemble"):
        compute_oof_holdout_predictions(
            component_models=[],
            component_names=[],
            component_specs=[],
            train_X=train_X,
            y_train_full=y_train_full,
            base_train_full_per_spec=base_train_full_per_spec,
            holdout_frac=0.2,
            random_state=0,
            time_ordering=time_ordering,
            kfold=1,
        )
    # Explicit path log message does NOT contain "auto-detected" - it should say "time_ordering signal".
    explicit = [r.getMessage() for r in caplog.records if "time_ordering signal" in r.getMessage()]
    auto = [r.getMessage() for r in caplog.records if "auto-detected" in r.getMessage()]
    assert explicit, "expected the explicit-time_ordering INFO log"
    assert not auto, "auto-detect log must not fire when time_ordering is explicit"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-xvs", "--no-cov"])
