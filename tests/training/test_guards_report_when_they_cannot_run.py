"""A guard that cannot run must say so, not return as if it had nothing to do.

Two production sites where a configured guard silently no-opped:

`apply_base_leakage_guard` indexes `time_ordering` by the training indices, and returned the candidates
untouched when the ordering was shorter than those indices reach. The caller had asked for the guard via
`config.detect_base_leakage`, so its inability to run is information they need -- otherwise same-time
re-encodings of `y` enter discovery as bases with nothing in the log to explain it.

`fit_missing_indicator_imputation` excludes `group_col` from the columns it imputes. When the auto-derived
column list contained only that column, the exclusion emptied it and the fit loop did no work, returning an
identity state to a caller who asked for imputation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest


def test_base_leakage_guard_warns_when_the_time_ordering_is_too_short(caplog):
    """The guard is configured on and cannot run; that must reach the log."""
    from mlframe.training.composite.discovery._fit_temporal import apply_base_leakage_guard

    n = 60
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"b0": rng.normal(size=n), "b1": rng.normal(size=n)})
    y_train = rng.normal(size=n)
    train_idx = np.arange(n)
    short_ordering = np.arange(n // 2)  # half as long as the training indices reach

    with caplog.at_level(logging.WARNING):
        kept = apply_base_leakage_guard(None, df, ["b0", "b1"], train_idx, y_train, short_ordering)

    assert kept == ["b0", "b1"], "a guard that cannot run must still return the candidates untouched"
    assert any(
        "base-leakage guard SKIPPED" in r.getMessage() for r in caplog.records
    ), f"the guard skipped silently; warnings seen: {[r.getMessage()[:70] for r in caplog.records]}"
    msg = next(r.getMessage() for r in caplog.records if "SKIPPED" in r.getMessage())
    assert "30" in msg and "60" in msg, f"the warning must name both lengths so the caller can see the mismatch: {msg}"


def test_base_leakage_guard_stays_quiet_when_it_can_run(caplog):
    """A guard that cries on every call gets muted; the warning must be specific to the skip."""
    from mlframe.training.composite.discovery._fit_temporal import apply_base_leakage_guard

    n = 60
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"b0": rng.normal(size=n)})
    y_train = rng.normal(size=n)
    train_idx = np.arange(n)

    with caplog.at_level(logging.WARNING):
        apply_base_leakage_guard(None, df, ["b0"], train_idx, y_train, np.arange(n))

    assert not [r for r in caplog.records if "SKIPPED" in r.getMessage()], "the guard reported a skip on an ordering it could index"


def test_missing_indicator_fit_rejects_a_column_list_that_is_only_the_group_key():
    """The exclusion emptied the list, and the loop then did nothing while the caller expected imputation."""
    from mlframe.preprocessing.missing_indicator_pairing import fit_missing_indicator_imputation

    df = pd.DataFrame({"g": [1.0, np.nan, 2.0, np.nan], "other": [1.0, 2.0, 3.0, 4.0]})
    # `columns=None` derives the list from the NaN-bearing columns, which here is exactly the group key.
    with pytest.raises(ValueError, match="no columns left to impute"):
        fit_missing_indicator_imputation(df, columns=None, group_col="g")


def test_missing_indicator_fit_still_works_when_other_columns_remain():
    """The guard must only fire on the degenerate case, not on any frame whose group key has NaNs."""
    from mlframe.preprocessing.missing_indicator_pairing import fit_missing_indicator_imputation

    df = pd.DataFrame({"g": [1.0, np.nan, 2.0, np.nan], "v": [1.0, np.nan, 3.0, 4.0]})
    state = fit_missing_indicator_imputation(df, columns=None, group_col="g")
    assert state, "a frame with another NaN-bearing column must still produce fit state"
