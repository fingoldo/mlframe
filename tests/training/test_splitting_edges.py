"""Edge-case tests for `make_train_test_split`.

Proactive probes surfaced five latent issues on 2026-04-19:

1. ``test_size=1.0`` + timestamps -> NaT strftime crash on empty train.
2. Negative ``val_size`` / ``test_size`` -> silent no-op (no validation).
3. ``trainset_aging_limit=0`` -> silent no-op, contradicted the strict
   ``(0, 1)`` validator.
4. Single-date frame (or very small n_rows) with ``wholeday_splitting=True``
   silently produced empty val/test with no warning.
5. Empty / single-row DF -> crashes inside sklearn (left as-is; the sklearn
   error is descriptive enough, and the suite's upstream input validation
   catches it earlier in realistic cases).

The tests below are sensors for all of those. They will fail again if
the validation or the NaT guard regresses.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training.splitting import make_train_test_split


# ---------------------------------------------------------------------------
# Input validation — negative / out-of-range args must fail loudly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, error_frag", [
    ({"test_size": -0.1},              "test_size"),
    ({"test_size": 1.5},               "test_size"),
    ({"val_size":  -0.1},              "val_size"),
    ({"val_size":   1.5},              "val_size"),
    ({"trainset_aging_limit": 0},      "trainset_aging_limit"),
    ({"trainset_aging_limit": 1.0},    "trainset_aging_limit"),
    ({"trainset_aging_limit": -0.5},   "trainset_aging_limit"),
    ({"trainset_aging_limit": 1.5},    "trainset_aging_limit"),
])
def test_invalid_args_raise_with_clear_message(kwargs, error_frag):
    df = pd.DataFrame({"x": range(50)})
    with pytest.raises(ValueError, match=error_frag):
        make_train_test_split(df, **kwargs)


def test_trainset_aging_limit_none_is_noop():
    """``None`` is explicitly the "no aging" signal and must not be
    rejected by the strict-range validator."""
    df = pd.DataFrame({"x": range(50)})
    out = make_train_test_split(df, trainset_aging_limit=None)
    assert len(out[0]) > 0  # train not empty


# ---------------------------------------------------------------------------
# NaT strftime on empty train (formerly crashed)
# ---------------------------------------------------------------------------

def test_test_size_1_with_timestamps_does_not_crash_on_empty_train():
    """Regression sensor: ``test_size=1.0`` with timestamps lands the full
    dataset in the test split. Formatting train_details on the empty
    train index used to hit
    ``ValueError: NaTType does not support strftime`` because
    ``idx.min()`` on empty returns NaT.
    """
    df = pd.DataFrame({"x": range(50)})
    ts = pd.Series(pd.date_range("2024-01-01", periods=50))
    train_idx, val_idx, test_idx, train_det, val_det, test_det = (
        make_train_test_split(df, test_size=1.0, val_size=0.0, timestamps=ts)
    )
    assert len(train_idx) == 0
    assert len(test_idx) == 50
    assert train_det == "(empty)"


def test_empty_train_details_also_guarded_for_row_timestamps():
    """Same NaT guard must cover the row-timestamp branch (wholeday_splitting=False)."""
    df = pd.DataFrame({"x": range(50)})
    ts = pd.Series(pd.date_range("2024-01-01", periods=50))
    _, _, _, train_det, _, _ = make_train_test_split(
        df, test_size=1.0, val_size=0.0, timestamps=ts, wholeday_splitting=False,
    )
    assert train_det == "(empty)"


# ---------------------------------------------------------------------------
# Silent-empty-split warning
# ---------------------------------------------------------------------------

def test_single_date_wholeday_warns_on_empty_val(caplog):
    """When all rows share a single date and ``wholeday_splitting=True``,
    ``int(1 * 0.1) == 0`` collapses val/test to empty even though the user
    asked for non-zero fractions. A warning must fire so the user notices
    the split silently lost what they requested.
    """
    df = pd.DataFrame({"x": range(100)})
    ts = pd.Series([pd.Timestamp("2024-01-01")] * 100)
    with caplog.at_level(logging.WARNING, logger="mlframe.training.splitting"):
        _, val_idx, test_idx, *_ = make_train_test_split(
            df, test_size=0.1, val_size=0.1, timestamps=ts,
        )
    assert len(val_idx) == 0
    assert len(test_idx) == 0
    warnings_text = " ".join(r.message for r in caplog.records if r.levelname == "WARNING")
    assert "val split is empty" in warnings_text or "val_size" in warnings_text.lower()
    assert "test split is empty" in warnings_text or "test_size" in warnings_text.lower()


def test_no_warning_when_split_intentionally_zero(caplog):
    """User passed ``val_size=0`` -> empty val is intentional, do NOT warn.
    The warning must trigger only on the mismatch "requested non-zero,
    got zero"."""
    df = pd.DataFrame({"x": range(100)})
    with caplog.at_level(logging.WARNING, logger="mlframe.training.splitting"):
        _, val_idx, _, *_ = make_train_test_split(df, test_size=0.1, val_size=0)
    assert len(val_idx) == 0
    msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert not any("val split is empty" in m for m in msgs), (
        "warning must not fire when val_size=0 is user-intended"
    )


# ---------------------------------------------------------------------------
# Pre-existing happy paths still work
# ---------------------------------------------------------------------------

def test_default_happy_path():
    """Sanity: the default call still produces sensible splits."""
    df = pd.DataFrame({"x": range(1000)})
    train, val, test, *_ = make_train_test_split(df)
    assert len(train) + len(val) + len(test) == 1000
    assert len(test) == 100  # 10% default
    assert len(val) == 90    # 10% of remaining 900


def test_seed_zero_is_deterministic():
    """``random_seed=0`` must round-trip (tests the ``is not None`` check,
    not the falsy short-circuit)."""
    df = pd.DataFrame({"x": range(100)})
    a = make_train_test_split(df, test_size=0.2, val_size=0.1, shuffle_test=True, shuffle_val=True, random_seed=0)
    b = make_train_test_split(df, test_size=0.2, val_size=0.1, shuffle_test=True, shuffle_val=True, random_seed=0)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[2], b[2])
