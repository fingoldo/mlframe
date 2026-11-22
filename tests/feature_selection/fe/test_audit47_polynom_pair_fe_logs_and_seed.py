"""Regression tests for polynom_pair_fe per-pair seed collisions and silent fallbacks."""
import logging

import numpy as np

from mlframe.feature_selection.filters import polynom_pair_fe as ppf
from mlframe.utils.log_throttle import reset_throttle_counts


def test_pair_rng_streams_do_not_collide_for_wide_frames():
    """(0, 2000) and (1, 1000) used to share seed + 2000."""
    a = ppf._pp_pair_rng(42, (0, 2000)).integers(0, 1 << 30, 8)
    b = ppf._pp_pair_rng(42, (1, 1000)).integers(0, 1 << 30, 8)
    assert not np.array_equal(a, b)


def test_pair_rng_keeps_the_legacy_stream_for_narrow_frames():
    """Below 1000 columns the draw must stay byte-identical to the historical seed formula."""
    got = ppf._pp_pair_rng(42, (3, 17)).integers(0, 1 << 30, 8)
    want = np.random.default_rng(42 + 3 * 1000 + 17).integers(0, 1 << 30, 8)
    assert np.array_equal(got, want)


def test_injection_failure_warns_even_when_not_verbose(caplog):
    """A survivor whose injection raised must be visible at WARNING with verbose=0."""
    reset_throttle_counts()
    with caplog.at_level(logging.WARNING, logger=ppf.logger.name):
        ppf._pp_log_injection_failure((1, 2), RuntimeError("boom"))
    assert any(r.levelno == logging.WARNING and "injection failed" in r.getMessage() for r in caplog.records)


def test_cheap_skip_corr_failure_warns(caplog):
    """An unreadable target in the linear-usability pre-check must log a throttled warning."""
    reset_throttle_counts()
    with caplog.at_level(logging.WARNING, logger=ppf.logger.name):
        ppf._pp_cheap_skip_applies((0, 1), 1.0, np.arange(4.0), ["a", "b", "c", "d"], 0.9, 0.9, {(0, 1): 1.0})
    assert any("trivial-correlation pre-check failed" in r.getMessage() for r in caplog.records)


def test_shared_subsample_idx_failure_warns(caplog):
    """An unreadable shared subsample index must log a throttled warning and fall back."""
    class Bad:
        """Object whose array conversion raises."""

        def __array__(self, *a, **k):
            """Always fail."""
            raise ValueError("no")

    reset_throttle_counts()
    v = np.arange(10.0)
    with caplog.at_level(logging.WARNING, logger=ppf.logger.name):
        ppf._pp_subsample_pair_rows((0, 1), v, v, np.zeros(10), Bad(), 0, 42, False)
    assert any("shared_subsample_idx validation failed" in r.getMessage() for r in caplog.records)
