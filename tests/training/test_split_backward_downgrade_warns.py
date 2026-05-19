"""val_placement='backward' silently downgrading to 'forward' is a temporal-honesty
loss: the caller wanted newest-data validation, got random-style instead. Pre-fix
this was an INFO log -> easy to miss in production runs. New behavior: WARNING
level with explicit reason and remediation hint.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest


def _make_tiny_df(n=200):
    return pd.DataFrame({"x": np.arange(n, dtype=np.float32), "y": np.arange(n, dtype=np.float32)})


@pytest.mark.fast
def test_backward_without_timestamps_warns(caplog):
    from mlframe.training.splitting import make_train_test_split

    df = _make_tiny_df()
    with caplog.at_level(logging.WARNING, logger="mlframe.training.splitting"):
        make_train_test_split(
            df=df,
            test_size=0.2,
            val_size=0.1,
            timestamps=None,
            val_placement="backward",
        )

    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("val_placement='backward' requested but downgraded" in m for m in msgs), (
        f"Expected WARNING about backward downgrade. Got: {msgs}"
    )
    assert any("Temporal honesty lost" in m for m in msgs), (
        f"Expected explicit consequence in warning. Got: {msgs}"
    )


@pytest.mark.fast
def test_backward_with_timestamps_no_warn(caplog):
    """When timestamps are provided backward placement is honored, no downgrade warning."""
    from mlframe.training.splitting import make_train_test_split

    n = 200
    df = _make_tiny_df(n)
    timestamps = pd.Series(pd.date_range("2026-01-01", periods=n, freq="h"))
    with caplog.at_level(logging.WARNING, logger="mlframe.training.splitting"):
        make_train_test_split(
            df=df,
            test_size=0.2,
            val_size=0.1,
            timestamps=timestamps,
            val_placement="backward",
        )

    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("downgraded" in m for m in msgs), (
        f"backward with timestamps should NOT warn about downgrade. Got: {msgs}"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "--no-cov", "-x", "-s", "--tb=short"]))
