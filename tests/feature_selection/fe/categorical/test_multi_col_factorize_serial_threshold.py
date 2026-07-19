"""Regression test for the site-4 joblib audit fix (2026-07-19): ``_multi_col_factorize_native``'s
joblib(threads) pool for non-Categorical columns must stay serial through 8 columns (raised from the
previous ``<= 1`` threshold), and only build the pool above that.

Isolated/warmed/best-of-3+ measurement: 2 cols -> 1.29x (but that case was already serial pre-fix, since the
old threshold was ``<= 1``), 8 cols -> 0.52x (loses to serial), 40 cols -> 0.93x (roughly even). No clean win
was found across the realistic 2-8 non-Categorical-column range, so it now stays serial end-to-end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.discretization import _multi_col_factorize_native


def _object_df(n_cols: int, n_rows: int = 200, seed: int = 0) -> pd.DataFrame:
    """A DataFrame of plain-object (non-Categorical) string columns, forcing every column into the
    ``needs_factorize`` list."""
    rng = np.random.default_rng(seed)
    data = {f"c{j}": rng.choice(["a", "b", "c", "d"], size=n_rows) for j in range(n_cols)}
    return pd.DataFrame(data)


def test_factorize_stays_serial_at_8_non_categorical_columns(monkeypatch):
    """8 non-Categorical columns (at the new threshold) must NOT construct a joblib.Parallel pool."""

    def _boom(*args, **kwargs):
        """Sensor stub: any call proves the serial fast path did not hold at n_cols<=8."""
        raise AssertionError("Parallel() must not be constructed for <=8 non-Categorical columns")

    # Parallel is imported lazily inside the function body (``from joblib import Parallel, ...``), so patch
    # the joblib module attribute itself rather than a module-level name in ``_disc_mod``.
    monkeypatch.setattr("joblib.Parallel", _boom)

    df = _object_df(8)
    out = _multi_col_factorize_native(df)
    assert out.shape == (200, 8)


def test_factorize_uses_pool_above_8_non_categorical_columns(monkeypatch):
    """Above the new threshold (9+ columns), the joblib pool IS still reachable -- the fix only changes
    the serial/parallel boundary, it does not remove the parallel path."""
    calls = []
    from joblib import Parallel as _RealParallel

    def _spy(*args, **kwargs):
        """Records that Parallel was constructed, then delegates to the real joblib.Parallel."""
        calls.append((args, kwargs))
        return _RealParallel(*args, **kwargs)

    monkeypatch.setattr("joblib.Parallel", _spy)

    df = _object_df(9)
    out = _multi_col_factorize_native(df)
    assert out.shape == (200, 9)
    assert len(calls) == 1, "Parallel must be constructed once above the new 8-column threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
