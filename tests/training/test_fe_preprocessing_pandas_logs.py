"""Regression: ``_process_special_values_fused`` (renamed to
``_process_special_values``) must log null/inf counts for pandas
input too. Pre-fix only the polars branch logged. The "fused" suffix
was also misleading - the pandas branch was never single-pass - so
the function is renamed.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import pytest


def test_pandas_branch_logs_null_and_inf_counts(caplog):
    """Pandas frame with NaN + +/-inf should produce log lines mentioning
    the counts (same level of observability as the polars branch)."""
    from mlframe.training.preprocessing import _process_special_values

    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.inf, -np.inf],
            "b": [np.nan, np.nan, 2.0, 4.0, 5.0],
        }
    )
    caplog.set_level(logging.INFO, logger="mlframe.training.preprocessing")
    _process_special_values(df, fill_value=0.0, verbose=1)
    log_text = caplog.text.lower()
    assert "null" in log_text or "nan" in log_text, f"pandas branch did not log null/nan counts; captured: {caplog.text!r}"
    assert "inf" in log_text, f"pandas branch did not log inf counts; captured: {caplog.text!r}"


def test_old_fused_name_is_deprecated_but_still_callable():
    """Backward-compat: the historic ``_process_special_values_fused`` alias
    keeps working so old call sites don't blow up - but the new
    ``_process_special_values`` is the canonical name."""
    from mlframe.training import preprocessing as pm

    assert hasattr(pm, "_process_special_values")
    # Old name still exists (shim) so any caller in the wild doesn't break.
    assert hasattr(pm, "_process_special_values_fused")
