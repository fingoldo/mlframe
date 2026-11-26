"""Cluster 24 regression: a raw column with no derived column present does not crash the verify path (audit 2026-11-20, finding 24-F44).

With ``min_derived_present=0`` a raw column can qualify with zero derived columns. The verify path then took ``max()`` over an empty
generator and raised ValueError. The embedding retains nothing, so the raw column must be kept and reported, not dropped and not a crash.
"""

import numpy as np
import pandas as pd

from mlframe.feature_selection.drop_raw_after_embedding import drop_raw_after_embedding


def test_raw_with_no_derived_column_is_kept_and_reported_not_crashed():
    """Verify mode with an empty derived list keeps the informative raw column and records why (24-F44)."""
    rng = np.random.default_rng(0)
    y = (rng.normal(size=200) > 0).astype(int)  # the verify helper is binary-only
    df = pd.DataFrame({"raw": y + 0.3 * rng.normal(size=200), "other": rng.normal(size=200)})
    report: dict = {}
    out = drop_raw_after_embedding(
        df,
        raw_to_derived={"raw": []},
        min_derived_present=0,
        verify_against=(y, 0.5),
        safety_report=report,
    )
    assert "raw" in out.columns
    assert "raw" in report
