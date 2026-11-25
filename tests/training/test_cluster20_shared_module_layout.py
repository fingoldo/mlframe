"""Cluster 20 regression: the chart shared modules keep __all__ at the end and separate their helpers (audit 2026-11-20, finding 20-F44).

``__all__`` sat in the middle of _error_analysis_shared, and _frame_rows followed _frame_columns with no blank line in _drift_shared. Both are
layout fixes only; the exported names are unchanged.
"""

from pathlib import Path

from mlframe.reporting.charts import _drift_shared as drift
from mlframe.reporting.charts import _error_analysis_shared as err


def test_all_is_at_the_end_of_error_analysis_shared():
    """``__all__`` is the last statement of the module, not mid-file (20-F44)."""
    lines = [ln for ln in Path(err.__file__).resolve().read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines[-1].startswith("__all__ = [")
    assert set(err.__all__) == {"DEFAULT_OVERLAY_BINS", "_DRIFT_Z", "_as_float_1d", "_pool_row_vectors"}


def test_frame_helpers_are_separated_in_drift_shared():
    """_frame_rows is preceded by a blank-line gap after _frame_columns (20-F44)."""
    src = Path(drift.__file__).resolve().read_text(encoding="utf-8")
    assert "    return [arr[:, i] for i in range(arr.shape[1])], names\n\n\ndef _frame_rows" in src
