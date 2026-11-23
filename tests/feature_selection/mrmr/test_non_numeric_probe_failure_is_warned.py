"""A failed dtype introspection must be warned about, not only logged at debug level (AUD23-42).

``_non_numeric_column_indices`` returns an empty set when it cannot read the dtypes, which marks every column numeric: a string column can then reach the numeric basis. A fallback that changes what a
fit does is a warning.
"""

from __future__ import annotations

import logging

from mlframe.feature_selection.filters._mrmr_fe_step._helpers import _non_numeric_column_indices


class _BrokenFrame:
    """A frame whose ``dtypes`` raise, as a half-implemented dataframe-like object may."""

    @property
    def dtypes(self):
        """Raise on access."""
        raise RuntimeError("dtype probe failed")


def test_a_failed_probe_warns_and_marks_every_column_numeric(caplog):
    """The result is the documented empty set, and the failure is visible at WARNING level."""
    with caplog.at_level(logging.WARNING):
        result = _non_numeric_column_indices(_BrokenFrame(), ["a", "b"])
    assert result == set()
    assert any(r.levelno >= logging.WARNING and "dtype introspection failed" in r.getMessage() for r in caplog.records)
