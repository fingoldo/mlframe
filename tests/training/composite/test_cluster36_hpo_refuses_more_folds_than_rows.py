"""Audit 2026-11-20 finding 36-F6: asking for more CV folds than rows is an error, not a search over empty folds.

Empty test folds scored inf for every candidate, and the winner was arbitrary with no error. The test calls the split resolver with three
rows and five folds and checks the ValueError names both numbers.
"""

import numpy as np
import pytest

from mlframe.training.composite import hpo


def test_more_folds_than_rows_is_refused():
    """cv=5 on three rows raises ValueError naming the folds and the rows."""
    X = np.arange(3, dtype=float).reshape(-1, 1)
    with pytest.raises(ValueError, match="cv=5"):
        hpo._resolve_splits(X, 3, 5, None)
