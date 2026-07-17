"""Regression: dummy-baseline plot path passed ``train_df.columns or []`` which
triggers ``pd.Index.__bool__`` -> ValueError ("truth value of a Index is
ambiguous"). The whole pre-training dummy floor report was lost silently.

Pre-fix repro:
    >>> import pandas as pd
    >>> idx = pd.Index(["a", "b"])
    >>> idx or []
    ValueError: The truth value of a Index is ambiguous. ...
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.mark.fast
def test_pandas_index_truthiness_raises_demonstration():
    """Sanity: ``pd.Index or []`` really does raise. Confirms the bug surface."""
    idx = pd.Index(["a", "b", "c"])
    with pytest.raises(ValueError, match="truth value of a Index is ambiguous"):
        _ = idx or []


@pytest.mark.fast
def test_columns_extraction_uses_explicit_none_check():
    """The fixed pattern: explicit None check, no boolean-or on Index.

    Both empty Index and populated Index must yield a list without raising.
    """
    for cols in [pd.Index([]), pd.Index(["a"]), pd.Index(["a", "b", "c"])]:
        attr = cols  # what ``getattr(df, "columns", None)`` returns
        # Fixed path:
        result = list(attr) if attr is not None else []
        assert isinstance(result, list)
        assert len(result) == len(cols)


@pytest.mark.fast
def test_none_df_returns_empty_columns():
    """When filtered_train_df is None, columns extraction returns [] without raising."""
    filtered_train_df = None
    cols_attr = getattr(filtered_train_df, "columns", None)
    result = list(cols_attr) if cols_attr is not None else []
    assert result == []


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "--no-cov", "-x", "-s", "--tb=short"]))
