"""Audit 2026-11-20 finding 26-F10: nearest-mode imputation works on a frame whose index labels repeat.

The replacement used to be reindexed by label, which pandas refuses on a duplicated index. It is now placed by position, and the
imputed values are the ones the positional fill would give.
"""

import pandas as pd

from mlframe.preprocessing.unseen_category_imputer import UnseenCategoryImputer


def test_nearest_mode_with_duplicated_index_labels():
    """An unseen category under a repeated index label is imputed by value, with no reindex error."""
    train = pd.DataFrame({"c": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]})
    imputer = UnseenCategoryImputer(columns=["c"], similarity_mode="nearest", value_column="v").fit(train)
    test = pd.DataFrame({"c": ["a", "zzz", "c"], "v": [1.1, 2.9, 3.0]}, index=[0, 0, 1])
    out = imputer.transform(test)
    assert list(out["c"]) == ["a", "c", "c"]
