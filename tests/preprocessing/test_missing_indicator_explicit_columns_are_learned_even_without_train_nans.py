"""An explicit column list is learned in full by fit_missing_indicator_imputation, so a column complete in train is still imputed in a later frame."""
import numpy as np
import pandas as pd

from mlframe.preprocessing.missing_indicator_pairing import apply_missing_indicator_imputation, fit_missing_indicator_imputation


def test_explicit_column_without_train_nans_is_imputed_and_flagged_at_apply_time():
    """Train {a: [1, 2, 3]} has no NaN; test {a: [NaN]} used to come back unfilled with no a_was_missing column."""
    state = fit_missing_indicator_imputation(pd.DataFrame({"a": [1.0, 2.0, 3.0]}), columns=["a"])
    out = apply_missing_indicator_imputation(pd.DataFrame({"a": [np.nan]}), state)
    assert float(out["a"].iloc[0]) == 2.0
    assert list(out["a_was_missing"]) == [1]
