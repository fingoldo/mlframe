"""Audit 2026-11-20 finding 26-F8: the binary-column fill records and restores the column's current dtype.

The fill used to take the dtype from a head snapshot taken before step 3 converts object columns to category. The test checks the
recorded dtype of a two-valued object column with a missing cell against the frame's own dtype at the time of the fill.
"""

import pandas as pd

from mlframe.preprocessing.cleaning import analyse_and_clean_features


def test_binary_fill_records_the_current_dtype():
    """The features_cleaning entry for a two-valued object column records the frame's dtype, object here."""
    df = pd.DataFrame({"b": pd.Series(["yes", "no", None, "yes", "no", "yes"] * 5, dtype=object)})
    cleaning = analyse_and_clean_features(df, update_data=True)
    assert df["b"].dtype == object
    assert str(cleaning["dtypes"]["b"]) == str(df["b"].dtype)
