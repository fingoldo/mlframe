"""augment_temporal_drift keeps a numeric label out of the standardised features when told which column it is, and never back-fills a NaN label from an earlier vintage."""
import pandas as pd

from mlframe.preprocessing.temporal_drift_augment import augment_temporal_drift


def test_label_cols_keeps_a_numeric_label_as_the_true_last_period_label():
    """Without label_cols the default feature list included y, and the synthetic row carried a z-score (about 0.58) instead of the entity's real label 1."""
    df = pd.DataFrame({"id": [1, 1, 1, 1], "t": [1, 2, 3, 4], "x": [1.0, 2.0, 3.0, 5.0], "y": [0, 0, 1, 1]})
    out = augment_temporal_drift(df, "id", "t", label_cols=["y"])
    synthetic = out.loc[out["_temporal_drift_augmented"]]
    assert list(synthetic["y"]) == [1]


def test_nan_label_in_the_last_period_is_not_replaced_by_the_earlier_vintage_label():
    """The last period's label is missing; the synthetic row must not take the earlier period's 'A'."""
    df = pd.DataFrame({"id": [1, 1, 1], "t": [1, 2, 3], "x": [1.0, 2.0, 4.0], "label": ["A", "A", None]})
    out = augment_temporal_drift(df, "id", "t")
    synthetic = out.loc[out["_temporal_drift_augmented"]]
    assert synthetic["label"].isna().all()
