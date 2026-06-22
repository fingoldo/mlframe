import pandas as pd

from mlframe.preprocessing.cleaning import apply_features_cleaning


def test_apply_features_cleaning_iterates_transforms_via_items():
    """features_transforms is a Dict[col -> repl_instructions]. Pre-fix code iterated the dict
    directly (yielding keys only) and unpacked `col, repl_instructions`, raising ValueError.
    It must iterate `.items()` and apply the replacement."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    features_cleaning = {
        "features_transforms": {"a": {2: 20}},
        "constant_features": [],
    }
    apply_features_cleaning(df, features_cleaning)
    assert df["a"].tolist() == [1, 20, 3]
