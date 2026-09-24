"""prepare_df_* must give later frames the SAME category codes as the training frame, and a stable column order."""

import pandas as pd

from mlframe.preprocessing.transforms import prepare_df_for_xgboost


def test_a_later_frame_is_coded_against_the_training_domain():
    """astype('category') on each frame coded 'c' as 2 in training and as 1 at predict: XGBoost read it as 'b'."""
    domains: dict = {}
    train = prepare_df_for_xgboost(pd.DataFrame({"g": ["a", "b", "c"]}), cat_features=["g"], out_category_domains=domains)
    assert domains == {"g": ["a", "b", "c"]}

    later = prepare_df_for_xgboost(pd.DataFrame({"g": ["a", "c"]}), cat_features=["g"], category_domains=domains)
    assert list(later["g"].cat.codes) == [0, 2], "'c' must keep the code it had in training"
    assert list(train["g"].cat.codes) == [0, 1, 2]


def test_an_unseen_value_becomes_missing_rather_than_borrowing_a_code():
    later = prepare_df_for_xgboost(pd.DataFrame({"g": ["a", "zzz"]}), cat_features=["g"], category_domains={"g": ["a", "b"]})
    assert list(later["g"].cat.codes) == [0, -1]


def test_an_already_categorical_column_is_recoded_too():
    """Its own categories may be ordered differently, which is exactly what moves the codes."""
    df = pd.DataFrame({"g": pd.Categorical(["c", "a"], categories=["c", "a"])})
    out = prepare_df_for_xgboost(df, cat_features=["g"], category_domains={"g": ["a", "c"]})
    assert list(out["g"].cat.codes) == [1, 0]


def test_cat_features_follow_the_frames_column_order():
    """Iterating a set appended detected categoricals in a hash-randomised order that differed between processes."""
    df = pd.DataFrame({name: pd.Categorical(["x"]) for name in ["zeta", "alpha", "mid", "beta"]})
    cats: list = []
    prepare_df_for_xgboost(df, cat_features=cats, inplace=True)
    assert cats == ["zeta", "alpha", "mid", "beta"]
