"""Determinism regression: ``metadata["cat_features"]`` must preserve input order.

Pre-fix, ``raw_cat_features = list(set(...))`` reordered the categorical feature
list per PYTHONHASHSEED, making a fixed-random_state suite non-reproducible (the
recipe + CatBoost Pool cat_features order changed run-to-run). The fix is an
order-preserving dedup. This test pins first-seen input order.
"""

import pandas as pd

from mlframe.training.core._phase_helpers_fit_split import _phase_auto_detect_feature_types


def _run(cat_features, cat_features_polars):
    # Many overlapping names so set-iteration order would visibly diverge from input order.
    cols = ["zzz", "mmm", "aaa", "qqq", "bbb", "kkk", "ccc", "ttt", "ddd", "www"]
    df = pd.DataFrame({c: [1, 2, 3] for c in cols})
    df["y"] = [0, 1, 0]
    metadata: dict = {}
    out = _phase_auto_detect_feature_types(
        train_df=df,
        val_df=None,
        test_df=None,
        train_df_polars_pre=None,
        val_df_polars_pre=None,
        test_df_polars_pre=None,
        cat_features=list(cat_features),
        cat_features_polars=list(cat_features_polars),
        was_polars_input=False,
        all_models_polars_native=False,
        pipeline_config=None,
        feature_types_config=None,
        metadata=metadata,
        verbose=False,
        train_df_pandas_pre_meta=None,
    )
    # cat_features is the 9th return element and also stored in metadata.
    return out[8], metadata["cat_features"]


def test_cat_features_preserves_first_seen_input_order():
    cat = ["zzz", "mmm", "aaa", "qqq", "bbb"]
    cat_polars = ["aaa", "kkk", "ccc", "zzz", "ttt"]  # overlaps + new names
    expected = ["zzz", "mmm", "aaa", "qqq", "bbb", "kkk", "ccc", "ttt"]
    ret, meta = _run(cat, cat_polars)
    assert ret == expected, ret
    assert meta == expected, meta


def test_cat_features_order_stable_across_runs():
    cat = ["ttt", "ddd", "www", "mmm", "ccc"]
    cat_polars = ["kkk", "aaa", "ttt", "bbb"]
    first, _ = _run(cat, cat_polars)
    for _ in range(5):
        again, _ = _run(cat, cat_polars)
        assert again == first, (first, again)
    # No dropped names, no duplicates.
    assert sorted(first) == sorted(set(cat) | set(cat_polars))
