"""Regression: train_mlframe_ranker_suite must label-encode object-
dtype categorical columns before invoking LGBMRanker / LGBMRanker.predict.

Pre-fix path (fuzz c0102_55b75e82):
1. ``train_mlframe_models_suite`` dispatches a learning_to_rank target to
   ``train_mlframe_ranker_suite``.
2. The FTE-emitted dataframe has object-dtype string categorical columns
   (cat_0..cat_N) on a pl_utf8 input that the classifier/regressor
   pre-pipeline's CatBoostEncoder normally converts upstream.
3. The ranker dispatch skips that pre-pipeline (the FTE frame is consumed
   directly), so the object-dtype columns reach LGB.
4. ``LGBMRanker.fit`` (or ``Booster.predict`` post-fit) rejects them at
   ``lightgbm/basic.py:805``::

       ValueError: pandas dtypes must be int, float or bool.
       Fields with bad pandas dtypes: cat_0: object, cat_1: object, ...

Post-fix: ``train_mlframe_ranker_suite`` label-encodes object-dtype cat
columns to int32 codes using a shared train+val+test vocabulary
*before* the inner ranker fit fires. Unseen values map to -1 (LGB
missing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_object_cat_frame(n: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    num = rng.standard_normal(n).astype(np.float64)
    cat0 = np.array([f"a{i % 3}" for i in range(n)], dtype=object)
    cat1 = np.array([f"b{i % 5}" for i in range(n)], dtype=object)
    df = pd.DataFrame({"num_0": num, "cat_0": cat0, "cat_1": cat1})
    return df


def test_object_cats_encoded_to_int_codes_via_shared_vocab() -> None:
    """Replay the ranker_suite encoding block on synthetic frames and
    assert the output has int32 dtype across all splits with consistent
    codes (same string -> same int across train/val/test)."""
    train_df = _make_object_cat_frame(30, seed=0)
    val_df = _make_object_cat_frame(15, seed=1)
    test_df = _make_object_cat_frame(10, seed=2)
    cat_features = ["cat_0", "cat_1"]

    # Replay the exact encoding logic from train_mlframe_ranker_suite.
    # Uses ``is_string_dtype`` (not ``dtype == object``) so pandas 2.1+
    # ``infer_string`` / pyarrow-backed string columns also get caught;
    # the prior ``dtype == object`` check failed on modern pandas because
    # the dataframe constructor auto-converted np.array(..., dtype=object)
    # of strings into pd.StringDtype.
    _to_encode = [c for c in cat_features if c in train_df.columns and pd.api.types.is_string_dtype(train_df[c])]
    assert _to_encode == cat_features

    _splits_for_vocab = [train_df, val_df, test_df]
    _vocabs: dict[str, dict] = {}
    for _c in _to_encode:
        _vals = set()
        for _split in _splits_for_vocab:
            _vals.update(v for v in _split[_c].dropna().tolist())
        _vocabs[_c] = {v: i for i, v in enumerate(sorted(_vals, key=lambda x: str(x)))}
    encoded = []
    for _df in _splits_for_vocab:
        _new = _df.copy()
        for _c in _to_encode:
            _new[_c] = _new[_c].map(_vocabs[_c]).fillna(-1).astype("int32")
        encoded.append(_new)
    enc_train, enc_val, enc_test = encoded

    # All cat columns must now be int32 in every split.
    for _df in (enc_train, enc_val, enc_test):
        for _c in cat_features:
            assert _df[_c].dtype == np.int32, f"{_c} expected int32, got {_df[_c].dtype}"

    # Shared vocab: same string -> same code in every split.
    code_a0_in_train = enc_train.loc[train_df["cat_0"] == "a0", "cat_0"].iloc[0]
    code_a0_in_val = enc_val.loc[val_df["cat_0"] == "a0", "cat_0"].iloc[0]
    code_a0_in_test = enc_test.loc[test_df["cat_0"] == "a0", "cat_0"].iloc[0]
    assert code_a0_in_train == code_a0_in_val == code_a0_in_test


def test_unseen_values_in_val_map_to_minus_one() -> None:
    """Values seen only at predict time (never in any of the vocab-
    contributing splits) must map to -1, the LGB-missing sentinel."""
    train_df = pd.DataFrame({"cat_0": np.array(["a", "b", "c"], dtype=object)})
    val_df = pd.DataFrame({"cat_0": np.array(["a", "b"], dtype=object)})
    test_df = pd.DataFrame({"cat_0": np.array(["a", "b"], dtype=object)})

    # Build vocab from train+val+test (the only values are a/b/c).
    _vocab = {v: i for i, v in enumerate(sorted({"a", "b", "c"}, key=str))}
    # Now simulate a predict-time frame with an unseen value.
    predict_df = pd.DataFrame({"cat_0": np.array(["a", "d_unseen"], dtype=object)})
    encoded = predict_df["cat_0"].map(_vocab).fillna(-1).astype("int32")
    assert encoded.iloc[0] == _vocab["a"]
    assert encoded.iloc[1] == -1


def test_lgb_ranker_smoke_with_object_cats() -> None:
    """End-to-end: train an LGBMRanker on a frame whose cat columns
    are encoded via the ranker_suite path. Predict must not raise."""
    pytest.importorskip("lightgbm")
    from mlframe.training.ranking.ranker_suite import train_mlframe_ranker_suite
    from mlframe.training.configs import LearningToRankConfig

    n = 60
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "num_0": rng.standard_normal(n).astype(np.float64),
            "cat_0": np.array([f"x{i % 4}" for i in range(n)], dtype=object),
            "qid": np.repeat(np.arange(n // 6), 6),
            "rel": rng.integers(0, 4, size=n).astype(np.int32),
        }
    )

    # Standard FTE contract per train_mlframe_ranker_suite: ``transform``
    # returns a 5+-tuple ``(df_features, target_by_type, group_ids_raw,
    # group_ids, timestamps, ...)``. The previous 3-tuple shape
    # ``(features, target, groups)`` always tripped the suite's
    # ``transformed[3]`` lookup with None and raised "FTE produced no
    # group_ids despite group_field being set", which the test then
    # swallowed via ``pytest.skip("unrelated error")`` -- masking the
    # actual object-cats-encoded-to-int contract this test is supposed
    # to verify. Now we return the correct shape so the test runs end
    # to end.
    from mlframe.training.configs import TargetTypes

    class _FTE:
        target_column = "rel"
        target_type = None
        regression = False
        group_field = "qid"
        ts_field = None
        doc_field = None
        target_carrier = "numpy"

        def transform(self, frame):
            features = frame.drop(columns=["rel"]) if "rel" in frame.columns else frame
            features = features.drop(columns=["qid"]) if "qid" in features.columns else features
            target = np.asarray(frame["rel"]) if "rel" in frame.columns else None
            groups = np.asarray(frame["qid"]) if "qid" in frame.columns else None
            target_by_type = {TargetTypes.LEARNING_TO_RANK: {"rel": target}}
            return features, target_by_type, groups, groups, None

    fte = _FTE()
    out = train_mlframe_ranker_suite(
        df=df,
        target_name="rel",
        model_name="ltr_test",
        features_and_targets_extractor=fte,
        mlframe_models=["lgb"],
        ranking_config=LearningToRankConfig(ensemble_method="rrf"),
        iterations=5,
        verbose=0,
    )
    assert out is not None, "ranker suite must complete without crashing on object cats"
