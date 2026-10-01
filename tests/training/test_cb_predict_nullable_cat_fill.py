"""CatBoost's polars fastpath rejects a categorical column that carries nulls; the predict boundary fills them instead of falling to pandas.

Probed on CatBoost 1.2.10 / polars 1.44 (one column, rest float): Int8..UInt64, Boolean, Float32/64 (all nullable), non-null Enum/Categorical
(204 categories too) all fit and predict natively. A nullable Categorical at predict raises ``TypeError: No matching signature found``, a
nullable String raises ``CatBoostError: Data with nulls is not supported`` and a nullable Enum aborts the interpreter, so only nullable
categorical columns are filled -- per column, never the whole frame.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts

catboost = pytest.importorskip("catboost")


def _frames(dtype, n=300, seed=0):
    """Frames."""
    rng = np.random.default_rng(seed)
    cats = [f"c{i}" for i in range(5)]
    y = (rng.random(n) > 0.5).astype(int)

    def build(nulls: bool) -> pl.DataFrame:
        """Build."""
        vals = [cats[i] for i in rng.integers(0, 5, n)]
        if nulls:
            for i in range(0, n, 7):
                vals[i] = None
        return pl.DataFrame({"a": pl.Series("a", vals, dtype=dtype), "b": pl.Series("b", rng.standard_normal(n))})

    return build(False), build(True), y


def _fitted(dtype):
    """Fitted."""
    fit_df, nullable_df, y = _frames(dtype)
    model = catboost.CatBoostClassifier(iterations=3, verbose=0, cat_features=["a"], thread_count=1, allow_writing_files=False)
    model.fit(fit_df, y)
    return model, nullable_df


@pytest.mark.parametrize("dtype", [pl.Categorical, pl.String])
def test_nullable_categorical_predict_stays_on_polars_fastpath(dtype, caplog):
    """A nullable categorical predict frame is scored natively: no dispatch miss, no sticky pandas flag, same values as the pandas retry."""
    from mlframe.training._predict_guards import _cb_polars_to_pandas
    from mlframe.training.trainer import _predict_with_fallback

    reset_throttle_counts()
    model, nullable_df = _fitted(dtype)
    with caplog.at_level(logging.INFO):
        got = _predict_with_fallback(model, nullable_df, method="predict_proba")

    assert not any("fastpath rejected" in r.getMessage() for r in caplog.records), [r.getMessage() for r in caplog.records]
    assert not getattr(model, "_mlframe_polars_fastpath_broken", False)
    assert any("filled nulls" in r.getMessage() for r in caplog.records)
    want = model.predict_proba(_cb_polars_to_pandas(model, nullable_df, "predict_proba"))
    np.testing.assert_allclose(got, want)


def test_predict_fill_does_not_mutate_callers_frame():
    """The fill returns a new frame; the caller's nulls stay nulls."""
    from mlframe.training.trainer import _predict_with_fallback

    model, nullable_df = _fitted(pl.Categorical)
    before = nullable_df["a"].null_count()
    _predict_with_fallback(model, nullable_df, method="predict_proba")
    assert before > 0 and nullable_df["a"].null_count() == before


def test_sticky_pandas_message_does_not_contradict_itself(caplog):
    """After an observed miss on a build whose probe passes, the message says the miss is data-specific instead of asserting both facts."""
    import mlframe.training._polars_native_support as pns
    from mlframe.training.trainer import _predict_with_fallback

    class _Fake:
        """Fake."""
        _mlframe_polars_fastpath_broken = True
        _mlframe_polars_fastpath_miss_observed = True
        feature_names_ = ["a"]

        def _get_cat_feature_indices(self):
            """Get cat feature indices."""
            return []

        def _get_text_feature_indices(self):
            """Get text feature indices."""
            return []

        def predict(self, X):
            """Predict."""
            return np.zeros(len(X))

    _Fake.__name__ = "CatBoostClassifier"
    reset_throttle_counts()
    orig = pns.accepts_polars
    pns.accepts_polars = lambda lib: True
    try:
        with caplog.at_level(logging.INFO):
            _predict_with_fallback(_Fake(), pl.DataFrame({"a": [1.0, 2.0]}), method="predict")
    finally:
        pns.accepts_polars = orig
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "converted to pandas from here on" in text
    assert "specific to this model's data" in text
    assert "DOES" not in text
