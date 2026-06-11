"""Regression: ``_apply_pre_pipeline_with_passthrough`` must not re-copy
the whole frame per passthrough column.

Two perf/memory bugs (both catastrophic on a 100GB frame):
  1. The re-attach loop called ``input_for_model.reset_index(drop=True)``
     INSIDE the loop over passthrough cols -> N full-frame resets (N copies)
     for an N-column passthrough. Fix hoists the reset OUT of the loop
     (once before it).
  2. Stashing passthrough cols from a polars source called
     ``get_column(_pc).to_pandas()`` PER column -> N polars->pandas
     conversions. Fix batches it to ONE ``select(cols).to_pandas()``.

Both fixes are behaviour-preserving: the same columns are re-attached.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


class _DropPassthroughPP(BaseEstimator, TransformerMixin):
    """Fitted pre_pipeline that drops the passthrough cols at transform."""

    def __init__(self, drop_cols) -> None:
        self.is_fitted_ = True
        self._drop_cols = list(drop_cols)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in X.columns if c in self._drop_cols])


def _make_inputs(n=5):
    cols = ["txt0", "txt1", "emb0"]
    df = pd.DataFrame(
        {
            "x": np.arange(n, dtype=np.float64),
            "txt0": [f"a{i}" for i in range(n)],
            "txt1": [f"b{i}" for i in range(n)],
            "emb0": [f"c{i}" for i in range(n)],
        }
    )
    metadata = {"text_features": ["txt0", "txt1"], "embedding_features": ["emb0"]}
    return df, cols, metadata


class _ModelObj:
    def __init__(self, pp):
        self.pre_pipeline = pp


def test_reattach_resets_frame_index_at_most_once(monkeypatch) -> None:
    """3+ passthrough cols must trigger <=1 full-frame reset_index, not N."""
    from mlframe.training.core import _predict_pre_pipeline as mod

    df, cols, metadata = _make_inputs()
    pp = _DropPassthroughPP(drop_cols=cols)

    calls = {"n": 0}
    _orig_reset = pd.DataFrame.reset_index

    def _counting_reset(self, *a, **k):
        calls["n"] += 1
        return _orig_reset(self, *a, **k)

    monkeypatch.setattr(pd.DataFrame, "reset_index", _counting_reset)

    out = mod._apply_pre_pipeline_with_passthrough(
        df.copy(),
        model=object(),
        model_obj=_ModelObj(pp),
        pipeline=None,
        df=df,
        df_pre_pipeline=df,
        metadata=metadata,
        model_name="mem_test",
        verbose=0,
    )

    # Behaviour preserved: all 3 passthrough cols re-attached correctly.
    assert isinstance(out, pd.DataFrame)
    for c in cols:
        assert c in out.columns
        assert list(out[c]) == list(df[c])

    # The whole-frame reset must happen at most once despite 3 passthrough cols.
    # Pre-fix this was 3 (one per column). The per-column Series reset is a
    # separate (cheap) object and does not count against the frame here.
    assert calls["n"] <= 1, f"frame reset_index called {calls['n']} times (expected <=1)"


def test_polars_stash_does_one_to_pandas(monkeypatch) -> None:
    """A polars stash source must convert to pandas ONCE, not per column."""
    from mlframe.training.core import _predict_pre_pipeline as mod

    _df_pd, cols, metadata = _make_inputs()
    df_pl = pl.from_pandas(_df_pd)
    pp = _DropPassthroughPP(drop_cols=cols)

    calls = {"n": 0}
    _orig_to_pandas = pl.DataFrame.to_pandas

    def _counting_to_pandas(self, *a, **k):
        calls["n"] += 1
        return _orig_to_pandas(self, *a, **k)

    monkeypatch.setattr(pl.DataFrame, "to_pandas", _counting_to_pandas)

    # input_for_model is pandas (post main pipeline); the broad stash source is polars.
    out = mod._apply_pre_pipeline_with_passthrough(
        _df_pd.copy(),
        model=object(),
        model_obj=_ModelObj(pp),
        pipeline=None,
        df=df_pl,
        df_pre_pipeline=df_pl,
        metadata=metadata,
        model_name="mem_test_pl",
        verbose=0,
    )

    assert isinstance(out, pd.DataFrame)
    for c in cols:
        assert c in out.columns
        assert list(out[c]) == list(_df_pd[c])

    # Pre-fix: one to_pandas per passthrough col (3). Post-fix: exactly 1.
    assert calls["n"] == 1, f"polars to_pandas called {calls['n']} times (expected 1)"
