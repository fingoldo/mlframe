"""Cluster 28 regressions: tests for helpers the audit found with no test reference (audit 2026-11-20, 28-F76, part).

Covers ``_oof_train_timestamps`` (numpy / pandas / tz-aware / absent inputs) and ``retry_params_after_empty_text_dictionary`` (no text features, a
unigram rescue, and dropping unusable columns into ``cat_features``).
"""

import numpy as np
import pandas as pd

from mlframe.training import _training_loop_text_retry as tr
from mlframe.training._trainer_train_and_evaluate_parts import _oof_train_timestamps


def test_oof_timestamps_none_and_no_index_pass_through():
    """No timestamps gives None; no train index returns the timestamps unchanged (28-F76)."""
    assert _oof_train_timestamps(None, np.array([0, 1])) is None
    ts = np.arange(5)
    assert _oof_train_timestamps(ts, None) is ts


def test_oof_timestamps_select_train_rows_for_numpy_and_keep_tz_for_pandas():
    """A numpy array is indexed by the train rows; a tz-aware pandas Series stays a Series with its dtype (28-F76)."""
    np.testing.assert_array_equal(_oof_train_timestamps(np.array([10, 11, 12, 13]), [3, 1]), [13, 11])
    s = pd.Series(pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"))
    out = _oof_train_timestamps(s, [2, 0])
    assert isinstance(out, pd.Series) and str(out.dtype) == str(s.dtype)
    assert out.iloc[0] == s.iloc[2] and out.iloc[1] == s.iloc[0]


def test_retry_without_text_features_does_not_try_again():
    """An empty-dictionary error with no text features is an unexpected variant: no retry, params unchanged (28-F76)."""
    params = {"cat_features": ["a"]}
    again, out = tr.retry_params_after_empty_text_dictionary(object(), None, None, params)
    assert again is False and out == params


class _Model:
    """Records the set_params call of the unigram rescue."""

    def __init__(self):
        self.params = {}

    def set_params(self, **kw):
        """Remember the parameters."""
        self.params.update(kw)


def test_unigram_rescue_keeps_every_text_feature(monkeypatch):
    """When unigrams rescue the columns, text_processing is set on the model and fit_params keep their text features (28-F76)."""
    import mlframe.training.cb as cb

    monkeypatch.setattr(cb, "unigram_rescues_text_features", lambda *a, **k: True)
    monkeypatch.setattr(cb, "unigram_text_processing", lambda: {"tokenizers": "unigram"})
    model = _Model()
    again, out = tr.retry_params_after_empty_text_dictionary(model, None, None, {"text_features": ["t1", "t2"]})
    assert again is True
    assert model.params == {"text_processing": {"tokenizers": "unigram"}}
    assert out["text_features"] == ["t1", "t2"]


def test_unusable_columns_are_moved_to_cat_features(monkeypatch):
    """Without a rescue, the unusable text column leaves text_features and joins cat_features, the rest are kept (28-F76)."""
    import mlframe.training.cb as cb

    monkeypatch.setattr(cb, "unigram_rescues_text_features", lambda *a, **k: False)
    monkeypatch.setattr(cb, "unusable_text_features", lambda *a, **k: {"t2": "one token per row"})
    again, out = tr.retry_params_after_empty_text_dictionary(_Model(), None, None, {"text_features": ["t1", "t2"], "cat_features": ["c"]})
    assert again is True
    assert out["text_features"] == ["t1"]
    assert out["cat_features"] == ["c", "t2"]
