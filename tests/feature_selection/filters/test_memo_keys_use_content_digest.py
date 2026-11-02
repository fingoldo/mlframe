"""In-process memo keys over target arrays use the 128-bit content digest, not the builtin 64-bit ``hash`` (tracker 2026-11-02, CRY5-2)."""

import numpy as np

from mlframe.feature_selection.filters import _fe_accuracy_gate as ag
from mlframe.utils import disk_cache


def test_infer_classification_keys_its_memo_on_the_content_digest(monkeypatch):
    """The memo key comes from ``hash_array_content`` (called once per new array) and the verdict is unchanged."""
    calls = []
    real = disk_cache.hash_array_content

    def spy(arr):
        """Record the call and delegate."""
        calls.append(arr.shape)
        return real(arr)

    monkeypatch.setattr(disk_cache, "hash_array_content", spy)
    ag._INFER_CLS_MEMO.clear()
    y_cls = np.tile(np.array([0, 1, 2], dtype=np.int64), 40)
    y_reg = np.linspace(0.0, 1.0, 120)
    assert ag.infer_classification(y_cls) is True
    assert ag.infer_classification(y_reg) is False
    assert len(calls) == 2


def test_two_arrays_that_differ_in_one_middle_element_get_different_keys():
    """The digest sees every element, so a one-element change in the middle gives a different key."""
    a = np.arange(1000, dtype=np.int64)
    b = a.copy()
    b[500] += 1
    assert disk_cache.hash_array_content(a) != disk_cache.hash_array_content(b)
