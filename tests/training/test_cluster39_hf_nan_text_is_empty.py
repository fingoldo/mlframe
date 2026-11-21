"""Audit 2026-11-20 finding 39-F18: a missing text (NaN) is embedded as the empty string, not as the word 'nan'."""

from types import SimpleNamespace

import numpy as np

from mlframe.training.feature_handling.hf_provider import HuggingFaceProvider


def _provider_with_recorded_inference():
    """A provider whose model is never loaded: the inference step is replaced by a recorder of what it receives."""
    prov = HuggingFaceProvider.__new__(HuggingFaceProvider)
    prov._is_loaded = True
    prov._auto_prefix = None
    prov._cfg = SimpleNamespace(params={"batch_size": 4, "max_length": 16})
    seen = {}

    def _record(texts, batch_size, max_length, pool):
        """Store the texts the transform hands to inference and return a placeholder matrix."""
        seen["texts"] = list(texts)
        return np.zeros((len(texts), 2))

    prov._batched_inference = _record
    prov._resolve_batch_size = lambda bs: int(bs)
    return prov, seen


def test_nan_and_none_texts_become_empty_strings_not_nan_words():
    """NaN and None are embedded as '' so the model does not see the word 'nan' as content."""
    prov, seen = _provider_with_recorded_inference()
    prov.transform(["hello", float("nan"), None, "bye"])
    assert seen["texts"] == ["hello", "", "", "bye"]
