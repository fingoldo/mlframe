"""A pickle entry of the feature cache is verified against its sidecar before ``np.load(allow_pickle=True)`` unpickles it.

``np.load`` with ``allow_pickle=True`` unpickles a pure-pickle file inside numpy, so the sidecar check that followed it never guarded that read. A
tampered entry was executed, and the mismatch was only found afterwards, if at all.
"""

from __future__ import annotations

import pickle

import pytest

from mlframe.training.feature_handling.cache import _deserialize
from mlframe.utils.safe_pickle import PickleVerificationError, write_sidecar


def test_tampered_pickle_entry_is_refused_before_it_is_unpickled(tmp_path, monkeypatch):
    """With a sidecar recorded for the original bytes, replacing the entry makes the read fail with a verification error."""
    monkeypatch.delenv("MLFRAME_FEATURE_CACHE_STRICT", raising=False)
    entry = tmp_path / "entry.pkl"
    entry.write_bytes(pickle.dumps({"a": 1}))
    write_sidecar(str(entry))
    entry.write_bytes(pickle.dumps({"a": 2}))
    with pytest.raises(PickleVerificationError):
        _deserialize(str(entry), allow_pickle=True)


def test_untampered_pickle_entry_still_loads(tmp_path, monkeypatch):
    """An entry whose bytes match its sidecar loads as before."""
    monkeypatch.delenv("MLFRAME_FEATURE_CACHE_STRICT", raising=False)
    entry = tmp_path / "entry.pkl"
    entry.write_bytes(pickle.dumps({"a": 1}))
    write_sidecar(str(entry))
    assert _deserialize(str(entry), allow_pickle=True) == {"a": 1}
