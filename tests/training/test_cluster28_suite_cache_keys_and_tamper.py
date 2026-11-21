"""Audit 2026-11-20 findings 28-F29 and 28-F28 in the suite artefact cache.

28-F29: two different large arrays that share their head and tail get different key fragments, because the fragment digests the bytes.
28-F28: a digest mismatch on read is a WARNING and deletes the entry, rather than a debug line that leaves it to be recomputed forever.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mlframe.training import suite_artefact_cache as sac


def test_arrays_with_equal_head_and_tail_get_different_fragments():
    """The middle of a large array changes the fingerprint, where repr() would have elided it."""
    a = np.zeros(100_000)
    b = a.copy()
    b[50_000] = 1.0
    assert sac._arg_fingerprint(a) != sac._arg_fingerprint(b)


def test_same_array_gives_the_same_fragment():
    """The fingerprint is stable for equal content."""
    a = np.arange(10.0)
    assert sac._arg_fingerprint(a) == sac._arg_fingerprint(a.copy())


def test_pandas_frame_fingerprint_follows_content():
    """A pandas frame is fingerprinted by its values, so a changed cell changes the fragment."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    other = df.copy()
    other.iloc[1, 0] = 9.0
    assert sac._arg_fingerprint(df) != sac._arg_fingerprint(other)


def test_tampered_entry_is_logged_as_warning_and_deleted(tmp_path, monkeypatch, caplog):
    """A digest mismatch on get() logs at WARNING and removes the file."""
    cache = sac.SuiteArtefactCache(str(tmp_path))
    key = "k1"
    path = Path(cache._path(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"tampered")

    def raise_verification(_path):
        """Simulate the verifying loader rejecting the entry."""
        raise sac.PickleVerificationError("digest mismatch")

    monkeypatch.setattr(sac, "_safe_pickle_load", raise_verification)
    with caplog.at_level(logging.WARNING):
        assert cache.get(key, default="fallback") == "fallback"
    assert not path.exists()
    assert any(r.levelno == logging.WARNING and "verification" in r.getMessage() for r in caplog.records)
