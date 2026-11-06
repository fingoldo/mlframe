"""A feature-cache pickle entry is unpickled from the bytes its sidecar was checked against, and a missing sidecar is refused by default.

The entry used to be verified by path and then handed to numpy by path, so a file replaced between the two was unpickled. A sidecar-less
entry was also accepted unless strict mode was set.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from mlframe.training.feature_handling import cache as cache_mod
from mlframe.utils.safe_pickle import PickleVerificationError, write_sidecar


def test_entry_replaced_after_the_digest_check_is_not_unpickled(tmp_path, monkeypatch):
    """A payload swapped on disk while numpy reads the path is not the one returned."""
    monkeypatch.delenv("MLFRAME_FEATURE_CACHE_STRICT", raising=False)
    entry = tmp_path / "entry.pkl"
    entry.write_bytes(pickle.dumps({"a": 1}))
    write_sidecar(str(entry))
    real_load = np.load

    def swap_then_load(file, *args, **kwargs):
        """Replace the entry on disk when numpy is handed the path, as an attacker racing the check would."""
        if isinstance(file, str):
            entry.write_bytes(pickle.dumps({"a": 2}))
        return real_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", swap_then_load)
    assert cache_mod._deserialize(str(entry), allow_pickle=True) == {"a": 1}


def test_sidecar_less_pickle_entry_is_refused_by_default(tmp_path, monkeypatch):
    """With pickle mode on and no sidecar, the entry is refused unless the strict flag is explicitly turned off."""
    monkeypatch.delenv("MLFRAME_FEATURE_CACHE_STRICT", raising=False)
    entry = tmp_path / "legacy.pkl"
    entry.write_bytes(pickle.dumps({"a": 1}))
    with pytest.raises(PickleVerificationError):
        cache_mod._deserialize(str(entry), allow_pickle=True)


def test_sidecar_less_entry_loads_only_with_the_explicit_opt_out(tmp_path, monkeypatch):
    """Setting the strict flag to a false value is the opt-out for entries written before the sidecar existed."""
    monkeypatch.setenv("MLFRAME_FEATURE_CACHE_STRICT", "0")
    entry = tmp_path / "legacy.pkl"
    entry.write_bytes(pickle.dumps({"a": 1}))
    assert cache_mod._deserialize(str(entry), allow_pickle=True) == {"a": 1}
