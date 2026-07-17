"""Regression sensors for ``_key_bank`` save/load.

LEAK4 (``save_key_bank``): the np.save x3 + pickle write block had no cleanup, so any write error left the
multi-hundred-MB UUID tmp dir orphaned forever. The fix wraps the writes in ``try/except: rmtree(tmp_dir); raise``.

SEC2 (``try_load_key_bank``): bare ``pickle.load`` on cache files was an RCE surface. Loads now go through
``safe_pickle.safe_load`` (sha256 sidecar verified, written by ``save_key_bank`` via ``safe_dump``); a tampered
pickle is refused (the loader treats the verification failure as a cache miss).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer._key_bank import (
    KeyBank,
    save_key_bank,
    try_load_key_bank,
)
from mlframe.feature_engineering.transformer._row_attention_ann import _AnnIndex


def _make_bank() -> KeyBank:
    """Helper: Make bank."""
    n_heads, n_train, head_dim, d_input = 2, 5, 3, 4
    projections = np.zeros((n_heads, d_input, head_dim), dtype=np.float32)
    k_proj = np.zeros((n_heads, n_train, head_dim), dtype=np.float32)
    y_train = np.zeros(n_train, dtype=np.float32)
    bank = KeyBank(projections=projections, k_proj=k_proj, y_train=y_train)
    # A trivially picklable per-head index stand-in (pynndescent backend path -> goes through safe_dump).
    bank.ann_indices = [_AnnIndex(backend="pynndescent", obj={"head": h}, metric="cosine", head_dim=head_dim) for h in range(n_heads)]
    return bank


def test_save_key_bank_cleans_tmp_dir_on_write_failure(tmp_path, monkeypatch):
    """A write error mid-save must leave no orphan ``<fingerprint>.tmp.*`` dir behind."""
    import mlframe.feature_engineering.transformer._key_bank as kb

    bank = _make_bank()
    cache_dir = tmp_path / "cache"

    # Make the second np.save raise (after the tmp dir + first file already exist) to simulate a real mid-write failure.
    calls = {"n": 0}
    real_save = np.save

    def boom_save(file, arr, *a, **k):
        """Boom save."""
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated disk-full mid-write")
        return real_save(file, arr, *a, **k)

    monkeypatch.setattr(kb.np, "save", boom_save)

    with pytest.raises(OSError):
        save_key_bank(bank, cache_dir, "fp_fail")

    orphans = list(cache_dir.glob("fp_fail.tmp.*"))
    assert orphans == [], f"orphan tmp dir(s) left behind: {orphans}"


def test_save_load_round_trip_via_safe_pickle(tmp_path):
    """Happy path: save writes sidecars, load verifies + reconstructs the bank."""
    bank = _make_bank()
    cache_dir = tmp_path / "cache"
    save_key_bank(bank, cache_dir, "fp_ok")

    bank_dir = cache_dir / "fp_ok"
    assert (bank_dir / "metadata.pkl.sha256").is_file(), "save must write the sha256 sidecar for safe_load"

    loaded = try_load_key_bank(cache_dir, "fp_ok")
    assert loaded is not None
    assert loaded.n_heads == bank.n_heads
    assert len(loaded.ann_indices) == bank.n_heads


def test_tampered_pickle_is_refused(tmp_path):
    """A cache pickle tampered after save (sidecar no longer matches) must be refused -> treated as a miss."""
    bank = _make_bank()
    cache_dir = tmp_path / "cache"
    save_key_bank(bank, cache_dir, "fp_tamper")

    meta = cache_dir / "fp_tamper" / "metadata.pkl"
    meta.write_bytes(meta.read_bytes() + b"tampered-tail-bytes")

    # safe_load raises PickleVerificationError on digest mismatch; try_load_key_bank swallows it as a cache miss.
    assert try_load_key_bank(cache_dir, "fp_tamper") is None
