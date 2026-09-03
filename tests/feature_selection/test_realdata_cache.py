"""Tests for the offline, checksummed real-data cache backing the fs_hybrid real-data bench.

The cache is a developer artifact, not a shipped resource, so every test that needs actual data skips cleanly
when the cache is empty. The one contract worth pinning unconditionally is the checksum enforcement: a payload
that no longer matches its recorded digest must raise rather than silently feed corrupt data into a benchmark.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "fs_hybrid" / "_realdata_cache.py"


def _load_module() -> Any:
    """Import `_realdata_cache` by path: `_benchmarks/fs_hybrid` is not an importable package from the test tree."""
    spec = importlib.util.spec_from_file_location("_fs_hybrid_realdata_cache", _MODULE_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"cannot load {_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cache_module() -> Any:
    """The `_realdata_cache` module, loaded once per test module."""
    if not _MODULE_PATH.exists():
        pytest.skip(f"{_MODULE_PATH} is absent")
    return _load_module()


@pytest.fixture()
def populated_name(cache_module: Any) -> str:
    """Name of one dataset present in the real cache, or a skip when the cache has not been filled on this machine."""
    entries = cache_module.available()
    if not entries:
        pytest.skip("real-data cache is empty; run _realdata_cache.fill_cache() with network access to populate it")
    return str(entries[0]["name"])


def test_available_returns_list_without_network(cache_module: Any) -> None:
    """`available()` is pure local I/O and returns a list even when the cache directory does not exist."""
    assert isinstance(cache_module.available(), list)
    assert isinstance(cache_module.available(cache_dir="does-not-exist-anywhere"), list)


def test_load_cached_missing_dataset_raises_with_fill_cache_hint(cache_module: Any, tmp_path: Path) -> None:
    """A cache miss names the remedy instead of failing with a bare KeyError deep inside numpy."""
    with pytest.raises(FileNotFoundError, match="fill_cache"):
        cache_module.load_cached("no_such_dataset", cache_dir=tmp_path)


def test_load_cached_returns_frame_array_and_meta(cache_module: Any, populated_name: str) -> None:
    """A populated cache round-trips to a DataFrame, an integer target and metadata consistent with the payload."""
    X, y, meta = cache_module.load_cached(populated_name)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0] == meta["n_rows"]
    assert X.shape[1] == meta["n_cols"]
    assert np.issubdtype(y.dtype, np.integer)


def test_load_cached_rejects_corrupted_payload(cache_module: Any, populated_name: str, tmp_path: Path) -> None:
    """Flipping bytes in a copied `.npz` must make the loader raise, not hand back silently wrong benchmark data."""
    root = cache_module.cache_dir_default()
    npz_src = root / f"{populated_name}.npz"
    json_src = root / f"{populated_name}.json"
    npz_dst = tmp_path / npz_src.name
    shutil.copy2(npz_src, npz_dst)
    shutil.copy2(json_src, tmp_path / json_src.name)

    cache_module.load_cached(populated_name, cache_dir=tmp_path)  # the untouched copy still verifies

    payload = bytearray(npz_dst.read_bytes())
    payload[-1] ^= 0xFF
    npz_dst.write_bytes(bytes(payload))

    with pytest.raises(ValueError, match="corrupt"):
        cache_module.load_cached(populated_name, cache_dir=tmp_path)


def test_load_cached_rejects_tampered_sidecar_digest(cache_module: Any, populated_name: str, tmp_path: Path) -> None:
    """The digest is enforced in both directions: a rewritten sidecar hash fails just as a corrupt payload does."""
    root = cache_module.cache_dir_default()
    shutil.copy2(root / f"{populated_name}.npz", tmp_path / f"{populated_name}.npz")
    json_dst = tmp_path / f"{populated_name}.json"
    meta = json.loads((root / f"{populated_name}.json").read_text(encoding="utf-8"))
    meta["sha256"] = "0" * 64
    json_dst.write_bytes(json.dumps(meta, indent=2, sort_keys=True).encode("utf-8"))

    with pytest.raises(ValueError, match="corrupt"):
        cache_module.load_cached(populated_name, cache_dir=tmp_path)
