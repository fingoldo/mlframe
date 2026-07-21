"""Sensor: ``_param_oracle.py`` store carve into ``_param_oracle_store.py``.

Verifies parent re-export identity AND calls into the moved bodies (ParquetStore
round-trip + aggregation, median, canonical JSON) so a missing import would fail
at runtime, not pass an import-only check.
"""

from __future__ import annotations

import os
import tempfile


def test_store_reexport_identity():
    """Store reexport identity."""
    from mlframe.utils import _param_oracle_store as sib
    from mlframe.utils import _param_oracle as parent

    for nm in ("_ParquetStore", "_median", "_stable_json", "stable_json", "SCHEMA_VERSION", "_STORE_COLUMNS"):
        assert getattr(parent, nm) is getattr(sib, nm)


def test_parquet_store_roundtrip_and_aggregate():
    """Parquet store roundtrip and aggregate."""
    from mlframe.utils._param_oracle_store import _ParquetStore, _stable_json

    pytest_importorskip_pyarrow()
    d = tempfile.mkdtemp()
    store = _ParquetStore(os.path.join(d, "oracle.parquet"))
    assert store.read_rows() == []

    row = {
        "schema_version": 1,
        "fn_name": "f",
        "host": "h",
        "fp_bucket_json": "{}",
        "param_combo_json": _stable_json({"a": 1}),
        "objective_json": _stable_json({"t": 2.0}),
        "n_obs": 1,
        "ts": "2026",
    }
    store.append([row])
    assert len(store.read_rows()) == 1
    # same key re-append folds to one aggregated row, summing n_obs
    store.append([row])
    got = store.read_rows()
    assert len(got) == 1 and got[0]["n_obs"] == 2


def test_median_and_stable_json_bodies():
    """Median and stable json bodies."""
    from mlframe.utils._param_oracle_store import _median, _stable_json

    assert _median([3.0, 1.0, 2.0]) == 2.0
    assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5
    # sort_keys canonical form (deterministic for hashing/dedup)
    assert _stable_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def pytest_importorskip_pyarrow():
    """Returns ``pytest.importorskip('pyarrow')`` (after 1 setup step)."""
    import pytest

    return pytest.importorskip("pyarrow")
