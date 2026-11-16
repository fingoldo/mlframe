"""Regressions from the 2026-11-17 utils audit (param-oracle store, log sanitizer)."""
import glob
import logging
import os

import pytest

pytest.importorskip("pyarrow")

from mlframe.utils import log_sanitize
from mlframe.utils._param_oracle_store import _ParquetStore, _weighted_median


def test_weighted_median_even_total_is_mean_of_middle_pair():
    """An even total weight returns the mean of the middle pair."""
    assert _weighted_median([(1.0, 1), (3.0, 1)]) == 2.0
    assert _weighted_median([(1.0, 2), (5.0, 2)]) == 3.0
    assert _weighted_median([(1.0, 1), (3.0, 1), (9.0, 1)]) == 3.0


def _row(n=1):
    """Build a minimal row fixture."""
    return {"fn_name": "f", "host": "h", "fp_bucket_json": "{}", "param_combo_json": "{}", "objective_json": '{"t": 1.0}', "n_obs": n, "ts": "1"}


def test_unreadable_store_is_kept_aside_not_overwritten(tmp_path):
    """An unreadable store is moved aside before a fresh write."""
    path = str(tmp_path / "s.parquet")
    with open(path, "wb") as f:
        f.write(b"not parquet at all")
    _ParquetStore(path).append([_row()])
    kept = glob.glob(path + ".corrupt-*")
    assert len(kept) == 1
    with open(kept[0], "rb") as f:
        assert f.read() == b"not parquet at all"
    assert len(_ParquetStore(path).read_rows()) == 1


def test_failed_write_leaves_no_temp_file(tmp_path, monkeypatch):
    """A failed write removes its temp shard."""
    import pyarrow.parquet as pq

    def boom(tbl, where, *a, **k):
        """Fail the write after the temp file is created."""
        open(where, "wb").write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(pq, "write_table", boom)
    store = _ParquetStore(str(tmp_path / "s.parquet"))
    with pytest.raises(OSError):
        store.append([_row()])
    assert [n for n in os.listdir(tmp_path) if n.endswith(".tmp")] == []


def test_installed_factory_survives_uninstall():
    """A record built after uninstall still uses the previous factory."""
    log_sanitize.uninstall_log_sanitizer()
    assert log_sanitize.install_log_sanitizer()
    factory = logging.getLogRecordFactory()
    try:
        assert log_sanitize.uninstall_log_sanitizer()
        rec = factory("mlframe.x", logging.INFO, __file__, 1, "m %s", ("a\nb",), None)
        assert rec.args == ("a" + chr(92) + "nb",)
    finally:
        log_sanitize.uninstall_log_sanitizer()
