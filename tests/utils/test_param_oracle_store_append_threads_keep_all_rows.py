"""U5: concurrent in-process appends must not lose rows when filelock is absent."""
import sys
import threading

from mlframe.utils._param_oracle_store import _ParquetStore


def _row(i):
    """One distinct observation row."""
    return {"schema_version": 1, "fn_name": "f", "host": "h", "fp_bucket_json": "{}", "param_combo_json": f'{{"i": {i}}}', "objective_json": '{"t": 1.0}', "n_obs": 1, "ts": "2026"}


def test_append_threads_without_filelock_lose_no_rows(tmp_path, monkeypatch):
    """Eight threads released by a barrier append one row each with filelock unavailable; all eight rows survive."""
    import pyarrow.parquet  # noqa: F401  (import once up front so the threads do not race the import itself)

    monkeypatch.setitem(sys.modules, "filelock", None)
    store = _ParquetStore(str(tmp_path / "s.parquet"))
    n = 8
    barrier = threading.Barrier(n)
    errors = []

    def _work(i):
        """Wait at the barrier, then append one row."""
        try:
            barrier.wait(timeout=5)
            store.append([_row(i)])
        except Exception as e:  # surfaced below
            errors.append(e)

    threads = [threading.Thread(target=_work, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert not errors, errors
    assert len(store.read_rows()) == n
