"""Cluster 29 regressions: probe flags, NaN bucket markers, fingerprint row identity, structured-dtype hashing, and array-like saves (audit 2026-11-20, findings 29-F1..F4, 29-F6)."""

import numpy as np
import pandas as pd

from mlframe.utils._param_oracle import _as_2d_numeric, linear_bucket, log_bucket
from mlframe.utils.disk_cache import hash_array_summary
from mlframe.utils.native_import_probe import _probe_child
from mlframe.training.utils import save_series_or_df


def test_probe_child_runs_on_this_interpreter():
    """The child is started with -I, which exists on every supported Python, not -P (29-F1)."""
    result = _probe_child("json")
    assert result is not None and result.returncode == 0


def test_unmeasurable_dimensions_keep_a_nan_marker_in_the_bucket():
    """A NaN input used to bucket to 0.0, erasing the incomplete-fingerprint marker (29-F3)."""
    assert log_bucket(float("nan")) == "nan"
    assert linear_bucket(float("inf")) == "nan"


def test_fingerprint_keeps_the_true_row_count_when_it_samples(monkeypatch):
    """Pre-sampling must not change the identity: n is the full frame's row count (29-F2)."""
    import mlframe.utils._param_oracle as oracle

    monkeypatch.setattr(oracle, "_FINGERPRINT_MAX_CELLS", 10_000)  # shrink the cell budget so a small frame is sampled
    n_rows = 5_000
    df = pd.DataFrame({f"c{i}": np.arange(n_rows, dtype=np.float64) for i in range(10)})
    arr2d, n, p, _kind = _as_2d_numeric(df)
    assert n == n_rows
    assert p == 10
    assert arr2d.shape[0] < n_rows


def test_structured_dtype_hashes_the_whole_buffer():
    """Only head and tail rows used to be hashed for void dtype, so a middle-row change collided (29-F4)."""
    dt = np.dtype([("a", np.int64), ("b", np.float64)])
    base = np.zeros(1000, dtype=dt)
    changed = base.copy()
    changed[500]["b"] = 3.5
    assert hash_array_summary(base) != hash_array_summary(changed)


def test_numpy_array_is_written_not_silently_skipped(tmp_path):
    """A numpy input used to fall through without writing anything (29-F6)."""
    target = tmp_path / "artifact.parquet"
    save_series_or_df(np.arange(5), str(target), compression="snappy")
    assert target.exists()
