"""Audit 2026-11-20 finding 28-F7: a re-run on the same directory rewrites split artifacts, so no file from an earlier split survives."""

import numpy as np
import pandas as pd

from mlframe.training.preprocessing import save_split_artifacts


def test_second_run_overwrites_the_first_runs_split_artifacts(tmp_path):
    """Two runs write different validation timestamps into the same directory; the file on disk holds the second run's values."""
    timestamps = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
    first = dict(train_idx=np.arange(6), val_idx=np.arange(6, 8), test_idx=np.arange(8, 10))
    second = dict(train_idx=np.arange(4), val_idx=np.arange(4, 7), test_idx=np.arange(7, 10))
    for split in (first, second):
        save_split_artifacts(
            train_idx=split["train_idx"], val_idx=split["val_idx"], test_idx=split["test_idx"], timestamps=timestamps,
            group_ids_raw=None, artifacts=None, data_dir=str(tmp_path), models_dir="models", target_name="t", model_name="m",
        )
    split_dir = next(tmp_path.rglob("val_timestamps.parquet")).parent
    val_on_disk = pd.read_parquet(split_dir / "val_timestamps.parquet")
    assert len(val_on_disk) == 3
