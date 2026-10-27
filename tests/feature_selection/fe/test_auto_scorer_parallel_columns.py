"""The auto-scorer's per-column bootstrap tables are the same with one thread and with several."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orth_auto_scorer_fe import select_best_scorer_per_column


def _frames(n=1500, p=6, seed=0):
    """Raw columns, engineered transforms of them, and a binary target tied to one raw column."""
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    eng = pd.DataFrame({f"x{i}__sq": raw[f"x{i}"] ** 2 for i in range(p)} | {f"x{i}__sin": np.sin(2 * raw[f"x{i}"]) for i in range(p)})
    y = (np.sin(2 * raw["x1"]) + 0.3 * rng.normal(size=n) > 0).astype(np.int64).to_numpy()
    return raw, eng, y


def test_threaded_tables_equal_serial_tables():
    """Same winners, LCBs and per-scorer dicts for n_jobs=1 and n_jobs=4: the seeds depend on the bootstrap index only, never on the thread."""
    raw, eng, y = _frames()
    serial = select_best_scorer_per_column(raw, eng, y, n_boot=3, n_jobs=1)
    threaded = select_best_scorer_per_column(raw, eng, y, n_boot=3, n_jobs=4)
    pd.testing.assert_frame_equal(serial, threaded)
    assert serial.iloc[0]["source_col"] == "x1"
