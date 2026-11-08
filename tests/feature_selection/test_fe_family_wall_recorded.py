"""The wall of every FE family of one fit is stored on the estimator, in the provenance frame and in the report; gate laps are kept per thread."""

import threading
import time

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _fe_family_timing as timing
from mlframe.feature_selection.filters.mrmr import MRMR


def test_gate_laps_of_two_threads_do_not_close_each_other():
    """Two threads walking their own gates in turn each get their own lap, and ``close_thread_laps`` ends only the calling thread's lap."""
    timer = timing.FamilyLapTimer()
    before = timing.get_fe_family_wall()
    started, release = threading.Event(), threading.Event()

    def worker() -> None:
        """Open a lap on 'worker_family', wait while the main thread works, then end the thread's laps."""
        timer.gate("worker_family", True)
        started.set()
        release.wait(5)
        time.sleep(0.05)
        timing.close_thread_laps()

    thread = threading.Thread(target=worker)
    thread.start()
    assert started.wait(5)
    timer.gate("main_family", True)  # does not end the worker's lap
    time.sleep(0.02)
    timer.close()
    release.set()
    thread.join(5)
    rows = timing.fe_family_wall_since(before)
    assert rows["gate:main_family"]["calls"] == 1 and rows["gate:main_family"]["wall_s"] >= 0.02 - 1e-3
    assert rows["gate:worker_family"]["calls"] == 1 and rows["gate:worker_family"]["wall_s"] >= 0.05 - 1e-3


def test_fit_records_the_wall_of_each_family(monkeypatch):
    """After a fit with FE on, ``fe_family_wall_`` holds timed families (longest first), mirrors ``fe_provenance_.attrs`` and is printed by ``get_fe_report``."""
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({f"x{i}": rng.standard_normal(n) for i in range(6)})
    y = (X["x0"] * X["x1"] + 0.3 * rng.standard_normal(n) > 0).astype(int)
    sel = MRMR(verbose=0, random_seed=0, n_jobs=1, fe_max_steps=1).fit(X, y)
    wall = sel.fe_family_wall_
    assert isinstance(wall, dict) and wall, "no family was timed"
    walls = [row["wall_s"] for row in wall.values()]
    assert walls == sorted(walls, reverse=True)
    assert all(row["calls"] >= 1 and row["wall_s"] >= 0.0 for row in wall.values())
    assert sel.fe_provenance_.attrs["family_wall"] == wall
    first = next(iter(wall))
    assert first in sel.get_fe_report() or sel.fe_provenance_.empty
