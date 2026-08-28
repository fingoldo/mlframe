"""cProfile harness for :meth:`TrainingProgressWidget.update` -- the cost a live widget adds to a fit.

The widget repaints on a wall-clock throttle, so what matters is not the cost of ONE update but how that cost
scales as the history grows: a refresh that re-sends the entire history is O(n) per repaint and O(n^2) over a
run, which is exactly how a naive live plot ends up slower than the model it is watching.

Run::

    python -m mlframe.training.callbacks._benchmarks.bench_progress_widget_update
    python -m mlframe.training.callbacks._benchmarks.bench_progress_widget_update --save-stats out.prof

Reports per-refresh wall time at several history lengths (so the scaling is visible, not just one number)
plus a cumtime profile of the hot path.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
from timeit import default_timer as timer
from typing import Dict, List, Tuple
from unittest.mock import patch

import numpy as np

# Metrics / datasets matching a realistic multi-metric fit: two metrics of opposite direction over three splits.
N_METRICS = 2
DATASETS = ("train", "validation", "test")
HISTORY_LENGTHS = (100, 500, 2_000, 10_000)
REPEATS = 30


def _build_history(n: int) -> Tuple[List[int], Dict[str, Dict[str, List[float]]], List[Tuple[int, float]]]:
    """A history of ``n`` reported points over 3 splits x 2 metrics, plus a sparsely sampled RAM series."""
    rng = np.random.default_rng(0)
    hist: Dict[str, Dict[str, List[float]]] = {ds: {"ICE": [], "AUC": []} for ds in DATASETS}
    iters: List[int] = []
    ram: List[Tuple[int, float]] = []
    for i in range(n):
        iters.append(i * 9)
        for off, ds in enumerate(DATASETS):
            hist[ds]["ICE"].append(-0.2 - 0.0001 * i + 0.01 * off + float(rng.normal(0, 0.001)))
            hist[ds]["AUC"].append(0.7 + 0.00002 * i - 0.01 * off + float(rng.normal(0, 0.001)))
        if i % 20 == 0:
            ram.append((i * 9, 50.0 + i * 0.001))
    return iters, hist, ram


def _make_widget():
    """A widget forced into its enabled path without a real notebook frontend."""
    from mlframe.training.callbacks.progress_widget import TrainingProgressWidget

    with patch("mlframe.training.callbacks.progress_widget.in_notebook", return_value=True):
        return TrainingProgressWidget(refresh_secs=0.0)


def time_refresh_scaling() -> None:
    """Per-refresh wall time as the history grows -- the number that decides if this is usable at 10k iterations."""
    print(f"{'history':>9} {'per refresh':>13} {'per point':>12}")
    for n in HISTORY_LENGTHS:
        iters, hist, ram = _build_history(n)
        widget = _make_widget()
        with patch("IPython.display.display", lambda *a, **k: None):
            widget.monitor_dataset = "validation"
            widget.update(iters, hist, ram, {"ICE": "min", "AUC": "max"}, force=True)  # warm: build tabs/traces
            start = timer()
            for _ in range(REPEATS):
                widget.update(iters, hist, ram, {"ICE": "min", "AUC": "max"}, force=True)
            elapsed = (timer() - start) / REPEATS
        print(f"{n:>9,} {elapsed * 1e3:>10.2f} ms {elapsed / n * 1e6:>9.3f} us")


def profile_refresh(n: int = 10_000, save_stats: str | None = None) -> None:
    """cProfile one refresh at a large history, sorted by cumulative time."""
    iters, hist, ram = _build_history(n)
    widget = _make_widget()
    with patch("IPython.display.display", lambda *a, **k: None):
        widget.monitor_dataset = "validation"
        widget.update(iters, hist, ram, {"ICE": "min", "AUC": "max"}, force=True)

        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(REPEATS):
            widget.update(iters, hist, ram, {"ICE": "min", "AUC": "max"}, force=True)
        profiler.disable()

    stats = pstats.Stats(profiler).sort_stats("cumulative")
    print(f"\ncProfile: {REPEATS} refreshes at a {n:,}-point history, by cumtime")
    stats.print_stats(25)
    if save_stats:
        stats.dump_stats(save_stats)
        print(f"stats written to {save_stats}")


def main() -> None:
    """Run the scaling table and the profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-stats", default=None, help="write the raw pstats file here")
    parser.add_argument("--n", type=int, default=10_000, help="history length to profile")
    args = parser.parse_args()
    time_refresh_scaling()
    profile_refresh(args.n, args.save_stats)


if __name__ == "__main__":
    main()
