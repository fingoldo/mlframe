"""cProfile harness for ``apply_event_proximity_decay_composite_fe``.

Run: ``python -m mlframe.training.pipeline._benchmarks.bench_event_proximity_decay_composite_fe``
"""
from __future__ import annotations

import cProfile
import pstats
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline._event_proximity_decay_composite_fe import apply_event_proximity_decay_composite_fe


def _run(n_rows: int, n_events: int):
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="h")
    df = pd.DataFrame({"dummy": np.arange(n_rows)})
    event_dates = pd.date_range("2020-01-01", periods=n_events, freq="17D").strftime("%Y-%m-%d").tolist()
    cfg = PreprocessingExtensionsConfig(event_proximity_decay_event_dates=event_dates, event_proximity_decay_cap=10)
    apply_event_proximity_decay_composite_fe(df, None, None, cfg, np.asarray(dates), np.arange(n_rows), None, None, metadata={}, verbose=0)


if __name__ == "__main__":
    for n_rows, n_events in [(50_000, 20), (300_000, 50)]:
        pr = cProfile.Profile()
        pr.enable()
        _run(n_rows, n_events)
        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        print(f"=== n_rows={n_rows} n_events={n_events} ===")
        print(s.getvalue())
