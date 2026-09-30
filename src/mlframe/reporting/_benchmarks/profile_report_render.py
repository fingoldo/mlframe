"""cProfile + per-figure timers for ONE model's post-fit report on synthetic predictions (no training involved).

Drives the repo's own reporting entry point (``report_model_perf`` -> calibration / panel / diagnostic charts, exactly what the suite
calls after ``predict``) on a synthetic val (60k rows) and test (68k rows) prediction set for binary, multiclass and regression,
with ``model=None`` and precomputed preds/probs (the ``just_evaluate`` shape). Prints per split: wall time, render seconds by backend,
the top charts, and (``--profile``) the cumulative-time top functions split into matplotlib draw/save, plotly serialisation and
everything else. ``--queue thread|process`` runs the same report through a ``ReportRenderQueue`` and prints how long the submitting
thread was blocked vs how long the queue needed.

Run:  ``python -m mlframe.reporting._benchmarks.profile_report_render --task binary --profile``
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import shutil
import tempfile
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def make_predictions(task: str, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, str]:
    """``(y, probs_or_preds, feature_frame, target_type)`` for one split."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.standard_normal(n).astype("float32") for i in range(10)})
    X["cat"] = pd.Categorical(rng.choice(["a", "b", "c", "d"], n))
    signal = X["f0"].to_numpy() - 0.5 * X["f1"].to_numpy()
    if task == "regression":
        y = signal + rng.standard_normal(n) * 0.5
        return y, y + rng.standard_normal(n) * 0.4, X, "regression"
    if task == "multiclass":
        s = signal + rng.standard_normal(n) * 0.7
        y = np.digitize(s, np.quantile(s, [0.33, 0.66])).astype(np.int64)
        logits = np.stack([-(s + 1) ** 2, -(s) ** 2, -(s - 1) ** 2], axis=1) + rng.standard_normal((n, 3)) * 0.3
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return y, e / e.sum(axis=1, keepdims=True), X, "multiclass_classification"
    y = (signal + rng.standard_normal(n) * 0.8 > 0).astype(np.int64)
    p1 = 1.0 / (1.0 + np.exp(-(1.4 * signal + rng.standard_normal(n) * 0.3)))
    return y, np.column_stack([1 - p1, p1]), X, "binary_classification"


def report_one_split(task: str, n: int, seed: int, out: str, split: str) -> Dict[str, Any]:
    """Run ``report_model_perf`` for one synthetic split and return its metrics dict."""
    from mlframe.training.configs import ReportingConfig
    from mlframe.training.reporting._reporting import report_model_perf

    y, pred, X, target_type = make_predictions(task, n, seed)
    cfg = ReportingConfig(show_perf_chart=False, async_render=False)
    metrics: Dict[str, Any] = {}
    kwargs: Dict[str, Any] = dict(
        targets=y, columns=list(X.columns), model_name=f"synthetic-{task}", model=None, df=X, print_report=False, show_perf_chart=False,
        show_fi=False, plot_file=os.path.join(out, f"m_{split}"), plot_outputs=cfg.plot_outputs, metrics=metrics, target_type=target_type,
        binary_panels=cfg.binary_panels, multiclass_panels=cfg.multiclass_panels, reporting_config=cfg,
    )
    if task == "regression":
        kwargs["preds"] = pred
    else:
        kwargs["probs"] = pred
        if task == "binary":
            kwargs["custom_ice_metric"] = None
    report_model_perf(**kwargs)
    return metrics


def main(argv: List[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="binary", choices=["binary", "multiclass", "regression"])
    ap.add_argument("--val", type=int, default=60_000)
    ap.add_argument("--test", type=int, default=68_000)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--queue", default="", choices=["", "thread", "process"])
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args(argv)
    from mlframe.reporting._async_render_hooks import render_queue_scope
    from mlframe.reporting.renderers import chart_timings_snapshot, reset_chart_timings
    from mlframe.reporting.renderers.save import set_format_subfolders

    out = tempfile.mkdtemp(prefix="profile_report_")
    set_format_subfolders(True)
    queue = None
    if a.queue:
        from mlframe.reporting._async_render import ReportRenderQueue

        queue = ReportRenderQueue(backend=a.queue, workers=1)
        if a.queue == "process":
            queue.warm()
    prof = cProfile.Profile() if a.profile else None
    # one untimed pass so imports / numba / font caches do not land in the numbers
    report_one_split(a.task, 4000, 1, os.path.join(out, "warm"), "warm")
    reset_chart_timings()
    if prof:
        prof.enable()
    t_total = time.perf_counter()
    for split, n, seed in (("val", a.val, 11), ("test", a.test, 12)):
        t0 = time.perf_counter()
        if queue is not None:
            with render_queue_scope(queue):
                report_one_split(a.task, n, seed, out, split)
            blocked = time.perf_counter() - t0
            print(f"[{split}] submit-side wall (caller blocked): {blocked:.2f}s, pending {queue.pending()}")
        else:
            report_one_split(a.task, n, seed, out, split)
            print(f"[{split}] wall: {time.perf_counter() - t0:.2f}s")
    if queue is not None:
        t_join = time.perf_counter()
        summary = queue.close()
        print(f"queue drain after the last submit: {time.perf_counter() - t_join:.2f}s | {summary.line()}")
    print(f"TOTAL wall: {time.perf_counter() - t_total:.2f}s")
    if prof:
        prof.disable()
    rows = chart_timings_snapshot()
    by: Dict[str, float] = {}
    for r in rows:
        tag = "plotly" if str(r["chart"]).endswith("[plotly]") else ("matplotlib" if str(r["chart"]).endswith("[matplotlib]") else "other")
        by[tag] = by.get(tag, 0.0) + float(r["seconds"])
    print(f"figures: {sum(int(r['count']) for r in rows)}, render seconds by backend (summed over threads): " + ", ".join(f"{k}={v:.2f}s" for k, v in sorted(by.items())))
    for r in rows[: a.top // 2]:
        print(f"  {r['seconds']:7.3f}s x{r['count']:<2d} {r['chart']}")
    if prof:
        st = pstats.Stats(prof)
        st.sort_stats("cumulative").print_stats(a.top)
        print("note: cProfile sees only the calling thread; matplotlib/plotly drawing runs on the per-backend render threads, which shows up above as time blocked in lock.acquire / Future.result")
    shutil.rmtree(out, ignore_errors=True)
    set_format_subfolders(None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
