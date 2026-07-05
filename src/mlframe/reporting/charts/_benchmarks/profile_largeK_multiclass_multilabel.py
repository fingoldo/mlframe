"""cProfile the large-K multiclass / multilabel composers at K=10/50/100.

The per-class one-vs-rest overlay panels (ROC, PR, reliability) draw K curves in one panel; at K=50/100 that is both slow
(K sklearn curve fits + K artists) and unreadable. This bench measures compose time per K so the top-N-worst + macro-avg
auto-switch can be validated as a real win. Run: python -m mlframe.reporting.charts._benchmarks.profile_largeK_multiclass_multilabel
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np


def make_multiclass(n: int, K: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, size=n)
    logits = rng.normal(0.0, 1.0, size=(n, K))
    logits[np.arange(n), y] += 2.0
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba = e / e.sum(axis=1, keepdims=True)
    return y, proba, list(range(K))


def make_multilabel(n: int, K: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    yt = (rng.random((n, K)) < 0.3).astype(np.int8)
    proba = np.clip(0.15 + 0.7 * yt + rng.normal(0.0, 0.2, size=(n, K)), 0.0, 1.0)
    return yt, proba, [f"lbl{k}" for k in range(K)]


def _time(fn, repeat: int = 3) -> float:
    fn()  # warm
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench(n: int = 200_000, ks=(10, 50, 100), **compose_kwargs) -> None:
    from mlframe.reporting.charts.multiclass import compose_multiclass_figure
    from mlframe.reporting.charts.multilabel import compose_multilabel_figure

    print(f"=== compose times (n={n:_}), kwargs={compose_kwargs} ===")
    for K in ks:
        y, proba, classes = make_multiclass(n, K)
        mc = _time(lambda: compose_multiclass_figure(y, proba, classes, panels_template="CONFUSION ROC PR_CURVES CALIB_GRID", **compose_kwargs))
        yt, mlp, labels = make_multilabel(n, K)
        ml = _time(lambda: compose_multilabel_figure(yt, mlp, labels, panels_template="ROC CALIB_GRID PR_F1", **compose_kwargs))
        print(f"  K={K:3d}  multiclass={mc * 1e3:8.1f} ms   multilabel={ml * 1e3:8.1f} ms")


def profile_one(n: int = 200_000, K: int = 100) -> None:
    from mlframe.reporting.charts.multiclass import compose_multiclass_figure

    y, proba, classes = make_multiclass(n, K)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(3):
        compose_multiclass_figure(y, proba, classes, panels_template="CONFUSION ROC PR_CURVES CALIB_GRID")
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
    print(f"=== cProfile multiclass K={K} n={n:_} (3 composes) ===")
    print(s.getvalue())


if __name__ == "__main__":
    bench()
    profile_one()
