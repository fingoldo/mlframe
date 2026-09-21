"""cProfile harness for ``training.composite.hurdle.HurdleRegressor``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_hurdle_regressor``

Measures fit + predict wall time across sizes for each smearing mode, then profiles the default at the largest size.
The question it answers is whether the hurdle's own orchestration (row subsetting, the smearing pass) costs anything
next to the two estimator fits it wraps, and how much ``smearing="oof"``'s K extra magnitude fits add.

Verdict (n=200k, 8 features, HGB x100 iterations; cProfile, cumulative): no actionable speedup in the hurdle itself.
``fit`` took 2.98 s of which the two wrapped HGB fits were 2.82 s, so the orchestration is ~0.16 s (~5%); ``predict``
took 0.94 s, all of it the two HGB ``predict`` calls. The one cost the hurdle adds is a third HGB ``predict`` over the
event rows during ``fit``, which in-sample Duan smearing needs for its residuals (~0.35 s here) -- inseparable from the
estimator, since HGB exposes no training-time predictions. ``smearing="oof"`` adds ``smearing_cv`` (default 5) extra
magnitude fits, as designed. The single-shot wall timings printed above are too noisy to rank the modes against each
other on a contended host (one run had ``"none"`` slower than ``"insample"`` at 50k); read the cProfile attribution,
not them.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from mlframe.training.composite.hurdle import HurdleRegressor


def _make_data(n: int, seed: int = 0, zero_frac: float = 0.71):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8)).astype(np.float32)
    happens = rng.random(n) < 1.0 / (1.0 + np.exp(-(1.6 * X[:, 0] - 1.2 * X[:, 1] + np.log((1 - zero_frac) / zero_frac))))
    y = np.where(happens, np.exp(3.5 + 0.9 * X[:, 2] - 0.6 * X[:, 3] + rng.normal(0, 1.2, n)), 0.0)
    return X, y


def _model(smearing: str) -> HurdleRegressor:
    return HurdleRegressor(
        classifier=HistGradientBoostingClassifier(max_iter=100, random_state=0),
        regressor=HistGradientBoostingRegressor(max_iter=100, random_state=0),
        smearing=smearing,
        random_state=0,
    )


def _run(n: int, smearing: str) -> None:
    X, y = _make_data(n)
    _model(smearing).fit(X, y).predict(X)


if __name__ == "__main__":
    _run(2000, "insample")  # warm imports and HGB's first-call setup so the timings below compare like with like
    for n in (5_000, 50_000, 200_000):
        for smearing in ("insample", "oof", "none"):
            t0 = time.perf_counter()
            _run(n, smearing)
            print(f"n={n:>7} smearing={smearing:<8} -> {(time.perf_counter() - t0) * 1000:10.1f} ms")

    pr = cProfile.Profile()
    pr.enable()
    _run(200_000, "insample")
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
    print(s.getvalue())
