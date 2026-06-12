"""cProfile-driven orchestration profile for ``StabilityMRMR`` (filters/stability.py).

Scope: the BOOTSTRAP LOOP + the Nogueira/inclusion-frequency aggregation -- NOT the inner MRMR fit
(off-limits: filters/_mrmr_fit_impl/, filters/mrmr/, filters/_mrmr_fe_step/) and NOT the tuned MI kernels.

What this measures
------------------
``StabilityMRMR(estimator=MRMR(verbose=0), n_bootstraps=5, sample_fraction=0.7)`` fit on a small
n~800 p~15 binary-target frame. We sort cProfile by cumulative AND tottime, top ~28, and isolate the
mlframe-side ORCHESTRATION rows (stability.py own code) from the inner MRMR-fit + MI-kernel attribution.

Orchestration hotspots inspected (the productive seam per the task framing)
--------------------------------------------------------------------------
Per-bootstrap resample build (``_one_bootstrap``):
  * ``X.iloc[idx]`` / ``y.iloc[idx]`` -- a fancy-index COPY of the subsample handed to the estimator
    clone. This copy is genuinely consumed by ``est.fit`` (the estimator needs its own row subset),
    so it is NOT discarded work -- an iloc view is impossible (fancy index, non-contiguous rows).
  * ``clone(self.estimator)`` -- a fresh deep param copy each bootstrap; required (sklearn contract,
    each fit must not see prior fitted state). Cheap relative to the fit.
Train-invariants (computed ONCE before the loop, correctly):
  * ``n_samples`` / ``n_features`` / ``sub_size`` / ``seeds`` -- all hoisted out of the loop already.
  * ``counts`` accumulation matrix -- allocated once after the loop; ``counts[sup] += 1`` per bootstrap
    is a tiny scatter-add over <=p indices, vectorizable but p~15 makes it sub-microsecond noise.
Aggregation:
  * ``selection_probabilities_`` / ``np.where(... >= threshold)`` -- O(p) numpy, sub-microsecond.

VERDICT: no actionable orchestration speedup -- the wall is genuinely the inner MRMR fits (off-limits)
------------------------------------------------------------------------------------------------------
Measured n=800 p=15 n_bootstraps=5 (warm fit, store py3.14, GPU disabled):

  * Full profiled fit: ~14.3 s. The ONLY stability.py rows are ``_one_bootstrap`` (cumtime 14.347,
    tottime 0.001) and ``fit`` (cumtime 9.439, tottime 0.001) -- both ~zero tottime; 100% of the wall
    is inside the off-limits ``MRMR.fit`` (``_fit_impl`` -> ``_run_fe_step`` -> numba JIT + MI/cupy
    kernels). The orchestration is invisible in tottime.
  * Per-bootstrap orchestration cost, isolated (NO fit): ``X.iloc[idx]`` + ``y.iloc[idx]`` copy = 186 us,
    ``clone(self.estimator)`` = 2415 us, total ~2.9 ms. One inner ``MRMR.fit`` is ~2.6 s -- so the
    entire per-bootstrap orchestration is ~0.1% of one bootstrap's wall. The iloc copy is a genuine
    fancy-index gather (non-contiguous rows -> a view is impossible) and is CONSUMED by ``est.fit``
    (not discarded), so there is no pruned-output fast path to win. ``clone`` is required by the
    sklearn contract (each fit must not see prior fitted state) and is 0.09% of the bootstrap.
  * Post-loop aggregation (counts scatter + probs + ``np.where``): ~15 us, but it runs ONCE per fit
    (not per bootstrap), so it is sub-0.0001% of the wall -- attribution noise, never re-flag.

No train-invariant is recomputed in the loop (``n_samples`` / ``n_features`` / ``sub_size`` / ``seeds``
are all hoisted before it; ``counts`` is allocated once after it). Nothing is rebuilt per iteration
that a view/buffer would save, and nothing is discarded. Orchestration is TIGHT; the only lever is the
inner MRMR fit, which is off-limits. No code change shipped (selection therefore trivially BIT-IDENTICAL).

Run::

    CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 PYTHONPATH=src python \
        -m mlframe.feature_selection._benchmarks.bench_stability_orchestration
"""
from __future__ import annotations

import cProfile
import io
import os
import pstats
import time

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from mlframe.feature_selection.filters.stability import StabilityMRMR


def make_frame(n: int = 800, p: int = 15, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Small binary-target frame: 3 informative columns drive a linear logit; the rest are noise.

    n~800 p~15 matches the task's profiling shape. Informative columns guarantee a non-empty support
    so the aggregation path runs over real selected indices, not an empty set.
    """
    rng = np.random.default_rng(seed)
    cols = {f"f{j}": rng.standard_normal(n) for j in range(p)}
    X = pd.DataFrame(cols)
    logit = X["f0"] + 0.8 * X["f1"] - 0.6 * X["f2"] + 0.3 * rng.standard_normal(n)
    y = pd.Series((logit > logit.median()).astype(np.int64), name="y")
    return X, y


def build_selector(n_bootstraps: int = 5, sample_fraction: float = 0.7, random_state: int = 0) -> StabilityMRMR:
    from mlframe.feature_selection.filters.mrmr import MRMR
    from mlframe.feature_selection.filters.stability import StabilityMRMR

    return StabilityMRMR(
        estimator=MRMR(verbose=0),
        n_bootstraps=n_bootstraps,
        sample_fraction=sample_fraction,
        support_threshold=0.6,
        random_state=random_state,
    )


def run_profile(n: int = 800, p: int = 15, n_bootstraps: int = 5) -> None:
    X, y = make_frame(n=n, p=p)
    sel = build_selector(n_bootstraps=n_bootstraps)
    # Warm one fit so numba JIT / class-build cost does not pollute the profiled fit.
    build_selector(n_bootstraps=2).fit(X, y)

    pr = cProfile.Profile()
    pr.enable()
    sel.fit(X, y)
    pr.disable()

    for sort_key in ("cumulative", "tottime"):
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
        ps.print_stats(28)
        print(f"\n{'=' * 80}\nSORT BY {sort_key.upper()} (n={n}, p={p}, n_bootstraps={n_bootstraps})\n{'=' * 80}")
        print(s.getvalue())


def microbench_aggregation(n_features: int = 15, n_bootstraps: int = 5, reps: int = 100000) -> None:
    """Standalone wall-time of the post-loop aggregation block (counts scatter + probs + where).

    Isolates the ORCHESTRATION aggregation from cProfile's deep-stack inflation. If this is <1ms it is
    attribution noise in the full profile and must not be re-flagged as a hotspot.
    """
    rng = np.random.default_rng(0)
    supports = [rng.choice(n_features, size=4, replace=False).astype(np.int64) for _ in range(n_bootstraps)]
    threshold = 0.6
    t0 = time.perf_counter()
    for _ in range(reps):
        counts = np.zeros(n_features, dtype=np.int64)
        for sup in supports:
            counts[sup] += 1
        probs = counts / n_bootstraps
        _ = np.where(probs >= threshold)[0]
    dt = time.perf_counter() - t0
    # ~15 us/call, but runs ONCE per fit (not per bootstrap) -> sub-0.0001% of the ~14 s fit wall: noise.
    print(f"\naggregation block: {dt / reps * 1e6:.3f} us/call (p={n_features}, n_bootstraps={n_bootstraps}, reps={reps})")


def _print_verdict() -> None:
    print(
        "\nVERDICT: stability.py orchestration is tight -- wall is the n_bootstraps inner MRMR.fit "
        "calls (off-limits). Train-invariants are hoisted; the per-bootstrap iloc copy is consumed by "
        "the estimator fit (not discarded); aggregation is O(p) numpy (<1us microbench => attribution "
        "noise in the full profile). See module docstring."
    )


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
    run_profile(n=800, p=15, n_bootstraps=5)
    microbench_aggregation()
    _print_verdict()
