"""Bench the cached fit-signature lookup vs per-call inspect.signature in the RFECV fold loop.

Run: python -m mlframe.feature_selection.wrappers.rfecv._benchmarks.bench_fit_sig_cache
"""
from __future__ import annotations

import inspect
import time

from sklearn.ensemble import RandomForestClassifier

from mlframe.feature_selection.wrappers.rfecv._fit_fold import _fit_accepts_sample_weight


def old_check(est):
    try:
        sig = inspect.signature(est.fit)
    except (TypeError, ValueError):
        return False
    p = sig.parameters
    return "sample_weight" in p or any(x.kind == inspect.Parameter.VAR_KEYWORD for x in p.values())


def new_check(est):
    key = getattr(est.fit, "__func__", est.fit)
    return _fit_accepts_sample_weight(key)


def bestof(fn, est, n=20000):
    fn(est)
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter()
        fn(est)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    est = RandomForestClassifier()
    assert old_check(est) == new_check(est)  # nosec B101 - internal invariant check in src/mlframe/feature_selection/wrappers/rfecv/_benchmarks, not reachable with untrusted input
    o = bestof(old_check, est)
    nv = bestof(new_check, est)
    print(f"old (inspect.signature/call) {o*1e6:7.3f}us  new (lru_cache) {nv*1e6:7.3f}us  speedup {o/nv:.1f}x")
    # total saving estimate: calls = n_estimators * n_folds * n_elimination_steps
    per_call_saved = o - nv
    print(f"per-call saving ~{per_call_saved*1e6:.2f}us; at 1 est x 5 folds x 30 steps = 150 calls -> ~{per_call_saved*150*1e3:.3f}ms")


if __name__ == "__main__":
    main()
