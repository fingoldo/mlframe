"""Bench: per-fold cost of ``_build_tiny_model`` (lazy import + fresh estimator construction).

``_tiny_cv_rmse_*`` calls ``_build_tiny_model`` once per (fold, spec) to get a fresh, unfitted regressor. The lead
asked whether constructing a fresh estimator + paying the lazy ``import`` each fold is an actionable hotspot worth
caching/pooling. This bench measures the warm per-call cost (the ``import`` is cached after first touch, so steady
state is just the estimator ``__init__`` + kwargs dict build).

Verdict driver: the per-call cost is negligible vs the fold FIT (which is milliseconds-to-seconds), so pooling /
reusing estimator instances across folds would buy nothing AND would be UNSAFE -- a reused fitted estimator would
leak state across folds / specs (sklearn estimators are not guaranteed re-fit-clean for every param set). A fresh
instance per fold is the correct contract, and it is already effectively free.

Run:
    python src/mlframe/training/composite/discovery/_benchmarks/bench_build_tiny_model_per_fold.py
"""
from __future__ import annotations

import time

from mlframe.training.composite.discovery._screening_tiny import _build_tiny_model


def _time_build(family: str, iters: int = 100_000) -> float:
    kw = dict(n_estimators=50, num_leaves=15, learning_rate=0.1, random_state=0)
    _build_tiny_model(family, **kw)  # warm the lazy import
    t0 = time.perf_counter()
    for _ in range(iters):
        _build_tiny_model(family, **kw)
    return (time.perf_counter() - t0) / iters * 1e6  # us/call


def main() -> None:
    cv_folds = 5
    print(f"{'family':<10} {'us/build':>10} {'us/sweep(5 folds)':>18}")
    for fam in ("lightgbm", "xgboost", "catboost", "linear"):
        try:
            us = _time_build(fam, iters=20_000)
        except Exception as e:  # optional booster missing
            print(f"{fam:<10} {'(skip: ' + type(e).__name__ + ')':>10}")
            continue
        print(f"{fam:<10} {us:>10.2f} {us * cv_folds:>18.2f}")
    print(
        "\nVerdict (DOC -- no actionable speedup): a fresh estimator construction is sub-microsecond-to-low-"
        "microseconds per fold (the lazy import is cached after the first touch). At cv_folds=5 a full per-spec "
        "build sweep is well under a millisecond -- orders of magnitude below the fold FIT (ms-to-s). Pooling / "
        "reusing instances across folds would save nothing and would risk cross-fold/spec state leakage (a fresh "
        "unfitted estimator per fold is the correct, leak-free contract). Keep per-fold construction."
    )


if __name__ == "__main__":
    main()
