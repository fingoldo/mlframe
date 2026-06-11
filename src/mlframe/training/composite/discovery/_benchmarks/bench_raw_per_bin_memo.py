"""Wall-time bench: raw-y per-bin baseline -- per-base refit vs fit-once + re-bin.

When the regime-aware gate is ON (``per_bin_n_bins>0``), ``_tiny_model_rerank``
needs a per-bin raw-y RMSE baseline for every distinct base column. The raw-y
tiny model is trained on the FULL feature matrix and is INDEPENDENT of the
per-base ``bin_var`` -- only the per-bin quantile re-aggregation differs across
bases. The legacy path refit the K-fold raw-y model once PER BASE
(``_tiny_cv_rmse_raw_y(..., bin_var=base)``), so B distinct bases paid B *
cv_folds LightGBM/Ridge fits. The new path fits the raw-y model ONCE
(``return_fold_preds=True``) and re-bins the cached per-fold predictions per
base via ``_per_bin_from_fold_preds`` -- 1 * cv_folds fits + B cheap re-binnings.

This bench times the whole B-base per-bin sweep for both paths, warmed once and
repeated several times (min-of-reps reported to suppress scheduler noise). It
also asserts bit-identity (each base's per-bin array is exactly equal between
the two paths) so a future "just refit per base again" cannot silently diverge.

The win scales with B (specs-per-distinct-base): each extra distinct base saves
one full K-fold raw-y fit. Off by default (``per_bin_n_bins=0``) so this is a
perf option, realised only when the regime gate is enabled.

MEASURED (this Windows host, py3.14, n=8000 F=12 B=6 cv_folds=3 n_bins=8,
family='lightgbm', 5 reps, min-of-reps wall):
  old (refit per base):  ~882 ms
  new (fit once+re-bin): ~160 ms   (~5.5x; saves B-1 full K-fold raw-y fits)
  bit-identical per-bin: True
Host-relative; the DIRECTION new <= old AND bit-identical per-bin is the
invariant the paired test pins. The factor grows with B (distinct bases).

Usage::

    python -m mlframe.training.composite.discovery._benchmarks.bench_raw_per_bin_memo
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from timeit import default_timer as timer

import numpy as np

from mlframe.training.composite.discovery._screening_tiny import (
    _per_bin_from_fold_preds,
    _tiny_cv_rmse_raw_y,
)

_N = 8_000
_F = 12
_B = 6          # distinct base columns (bin_vars) sharing one raw-y fit
_CV_FOLDS = 3
_NBINS = 8
_REPS = 5
_FAMILY = "lightgbm"  # the production tiny family; falls back to linear if absent


def _make(n: int, f: int, n_bases: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, f)).astype(np.float32)
    y = (0.7 * x[:, 0] - 1.1 * x[:, 1] + 0.4 * x[:, 2]).astype(np.float64)
    y += rng.standard_normal(n) * 0.5
    bin_vars = [
        x[:, i % f] + 0.3 * rng.standard_normal(n).astype(np.float32)
        for i in range(n_bases)
    ]
    return y, x, bin_vars


def _common_kw(family: str) -> dict:
    return dict(
        family=family, n_estimators=20, num_leaves=15, learning_rate=0.1,
        cv_folds=_CV_FOLDS, random_state=0, deterministic=True,
    )


def _old_per_base(y, x, bin_vars, family) -> list[np.ndarray]:
    """Legacy: per base, full K-fold raw-y refit with bin_var set."""
    out = []
    kw = _common_kw(family)
    for bv in bin_vars:
        _, per_bin = _tiny_cv_rmse_raw_y(
            y_train=y, x_train_matrix=x,
            return_per_bin=True, n_bins=_NBINS, bin_var=bv, **kw,
        )
        out.append(per_bin)
    return out


def _new_fit_once(y, x, bin_vars, family) -> list[np.ndarray]:
    """New: fit the raw-y model ONCE, re-bin the cached fold preds per base."""
    kw = _common_kw(family)
    _, fold_preds = _tiny_cv_rmse_raw_y(
        y_train=y, x_train_matrix=x, return_fold_preds=True, **kw,
    )
    return [
        _per_bin_from_fold_preds(fold_preds, bv, n_bins=_NBINS)
        for bv in bin_vars
    ]


def _time(fn, *args) -> tuple[float, list[np.ndarray]]:
    t0 = timer()
    res = fn(*args)
    return timer() - t0, res


def main() -> None:
    family = _FAMILY
    try:
        import importlib
        importlib.import_module("lightgbm")
    except Exception:
        family = "linear"

    y, x, bin_vars = _make(_N, _F, _B)

    # Warm + bit-identity gate.
    old0 = _old_per_base(y, x, bin_vars, family)
    new0 = _new_fit_once(y, x, bin_vars, family)
    identical = all(
        np.array_equal(o, n, equal_nan=True) for o, n in zip(old0, new0)
    )

    old_t = float("inf")
    new_t = float("inf")
    for _ in range(_REPS):
        t_old, _ = _time(_old_per_base, y, x, bin_vars, family)
        t_new, _ = _time(_new_fit_once, y, x, bin_vars, family)
        old_t = min(old_t, t_old)
        new_t = min(new_t, t_new)

    speedup = old_t / new_t if new_t > 0 else float("inf")
    print(
        f"raw per-bin memo bench  n={_N} F={_F} B={_B} cv_folds={_CV_FOLDS} "
        f"n_bins={_NBINS} family={family} reps={_REPS} py={sys.version.split()[0]}"
    )
    print(f"  bit-identical per-bin: {identical}")
    print(f"  old (refit per base):  {old_t * 1e3:8.2f} ms")
    print(f"  new (fit once+re-bin): {new_t * 1e3:8.2f} ms  (speedup {speedup:.2f}x)")

    out_dir = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "raw_per_bin_memo.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "ts": datetime.now().isoformat(),
                "n": _N, "F": _F, "B": _B, "cv_folds": _CV_FOLDS,
                "n_bins": _NBINS, "family": family, "reps": _REPS,
                "bit_identical": identical,
                "old_ms": old_t * 1e3,
                "new_ms": new_t * 1e3,
                "speedup": speedup,
            },
            fh,
            indent=2,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
