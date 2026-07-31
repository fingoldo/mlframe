"""A/B correctness + speed check for the fused auc/brier/log_loss/ece batch bootstrap (2026-07-31).

Compares ``bootstrap_auc_brier_ll_ece_batch`` (one prange-parallel njit pass over the whole
bootstrap distribution) against ``bootstrap_metrics`` (the generic per-resample Python-dispatch
loop honest_diagnostics.py currently calls) on the EXACT metric bundle and calling convention
``honest_diagnostics._bootstrap_block`` uses (stratify=y_true, method="bca").

Confirms point/lo/hi/samples parity for all four metrics and measures wall-time at the
honest_diagnostics shape (n=300k, R=1000, stratified).

Usage:
    python profiling/bench_bootstrap_fused_binary_bundle.py
"""

from __future__ import annotations

import time

import numpy as np

from mlframe.evaluation._bootstrap_fused_binary_bundle import bootstrap_auc_brier_ll_ece_batch
from mlframe.evaluation.bootstrap import bootstrap_metrics
from mlframe.evaluation._bootstrap_jackknife import _jackknife_auc
from mlframe.metrics.core import (
    fast_brier_score_loss as _fast_brier,
    fast_log_loss as _fast_ll,
    make_bootstrap_auc_resampler,
)
from mlframe.calibration.policy import _ece_score


def _run(n: int, n_bootstrap: int, seed: int = 0):
    rng = np.random.default_rng(1)
    y_true = (rng.random(n) < 0.3).astype(np.float64)
    # continuous, tie-free scores (no clip -- clipping to a shared boundary creates ties and trips
    # the tie-free gate); a logistic squash keeps values in (0, 1) without collapsing distinct inputs.
    raw = y_true * 1.5 + rng.normal(size=n) * 0.5
    p_pos = 1.0 / (1.0 + np.exp(-raw))

    def _brier(yy, pp):
        return float(_fast_brier(yy, pp))

    def _ll(yy, pp):
        return float(_fast_ll(yy, pp))

    def _ll_per_row(yy, pp):
        _eps = np.finfo(np.asarray(pp).dtype).eps
        _pc = np.clip(pp, _eps, 1.0 - _eps)
        return np.where(np.asarray(yy) == 1, -np.log(_pc), -np.log(1.0 - _pc))

    def _brier_per_row(yy, pp):
        _d = np.asarray(pp, dtype=np.float64) - np.asarray(yy, dtype=np.float64)
        return _d * _d

    metric_fns = {"brier": _brier, "log_loss": _ll, "ece": lambda yy, pp: _ece_score(yy, pp)}
    per_row_fns = {"log_loss": (_ll_per_row, True, None), "brier": (_brier_per_row, False, None)}
    jackknife_fns = {"roc_auc": lambda yy, ss: _jackknife_auc(yy, ss)}
    metric_fns_idx = {"roc_auc": make_bootstrap_auc_resampler(y_true, p_pos)}

    t0 = time.perf_counter()
    ref = bootstrap_metrics(
        y_true, p_pos, metric_fns, n_bootstrap=n_bootstrap, alpha=0.05, stratify=y_true,
        random_state=seed, metric_fns_idx=metric_fns_idx, per_row_fns=per_row_fns, jackknife_fns=jackknife_fns,
    )
    t_ref = time.perf_counter() - t0

    # warm the fused kernel's JIT compilation before timing
    bootstrap_auc_brier_ll_ece_batch(y_true[:1000], p_pos[:1000], n_bootstrap=4, stratify=y_true[:1000], random_state=999, chunk_size=4)

    t0 = time.perf_counter()
    fused = bootstrap_auc_brier_ll_ece_batch(y_true, p_pos, n_bootstrap=n_bootstrap, alpha=0.05, stratify=y_true, random_state=seed)
    t_fused = time.perf_counter() - t0

    print(f"n={n:,} n_bootstrap={n_bootstrap}")
    print(f"bootstrap_metrics (generic):        {t_ref:.4f}s")
    print(f"bootstrap_auc_brier_ll_ece_batch:    {t_fused:.4f}s")
    print(f"speedup: {t_ref / t_fused:.2f}x")
    assert fused is not None, "fused path unexpectedly returned None (tie gate)"
    # roc_auc/ece use the SAME reduction order in both paths (bit-identical); brier/log_loss's
    # PUBLIC dispatchers (fast_brier_score_loss/fast_log_loss, used by bootstrap_metrics's metric_fns)
    # auto-pick the parallel-reduction kernel at n>=100k, while this fused kernel always calls the
    # sequential variant (nested parallelism inside its own prange loop is unsupported) -- a
    # ~1e-14 FP-reorder difference at that size, not a correctness bug (tolerance matches the
    # project's own ~1e-9 FP-reorder bar).
    all_ok = True
    for name in ("roc_auc", "brier", "log_loss", "ece"):
        r, f = ref[name], fused[name]
        tol = 1e-9
        point_ok = abs(r["point"] - f["point"]) < tol
        lo_ok = abs(r["lo"] - f["lo"]) < tol
        hi_ok = abs(r["hi"] - f["hi"]) < tol
        samples_ok = np.allclose(np.sort(r["samples"]), np.sort(f["samples"]), atol=tol)
        ok = point_ok and lo_ok and hi_ok and samples_ok
        all_ok &= ok
        print(f"  {name}: point_ok={point_ok} lo_ok={lo_ok} hi_ok={hi_ok} samples_ok={samples_ok} " f"(ref point={r['point']:.6f} fused point={f['point']:.6f})")
    print()
    return all_ok


def main():
    ok1 = _run(n=50_000, n_bootstrap=500)
    ok2 = _run(n=300_000, n_bootstrap=1000)
    assert ok1 and ok2, "fused bundle diverged from bootstrap_metrics"


if __name__ == "__main__":
    main()
