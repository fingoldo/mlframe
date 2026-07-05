"""Validate + benchmark the n_jobs parallelization of bootstrap_metrics (threading backend over nogil njit kernels).

Checks, mirroring honest_diagnostics' real call (stratified binary, brier+log_loss slice metrics + AUC idx-aware):
1. n_jobs=1 is the unchanged serial path (reference CIs).
2. n_jobs>1 CIs are statistically equivalent to serial (|drift| within seed-to-seed MC variability of the serial path).
3. Speedup across core counts.
4. Reproducibility: same seed + n_jobs twice -> identical CIs, AND independent of worker count (per-resample seeding).
"""
import time
import numpy as np

from mlframe.evaluation.bootstrap import bootstrap_metrics
from mlframe.metrics.core import fast_brier_score_loss, fast_log_loss, make_bootstrap_auc_resampler

N = 200_000
R = 1000  # honest_diagnostics' actual n_bootstrap
rng0 = np.random.default_rng(0)
y = (rng0.random(N) < 0.3).astype(np.float64)
p = np.clip(0.15 + 0.5 * y + rng0.standard_normal(N) * 0.25, 1e-6, 1 - 1e-6)


def _build(seed):
    mf = {"brier": lambda yy, pp: float(fast_brier_score_loss(yy, pp)), "log_loss": lambda yy, pp: float(fast_log_loss(yy, pp))}
    mfi = {"roc_auc": make_bootstrap_auc_resampler(y, p)}
    return dict(y_true=y, y_pred=p, metric_fns=mf, metric_fns_idx=mfi, n_bootstrap=R, stratify=y, random_state=seed, method="bca")


def _ci(res):
    return {k: (v.get("lo"), v.get("hi")) for k, v in res.items()}


if __name__ == "__main__":
    # warm numba
    bootstrap_metrics(**_build(1))

    t = time.perf_counter(); r_ser = bootstrap_metrics(**_build(42), n_jobs=1); t_ser = time.perf_counter() - t
    ci_ser = _ci(r_ser)
    print(f"serial (n_jobs=1): {t_ser:.1f}s  " + "  ".join(f"{k}=[{lo:.5f},{hi:.5f}]" for k, (lo, hi) in ci_ser.items()))

    # seed-to-seed MC variability of the SERIAL path (the reference for "statistically equivalent")
    alt = _ci(bootstrap_metrics(**_build(43), n_jobs=1))
    mc_ref = {k: (abs(alt[k][0] - ci_ser[k][0]), abs(alt[k][1] - ci_ser[k][1])) for k in ci_ser}

    for nj in (4, 8, 16):
        t = time.perf_counter(); r_par = bootstrap_metrics(**_build(42), n_jobs=nj); t_par = time.perf_counter() - t
        ci_par = _ci(r_par)
        drift = {k: (abs(ci_par[k][0] - ci_ser[k][0]), abs(ci_par[k][1] - ci_ser[k][1])) for k in ci_ser}
        ok = all(max(drift[k]) <= 3 * max(mc_ref[k]) + 1e-9 for k in ci_ser)
        print(f"parallel x{nj:<2}: {t_par:.1f}s  speedup {t_ser/t_par:.1f}x  "
              + "  ".join(f"{k}:drift={max(drift[k]):.2e}(mc {max(mc_ref[k]):.2e})" for k in ci_ser)
              + f"  {'OK' if ok else 'CHECK'}")

    # reproducibility + worker-count independence
    a = _ci(bootstrap_metrics(**_build(7), n_jobs=8))
    b = _ci(bootstrap_metrics(**_build(7), n_jobs=8))
    c = _ci(bootstrap_metrics(**_build(7), n_jobs=4))  # different worker count, same seed
    same_repro = all(np.allclose(a[k], b[k]) for k in a)
    same_workers = all(np.allclose(a[k], c[k]) for k in a)
    print(f"reproducible (x8 twice): {same_repro}   worker-count-independent (x8 vs x4): {same_workers}")
