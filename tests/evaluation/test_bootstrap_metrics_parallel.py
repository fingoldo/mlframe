"""Regression: bootstrap_metrics gained an opt-in n_jobs resample-loop parallelization (threading over nogil njit
kernels, per-resample seeding). Pins the correctness contract that lets it be safe:
  - n_jobs=1 is the unchanged serial path and is deterministic for a fixed seed.
  - n_jobs>1 uses per-resample seeds so the CIs are (a) reproducible run-to-run and (b) INDEPENDENT of the worker /
    host-core count -- the property that makes the parallel result machine-portable.
  - the parallel CI is statistically equivalent to serial (bounds agree well within the metric's own scale).
The default (n_jobs=1) must never change the serial numbers, so it is pinned separately."""

import numpy as np

from mlframe.evaluation.bootstrap import bootstrap_metrics

_N = 8000  # above the n>=5000 parallel gate
_R = 400  # above the n_bootstrap>=256 gate


def _data(seed=0):
    """Helper that data."""
    rng = np.random.default_rng(seed)
    y = (rng.random(_N) < 0.35).astype(np.float64)
    p = np.clip(0.2 + 0.5 * y + rng.standard_normal(_N) * 0.3, 1e-6, 1 - 1e-6)
    return y, p


def _mf():
    """Helper that mf."""
    return {"brier": lambda yy, pp: float(np.mean((yy - pp) ** 2)), "mean_p": lambda yy, pp: float(np.mean(pp))}


def _ci(res):
    """Helper that ci."""
    return {k: (round(v["lo"], 10), round(v["hi"], 10)) for k, v in res.items()}


def test_serial_n_jobs_1_is_deterministic():
    """Serial n jobs 1 is deterministic."""
    y, p = _data()
    a = _ci(bootstrap_metrics(y, p, _mf(), n_bootstrap=_R, stratify=y, random_state=7, n_jobs=1))
    b = _ci(bootstrap_metrics(y, p, _mf(), n_bootstrap=_R, stratify=y, random_state=7, n_jobs=1))
    assert a == b


def test_parallel_reproducible_and_worker_count_independent():
    """Parallel reproducible and worker count independent."""
    y, p = _data()
    r4a = _ci(bootstrap_metrics(y, p, _mf(), n_bootstrap=_R, stratify=y, random_state=7, n_jobs=4))
    r4b = _ci(bootstrap_metrics(y, p, _mf(), n_bootstrap=_R, stratify=y, random_state=7, n_jobs=4))
    r2 = _ci(bootstrap_metrics(y, p, _mf(), n_bootstrap=_R, stratify=y, random_state=7, n_jobs=2))
    assert r4a == r4b, "same seed + same n_jobs must reproduce identical CIs"
    assert r4a == r2, "per-resample seeding must make the CI independent of the worker count"


def test_parallel_is_statistically_equivalent_to_serial():
    """Parallel is statistically equivalent to serial."""
    y, p = _data()
    ser = bootstrap_metrics(y, p, _mf(), n_bootstrap=_R, stratify=y, random_state=7, n_jobs=1)
    par = bootstrap_metrics(y, p, _mf(), n_bootstrap=_R, stratify=y, random_state=7, n_jobs=4)
    for name in ser:
        s, q = ser[name], par[name]
        scale = max(abs(s["point"]), 1e-6)
        # bounds agree to well within 5% of the metric scale -- both are MC estimates of the same CI
        assert abs(s["lo"] - q["lo"]) < 0.05 * scale and abs(s["hi"] - q["hi"]) < 0.05 * scale, (
            f"{name}: serial [{s['lo']:.5f},{s['hi']:.5f}] vs parallel [{q['lo']:.5f},{q['hi']:.5f}]"
        )
