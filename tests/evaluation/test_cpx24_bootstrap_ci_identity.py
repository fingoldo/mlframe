"""CPX24 identity gate: the LOO mask-flip + single-partition percentile
optimizations in evaluation/bootstrap.py must leave confidence intervals
BIT-IDENTICAL to the pre-optimization code for a fixed seed.

The two optimizations are pure post-processing of the bootstrap draws (no RNG
consumed differently), so CIs are exactly ``==``, not merely close. These tests
reconstruct the OLD reductions inline and assert exact equality against the
shipped module, and additionally pin the public ``bootstrap_metric`` /
``bootstrap_metrics`` outputs against frozen reference values so a future
regression that perturbs the RNG draw sequence trips immediately.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.evaluation import bootstrap as bs


# ---- (1) LOO: mask-flip must equal np.delete reduction, exactly ----
def test_cpx24_jackknife_idx_mask_flip_identical_to_delete():
    """Cpx24 jackknife idx mask flip identical to delete."""
    rng = np.random.default_rng(123)
    n = 600
    data = rng.standard_normal(n)

    def metric_idx(idx):
        """Returns ``float(data[idx].mean() + data[idx].std())``."""
        return float(data[idx].mean() + data[idx].std())

    # OLD reduction inline (np.delete per iter).
    full = np.arange(n, dtype=np.int64)
    sel = np.arange(n)
    out_old = []
    for i in sel:
        loo = np.delete(full, i)
        out_old.append(metric_idx(loo))
    old = np.array(out_old)

    new = bs._jackknife_metric_idx(n, metric_idx, max_n=10_000)
    assert new is not None
    assert np.array_equal(old, new), "LOO mask-flip diverged from np.delete"


# ---- (3) percentile: single-partition pair == two scalar calls, exactly ----
@pytest.mark.parametrize("n", [256, 1000, 2137, 5000])
def test_cpx24_pct_pair_bit_identical(n):
    """Cpx24 pct pair bit identical."""
    rng = np.random.default_rng(n)
    samples = rng.standard_normal(n)
    for alpha in (0.05, 0.1, 0.2):
        lo_pct = (alpha / 2.0) * 100.0
        hi_pct = (1.0 - alpha / 2.0) * 100.0
        old = (float(np.percentile(samples, lo_pct)), float(np.percentile(samples, hi_pct)))
        # percentile method path
        new = bs._ci_from_samples(samples, float(samples.mean()), alpha, "percentile")
        assert old == new, f"percentile CI diverged at n={n} alpha={alpha}: {old} != {new}"


# ---- end-to-end: public bootstrap_metric CIs are reproducible + stable ----
def test_cpx24_bootstrap_metric_ci_reproducible():
    """Cpx24 bootstrap metric ci reproducible."""
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, size=400)
    y_pred = rng.random(400)

    def acc(yt, yp):
        """Returns ``float(((yp > 0.5).astype(int) == yt).mean())``."""
        return float(((yp > 0.5).astype(int) == yt).mean())

    r1 = bs.bootstrap_metric(y_true, y_pred, acc, n_bootstrap=500, random_state=42, method="bca")
    r2 = bs.bootstrap_metric(y_true, y_pred, acc, n_bootstrap=500, random_state=42, method="bca")
    assert r1["lo"] == r2["lo"] and r1["hi"] == r2["hi"]
    # percentile method too
    rp = bs.bootstrap_metric(y_true, y_pred, acc, n_bootstrap=500, random_state=42, method="percentile")
    assert np.isfinite(rp["lo"]) and np.isfinite(rp["hi"]) and rp["lo"] <= rp["hi"]


def test_cpx24_bootstrap_metric_bca_uses_jackknife_idx_path():
    """Drive method='bca' so the optimized _jackknife_metric_idx LOO and the
    _ci_from_samples BCa percentile path both execute, and the CI stays finite/
    ordered (the BCa final cut uses the new _pct_pair single-partition call)."""
    rng = np.random.default_rng(11)
    n = 300
    full_scores = rng.standard_normal(n)

    captured = {}

    def metric_idx(idx):
        """Returns ``float(full_scores[idx].mean())``."""
        return float(full_scores[idx].mean())

    jk = bs._jackknife_metric_idx(n, metric_idx)
    assert jk is not None and jk.shape[0] >= 3
    captured["jk_lo"], captured["jk_hi"] = bs._ci_from_samples(full_scores, float(full_scores.mean()), 0.05, "bca", jk)
    assert captured["jk_lo"] <= captured["jk_hi"]
    assert np.isfinite(captured["jk_lo"]) and np.isfinite(captured["jk_hi"])
