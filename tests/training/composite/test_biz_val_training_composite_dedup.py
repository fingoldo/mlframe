"""biz_value verdict-pin for the qual-19 REJECT of the cross-target ensemble residual-dedup default.

Pins the measured outcome that flipping ``ct_ensemble_dedup_enabled`` ON does NOT improve honest-holdout test RMSE on a
redundant pool -- it is neutral under NNLS stacking (the stacker already learns the optimal combined weight regardless
of how near-duplicate members split it) and HARMFUL under uniform-mean ensembling (the redundant cluster's average is a
lower-variance estimate of the shared signal, so pruning it raises variance). If a future change made dedup measurably
beat the full pool here -- the precondition the project's majority-of-scenarios flip rule would need -- this test fails,
flagging that the qual-19 REJECT (default OFF) should be re-examined.

See bench: ``mlframe/training/_benchmarks/bench_ct_ensemble_residual_dedup_scenarios.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import nnls

from mlframe.training.composite import residual_dedup_indices
from mlframe.training._composite_target_discovery_config import CompositeTargetDiscoveryConfig


def _make_pool(seed: int, n: int = 4000, n_dup: int = 4, jitter: float = 0.02):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 2.0 * x + 1.5 * z + rng.normal(scale=1.0, size=n)
    members = [2.0 * x + rng.normal(scale=0.8, size=n),
               1.5 * z + rng.normal(scale=0.8, size=n),
               1.8 * x + 1.3 * z + rng.normal(scale=1.2, size=n)]
    strongest = members[-1]
    for _ in range(n_dup):
        members.append(strongest + rng.normal(scale=jitter, size=n))
    return np.column_stack(members), y


def _split(M, y, frac=0.5):
    cut = int(M.shape[0] * frac)
    return M[:cut], y[:cut], M[cut:], y[cut:]


def _nnls_rmse(oof, y_oof, test, y_test):
    w, _ = nnls(oof, y_oof)
    return float(np.sqrt(np.mean((test @ w - y_test) ** 2)))


def _mean_rmse(test, y_test):
    return float(np.sqrt(np.mean((test.mean(axis=1) - y_test) ** 2)))


def test_biz_val_ct_ensemble_dedup_default_is_off():
    """The corrective-mechanism flag stays OFF: qual-19 measured it neutral-or-harmful, so OFF is the most-accurate default."""
    assert CompositeTargetDiscoveryConfig().ct_ensemble_dedup_enabled is False


def test_biz_val_ct_ensemble_dedup_does_not_help_nnls_or_mean_on_redundant_pool():
    """Dedup must NOT win the majority of seeds under EITHER strategy on the redundant pool (the qual-19 REJECT precondition).

    NNLS: neutral (~1e-4, dedup wins <=1/10 -- stacker absorbs the member split). Mean: harmful (dedup raises RMSE on
    EVERY seed). A regression that made dedup beat the full pool would trip the project's flip rule; pin both sides.
    """
    seeds = range(10)
    nnls_dedup_wins = 0
    mean_dedup_worse = 0
    mean_full_total = 0.0
    mean_dedup_total = 0.0
    for s in seeds:
        M, y = _make_pool(s)
        M_oof, y_oof, M_test, y_test = _split(M, y)
        oof_rmses = np.sqrt(np.mean((M_oof - y_oof[:, None]) ** 2, axis=0))
        resid = M_oof - y_oof[:, None]
        keep, drop = residual_dedup_indices(resid, oof_rmses, corr_threshold=0.95)
        assert drop, "redundant pool should expose at least one near-duplicate to drop"

        nnls_full = _nnls_rmse(M_oof, y_oof, M_test, y_test)
        nnls_dd = _nnls_rmse(M_oof[:, keep], y_oof, M_test[:, keep], y_test)
        if nnls_dd < nnls_full:
            nnls_dedup_wins += 1

        mean_full = _mean_rmse(M_test, y_test)
        mean_dd = _mean_rmse(M_test[:, keep], y_test)
        mean_full_total += mean_full
        mean_dedup_total += mean_dd
        if mean_dd > mean_full:
            mean_dedup_worse += 1

    # NNLS: dedup does NOT win the majority (measured 1/10).
    assert nnls_dedup_wins <= len(list(seeds)) // 2, (
        f"dedup unexpectedly beat the full NNLS stack on {nnls_dedup_wins}/10 seeds -- re-examine the qual-19 REJECT"
    )
    # Mean: dedup is harmful on EVERY seed; aggregate worse by a clear margin (measured +2.8%).
    assert mean_dedup_worse == len(list(seeds)), (
        f"dedup expected to hurt the uniform-mean ensemble on all seeds; hurt only {mean_dedup_worse}/10"
    )
    assert mean_dedup_total > mean_full_total * 1.01, (
        "dedup should measurably RAISE uniform-mean holdout RMSE on the redundant pool (lower-variance cluster removed)"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--no-cov"]))
