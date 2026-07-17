"""Regression: float64 pre-cast in ``_bootstrap_block`` (lead5).

``_brier`` / ``_ll`` used to cast ``yy``/``pp`` to float64 INSIDE the 1000-resample
loop. On int labels ``astype(np.float64, copy=False)`` still copies every call
(~4000 in-loop copies / block). ``_bootstrap_block`` now pre-casts ``y_true`` /
``p_pos`` to float64 ONCE before the loop, so the resampled views are already
float64 and the kernels skip the per-call cast.

The bootstrap CI values must be UNCHANGED by this hoist (same numbers fed to the
same numba kernels), which this test pins against a reconstruction of the legacy
in-loop-astype path.

CI-bound tolerance note: the BCa acceleration term is now computed via the exact O(n) algebraic leave-one-out
jackknife (``_jackknife_mean_metric``: ``LOO_i = (sum - per_row_i)/(n-1)``) instead of re-gathering n-1 rows and
re-running the metric per leave-out point. That is a floating-point sum-reduction-order difference (~1e-15 on the
per-point LOO values), which propagates to <=~1e-13 on the CI bounds -- so the CI is asserted FP-close (rtol 1e-9),
not bit-exact, while the POINT estimate (untouched by the jackknife) stays exactly equal.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.fast]


def _legacy_brier_ll_cis(y, p_pos, rng_seed, n_bootstrap=400):
    """Reconstruct the pre-fix path: int arrays + in-loop astype in brier/ll."""
    from mlframe.evaluation.bootstrap import bootstrap_metrics
    from mlframe.metrics.core import fast_brier_score_loss as _b, fast_log_loss as _l
    from mlframe.calibration.policy import _ece_score

    def br(yy, pp):
        return float(_b(yy.astype(np.float64, copy=False), pp.astype(np.float64, copy=False)))

    def ll(yy, pp):
        return float(_l(yy.astype(np.float64, copy=False), pp.astype(np.float64, copy=False)))

    mf = {"brier": br, "log_loss": ll, "ece": lambda yy, pp: _ece_score(yy, pp)}
    return bootstrap_metrics(y, p_pos, mf, n_bootstrap=n_bootstrap, alpha=0.05, stratify=y, random_state=rng_seed)


def test_bootstrap_block_precast_ci_unchanged():
    from mlframe.training.honest_diagnostics import _bootstrap_block

    rng = np.random.default_rng(3)
    n = 5000
    raw = rng.uniform(0, 1, n)
    true_p = 1.0 / (1.0 + np.exp(-4.0 * (raw - 0.5)))
    y = (rng.uniform(0, 1, n) < true_p).astype(np.int64)
    probs = np.column_stack([1.0 - raw, raw])

    out = _bootstrap_block(y, probs, rng_seed=42)
    legacy = _legacy_brier_ll_cis(y, raw, rng_seed=42, n_bootstrap=1000)

    for m in ("brier", "log_loss"):
        # Point is untouched by the jackknife -> exactly equal. CI bounds match to FP precision: the O(n) algebraic
        # BCa jackknife differs from the gather jackknife only by sum-reduction order (~1e-15 per LOO value, <=~1e-13
        # on the bound) -- far below any decision-relevant scale (see module docstring).
        assert out[m]["point"] == legacy[m]["point"], m
        assert np.isclose(out[m]["ci_lo"], legacy[m]["lo"], rtol=1e-9, atol=0.0), (m, out[m]["ci_lo"], legacy[m]["lo"])
        assert np.isclose(out[m]["ci_hi"], legacy[m]["hi"], rtol=1e-9, atol=0.0), (m, out[m]["ci_hi"], legacy[m]["hi"])


def test_bootstrap_block_handles_int_and_float_labels_identically():
    """Pre-cast must yield identical CIs whether the caller passed int or float labels."""
    from mlframe.training.honest_diagnostics import _bootstrap_block

    rng = np.random.default_rng(7)
    n = 4000
    raw = rng.uniform(0, 1, n)
    y_int = (rng.uniform(0, 1, n) < raw).astype(np.int64)
    probs = np.column_stack([1.0 - raw, raw])

    out_int = _bootstrap_block(y_int, probs, rng_seed=99)
    out_float = _bootstrap_block(y_int.astype(np.float64), probs, rng_seed=99)
    for m in ("brier", "log_loss", "ece"):
        assert out_int[m]["point"] == out_float[m]["point"], m
        assert out_int[m]["ci_lo"] == out_float[m]["ci_lo"], m
        assert out_int[m]["ci_hi"] == out_float[m]["ci_hi"], m
