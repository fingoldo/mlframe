"""biz_val + regression tests for RFECV ``smooth_perf`` on ``select_optimal_nfeatures_``.

Two locked facts (measured by ``_benchmarks/fs_hybrid/round5_smooth_perf_bench.py``):

1. Under the PRODUCTION default rule ``auto`` (-> ``one_se_max``) with ``feature_cost=0``, ``smooth_perf`` is INERT: the
   1-SE band is built off the RAW ``cv_mean_perf``, so smoothing only touches ``base_perf`` (used for the plot alone) and
   the selected N is bit-identical for every ``smooth_perf``. This pins WHY the production default 0 is safe -- a future
   refactor that accidentally routes smoothed values into the band would change the pick and fail this test.
2. Under ``rule='argmax'`` smoothing DENOISES the curve argmax reads, a clean honest-score win on a sharp-peak curve. Pins
   the tunable pairing so removing ``smooth_perf`` can't silently kill it.

Exercises the REAL ``select_optimal_nfeatures_`` kernel (bound onto RFECV) via a minimal carrier of the attrs it reads.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.wrappers.rfecv._stability_select import select_optimal_nfeatures_

NGRID = list(range(2, 42, 2))


class _Mock:
    """Groups tests covering Mock."""
    def __init__(self, smooth_perf, rule):
        self.mean_perf_weight = 1.0
        self.std_perf_weight = 0.1
        self.n_features_selection_rule = rule
        self.max_nfeatures = None
        self.conduct_final_voting = False
        self.fi_missing_policy = "worst"
        self.n_features_in_ = max(NGRID)
        self.feature_names_in_ = [str(i) for i in range(self.n_features_in_)]
        self.selected_features_ = {int(n): [str(i) for i in range(int(n))] for n in NGRID}
        self.smooth_perf = smooth_perf
        self.cv_results_ = {}


def _peak_curve():
    """Peak curve."""
    nf = np.array(NGRID, dtype=float)
    x = nf / nf.max()
    return nf, 0.5 + 0.30 * np.exp(-((x - 0.4) ** 2) / (2 * 0.12**2))


def _pick(rule, smooth_perf, obs, std):
    """Helper that pick."""
    m = _Mock(smooth_perf, rule)
    select_optimal_nfeatures_(m, np.array(NGRID, dtype=float), obs.copy(), std.copy(), smooth_perf=smooth_perf, verbose=False, show_plot=False)
    return int(m.n_features_)


def test_smooth_perf_inert_under_auto_default_rule():
    """Production default rule auto(->one_se_max): smooth_perf must NOT change the selected N. Guards the default 0."""
    _nf, t = _peak_curve()
    std = np.full_like(t, 0.06)
    for sd in range(25):
        rng = np.random.default_rng(sd)
        obs = t + rng.normal(0.0, 0.06, size=t.shape)
        base = _pick("auto", 0, obs, std)
        for sp in (1, 3, 5):
            assert _pick("auto", sp, obs, std) == base, f"smooth_perf={sp} altered the auto-rule pick (seed {sd}): one_se_max band must read raw cv_mean_perf"


def test_smooth_perf_denoises_argmax_pick():
    """rule='argmax': smoothing (sp=3) yields higher TRUE-score-at-pick than no smoothing on a sharp-peak noisy curve.

    Measured (40 seeds, noise 0.05): sp=0 0.7831 -> sp=3 0.7919 on the peak curve. Floor the delta at +0.004 (well below
    the ~0.009 measured margin) so a real regression in the smoothing path trips while seed noise does not.
    """
    _nf, t = _peak_curve()
    std = np.full_like(t, 0.05)
    true_sp0, true_sp3 = [], []
    for sd in range(40):
        rng = np.random.default_rng(sd)
        obs = t + rng.normal(0.0, 0.05, size=t.shape)
        true_sp0.append(float(t[NGRID.index(_pick("argmax", 0, obs, std))]))
        true_sp3.append(float(t[NGRID.index(_pick("argmax", 3, obs, std))]))
    delta = float(np.mean(true_sp3) - np.mean(true_sp0))
    assert delta >= 0.004, f"argmax smoothing win regressed: sp3-sp0 TRUE-score delta={delta:.4f} (expected >= 0.004)"
