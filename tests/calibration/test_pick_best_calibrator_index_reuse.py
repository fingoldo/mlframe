"""Regression: shared resample-index reuse in ``pick_best_calibrator`` (lead4).

``pick_best_calibrator`` benches ~3-4 calibrator candidates; each previously
called ``bootstrap_metric`` which REGENERATED the identical stratified resample
indices (same n / stratify / seed) per candidate. The index matrix is now built
ONCE outside the candidate loop and reused. These tests pin:

  * BIT-IDENTITY: the shared-index helper produces percentile CIs identical to a
    fresh per-candidate ``bootstrap_metric`` call, on BOTH stratified and
    unstratified resampling.
  * SELECTION STABILITY: the chosen calibrator + every alternative's ECE mean/CI
    match the per-candidate ``bootstrap_metric`` reconstruction exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.fast]


def _make_miscalibrated(n: int, seed: int = 11):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=n)
    true_p = 1.0 / (1.0 + np.exp(-6.0 * (raw - 0.5)))
    y = (rng.uniform(0.0, 1.0, size=n) < true_p).astype(np.int64)
    return raw, y


@pytest.mark.parametrize("stratified", [True, False])
def test_shared_index_helper_bit_identical_to_bootstrap_metric(stratified):
    from mlframe.calibration import policy
    from mlframe.evaluation.bootstrap import bootstrap_metric

    rng = np.random.default_rng(0)
    n = 2500
    y = (rng.uniform(0, 1, n) < 0.3).astype(np.int64)
    p = np.clip(rng.uniform(0, 1, n) + 0.1 * y, 0.0, 1.0)
    strat = y if stratified else None
    mf = lambda a, b: policy._ece_score(a, b, n_bins=15)

    idx = policy._build_resample_indices(n, 400, strat, 7)
    new = policy._bootstrap_ece_with_indices(y, p, idx, mf, 0.05)
    old = bootstrap_metric(y, p, metric_fn=mf, n_bootstrap=400, alpha=0.05, stratify=strat, random_state=7)

    assert new["point"] == old["point"]
    assert new["lo"] == old["lo"]
    assert new["hi"] == old["hi"]


def test_pick_best_calibrator_selection_identical_to_per_candidate_bootstrap(monkeypatch):
    """End-to-end: chosen calibrator + alternatives match a per-candidate
    ``bootstrap_metric`` reconstruction (the pre-reuse behaviour)."""
    from mlframe.calibration import policy
    from mlframe.calibration.policy import pick_best_calibrator
    from mlframe.evaluation.bootstrap import bootstrap_metric

    raw, y = _make_miscalibrated(n=2000, seed=11)
    classes = np.unique(y)
    strat = y if classes.size == 2 else None

    out_new = pick_best_calibrator(
        probs=None,
        y=None,
        oof_probs=raw,
        oof_y=y,
        n_bootstrap=300,
        random_state=11,
    )

    # Reconstruct the legacy per-candidate path: bootstrap_metric per candidate.
    def _legacy_eci(yt, yp, idx, mf, alpha, n_bins=None):
        ci = bootstrap_metric(yt, yp, metric_fn=mf, n_bootstrap=300, alpha=alpha, stratify=strat, random_state=11)
        return {"point": ci["point"], "lo": ci["lo"], "hi": ci["hi"]}

    monkeypatch.setattr(policy, "_bootstrap_ece_with_indices", _legacy_eci)
    out_old = pick_best_calibrator(
        probs=None,
        y=None,
        oof_probs=raw,
        oof_y=y,
        n_bootstrap=300,
        random_state=11,
    )

    assert out_new["chosen"] == out_old["chosen"]
    assert out_new["rule"] == out_old["rule"]
    assert set(out_new["alternatives"]) == set(out_old["alternatives"])
    for name, info in out_new["alternatives"].items():
        ref = out_old["alternatives"][name]
        assert info["ece_mean"] == ref["ece_mean"], name
        assert info["ece_ci"] == ref["ece_ci"], name
