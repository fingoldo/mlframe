"""ACE's masking rounds re-test the same features, so alpha must be shared across them."""

import numpy as np

import mlframe.feature_selection.ace as ace


def test_each_round_tests_at_a_share_of_alpha(monkeypatch):
    """Every round used to test at nominal alpha and OR the acceptances, several times the nominal FDR on noise."""
    seen_alphas = []
    real_bh = ace._benjamini_hochberg_reject

    def spy(pvals, alpha):
        seen_alphas.append(alpha)
        return real_bh(pvals, alpha)

    monkeypatch.setattr(ace, "_benjamini_hochberg_reject", spy)
    # Every feature just misses a nominal 0.05 cut each round, so the rounds keep going and each one is observed.
    monkeypatch.setattr(ace, "_run_ace_round", lambda est, X, y, **kw: (np.ones((3, X.shape[1])), np.zeros(X.shape[1])))
    monkeypatch.setattr(ace, "_ttest_greater", lambda real, thr: np.full(real.shape[1], 0.02))

    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(60, 4)), rng.integers(0, 2, 60)
    out = ace.ace_select(X, y, n_masking_rounds=5, alpha=0.05, fdr_control=True, mask_redundant=False)
    assert seen_alphas and all(a == 0.01 for a in seen_alphas), f"per-round alpha must be alpha / rounds, got {seen_alphas}"
    assert not out.accepted.any(), "p = 0.02 clears nominal 0.05 but not the 0.01 each of five rounds may spend"
