"""biz_val: ``parsimony_tol`` controls recall of a genuinely weak-but-real signal end-to-end.

Traced during a wide-dataframe investigation (see
``_benchmarks/PLAN_wide_dataframe_improvements.md``): on a fixture with 6 strong features
(weight 1.0) and 6 weak-but-real features (weight 0.25), the weak features survive every stage
of the pipeline up to and including the search's winning candidate, but ``within_cluster_refine``
greedily drops them because removing them keeps the honest holdout loss within the default
``parsimony_tol=0.02`` (2%) of the best seen. This is the selector's DOCUMENTED precision/recall
dial (see the ``parsimony_tol`` docstring on ``ShapProxiedFS.__init__``), not a bug -- this test
pins the confirmed direction end-to-end through the real ``fit()`` pipeline so it cannot silently
regress or be "fixed away" by a future change that assumes the gap was a defect.

Confirmed sweep (this session): tol=0.02 -> 0/6 weak recovered; tol=0.005 -> 1/6; tol=0.0 -> 1/6
(plus the 6th strong feature, previously unexplained-missing, also reappears at tol=0.0). The
absolute weak-recall counts are fixture/seed-specific and not asserted directly; the test instead
pins the ROBUST, seed-independent claim -- looser tolerance never selects FEWER features than a
tighter one -- which is what the "tunable dial" contract promises regardless of exact recall counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")

from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS


def _make_mixed_strength_fixture(seed=0, n=3000, p=3000, n_strong=6, n_weak=6, strong_weight=1.0, weak_weight=0.25):
    """Make mixed strength fixture."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    strong = list(range(n_strong))
    weak = list(range(50, 50 + n_weak))
    logit = strong_weight * X[:, strong].sum(axis=1) + weak_weight * X[:, weak].sum(axis=1)
    logit = logit / logit.std() * 2.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    cols = [f"f{i}" for i in range(p)]
    Xdf = pd.DataFrame(X, columns=cols)
    return Xdf, pd.Series(y), strong, weak


def _fit_selected(X, y, parsimony_tol, seed=0):
    """Fit selected."""
    s = ShapProxiedFS(classification=True, random_state=seed, verbose=False, prescreen_ladder_mode="off", parsimony_tol=parsimony_tol, n_jobs=1)
    s.fit(X, y)
    return set(s.selected_features_)


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_biz_val_looser_parsimony_tol_never_selects_fewer_features():
    """The core, seed-robust claim: n_selected is monotone non-decreasing as parsimony_tol shrinks
    toward 0 (looser tolerance for a loss-increase => refine keeps more members). This is the
    mechanical guarantee ``parsimony_tol`` is documented to provide, independent of which specific
    features end up recovered on any one fixture/seed."""
    X, y, _strong, weak = _make_mixed_strength_fixture()

    sel_default = _fit_selected(X, y, parsimony_tol=0.02)
    sel_loose = _fit_selected(X, y, parsimony_tol=0.005)
    sel_off = _fit_selected(X, y, parsimony_tol=0.0)

    assert len(sel_loose) >= len(sel_default), (
        f"parsimony_tol=0.005 selected fewer features ({len(sel_loose)}) than the default "
        f"parsimony_tol=0.02 ({len(sel_default)}) -- looser tolerance must never REDUCE recall"
    )
    assert len(sel_off) >= len(sel_loose), (
        f"parsimony_tol=0.0 selected fewer features ({len(sel_off)}) than parsimony_tol=0.005 ({len(sel_loose)}) -- looser tolerance must never REDUCE recall"
    )
    # The default's selection must be a SUBSET-compatible starting point: every stage up to search
    # already carries the strong features, so the default should never drop MORE strong signal
    # than the loosened settings.
    weak_names = {f"f{i}" for i in weak}
    default_weak_recall = len(weak_names & sel_default)
    off_weak_recall = len(weak_names & sel_off)
    assert off_weak_recall >= default_weak_recall, (
        f"parsimony_tol=0.0 recovered fewer weak features ({off_weak_recall}) than the tighter "
        f"default ({default_weak_recall}) -- contradicts the documented recall dial"
    )


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_biz_val_default_parsimony_tol_favours_precision_on_marginal_signal():
    """Confirms the OTHER side of the tradeoff the default is tuned for: with the precision-favouring
    default (0.02), the selected subset is not inflated with every technically-nonzero-SHAP feature --
    it stays close to the small, high-confidence strong-only core. This is the flip side of the above
    monotonicity test and guards against a future change accidentally making the default "greedy"
    (selecting everything that clears search) instead of parsimonious."""
    X, y, strong, _weak = _make_mixed_strength_fixture()
    sel_default = _fit_selected(X, y, parsimony_tol=0.02)
    strong_names = {f"f{i}" for i in strong}
    # The default should stay close to the strong-only core: at most 2 extra features beyond the
    # strong set (generous headroom for seed variance), not the full 12-feature strong+weak union.
    assert len(sel_default) <= len(strong_names) + 2, (
        f"default parsimony_tol=0.02 selected {len(sel_default)} features, expected it to stay "
        f"close to the {len(strong_names)}-feature strong-only core (precision-favouring default)"
    )
