"""Unit + biz_val for the proxy bias corrector (levers #3 + #6).

biz_val claim: when the proxy's gap to honest loss is driven by subset redundancy (the documented
failure mode), a corrector fit on a few (proxy, honest, cardinality, redundancy) anchors recovers the
honest ranking better than the raw proxy ordering. Locks that the corrector adds real value exactly
where the proxy is biased, and degrades to a rank-preserving identity when anchors are too few.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from mlframe.feature_selection._shap_proxy_calibrate import (
    ProxyCorrector, fit_proxy_corrector, rerank_candidates, subset_redundancy,
    subset_redundancy_many)


def test_subset_redundancy_basic():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(500, 1))
    # 3 near-duplicate columns (high redundancy) vs 3 independent (low)
    dup = np.hstack([base + 0.01 * rng.normal(size=(500, 1)) for _ in range(3)])
    indep = rng.normal(size=(500, 3))
    phi = np.hstack([dup, indep])
    assert subset_redundancy(phi, [0, 1, 2]) > 0.9   # duplicates
    assert subset_redundancy(phi, [3, 4, 5]) < 0.3   # independent
    assert subset_redundancy(phi, [0]) == 0.0        # singleton


def test_subset_redundancy_many_matches_single():
    """The batched helper (transpose-once contiguous gather) must be bit-identical to looping the
    single subset_redundancy. Locks the iter108 layout optimisation against drift, incl. the
    singleton (->0.0) and the empty-subset edge cases."""
    rng = np.random.default_rng(3)
    phi = rng.normal(size=(800, 20))
    idx_list = [[0, 1, 2], [5], [3, 7, 11, 15], [], list(range(20)), [4, 4 + 1]]
    batch = subset_redundancy_many(phi, idx_list)
    single = np.array([subset_redundancy(phi, idx) for idx in idx_list], dtype=np.float64)
    np.testing.assert_array_equal(batch, single)


def test_biz_val_corrector_beats_raw_proxy_under_redundancy_bias():
    rng = np.random.default_rng(0)
    n = 160
    # Anchor design matching the documented failure mode: the honest loss is DOMINATED by a
    # redundancy-dependent term (high-redundancy subsets are honestly much better than the proxy
    # thinks -- a retrain leans on the correlated survivors the proxy can't). The raw proxy, blind to
    # redundancy, therefore mis-ranks; the corrector (which sees redundancy) should recover the order.
    proxy = rng.uniform(0.1, 1.0, size=n)
    redund = rng.uniform(0.0, 1.0, size=n)
    cards = rng.integers(2, 10, size=n).astype(float)
    honest = 0.3 * proxy - 0.9 * redund + 0.03 * rng.normal(size=n)

    tr, te = slice(0, 110), slice(110, n)
    corrector = fit_proxy_corrector(proxy[tr], honest[tr], cards[tr], redund[tr])
    assert not corrector.fallback
    pred_te = corrector.predict(proxy[te], cards[te], redund[te])

    sp_proxy = abs(spearmanr(proxy[te], honest[te]).statistic)
    sp_corr = abs(spearmanr(pred_te, honest[te]).statistic)
    # Raw proxy genuinely struggles here; the corrector must recover the honest ranking clearly.
    assert sp_corr > sp_proxy + 0.2, f"corrector spearman {sp_corr:.3f} not clearly above proxy {sp_proxy:.3f}"
    assert sp_corr > 0.85, f"corrector spearman {sp_corr:.3f} too low"


def test_corrector_falls_back_with_too_few_anchors():
    rng = np.random.default_rng(1)
    proxy = rng.uniform(0, 1, size=5)
    corrector = fit_proxy_corrector(proxy, proxy + 0.1, np.full(5, 3.0), np.zeros(5), min_anchors=12)
    assert corrector.fallback
    # fallback == identity on proxy (rank preserving)
    np.testing.assert_allclose(corrector.predict(proxy, np.full(5, 3.0), np.zeros(5)), proxy)


def test_rerank_preserves_stored_proxy_loss():
    rng = np.random.default_rng(2)
    phi = rng.normal(size=(200, 6))
    candidates = [(0.5, (0, 1)), (0.3, (2, 3)), (0.7, (4, 5))]
    corrector = ProxyCorrector(fallback=True)  # identity -> order by proxy_loss ascending
    out = rerank_candidates(corrector, candidates, phi)
    assert [c[0] for c in out] == [0.3, 0.5, 0.7]
    assert {c[1] for c in out} == {(0, 1), (2, 3), (4, 5)}  # same elements, reordered
