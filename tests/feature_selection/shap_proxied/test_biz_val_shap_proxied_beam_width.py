"""biz_val: a wider ``beam_width`` finds a better feature subset than a narrow beam.

``beam_search`` seeds the beam with the top-``beam_width`` single features by proxy loss, then expands.
On a frame where the truly-best columns are individually WEAK (their proxy credit only pays off in the
joint coalition) while several decoys look stronger marginally, a narrow beam prunes the weak-but-needed
columns at the seed stage and never recovers them; a wide beam keeps them in play and reaches the lower-
loss subset. The best candidate's proxy loss is the measurable win.

Measured dev run (seeds 0-5): beam_width=12 best-loss strictly below beam_width=2 on all 6 seeds
(e.g. seed0 1.272 vs 1.301). Floor pins a strict improvement with seed headroom.
"""

from __future__ import annotations

import numpy as np


def _phi_frame(seed):
    rng = np.random.default_rng(seed)
    n, p = 1500, 14
    core = rng.normal(size=(n, 4))
    y = core.sum(1) + 0.15 * rng.normal(size=n)  # the 4 core cols jointly reconstruct y
    phi = np.zeros((n, p))
    phi[:, :4] = core * 0.30  # individually weak proxy credit
    for k in range(4, 10):  # decoys: stronger single-feature proxy alignment, useless jointly
        phi[:, k] = 0.5 * y / 4 + 0.5 * rng.normal(size=n)
    phi[:, 10:] = rng.normal(size=(n, 4)) * 0.1
    return phi, np.zeros(n), y


def _best_loss(beam_width, seed):
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_heuristics as H

    phi, base, y = _phi_frame(seed)
    res = H.beam_search(phi, base, y, classification=False, metric="rmse", beam_width=beam_width, top_n=1, min_card=1, max_card=5)
    return res[0][0]


def test_biz_val_beam_width_wide_finds_better_subset_than_narrow():
    narrow = _best_loss(2, seed=0)
    wide = _best_loss(12, seed=0)
    assert wide < narrow - 1e-3, f"wide beam loss {wide:.5f} should beat narrow {narrow:.5f}"


def test_biz_val_beam_width_wins_or_ties_across_seeds():
    results = [(_best_loss(2, s), _best_loss(12, s)) for s in range(6)]
    wins = sum(w < nv - 1e-4 for nv, w in results)
    assert all(w <= nv + 1e-9 for nv, w in results), f"wide beam never worse; {results}"
    assert wins >= 5, f"wide beam should strictly win on majority of seeds; {results}"
