"""A NaN proxy loss must not occupy a top-N slot in the CPU exhaustive scan.

``_insert_top`` admitted ``loss`` unless ``loss >= worst``. A NaN loss fails that comparison, so it was written into the last slot and
displaced a finite candidate; the non-finite entry was then filtered out afterwards, leaving the cardinality one candidate short.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n


def test_nan_feature_loss_does_not_displace_finite_candidates():
    """A subset with a NaN loss leaves the remaining top-N slots filled with finite candidates."""
    rng = np.random.default_rng(0)
    n, f = 200, 4
    phi = rng.normal(size=(n, f))
    phi[:, 0] = np.nan  # every subset containing feature 0 has a NaN loss
    base = np.zeros(n)
    y = rng.normal(size=n)

    result = brute_force_top_n(phi, base, y, classification=False, metric="mse", min_card=1, max_card=1, top_n=3, parallel=False)

    combos = sorted(comb for _, comb in result)
    assert combos == [(1,), (2,), (3,)], combos
    assert all(np.isfinite(loss) for loss, _ in result)
