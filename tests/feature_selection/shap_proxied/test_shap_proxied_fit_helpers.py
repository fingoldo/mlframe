"""Regression tests for two ShapProxiedFS fit-pipeline helpers.

* ``_inject_operand_pairs``: a synergistic operand pair with no measured proxy loss must NOT be
  injected with a fabricated optimistic loss (the pre-fix ``candidates[0][0]`` stand-in) that could
  sort to the front and win selection; it goes in at +inf so it can never win.
* ``_apply_min_selected_ratio``: the ``len(c)/n_proxy_cols`` filter must not raise ZeroDivisionError
  when ``n_proxy_cols == 0``.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxied_fit import _apply_min_selected_ratio
from mlframe.feature_selection.shap_proxied_fs._shap_proxied_fit_interactions import _inject_operand_pairs
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import subset_loss


def _phi_fixture():
    """Build (phi, base, y) where phi columns 0/1 carry strong class signal and 3/4 are pure noise."""
    rng = np.random.default_rng(0)
    n = 600
    y = (np.arange(n) % 2).astype(float)
    signal = np.where(y == 1, 1.5, -1.5)
    phi = np.column_stack(
        [
            signal + 0.05 * rng.normal(size=n),  # 0: informative
            signal + 0.05 * rng.normal(size=n),  # 1: informative
            rng.normal(size=n),  # 2: noise
            rng.normal(size=n),  # 3: noise (injected-pair operand)
            rng.normal(size=n),  # 4: noise (injected-pair operand)
        ]
    ).astype(np.float64)
    base = np.zeros(n, dtype=np.float64)
    return phi, base, y


def test_injected_pair_gets_real_loss_and_cannot_win_on_fabricated_loss():
    phi, base, y = _phi_fixture()

    # Pre-existing winner: the informative coalition (0,1) with a genuinely low loss.
    win_loss = subset_loss(phi, base, y, [0, 1], "auc")
    candidates = [(win_loss, (0, 1)), (0.5, (2,))]
    merged = {tuple(sorted(c)): l for l, c in candidates}

    name_to_phi_idx = {"a": 3, "b": 4}
    usable_pairs = [(0.9, "a", "b")]  # noise operand pair (3,4), not yet in merged

    _inject_operand_pairs(merged, usable_pairs, name_to_phi_idx, phi=phi, base=base, y_phi=y, classification=True, metric="auc")

    injected_key = (3, 4)
    assert injected_key in merged
    # The injected pair carries its REAL (poor, noise) proxy loss -- NOT the winner's fabricated loss.
    assert merged[injected_key] > win_loss, "injected noise pair must carry a real loss worse than the winner"

    sorted_cands = sorted(((l, c) for c, l in merged.items()), key=lambda t: t[0])
    winner = sorted_cands[0]
    assert winner[1] == (0, 1), "an injected noise pair must not win the proxy sort on a fabricated loss"


def test_inject_keeps_real_loss_when_pair_already_present():
    phi, base, y = _phi_fixture()
    # The pair already carries a REAL (good) loss from upstream; injection must not overwrite it.
    merged = {(3, 4): 0.05, (0, 1): 0.1}
    _inject_operand_pairs(merged, [(0.9, "a", "b")], {"a": 3, "b": 4}, phi=phi, base=base, y_phi=y, classification=True, metric="auc")
    assert merged[(3, 4)] == 0.05


def test_min_selected_ratio_no_zero_division_on_empty_proxy():
    candidates = [(0.1, (0, 1)), (0.2, (0,))]
    out = _apply_min_selected_ratio(candidates, n_proxy_cols=0, min_selected_ratio=0.5)
    assert out == candidates  # no width to ratio against -> pass through unfiltered, no raise


def test_min_selected_ratio_filters_when_width_known():
    candidates = [(0.1, (0,)), (0.2, (0, 1, 2, 3))]
    out = _apply_min_selected_ratio(candidates, n_proxy_cols=4, min_selected_ratio=0.5)
    assert out == [(0.2, (0, 1, 2, 3))]  # 1/4 < 0.5 dropped, 4/4 kept
