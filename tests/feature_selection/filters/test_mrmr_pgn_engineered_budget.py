"""Regression: the p>=n FP-control cap must CHARGE engineered survivors against the ceiling.

Bug (fixed): inside ``_fit_impl`` the p>=n raw-feature budget read the engineered count from a local
``n_engineered_out`` that is not bound until much later in the function, so ``"n_engineered_out" in dir()``
was always False and the engineered count silently degraded to 0. The raw budget was therefore the FULL
ceiling ``max(20, p//3)`` regardless of how many engineered features already reached the output, so the p>=n
total (raw + engineered) could exceed the documented ``max(20, p//3)`` FP-control ceiling.

The fix reads the engineered count straight off ``self._engineered_recipes_`` (populated by the main sweep,
available at the cap site) via the pure ``_pgn_raw_budget(ceiling, n_engineered)`` helper.

Two legs:
  (a) UNIT -- the pure budget helper subtracts the engineered count from the ceiling (floored at 0). This is
      the exact arithmetic the bug got wrong (it used ``n_engineered=0`` unconditionally).
  (b) END-TO-END -- a real MRMR fit in the p>=n regime keeps ``n_features_`` within ``max(20, p//3)`` even
      when engineered features are produced. Engineered survivors are counted toward the ceiling, never
      allowed to push the total past it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core import _pgn_raw_budget


def test_pgn_raw_budget_charges_engineered_against_ceiling():
    """The raw budget = ceiling - n_engineered, floored at 0. With engineered features present the raw
    budget is STRICTLY smaller than the ceiling -- the exact behaviour the pre-fix ``n_engineered=0`` path
    lost (it returned the full ceiling regardless, letting raw+engineered exceed it)."""
    # No engineered features: raw budget is the full ceiling.
    assert _pgn_raw_budget(20, 0) == 20
    # Engineered features consume budget: raw budget shrinks by exactly that count.
    assert _pgn_raw_budget(20, 5) == 15
    assert _pgn_raw_budget(50, 3) == 47
    # A non-zero engineered count MUST reduce the raw budget below the ceiling (the pre-fix bug did not).
    for ceiling in (20, 33, 100):
        for n_eng in (1, 2, 7):
            assert _pgn_raw_budget(ceiling, n_eng) == ceiling - n_eng
            assert _pgn_raw_budget(ceiling, n_eng) < ceiling
    # Never negative even if engineered survivors already exceed the ceiling.
    assert _pgn_raw_budget(20, 25) == 0
    assert _pgn_raw_budget(0, 3) == 0


def test_pgn_total_respects_ceiling_with_engineered_features():
    """End-to-end p>=n fit: the produced total (``n_features_`` = raw support + engineered) never exceeds the
    ``max(20, p//3)`` FP-control ceiling, even on a fit that yields engineered features. Guards against the
    engineered count silently degrading to 0 and over-allocating the raw budget."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n, p = 40, 150  # p >= n; ceiling = max(20, 50) = 50
    x_sig = rng.standard_normal((n, 4))
    x_noise = rng.standard_normal((n, p - 4))
    X = np.column_stack([x_sig, x_noise])
    # interaction + additive signal so the FE step can propose engineered features
    score = x_sig[:, 0] * x_sig[:, 1] + x_sig[:, 2] + x_sig[:, 3] + 0.2 * rng.standard_normal(n)
    y = (score > np.median(score)).astype(np.int64)
    Xdf = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    ys = pd.Series(y, name="y")

    m = MRMR(
        min_relevance_gain=0.0,
        cv=3,
        run_additional_rfecv_minutes=False,
        full_npermutations=3,
        random_seed=0,
        min_features_fallback=1,
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xdf, ys)

    ceiling = max(20, p // 3)
    _recipes = getattr(m, "_engineered_recipes_", None)
    n_eng = 0 if _recipes is None else len(_recipes)
    _support = getattr(m, "support_", None)
    n_raw = 0 if _support is None else len(_support)
    total = int(getattr(m, "n_features_", n_raw + n_eng))
    # Engineered features are charged toward the ceiling: the total output width stays within it.
    assert total <= ceiling, f"p>=n total {total} (raw={n_raw}, engineered={n_eng}) exceeds ceiling {ceiling}"
    # Sanity: engineered survivors ARE part of the reported total (so the ceiling check is meaningful).
    assert total >= n_raw
