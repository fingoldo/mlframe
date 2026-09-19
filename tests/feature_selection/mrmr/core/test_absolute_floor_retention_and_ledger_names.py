"""The post-screen pure-form retention honours an explicit absolute relevance floor; the rejection ledger
resolves operand names when ``feature_names_in_`` is an ndarray.

The retention pass builds its own candidate pool outside the MI screen with a fixed 0.02 MI floor, so with
``min_relevance_gain_mode='absolute'`` and a floor no feature can clear, it still re-attached engineered pair
forms (e.g. ``sub(sqrt(f3),rint(row_extreme_top1_score))`` on a pure-noise target), and a caller asking for an
empty selection got one engineered column back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import MRMR
from mlframe.feature_selection.filters import _usability_aware_selection as _uas
from mlframe.feature_selection.filters._fe_rejection_ledger import _resolve_index_columns


def _frame(n: int = 300, seed: int = 0):
    """Regression frame with a pairwise interaction x0*x1 plus a linear x2 term."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"x{i}" for i in range(5)])
    y = 2.0 * X["x0"] * X["x1"] + X["x2"] + 0.1 * rng.standard_normal(n)
    return X, pd.Series(y)


def test_absolute_floor_reaches_retention_pool(monkeypatch):
    """Every engineered-form pool the retention builds under an absolute floor uses at least that floor."""
    seen: list[float] = []
    real = _uas.build_usability_candidate_pool

    def spy(*a, **k):
        """Record the mi_floor each engineered-candidate pool build receives, then delegate."""
        if k.get("max_pairs", 1) != 0:  # the raw-only pool (max_pairs=0) builds no engineered forms
            seen.append(float(k.get("mi_floor", 0.02)))
        return real(*a, **k)

    monkeypatch.setattr(_uas, "build_usability_candidate_pool", spy)
    X, y = _frame()
    MRMR(verbose=0, use_simple_mode=True, max_runtime_mins=0.3, min_relevance_gain_mode="absolute", min_relevance_gain=10.0, min_features_fallback=0).fit(X, y)
    assert seen, "retention pool was never built on this fixture; the test would pass vacuously"
    assert min(seen) >= 10.0, seen


def test_ledger_resolves_names_from_ndarray_feature_names():
    """The ledger maps operand indices to names from an ndarray ``feature_names_in_`` and marks engineered indices."""
    class _M:
        """Fitted-selector stand-in exposing only ``feature_names_in_``."""
        feature_names_in_ = np.array(["a", "b", "c"], dtype=object)

    df = pd.DataFrame({"operands": [(0, 2), np.array([1, 5]), None]})
    _resolve_index_columns(_M(), df)
    assert list(df["operand_names"]) == ["(a, c)", "(b, <engineered#5>)", ""]
