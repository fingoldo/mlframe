"""MRMR engineered-recipe replay must WARN (not silently fall back to a nameless ndarray)
when the predict-time X cannot be wrapped in the fitted feature_names_in_ -- a silent
fallback resolves src-names-by-name recipes against an unnamed frame -> wrong columns."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

from mlframe.feature_selection.filters._mrmr_validate_transform import _append_engineered


def test_name_mismatch_warns_not_silent(caplog):
    # feature_names_in_ has 3 names but the predict ndarray has 2 columns -> pd.DataFrame
    # wrap raises -> pre-fix swallowed it silently; post-fix logs a WARNING naming the risk.
    self = SimpleNamespace(feature_names_in_=["a", "b", "c"])
    base_out = np.zeros((5, 1))
    X = np.zeros((5, 2))  # width mismatch vs feature_names_in_
    # A non-empty recipes list is needed to pass the empty-short-circuit and reach the wrap; the
    # warn fires during the wrap (before any recipe is applied), so a downstream apply error is fine.
    recipes = [SimpleNamespace(name="dummy", src_names=("a",), extra={"chain_lookups": []}, verbose=0)]
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters.mrmr"):
        try:
            _append_engineered(self, base_out, X, recipes=recipes)
        except Exception:
            pass
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "unnamed frame" in msgs or "could not wrap" in msgs
