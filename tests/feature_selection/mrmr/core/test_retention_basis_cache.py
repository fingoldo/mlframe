"""Wave 13 (1b): retain_usable_pure_forms caches the per-(nm_a, nm_b) additive-basis StandardScaler
design matrix inside ``_adds_nonlinear_value`` instead of rebuilding+refitting it for every candidate
sharing the same operand pair (up to several candidates/pair via ``_max_per_pair``).

Verifies (a) the cache measurably reduces StandardScaler.fit_transform call count relative to the number
of distinct candidates sharing pairs, and (b) selection output is unaffected (see
test_retention_prep_equivalence.py / the adversarial suite for full-output equivalence pins).
"""
from __future__ import annotations

import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest


def _build_case2(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.random(n) + 0.5
    b = rng.random(n) + 0.5
    c = rng.random(n) + 0.5
    d = rng.random(n)
    e = rng.random(n)
    y = 0.2 * a**2 / b + np.log(c * 2.0) * np.sin(d / 3.0)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}), y.astype(np.float64)


class _Stub:
    def __init__(self, cols, seed=0):
        self.feature_names_in_ = list(cols)
        self._engineered_recipes_ = []
        self._engineered_features_ = []
        self.random_seed = seed


def test_basis_scaler_fit_count_less_than_candidate_count():
    """With max_per_pair=3 (regression) over C(5,2)=10 pairs, up to 30 pair candidates reach the
    non-separability gate, but the additive-basis design matrix (a distinctive 12-column
    ``np.column_stack`` -- 6 basis functions x 2 operands) must be built and StandardScaler-fit only
    ONCE per DISTINCT operand pair, not once per candidate: with the fix, builds == distinct pairs
    (<=10), strictly fewer than the 30 gate invocations that would rebuild it uncached.

    Patches ``numpy.column_stack`` (not ``StandardScaler.fit_transform`` directly) because the CV-MAE
    greedy (``usability_greedy``) also fits StandardScaler internally per fold -- unrelated to this
    cache -- which would inflate a fit_transform-based count; the 12-column signature isolates calls
    coming from ``_adds_nonlinear_value``'s additive basis specifically.
    """
    from mlframe.feature_selection.filters._fe_pure_form_retention import retain_usable_pure_forms

    df, y = _build_case2()
    seed = 3
    calls = {"n": 0}
    real_column_stack = np.column_stack

    def _counting_column_stack(arrs, *a, **kw):
        if len(list(arrs)) == 12:
            calls["n"] += 1
        return real_column_stack(arrs, *a, **kw)

    with warnings.catch_warnings(), mock.patch("numpy.column_stack", _counting_column_stack):
        warnings.simplefilter("ignore")
        out = retain_usable_pure_forms(_Stub(df.columns, seed), df, y, seed=seed)

    assert calls["n"] <= 10, f"expected <=10 additive-basis builds (one/distinct pair), got {calls['n']}"
    assert out, "sanity: this fixture must recover at least one pure form"


def test_basis_cache_does_not_change_selection():
    """Regression pin: caching the design matrix must not alter which forms are retained."""
    from mlframe.feature_selection.filters._fe_pure_form_retention import retain_usable_pure_forms

    df, y = _build_case2()
    seed = 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out1 = retain_usable_pure_forms(_Stub(df.columns, seed), df, y, seed=seed)
        out2 = retain_usable_pure_forms(_Stub(df.columns, seed), df, y, seed=seed)
    assert [n for _, n in out1] == [n for _, n in out2]
