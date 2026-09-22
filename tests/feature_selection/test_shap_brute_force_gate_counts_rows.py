"""The automatic brute-force dispatch must account for the rows it will score.

The gate compared only the subset COUNT against 80M ("~16 s at ~5M subsets/s"), but the kernel scores every row for
every subset: measured 0.32M subsets/s at 200 rows and 0.06M at 5000. A HybridSelector fit on a 1300-row interaction
bed spent 207 of its 245 s in an exhaustive search the gate believed would take seconds.
"""

from __future__ import annotations

import types

import pytest

from mlframe.feature_selection.shap_proxied_fs._shap_proxied_methods import ShapProxiedMethodsMixin
from mlframe.feature_selection.shap_proxied_fs._shap_proxied_resolvers import _resolve_brute_force_work_budget


def _resolver():
    return ShapProxiedMethodsMixin._resolve_optimizer


def _cfg(**kw):
    base = dict(optimizer="auto", brute_force_max_features=28, min_features=1, max_features=None, use_gpu=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_many_rows_route_a_large_search_to_beam():
    resolve = _resolver()
    assert resolve(_cfg(), 26, n_rows=1300) == "beam", "2^26 subsets x 1300 rows is minutes of work"


def test_few_rows_keep_the_exhaustive_search():
    resolve = _resolver()
    assert resolve(_cfg(), 18, n_rows=1300) == "bruteforce"


def test_the_subset_count_gate_still_applies_without_rows():
    """Legacy callers that pass no row count keep the old behaviour."""
    resolve = _resolver()
    assert resolve(_cfg(), 26) == "bruteforce"
    assert resolve(_cfg(), 27) == "beam"


def test_an_explicit_optimizer_is_never_overridden():
    resolve = _resolver()
    assert resolve(_cfg(optimizer="bruteforce"), 26, n_rows=100_000) == "bruteforce"


def test_the_budget_can_be_raised_per_host(monkeypatch):
    monkeypatch.setenv("MLFRAME_SHAP_BRUTE_FORCE_WORK_BUDGET", "1e12")
    assert _resolve_brute_force_work_budget() == 10**12
    resolve = _resolver()
    assert resolve(_cfg(), 26, n_rows=1300) == "bruteforce"
