"""TRAINING_COMPOSITE_CORE_A-3 regression test: factorize()'s no-pandas fallback must map NaN/null group
labels to code -1, matching the pandas branch's documented contract ("NaN / null labels map to code -1
(excluded downstream)").

The bug (fixed): the no-pandas fallback used plain np.unique(..., return_inverse=True), which gives NaN
its own real code (NaN sorts to the end of a float array and forms its own unique group) instead of -1 --
a downstream group report in the fallback-only (no-pandas) path would silently treat a NaN group as real
data.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite._composite_report_shared import factorize

pytestmark = pytest.mark.fast


def test_pandas_branch_maps_nan_to_minus_one():
    """Sanity: the (default, pandas-available) branch already maps NaN to -1."""
    codes, uniq = factorize(np.array([1.0, 2.0, np.nan, 1.0, np.nan]))
    assert codes[2] == -1
    assert codes[4] == -1
    assert codes[0] == codes[3]
    assert np.nan not in uniq


def test_no_pandas_fallback_maps_nan_to_minus_one(monkeypatch):
    """The no-pandas fallback must map NaN to -1, same as the pandas branch."""
    import mlframe.training.composite._composite_report_shared as shared_mod

    monkeypatch.setattr(shared_mod, "_HAVE_PANDAS", False)
    codes, uniq = factorize(np.array([1.0, 2.0, np.nan, 1.0, np.nan]))
    assert codes[2] == -1, "NaN group should map to code -1 in the no-pandas fallback"
    assert codes[4] == -1
    assert codes[0] == codes[3], "both '1.0' entries should share the same non-negative code"
    assert codes[0] >= 0
    assert codes[1] >= 0
    assert not any(np.isnan(u) if isinstance(u, float) else False for u in uniq), "NaN must not appear in the returned unique labels"


def test_no_pandas_fallback_maps_none_to_minus_one_for_object_array(monkeypatch):
    """The no-pandas fallback must map a None/object-array null to -1 too."""
    import mlframe.training.composite._composite_report_shared as shared_mod

    monkeypatch.setattr(shared_mod, "_HAVE_PANDAS", False)
    arr = np.array(["a", "b", None, "a"], dtype=object)
    codes, uniq = factorize(arr)
    assert codes[2] == -1, "None group should map to code -1 in the no-pandas fallback"
    assert codes[0] == codes[3]
    assert None not in uniq


def test_no_pandas_fallback_no_nan_matches_pandas_branch(monkeypatch):
    """Sanity: with no NaN present, both branches produce the same grouping structure (codes may differ
    in numeric assignment since np.unique sorts while pd.factorize preserves first-appearance order, but
    the number of distinct groups and which entries share a code must match)."""
    import mlframe.training.composite._composite_report_shared as shared_mod

    arr = np.array([3.0, 1.0, 2.0, 1.0, 3.0])
    codes_pd, uniq_pd = factorize(arr)

    monkeypatch.setattr(shared_mod, "_HAVE_PANDAS", False)
    codes_np, uniq_np = factorize(arr)

    assert len(uniq_pd) == len(uniq_np) == 3
    # Same-group structure: entries 1 and 3 (both value 1.0) share a code in both branches.
    assert codes_pd[1] == codes_pd[3]
    assert codes_np[1] == codes_np[3]
    assert (codes_pd[0] != codes_pd[1]) == (codes_np[0] != codes_np[1])
