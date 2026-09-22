"""The shared error-diagnostics prep must actually be shared between the two entry points.

Both callers sliced `y_true`/`y_pred` into fresh array objects before calling the cache, whose key is the `id()` of
those arrays, so two identical calls left two entries and zero hits: every run still paid both densifies while the
module docstring claimed the 0.6-5.1 s double densify was fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.reporting import _diagnostics_prep as prep
from mlframe.reporting.diagnostics_dispatch import _prepared_error_inputs


@pytest.fixture(autouse=True)
def _clean_cache():
    prep.clear_shared_error_prep()
    yield
    prep.clear_shared_error_prep()


def _bed(n: int = 200):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y_true = rng.normal(size=n)
    y_pred = y_true + rng.normal(0, 0.2, n)
    return df, y_true, y_pred


def _call(df, y_true, y_pred, n=None, builds=None):
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    n = n if n is not None else len(yt)
    if builds is not None:
        builds.append(1)
    return _prepared_error_inputs(df, yt[:n], yp[:n], "regression", 0, n, None, cache_key_arrays=(y_true, y_pred))


def test_two_entry_point_calls_build_once():
    df, y_true, y_pred = _bed()
    first = _call(df, y_true, y_pred)
    second = _call(df, y_true, y_pred)
    assert len(prep._PREP_CACHE) == 1, "the second call must hit, not add a second entry"
    assert first[2] is second[2], "both diagnostics must receive the SAME sub-frame object"


def test_a_different_row_count_is_a_different_entry():
    """The trimmed length changes the prep, so it must be part of the key rather than silently reused."""
    df, y_true, y_pred = _bed()
    a = _call(df, y_true, y_pred, n=200)
    b = _call(df, y_true, y_pred, n=100)
    assert len(a[0]) != len(b[0])
    assert len(prep._PREP_CACHE) == 2


def test_different_targets_do_not_share_a_prep():
    df, y_true, y_pred = _bed()
    other = y_pred + 1.0
    _call(df, y_true, y_pred)
    _call(df, y_true, other)
    assert len(prep._PREP_CACHE) == 2
