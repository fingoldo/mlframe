"""The two error diagnostics prepared the same inputs twice, then densified the same frame twice.

``render_split_error_diagnostics`` and ``render_slice_finder_diagnostic`` are separate entry points that a
report calls one after the other with the same frame, targets, task and seed. Each independently computed
the per-row error, drew the same bounded sample from it, capped the same columns and gathered the same
rows -- and then handed its own copy of that sub-frame to a builder that densified it. The densify alone
was 0.29s on 100k x 200 all-numeric and 2.55s with twenty object columns.

Both caches are keyed on object IDENTITY, which is only sound because the entries hold weak references and
every hit is confirmed against them: a freed object's id is reused, and an unconfirmed hit would hand one
report's frame to another report's diagnostic.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pandas as pd
import pytest

from mlframe.reporting import _diagnostics_prep as prep_mod
from mlframe.reporting.charts import error_analysis as ea_mod
from mlframe.reporting.charts.error_analysis import _resolve_feature_matrix, clear_feature_matrix_cache


@pytest.fixture(autouse=True)
def _clean_caches():
    """A leaked entry would make the next test pass for the wrong reason."""
    prep_mod.clear_shared_error_prep()
    clear_feature_matrix_cache()
    yield
    prep_mod.clear_shared_error_prep()
    clear_feature_matrix_cache()


def _frame(n: int = 2_000, p: int = 12) -> pd.DataFrame:
    """A small mixed-dtype frame; the caches do not care about size, only identity."""
    rng = np.random.default_rng(0)
    data = {f"f{i}": rng.normal(size=n) for i in range(p - 2)}
    data["cat"] = rng.choice(list("abcd"), n)
    data["flag"] = rng.random(n) < 0.5
    return pd.DataFrame(data)


def test_the_densify_is_reused_for_the_same_frame():
    """The second diagnostic gets the first one's matrix instead of rebuilding it."""
    df = _frame()
    first, _ = _resolve_feature_matrix(df, None)
    second, _ = _resolve_feature_matrix(df, None)
    assert first is second, "the same frame was densified twice"


def test_a_different_frame_is_densified_again():
    """Guard: the cache must key on the frame, not simply return whatever it saw last."""
    a, _ = _resolve_feature_matrix(_frame(), None)
    b, _ = _resolve_feature_matrix(_frame(n=2_001), None)
    assert a is not b and a.shape != b.shape


def test_a_stale_entry_under_a_recycled_id_is_not_served():
    """The reason the entries hold weak references, exercised deliberately rather than by waiting for luck."""
    df = _frame()
    honest, _ = _resolve_feature_matrix(df, None)
    clear_feature_matrix_cache()

    dead = _frame(n=2_002)
    other = ea_mod._resolve_feature_matrix_uncached(dead, None)
    dead_ref = weakref.ref(dead)
    del dead
    gc.collect()
    ea_mod._MATRIX_CACHE[(id(df), None)] = (dead_ref, other)

    served, _ = _resolve_feature_matrix(df, None)
    assert served.shape == honest.shape, "a stale entry under a recycled id was served as this frame's matrix"


def test_the_matrix_cache_does_not_pin_the_frame():
    """A cache entry that keeps a 100GB frame alive is worse than the densify it saves."""
    df = _frame()
    _resolve_feature_matrix(df, None)
    ref = weakref.ref(df)
    del df
    gc.collect()
    assert ref() is None, "the densify cache is holding a strong reference to the caller's frame"


def test_the_prep_is_built_once_per_input_set():
    """Counted, not inferred from timing: the same inputs must prepare once."""
    calls = {"n": 0}

    def build():
        """Count how often the preparation actually runs."""
        calls["n"] += 1
        return (1, 2, 3, 4)

    df, yt, yp = _frame(), np.arange(5.0), np.arange(5.0)
    assert prep_mod.shared_error_prep(df, yt, yp, "regression", 0, build) == (1, 2, 3, 4)
    assert prep_mod.shared_error_prep(df, yt, yp, "regression", 0, build) == (1, 2, 3, 4)
    assert calls["n"] == 1, f"the preparation ran {calls['n']} times for one set of inputs"


@pytest.mark.parametrize("changed", ["task", "seed", "targets"])
def test_a_different_input_set_is_prepared_again(changed):
    """Guard: sharing must not leak one diagnostic's inputs into another's."""
    calls = {"n": 0}

    def build():
        """Count how often the preparation actually runs."""
        calls["n"] += 1
        return (calls["n"],)

    df, yt, yp = _frame(), np.arange(5.0), np.arange(5.0)
    prep_mod.shared_error_prep(df, yt, yp, "regression", 0, build)
    kwargs = {"task": "regression", "seed": 0, "y_true": yt}
    if changed == "task":
        kwargs["task"] = "classification"
    elif changed == "seed":
        kwargs["seed"] = 1
    else:
        kwargs["y_true"] = np.arange(6.0)
    prep_mod.shared_error_prep(df, kwargs["y_true"], yp, kwargs["task"], kwargs["seed"], build)
    assert calls["n"] == 2, f"a change of {changed} was served from the cache"


def test_the_prep_cache_does_not_pin_the_frame():
    """An entry that keeps the caller's frame alive is worse than the work it saves."""
    df = _frame()
    prep_mod.shared_error_prep(df, np.arange(5.0), np.arange(5.0), "regression", 0, lambda: (1,))
    ref = weakref.ref(df)
    del df
    gc.collect()
    assert ref() is None, "the prep cache is holding a strong reference to the caller's frame"
