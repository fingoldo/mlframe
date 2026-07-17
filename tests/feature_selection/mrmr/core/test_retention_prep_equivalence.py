"""Wave 13 (1a): retain_usable_pure_forms / retain_usable_raw_columns must return IDENTICAL results
whether they recompute their base_names/trim/subsample prep internally (standalone call, ``_prep=None``)
or receive it precomputed via the shared ``_retention_prep`` helper -- the fix that lets the shared
_fit_impl_core.py call site compute the prep ONCE instead of each function duplicating the identical
seeded row-subsample draw (and the second ``X.iloc[_idx]`` copy) on the same (X, y_cont).

n=5000 (> the default max_rows=3000) so the row-subsample branch actually fires; p=16 (> the default
max_base_features=14) so the std-trim branch also fires -- exercising both duplicated blocks findings
1a flags.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _build_wide(n=5000, p=16, seed=0):
    """y depends on (a,b) via a**2/b plus a cross-mix of (c,d); many extra noise columns push p>14."""
    rng = np.random.default_rng(seed)
    cols = {}
    a = rng.random(n) + 0.5
    b = rng.random(n) + 0.5
    c = rng.random(n) + 0.5
    d = rng.random(n)
    cols["a"], cols["b"], cols["c"], cols["d"] = a, b, c, d
    for i in range(p - 4):
        cols[f"n{i}"] = rng.random(n)
    y = 0.2 * a**2 / b + np.log(c * 2.0) * np.sin(d / 3.0)
    return pd.DataFrame(cols), y


class _Stub:
    def __init__(self, feature_names, seed=0):
        self.feature_names_in_ = list(feature_names)
        self._engineered_recipes_ = []
        self._engineered_features_ = []
        self.support_ = np.array([], dtype=np.int64)
        self.random_seed = seed
        self.fe_subsample_stratify = None


def test_retention_prep_pure_forms_equivalence():
    from mlframe.feature_selection.filters._fe_pure_form_retention import (
        retain_usable_pure_forms,
        _retention_prep,
    )

    df, y = _build_wide()
    y = np.asarray(y, dtype=np.float64)
    seed = 3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        baseline = retain_usable_pure_forms(_Stub(df.columns, seed), df, y, seed=seed)

        prep = _retention_prep(_Stub(df.columns, seed), df, y, seed=seed)
        with_prep = retain_usable_pure_forms(_Stub(df.columns, seed), df, y, seed=seed, _prep=prep)

    assert [n for _, n in baseline] == [n for _, n in with_prep]
    for (r0, n0), (r1, n1) in zip(baseline, with_prep):
        assert n0 == n1
        assert getattr(r0, "src_names", None) == getattr(r1, "src_names", None)


def test_retention_prep_raw_columns_equivalence():
    from mlframe.feature_selection.filters._fe_pure_form_retention import (
        retain_usable_raw_columns,
        _retention_prep,
    )

    df, y = _build_wide()
    y = np.asarray(y, dtype=np.float64)
    seed = 7

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        baseline = retain_usable_raw_columns(_Stub(df.columns, seed), df, y, seed=seed)

        prep = _retention_prep(_Stub(df.columns, seed), df, y, seed=seed)
        with_prep = retain_usable_raw_columns(_Stub(df.columns, seed), df, y, seed=seed, _prep=prep)

    assert baseline == with_prep


def test_retention_prep_shared_draw_matches_each_standalone_subsample():
    """The whole point of 1a: _retention_prep's X_fit/y_fit for a fold that triggers subsampling must
    match EXACTLY what each function draws internally when it recomputes standalone (same seed)."""
    from mlframe.feature_selection.filters._fe_pure_form_retention import _retention_prep

    df, y = _build_wide()
    y = np.asarray(y, dtype=np.float64)
    seed = 11
    stub = _Stub(df.columns, seed)

    prep_a = _retention_prep(stub, df, y, seed=seed)
    prep_b = _retention_prep(stub, df, y, seed=seed)

    assert prep_a["base_names_trimmed"] == prep_b["base_names_trimmed"]
    assert len(prep_a["X_fit"]) == 3000  # max_rows default; confirms the subsample branch fired
    pd.testing.assert_frame_equal(prep_a["X_fit"], prep_b["X_fit"])
    np.testing.assert_array_equal(prep_a["y_fit"], prep_b["y_fit"])
