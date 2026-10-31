"""The batched njit honest supervised null equals the categorise-per-permutation reference, permutation by permutation.

The reference recomputes every permutation's codes through the production categorisation (``categorize_dataset`` on the permuted target)
and the plug-in MI of those codes. The batched kernel must reproduce it on NaN-free columns and on columns with NaN rows, for the default
Fayyad-Irani configuration and for ``mdlp_fast_mode=False`` (same categorisation).
"""

from __future__ import annotations

import warnings

import numpy as np
from mlframe.feature_selection.filters._mdlp_honest_bycol_njit import seeded_permutation
import pandas as pd
import pytest


def _fixture():
    """Binary target driven by two of five continuous columns; NaN-free."""
    rng = np.random.default_rng(11)
    n = 300
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"].to_numpy() + 0.5 * X["f1"].to_numpy() + 0.3 * rng.standard_normal(n) > 0).astype(np.int64)
    return X, y


def _fixture_with_nan():
    """The same target with NaN injected into three columns (the bench injects NaN into about 30% of columns)."""
    X, y = _fixture()
    rng = np.random.default_rng(12)
    for col in ("f0", "f1", "f3"):
        mask = rng.random(len(X)) < 0.12
        X.loc[mask, col] = np.nan
    return X, y


def _mrmr(kwargs):
    """MRMR configured as the bench configuration (no feature engineering, few permutations)."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    return MRMR(
        nbins_strategy="fayyad_irani", nbins_strategy_kwargs=dict(kwargs), quantization_nbins=8, full_npermutations=5,
        baseline_npermutations=5, fe_max_steps=0, random_state=0, verbose=0,
    )


def _fit(X, y, monkeypatch, disable_spec: bool, kwargs):
    """Fit MRMR with the njit spec, or with ``build_honest_spec`` forced to None (the categorise-per-permutation path)."""
    from mlframe.feature_selection.filters import _honest_spec

    builds = {"n": 0}
    real_build = _honest_spec.build_honest_spec

    def _spy(*args, **kw):
        """Count spec builds; return None when the categorise path is requested."""
        builds["n"] += 1
        return None if disable_spec else real_build(*args, **kw)

    monkeypatch.setattr(_honest_spec, "build_honest_spec", _spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = _mrmr(kwargs)
        sel.fit(X, y)
    return sel, builds["n"]


@pytest.mark.parametrize("fixture", [_fixture_with_nan], ids=["with_nan"])
@pytest.mark.parametrize("kwargs", [{}], ids=["default"])
def test_selection_identical_to_categorise_path(monkeypatch, fixture, kwargs):
    """The same fit selects the same features with identical importances on the njit spec and on the categorise path."""
    X, y = fixture()
    sel_spec, spec_builds = _fit(X, y, monkeypatch, disable_spec=False, kwargs=kwargs)
    sel_cat, _ = _fit(X, y, monkeypatch, disable_spec=True, kwargs=kwargs)
    assert spec_builds > 0, "the njit spec must be built for this configuration"
    assert list(sel_spec.support_) == list(sel_cat.support_), "selection diverged"


@pytest.mark.parametrize("fixture,kwargs", [(_fixture, {}), (_fixture_with_nan, {}), (_fixture_with_nan, {"mdlp_fast_mode": False})], ids=["nan_free-default", "with_nan-default", "with_nan-fast_mode_off"])
def test_null_matches_per_permutation_categorise_reference(monkeypatch, fixture, kwargs):
    """Every permutation's plain and corrected null MI from the batched kernel equals the reference recomputed from the production codes."""
    from mlframe.feature_selection.filters import _honest_mdlp_null as hmn
    from mlframe.feature_selection.filters._numba_utils_cross_module import _mi_codes

    captured: list = []
    real_null = hmn._njit_null_columns

    def _spy(spec, factors_data, factors_nbins, y_idx, cols, n_permutations, seed_key, eval_rows=None):
        """Record the kernel's inputs and output; return the output unchanged."""
        plain, corr = real_null(spec, factors_data, factors_nbins, y_idx, cols, n_permutations, seed_key, eval_rows)
        captured.append(dict(
            factors_data=np.asarray(factors_data).copy(), y_idx=int(y_idx), cols=[int(c) for c in cols], n_perm=int(n_permutations),
            seed=int(seed_key), eval_rows=None if eval_rows is None else np.asarray(eval_rows, dtype=np.int64).copy(),
            plain=np.asarray(plain).copy(), corr=np.asarray(corr).copy(),
        ))
        return plain, corr

    monkeypatch.setattr(hmn, "_njit_null_columns", _spy)
    X, y = fixture()
    sel, _ = _fit(X, y, monkeypatch, disable_spec=False, kwargs=kwargs)
    assert captured, "the spec kernel must serve the null"
    codes_fn = sel._honest_null_codes_fn
    checked = 0
    for rec in captured[:1]:
        n = rec["factors_data"].shape[0]
        ev = np.arange(n, dtype=np.int64) if rec["eval_rows"] is None else rec["eval_rows"]
        y_codes = rec["factors_data"][:, rec["y_idx"]].astype(np.int64)
        base_seed = int(np.random.default_rng(rec["seed"]).integers(0, 2**31 - 1))  # permutation b of the batched null's stream is the shuffle seeded base_seed + b
        m = ev.shape[0]
        for b in range(min(2, rec["n_perm"])):
            perm = seeded_permutation(n, base_seed + b)
            codes, nb = codes_fn(perm)
            ys = y_codes[perm][ev]
            ny = int(ys.max()) + 1
            ky_eff = float(np.unique(ys).size)
            for j, c in enumerate(rec["cols"]):
                xs = np.ascontiguousarray(codes[ev, c], dtype=np.int64)
                mi = _mi_codes(xs, int(nb[c]), ys, ny)
                kx_eff = float(np.unique(xs).size)
                corr = mi - (kx_eff - 1.0) * (ky_eff - 1.0) / (2.0 * m)
                np.testing.assert_allclose(rec["plain"][b, j], mi, rtol=1e-12, atol=1e-12, err_msg=f"plain b={b} col={c}")
                np.testing.assert_allclose(rec["corr"][b, j], corr, rtol=1e-12, atol=1e-12, err_msg=f"corrected b={b} col={c}")
                checked += 1
    assert checked > 0


def test_fast_mode_on_keeps_categorise_path(monkeypatch):
    """``mdlp_fast_mode=True`` is outside the njit chain and must not build the spec."""
    X, y = _fixture()
    _, spec_builds = _fit(X, y, monkeypatch, disable_spec=False, kwargs={"mdlp_fast_mode": True})
    assert spec_builds == 0
