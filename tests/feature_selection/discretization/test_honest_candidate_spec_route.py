"""Per-candidate honest null uses the batched kernel, and returns the same exceedance count as the per-permutation categorise route.

``honest_candidate_nfailed`` used to re-bin the whole frame (one ``categorize_dataset`` call per permutation) for every candidate, because it
ignored the njit spec the screen floor already uses. The batched route must give identical counts and observed MI, and must not call the
categorisation from inside the candidate test at all.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _fixture():
    """Binary target driven by two of five continuous columns; NaN-free and small."""
    rng = np.random.default_rng(31)
    n = 240
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"].to_numpy() + 0.6 * X["f1"].to_numpy() + 0.3 * rng.standard_normal(n) > 0).astype(np.int64)
    return X, y


def _fit_capturing(monkeypatch):
    """Fit MRMR once; capture the inputs of every honest candidate test and the honest codes closure."""
    from mlframe.feature_selection.filters import _honest_mdlp_null as hmn
    from mlframe.feature_selection.filters.mrmr import MRMR

    calls: list = []
    real = hmn.honest_candidate_nfailed

    def _spy(honest_codes_fn, full_factors, factors_nbins, y_idx, col, n_permutations, seed, eval_rows=None):
        """Record each candidate test's arguments and result."""
        res = real(honest_codes_fn, full_factors, factors_nbins, y_idx, col, n_permutations, seed, eval_rows)
        calls.append(
            dict(
                fn=honest_codes_fn,
                full=np.asarray(full_factors).copy(),
                nb=np.asarray(factors_nbins).copy(),
                y_idx=int(y_idx),
                col=int(col),
                n_perm=int(n_permutations),
                seed=seed,
                eval_rows=None if eval_rows is None else np.asarray(eval_rows).copy(),
                res=res,
            )
        )
        return res

    from mlframe.feature_selection.filters import _confirm_predictor as cp

    monkeypatch.setattr(cp, "honest_candidate_nfailed", _spy)
    X, y = _fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(
            nbins_strategy="fayyad_irani", quantization_nbins=8, full_npermutations=5, baseline_npermutations=5, fe_max_steps=0, random_state=0, verbose=0
        )
        sel.fit(X, y)
    return calls


def test_candidate_nfailed_identical_on_both_routes(monkeypatch):
    """The batched route and the per-permutation route return the same (nfailed, observed) for every candidate test of a fit."""
    from mlframe.feature_selection.filters import _honest_mdlp_null as hmn

    calls = _fit_capturing(monkeypatch)
    assert calls, "a fit must run honest candidate tests"
    checked = 0
    for rec in calls:
        if rec["res"] is None or not hasattr(rec["fn"], "spec"):
            continue

        def _no_spec(perm, _fn=rec["fn"]):
            """The same closure without the spec attribute: forces the per-permutation categorise route."""
            return _fn(perm)

        old = hmn.honest_candidate_nfailed(_no_spec, rec["full"], rec["nb"], rec["y_idx"], rec["col"], rec["n_perm"], rec["seed"], rec["eval_rows"])
        new = rec["res"]
        assert old is not None and new is not None
        assert old[0] == new[0], f"nfailed diverged for col {rec['col']}: categorise={old[0]} batched={new[0]}"
        assert old[1] == new[1], f"observed MI diverged for col {rec['col']}"
        checked += 1
    assert checked > 0, "the batched route must serve at least one candidate test"


def test_candidate_test_is_served_by_the_batched_kernel(monkeypatch):
    """Every candidate honest test with a spec runs the batched kernel for its column, not a per-permutation re-binning."""
    from mlframe.feature_selection.filters import _honest_mdlp_null as hmn

    single_col_kernel_calls: list = []
    real_null = hmn._njit_null_columns

    def _spy_null(spec, factors_data, factors_nbins, y_idx, cols, n_permutations, seed_key, eval_rows=None):
        """Record the single-column kernel calls that serve a candidate test."""
        if len(cols) == 1:
            single_col_kernel_calls.append(int(cols[0]))
        return real_null(spec, factors_data, factors_nbins, y_idx, cols, n_permutations, seed_key, eval_rows)

    monkeypatch.setattr(hmn, "_njit_null_columns", _spy_null)
    calls = _fit_capturing(monkeypatch)
    served = [rec for rec in calls if rec["res"] is not None and hasattr(rec["fn"], "spec")]
    assert served, "a fit must run spec-backed candidate tests"
    assert len(single_col_kernel_calls) == len(
        served
    ), f"{len(served)} candidate tests with a spec, but the batched kernel served {len(single_col_kernel_calls)}"
