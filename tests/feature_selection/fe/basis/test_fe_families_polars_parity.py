"""Every MRMR FE family runs on polars input, not just pandas.

Before the matrix-native FE seam, ~25 FE families guarded on ``isinstance(X, pd.DataFrame)`` and SILENTLY SKIPPED on a
polars frame -- so a polars-native suite ran with most of the FE arsenal disabled. The seam (``fe_decide_on_subsample`` +
``_fe_frame_ops``) makes each family format-agnostic: the pandas-native decision body is fed a pandas view of the frame
(``fe_to_pandas`` -- identity for pandas, subsample/full bridge for polars) and the engineered columns are appended back in
the source framework (``fe_append_columns``). Each case below pins that a polars fit engineers the SAME columns as pandas
and selects identically -- the fix is a no-op on selection, only removing the format restriction.
"""
import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


@pytest.fixture(autouse=True)
def _clear_mrmr_fit_cache():
    # The process-wide fit memo is keyed by X-content hash; a sibling test that fit the SAME synthetic could otherwise
    # replay a cached selection here and make the pandas/polars A/B compare against a stale entry. Drain it per test.
    MRMR._FIT_CACHE.clear()
    yield
    MRMR._FIT_CACHE.clear()

_FEATURE_LIST_ATTRS = (
    "hybrid_orth_features_", "mi_greedy_features_", "kfold_te_features_",
    "count_encoding_features_", "frequency_encoding_features_", "cat_num_interaction_features_",
    "pairwise_ratio_features_", "pairwise_log_ratio_features_", "grouped_delta_features_",
    "lagged_diff_features_",
)


def _engineered(sel):
    out = []
    for a in _FEATURE_LIST_ATTRS:
        out.extend(getattr(sel, a, None) or [])
    return sorted(map(str, set(out)))


def _quadratic(seed, n=1500):
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2, x3 = rng.standard_normal(n), rng.standard_normal(n)
    noise = rng.standard_normal((n, 3))
    # Pure symmetric univariate quadratic: raw x1 is ~uninformative (corr ~0), the He2 basis carries the signal, so the
    # univariate-basis / scorer families must engineer x1__He2 to recover it.
    y = (x1 * x1 > 1.0).astype(int)
    data = {"x1": x1, "x2": x2, "x3": x3,
            "n0": noise[:, 0], "n1": noise[:, 1], "n2": noise[:, 2]}
    return data, y


def _three_way(seed, n=2000):
    rng = np.random.default_rng(seed)
    x1, x2, x3 = (rng.standard_normal(n) for _ in range(3))
    noise = rng.standard_normal((n, 2))
    y = (np.sign(x1 * x2 * x3) > 0).astype(int)
    data = {"x1": x1, "x2": x2, "x3": x3, "n0": noise[:, 0], "n1": noise[:, 1]}
    return data, y


def _diff_pair(seed, n=2000):
    # Correlated pair (|corr| ~ 0.99) with a NONLINEAR signal in the residual diff, so the diff-basis family engineers it.
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = x1 + 0.15 * rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    noise = rng.standard_normal((n, 2))
    y = (((x1 - x2) * 7.0) ** 2 > 0.5).astype(int)
    data = {"x1": x1, "x2": x2, "x3": x3, "n0": noise[:, 0], "n1": noise[:, 1]}
    return data, y


def _cluster(seed, n=2000):
    # A 3-member correlated cluster whose SQUARED aggregate carries the signal, so the cluster-basis family engineers it.
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = x1 + 0.1 * rng.standard_normal(n)
    x3 = x1 + 0.1 * rng.standard_normal(n)
    noise = rng.standard_normal((n, 2))
    y = (((x1 + x2 + x3) / 3.0) ** 2 > 0.4).astype(int)
    data = {"x1": x1, "x2": x2, "x3": x3, "n0": noise[:, 0], "n1": noise[:, 1]}
    return data, y


def _ratio(seed, n=2000):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(1.0, 5.0, n)
    x2 = rng.uniform(1.0, 5.0, n)
    noise = rng.standard_normal((n, 2))
    y = ((x1 / x2) > 1.0).astype(int)
    data = {"x1": x1, "x2": x2, "n0": noise[:, 0], "n1": noise[:, 1]}
    return data, y


def _categorical(seed, n=1500):
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, 8, n)
    eff = np.array([-2, -1, 0, 1, 2, 3, 0, -3], dtype=float)[cat]
    num = rng.standard_normal(n)
    noise = rng.standard_normal((n, 2))
    y = (eff + 0.5 * num + rng.standard_normal(n) > 0).astype(int)
    data = {"cat": cat.astype("int64"), "num": num, "n0": noise[:, 0], "n1": noise[:, 1]}
    return data, y


def _hinge_target(seed, n=2000):
    # Clean MONOTONE slope-change (y depends on relu(x1 - tau)) so the held-out breakpoint search finds an UNAMBIGUOUS tau
    # -- a symmetric target gives no clean breakpoint and the SSE search near-ties become FP-noise-sensitive across formats.
    rng = np.random.default_rng(seed)
    x1, x2, x3 = (rng.standard_normal(n) for _ in range(3))
    noise = rng.standard_normal((n, 2))
    y = (0.2 * x1 + 3.0 * np.maximum(x1 - 0.4, 0.0) + 0.1 * x2 + 0.3 * rng.standard_normal(n) > 0.6).astype(int)
    data = {"x1": x1, "x2": x2, "x3": x3, "n0": noise[:, 0], "n1": noise[:, 1]}
    return data, y


def _cat_num(seed, n=2000):
    # Category MODULATES the numeric's slope (genuine cat x num interaction), so the residual-by-category encoding is engineered.
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, 6, n)
    slope = np.array([-3.0, -1.5, 0.0, 1.5, 3.0, -2.0])[cat]
    num = rng.standard_normal(n)
    noise = rng.standard_normal((n, 2))
    y = (slope * num + 0.2 * rng.standard_normal(n) > 0).astype(int)
    data = {"cat": cat.astype("int64"), "num": num, "n0": noise[:, 0], "n1": noise[:, 1]}
    return data, y


# family id -> (MRMR kwargs enabling ONLY this family, data-builder)
_SCORER_KW = dict(fe_univariate_basis_enable=True, fe_hybrid_orth_pair_enable=False,
                  fe_hybrid_orth_triplet_enable=False, fe_hybrid_orth_quadruplet_enable=False,
                  fe_hinge_enable=False, fe_kfold_te_enable=False,
                  fe_binned_numeric_agg_enable=False)

_CASES = {
    "adaptive_arity": (dict(fe_hybrid_orth_adaptive_arity_enable=True, **_SCORER_KW), _three_way),
    "adaptive_degree": (dict(fe_hybrid_orth_adaptive_degree_enable=True, **_SCORER_KW), _quadratic),
    "conditional_routing": (dict(fe_hybrid_orth_conditional_routing_enable=True, **_SCORER_KW), _quadratic),
    "diff_basis": (dict(fe_hybrid_orth_diff_basis_enable=True, **_SCORER_KW), _diff_pair),
    "cluster_basis": (dict(fe_hybrid_orth_cluster_basis_enable=True, **_SCORER_KW), _cluster),
    "bootstrap": (dict(fe_hybrid_orth_bootstrap_enable=True, **_SCORER_KW), _quadratic),
    "three_gate": (dict(fe_hybrid_orth_three_gate_enable=True, **_SCORER_KW), _quadratic),
    "ksg": (dict(fe_hybrid_orth_ksg_enable=True, **_SCORER_KW), _quadratic),
    "copula": (dict(fe_hybrid_orth_copula_enable=True, **_SCORER_KW), _quadratic),
    "dcor": (dict(fe_hybrid_orth_dcor_enable=True, **_SCORER_KW), _quadratic),
    "hsic": (dict(fe_hybrid_orth_hsic_enable=True, **_SCORER_KW), _quadratic),
    "jmim": (dict(fe_hybrid_orth_jmim_enable=True, **_SCORER_KW), _quadratic),
    "tc": (dict(fe_hybrid_orth_tc_enable=True, **_SCORER_KW), _quadratic),
    "cmim": (dict(fe_hybrid_orth_cmim_enable=True, **_SCORER_KW), _quadratic),
    "auto_scorer": (dict(fe_hybrid_orth_auto_scorer_enable=True, **_SCORER_KW), _quadratic),
    "ensemble": (dict(fe_hybrid_orth_ensemble_enable=True, **_SCORER_KW), _quadratic),
    "meta": (dict(fe_hybrid_orth_meta_enable=True, **_SCORER_KW), _quadratic),
    "mi_greedy": (dict(fe_mi_greedy_enable=True, **_SCORER_KW), _quadratic),
    "mi_greedy_cmi": (dict(fe_mi_greedy_cmi_enable=True, **_SCORER_KW), _quadratic),
    "hinge": (dict(fe_hinge_enable=True, fe_univariate_basis_enable=False,
                   fe_hybrid_orth_pair_enable=False, fe_hybrid_orth_triplet_enable=False,
                   fe_hybrid_orth_quadruplet_enable=False, fe_kfold_te_enable=False,
                   fe_binned_numeric_agg_enable=False), _hinge_target),
    "kfold_te": (dict(fe_kfold_te_enable=True, fe_kfold_te_cols=("cat",), **{k: v for k, v in _SCORER_KW.items() if k != "fe_kfold_te_enable"}), _categorical),
    "count_encoding": (dict(fe_count_encoding_enable=True, fe_count_encoding_cols=("cat",), **_SCORER_KW), _categorical),
    "frequency_encoding": (dict(fe_frequency_encoding_enable=True, fe_frequency_encoding_cols=("cat",), **_SCORER_KW), _categorical),
    "cat_num_interaction": (dict(fe_cat_num_interaction_enable=True, fe_cat_num_interaction_cat_cols=("cat",),
                                 fe_cat_num_interaction_num_cols=("num",), fe_local_mi_gate=False, **_SCORER_KW), _cat_num),
    "binned_numeric_agg": (dict(fe_binned_numeric_agg_enable=True, **{k: v for k, v in _SCORER_KW.items() if k != "fe_binned_numeric_agg_enable"}), _quadratic),
    "pairwise_ratio": (dict(fe_pairwise_ratio_enable=True, fe_pairwise_ratio_cols=("x1", "x2"), **_SCORER_KW), _ratio),
    "pairwise_log_ratio": (dict(fe_pairwise_log_ratio_enable=True, fe_pairwise_log_ratio_cols=("x1", "x2"), **_SCORER_KW), _ratio),
}


def _fit(data, y, kwargs, to_polars):
    if to_polars:
        pl = pytest.importorskip("polars")
        X = pl.DataFrame(data)
    else:
        X = pd.DataFrame(data)
    sel = MRMR(verbose=0, random_seed=0, **kwargs)
    sel.fit(X, pd.Series(y))
    return sel


def test_bucket_a_family_adds_no_whole_frame_to_pandas_on_polars(monkeypatch):
    """A CLOSED-FORM (bucket-A) FE family must decide via fe_decide_on_subsample (native subsample gather + per-column
    recipe replay) and NEVER whole-frame ``to_pandas`` a polars input. We force MRMR's polars->pandas FE bridge OFF so a
    NATIVE polars frame reaches the FE path, spy on ``pl.DataFrame.to_pandas`` heights, and assert the family adds ZERO
    full-height (== n) conversions over the family-off baseline. The pre-fix code (``family(fe_to_pandas(X), ...)``) copied
    the whole frame once per family -> this sensor fails on it. Only the small subsample gather (height <= subsample_n) is
    allowed. This is the key regression sensor that eager ``fe_to_pandas(X)`` is gone from bucket A."""
    pl = pytest.importorskip("polars")
    import mlframe.training.utils as U

    def _bridge_off(*_a, **_k):
        raise RuntimeError("polars->pandas FE bridge disabled for the spy")
    monkeypatch.setattr(U, "get_pandas_view_of_polars_df", _bridge_off)

    heights: list[int] = []
    orig_to_pandas = pl.DataFrame.to_pandas
    monkeypatch.setattr(pl.DataFrame, "to_pandas",
                        lambda self, *a, **k: (heights.append(self.height), orig_to_pandas(self, *a, **k))[1])

    rng = np.random.default_rng(0)
    n = 1500
    data = {"x1": rng.standard_normal(n), "x2": rng.standard_normal(n), "x3": rng.standard_normal(n)}
    y = pd.Series((data["x1"] ** 2 > 1.0).astype(int))

    def _full_count(kwargs):
        heights.clear()
        MRMR._FIT_CACHE.clear()
        sel = MRMR(verbose=0, random_seed=0, fe_check_pairs_subsample_n=500,
                   fe_univariate_basis_enable=True, **kwargs)
        sel.fit(pl.DataFrame(data), y)
        return sum(h == n for h in heights), sel

    base_full, _ = _full_count({})
    fam_full, sel_fam = _full_count({"fe_hybrid_orth_ksg_enable": True})

    assert fam_full == base_full, (
        f"bucket-A family added {fam_full - base_full} whole-frame to_pandas copy(ies) of the polars frame "
        f"(base={base_full}, with_family={fam_full}) -- it must subsample natively, never whole-copy"
    )
    assert getattr(sel_fam, "hybrid_orth_features_", None), "ksg engineered nothing -- fixture/routing broken"


@pytest.mark.parametrize("fam", list(_CASES))
def test_fe_family_runs_on_polars_and_matches_pandas(fam):
    pytest.importorskip("polars")
    kwargs, builder = _CASES[fam]
    # Deterministic per-family seed (never hash(fam) -- PYTHONHASHSEED randomises it, making the fixture non-reproducible).
    data, y = builder(list(_CASES).index(fam))

    sel_pd = _fit(data, y, kwargs, to_polars=False)
    sel_pl = _fit(data, y, kwargs, to_polars=True)

    eng_pd, eng_pl = _engineered(sel_pd), _engineered(sel_pl)
    sup_pd = sorted(map(str, sel_pd.support_))
    sup_pl = sorted(map(str, sel_pl.support_))

    # hinge defers its legs to support finalisation; cat_num_interaction's residual clears the FE seam (the family runs on
    # polars via fe_to_pandas + native append) but is dedup-dropped from the final engineered list on this synthetic for
    # BOTH formats. For these two the meaningful cross-format invariant is selection parity, not a non-empty engineered set.
    if fam == "hinge":
        # hinge legs surface via _hinge_features_ (deferred to support finalisation). Non-empty on polars proves the family
        # was NOT skipped; the clean monotone target makes the detected legs + selection deterministic across formats.
        legs_pd = sorted(map(str, getattr(sel_pd, "_hinge_features_", None) or []))
        legs_pl = sorted(map(str, getattr(sel_pl, "_hinge_features_", None) or []))
        assert legs_pd, "pandas baseline detected no hinge legs -- fixture does not trigger hinge"
        assert legs_pl, "polars input detected NO hinge legs -- the hinge family was SKIPPED (the bug this seam fixes)"
        assert legs_pl == legs_pd, f"hinge: polars legs diverged from pandas\n  pd={legs_pd}\n  pl={legs_pl}"
        assert sup_pl == sup_pd, f"hinge: polars vs pandas selection diverged\n  pd={sup_pd}\n  pl={sup_pl}"
        return
    if fam == "cat_num_interaction":
        assert sup_pl == sup_pd, f"{fam}: polars vs pandas selection diverged\n  pd={sup_pd}\n  pl={sup_pl}"
        return

    assert eng_pd, f"{fam}: pandas baseline engineered NO columns -- fixture does not trigger the family"
    assert eng_pl, f"{fam}: polars input engineered NO columns -- the FE family was SKIPPED (the bug this seam fixes)"
    assert eng_pl == eng_pd, f"{fam}: polars engineered columns diverged from pandas\n  pd={eng_pd}\n  pl={eng_pl}"
    assert sup_pl == sup_pd, f"{fam}: polars vs pandas selection diverged\n  pd={sup_pd}\n  pl={sup_pl}"
