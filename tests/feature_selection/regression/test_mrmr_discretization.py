"""Behavioural tests for MRMR discretisation, binning and bin-code dtype handling."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


def test_regression_adaptive_nbins_gate_restores_ctor_state_after_fit():
    """Pre-fix: MRMR.nbins_strategy/quantization_nbins stayed permanently mutated (None/20) after a fit
    that triggers the gate. Post-fix: restored to the constructor's original values, so get_params()/clone()
    round-trip correctly and a second .fit() re-evaluates the gate instead of being stuck.
    """
    from sklearn.base import clone

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=[f"x{i}" for i in range(4)])
    y = pd.Series(1.5 * X["x0"] - X["x1"] + 0.1 * rng.standard_normal(n))

    m = MRMR(
        nbins_strategy="mdlp",
        quantization_nbins=10,
        adaptive_nbins_large_n_reg=True,
        adaptive_nbins_large_n_reg_threshold=50,  # cheap trigger instead of the real 50_000 default
        fe_max_steps=0,
        full_npermutations=0,
        baseline_npermutations=0,
        random_seed=0,
        n_jobs=1,
        verbose=0,
        cv=2,
    )
    m.fit(X, y)
    assert getattr(m, "_adaptive_nbins_large_n_reg_fired_", False) is True, "gate must have engaged for this fit (n=300 >= threshold=50, detected regression)"
    assert m.nbins_strategy == "mdlp", "ctor state must be restored after fit(), not left mutated to None"
    assert m.quantization_nbins == 10, "ctor state must be restored after fit(), not left mutated to 20"

    # sklearn round-trip contract: get_params()/clone() must reproduce the constructor's own values.
    assert m.get_params()["nbins_strategy"] == "mdlp"
    assert m.get_params()["quantization_nbins"] == 10
    c = clone(m)  # must not raise, and must not silently inherit the gate's internal override
    assert c.nbins_strategy == "mdlp"
    assert c.quantization_nbins == 10

    # A second .fit() on the SAME instance must re-evaluate the gate (not be permanently stuck at
    # None/20 from the first call, which would silently and permanently freeze a config the gate's own
    # campaign data says loses at smaller n).
    m.fit(X, y)
    assert getattr(m, "_adaptive_nbins_large_n_reg_fired_", False) is True
    assert m.nbins_strategy == "mdlp"
    assert m.quantization_nbins == 10


# ---------------------------------------------------------------------------
# FIT_IMPL_B-2 (P1): the p>=n false-positive-control cap in _fit_impl_core.py
# was enforced exactly once, but usability-aware raw retention and raw-signal-
# retention augmentation run AFTER it and can each append more raw columns
# with no re-check, letting the final selected/n_features_ count silently
# exceed the documented max(20, p//3) ceiling.
# ---------------------------------------------------------------------------


def test_regression_mdlp_oos_validated_duplicate_rows_do_not_over_split():
    """Pre-fix: pure noise at dup_rate 0/10/50/90% gave 1/4/6/12 bins (over-split). Post-fix: collapses to
    1 bin at every duplication rate, matching the sibling mdlp_bin_edges_validated's behaviour."""
    from mlframe.feature_selection.filters._mdlp_validated_split import mdlp_bin_edges_oos_validated

    rng = np.random.default_rng(1)
    n_unique = 800
    x = rng.standard_normal(n_unique)
    y = rng.standard_normal(n_unique) * 1000.0
    for dup_rate in (0.0, 0.10, 0.50, 0.90):
        n_dup = round(dup_rate * n_unique)
        dup_idx = rng.integers(0, n_unique, n_dup) if n_dup else np.array([], dtype=np.int64)
        x_all = np.concatenate([x, x[dup_idx]])
        y_all = np.concatenate([y, y[dup_idx]])
        edges = mdlp_bin_edges_oos_validated(x_all, y_all, seed=1)
        n_bins = edges.size - 1
        assert n_bins <= 2, (dup_rate, n_bins)


# ---------------------------------------------------------------------------
# DISCRETIZATION-6 (P2): edges_optimal_joint masked rows by x's finiteness only, never independently
# checking y for NaN before _bin_y_for_mi's np.quantile call.
# ---------------------------------------------------------------------------


def test_regression_edges_optimal_joint_drops_nan_y_rows():
    """Pre-fix: a NaN-y row with finite x survived the mask and could propagate NaN into quantile edges.
    Post-fix: NaN-y rows are dropped too, same as NaN-x rows."""
    from mlframe.feature_selection.filters._adaptive_nbins import edges_optimal_joint

    rng = np.random.default_rng(0)
    n = 500
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    y[10] = np.nan  # finite x, NaN y -- the bug's exact scenario
    edges = edges_optimal_joint(x, y, n_splits=3, random_state=0)
    assert np.all(np.isfinite(edges)), f"NaN leaked into edges: {edges}"


# ---------------------------------------------------------------------------
# DISCRETIZATION-12 (P2): _mdlp_recurse_oos_validated's present/counts_parent branch handled a combination
# the recursive call sites never actually produce (present_parent set, counts_parent None) -- dead code.
# ---------------------------------------------------------------------------


def test_regression_mdlp_recurse_oos_validated_no_dead_present_parent_branch():
    """Post-fix: the dead present_parent-without-counts_parent branch is removed; passing both together
    (the only combination real call sites use) still works correctly."""
    from mlframe.feature_selection.filters._mdlp_validated_split import _mdlp_recurse_oos_validated

    rng = np.random.default_rng(0)
    n = 200
    x = np.sort(rng.standard_normal(n))
    y = (x > 0).astype(np.int64)
    splits: list = []
    counts_parent = np.bincount(y, minlength=2).astype(np.int64)
    present_parent = np.array([0, 1], dtype=np.int64)

    # Interleave rather than halve. `x` is SORTED, so `x[:100]` is the whole negative tail and `y[:100]` is
    # single-class -- the recursion correctly finds nothing to cut, which is why `splits` came back empty and
    # why passing it in and never looking at it hid that this call exercised no recursion at all. Odd/even
    # positions give both the train and the OOS slice both classes and a real boundary at x = 0.
    _tr = np.arange(0, n, 2)
    _oos = np.arange(1, n, 2)
    _mdlp_recurse_oos_validated(
        x[_tr],
        y[_tr],
        x[_oos],
        y[_oos],
        splits,
        0,
        5,
        8,
        0.3,
        counts_parent=counts_parent,
        present_parent=present_parent,
    )

    # `splits` was passed in and never inspected, so the "still works correctly" half of this test's own
    # docstring was unasserted -- and "does not raise" held before the dead-branch removal too, since the
    # combination exercised here (both counts_parent AND present_parent supplied) is the one that always worked.
    assert isinstance(splits, list), f"splits is not a list: {type(splits).__name__}"
    assert splits, "the MDLP recursion produced no cut points on a fixture with a clean class boundary at x=0"
    assert all(np.isfinite(c) for c in splits), f"non-finite cut point(s): {splits}"
    assert list(splits) == sorted(splits), f"cut points are not in ascending order: {splits}"
    assert len(set(splits)) == len(splits), f"duplicate cut point(s): {splits}"
    # The planted boundary is at x = 0; the recursion must find a cut near it rather than an arbitrary one.
    assert min(abs(float(c)) for c in splits) < 0.5, f"no cut point near the planted boundary at x=0: {splits}"


# ---------------------------------------------------------------------------
# DISCRETIZATION-13 (P2): cap_categorical_cardinality's "most frequent" selection used the non-stable
# default argsort, so two categories tied exactly at the cutoff boundary could swap non-deterministically
# across numpy versions/architectures.
# ---------------------------------------------------------------------------


def test_regression_cap_categorical_cardinality_stable_tie_break():
    """Post-fix: kind='stable' argsort on negated counts gives a deterministic, reproducible tie-break
    (lowest original index wins) for categories tied exactly at the cutoff."""
    from mlframe.feature_selection.filters.discretization import cap_categorical_cardinality

    # Exercises the PRODUCT, not numpy: asserting np.argsort(kind='stable') on its own only restates a numpy
    # guarantee and would keep passing if the function stopped using a stable sort altogether.
    # Codes 0 and 1 are tied at 10 occurrences, 3 has 20, 2 has 5. With a cap of 3 the two most frequent codes
    # keep distinct ids and the rest fold into the 'other' bucket, so the tie decides which of 0/1 survives:
    # a stable sort keeps the lower original code, deterministically, on every run and platform.
    col = np.array([3.0] * 20 + [0.0] * 10 + [1.0] * 10 + [2.0] * 5).reshape(-1, 1)
    capped = cap_categorical_cardinality(col, 3)

    # id 0 is the most frequent code (3), id 1 the tie-break winner (0), id 2 the folded 'other' bucket.
    assert capped[col[:, 0] == 3.0, 0].tolist() == [0.0] * 20
    assert capped[col[:, 0] == 0.0, 0].tolist() == [1.0] * 10, "the tie must resolve to the LOWER original code"
    assert set(capped[col[:, 0] == 1.0, 0].tolist()) == {2.0}, "the tie loser must fold into the other bucket"
    assert set(capped[col[:, 0] == 2.0, 0].tolist()) == {2.0}, "the rare code must fold into the other bucket"

    # Determinism: the same input must give the same assignment every call.
    np.testing.assert_array_equal(capped, cap_categorical_cardinality(col, 3))


# ---------------------------------------------------------------------------
# CLUSTERING_STABILITY-1 (P1): cluster_stability_selection / complementary_pairs_stability /
# stability_select_fe all silently returned an empty/"nothing is stable" result (only a logger.warning)
# when every single bootstrap failed, unlike the sibling StabilityMRMR.fit's post-B-14 RuntimeError.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["quantile", "uniform"])
def test_regression_binarize_aggregate_nan_gets_dedicated_code(method):
    """Pre-fix: a NaN row silently landed in a real bin (last bin for quantile, bin 0 for uniform).
    Post-fix: it gets the dedicated out-of-band code (n_bins), one past every real bin."""
    from mlframe.feature_selection.filters._dynamic_cluster_discovery._dcd_metrics import _binarize_aggregate

    rng = np.random.default_rng(0)
    values = rng.standard_normal(200)
    values[7] = np.nan
    n_bins = 5
    binned = _binarize_aggregate(values, method=method, n_bins=n_bins, dtype=np.int64)
    assert binned[7] == n_bins, f"method={method}: NaN row did not get the dedicated out-of-band code"
    assert np.all(binned[np.arange(200) != 7] < n_bins), f"method={method}: a real row leaked into the NaN code"


# ---------------------------------------------------------------------------
# CLUSTERING_STABILITY-8 (P2): ks_stability_filter's multi-split mode had no validation on split_frac; a
# caller-supplied split_frac > 1.0 crashed with an opaque numpy ValueError instead of a clear message.
# ---------------------------------------------------------------------------


def test_regression_score_plug_in_densifies_fractional_y_not_truncates():
    """Pre-fix: y=[0.1, 0.2] perfectly separated by x truncated both classes to 0 -> MI=0.0.
    Post-fix: densified via np.unique(return_inverse=True) -> a real, non-zero MI."""
    from mlframe.feature_selection.filters._orth_auto_scorer_fe import _score_plug_in

    rng = np.random.default_rng(0)
    n = 2000
    x = rng.standard_normal(n)
    y = np.where(x > 0, 0.1, 0.2)  # perfectly separated by x, but both truncate to class 0 pre-fix
    mi = _score_plug_in(x, y)
    assert mi > 0.1, f"expected a real, non-degenerate MI for a perfectly-separated fractional y, got {mi}"


# ---------------------------------------------------------------------------
# ORTH_SCORING_B-2 (P1): hybrid_orth_mi_auto_scorer_fe_with_recipes never froze the fit-time basis-
# preprocess params into emitted orth_univariate recipes -- the B-17 bug, reintroduced by the same
# split-before-fix blind spot as B-1.
# ---------------------------------------------------------------------------


def test_regression_kfold_target_encode_codes_rejects_n_folds_below_2():
    """Pre-fix: n_folds=0 raised a raw ZeroDivisionError; n_folds=1 silently produced a useless column.
    Post-fix: both raise a clear, named ValueError."""
    from mlframe.feature_selection.filters._cat_pair_fe import _kfold_target_encode_codes

    codes = np.array([0, 1, 0, 1, 2, 2], dtype=np.int64)
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    for bad_n_folds in (0, 1):
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            _kfold_target_encode_codes(codes, y, n_folds=bad_n_folds)


# ---------------------------------------------------------------------------
# CAT_INTERACTION_A-9 (P2): CatFEConfig.__post_init__ omitted range checks for 5 fields, letting e.g.
# bootstrap_ci_alpha=1.2 pass construction silently and later compute inverted CI bounds.
# ---------------------------------------------------------------------------


def test_regression_rolling_median_raises_clear_error_not_keyerror():
    """Pre-fix: stats=["median"] raised a bare KeyError deep in the call stack. Post-fix: a clear,
    actionable ValueError names the unsupported stat up front."""
    import pandas as pd

    from mlframe.feature_selection.filters._temporal_agg_fe_rolling import generate_rolling_window_agg_features

    X = pd.DataFrame(
        {
            "entity": [1, 1, 2, 2],
            "t": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]),
            "v": [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.raises(ValueError, match="median"):
        generate_rolling_window_agg_features(X, ["entity"], ["v"], "t", windows=("7D",), stats=["median"])


# ---------------------------------------------------------------------------
# CAT_INTERACTION_B-2 (P1): _global_value_for_stat's nunique/count fallback for an unseen group/composite
# key returned a whole-population-scale statistic instead of a per-group-scale one.
# ---------------------------------------------------------------------------


def test_regression_fit_impl_core_passes_random_seed_to_cmi_greedy(monkeypatch):
    """Post-fix: the fe_mi_greedy_cmi_enable call site threads self.random_seed through as the
    seed= kwarg to greedy_cmi_fe_construct_with_recipes, instead of leaving the hardcoded 0xC011
    default -- verified behaviorally (captured kwarg across two different random_seed values), not
    by matching source text (the call site has since moved out of _fit_impl_core.py during an
    unrelated monolith split, which broke the original inspect.getsource() assertion)."""
    import mlframe.feature_selection.filters._mi_greedy_cmi_fe as cmi_mod
    from mlframe.feature_selection.filters.mrmr import MRMR

    captured_seeds = []
    _orig = cmi_mod.greedy_cmi_fe_construct_with_recipes

    def _spy(*args, **kwargs):
        """Capture the seed= kwarg the call site passes, then delegate to the real implementation."""
        captured_seeds.append(kwargs.get("seed"))
        return _orig(*args, **kwargs)

    monkeypatch.setattr(cmi_mod, "greedy_cmi_fe_construct_with_recipes", _spy)

    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"x{i}": rng.standard_normal(200) for i in range(6)})
    y = (X["x0"] + X["x1"] > 0).astype(int).to_numpy()

    for seed in (11, 22):
        m = MRMR(
            verbose=0,
            max_runtime_mins=1,
            n_workers=1,
            quantization_nbins=5,
            use_simple_mode=True,
            random_seed=seed,
            fe_mi_greedy_cmi_enable=True,
        )
        m.fit(X, y)

    assert captured_seeds == [11, 22], f"seed= kwarg not threaded from self.random_seed at the fe_mi_greedy_cmi_enable call site: {captured_seeds}"


# ---------------------------------------------------------------------------
# MI_GREEDY_RECIPES-2 (P1): _greedy_score_and_select / greedy_mi_fe_construct / score_candidates_by_cmi /
# greedy_cmi_fe_construct all unconditionally `.astype(np.int64)` (TRUNCATING) any non-integer y before
# treating it as a classification label, destroying continuous-y signal confined to one integer bucket.
# ---------------------------------------------------------------------------


def _mi_greedy_densify_fixture():
    """Continuous y confined to [0, 1): a bare astype(np.int64) collapses it to one class, np.unique keeps every level."""
    rng = np.random.default_rng(0)
    n = 400
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    X = pd.DataFrame({"x0": x0, "x1": x1, "x2": rng.standard_normal(n)})
    y_cont = (x0 - x0.min()) / (x0.max() - x0.min()) * 0.999
    return X, y_cont


@pytest.mark.parametrize("entrypoint", ["score_candidates_by_cmi", "greedy_cmi_fe_construct", "greedy_mi_fe_construct", "_greedy_score_and_select"])
def test_regression_mi_greedy_functions_densify_not_truncate_y(entrypoint):
    """Post-fix: a continuous y in [0, 1) is densified via np.unique, not truncated to one integer class. MI-based
    scores depend on y only up to a bijective relabeling, so the result on the raw continuous y must equal the result
    on its densified integer codes. The pre-fix truncation collapses y to a single class and diverges from that."""
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe as cmi_mod
    from mlframe.feature_selection.filters import _mi_greedy_fe as greedy_mod

    X, y_cont = _mi_greedy_densify_fixture()
    y_dense = np.unique(y_cont, return_inverse=True)[1].astype(np.int64)

    def _run(y):
        """Return every output the entry point produces, so the comparison covers the scores, not just the augmented frame."""
        if entrypoint == "score_candidates_by_cmi":
            return [cmi_mod.score_candidates_by_cmi(X[["x0", "x1", "x2"]], y)]
        if entrypoint == "greedy_cmi_fe_construct":
            return list(cmi_mod.greedy_cmi_fe_construct(X, y))
        if entrypoint == "greedy_mi_fe_construct":
            return list(greedy_mod.greedy_mi_fe_construct(X, y))
        engineered = pd.DataFrame({"e_add": X["x0"] + X["x1"]})
        names, scores, _parsed = greedy_mod._greedy_score_and_select(
            X,
            engineered,
            [(("x0", "x1"), "add")],
            y,
            top_k=5,
            min_uplift=0.0,
            min_abs_mi_frac=0.0,
            nbins=10,
        )
        return [names, scores]

    out_cont = _run(y_cont)
    out_dense = _run(y_dense)
    assert len(out_cont) == len(out_dense)
    for a, b in zip(out_cont, out_dense):
        if isinstance(a, pd.DataFrame):
            pd.testing.assert_frame_equal(a, b)
        elif isinstance(a, pd.Series):
            pd.testing.assert_series_equal(a, b)
        else:
            assert a == b, f"{entrypoint}: outputs differ between continuous y and its densified codes"


def test_regression_compute_class_weights_rejects_huge_class_count(caplog, monkeypatch):
    """Post-fix: a y whose max value implies an enormous class count is rejected (logged, returns None)
    instead of allocating a bincount array of that size."""
    import logging

    from mlframe.feature_selection.filters._orthogonal_univariate_fe._imbalance_mi import compute_class_weights

    monkeypatch.setenv("MLFRAME_FE_IMBALANCE_MI", "on")
    y = np.array([0, 1, 2, 200_000], dtype=np.int64)
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters._orthogonal_univariate_fe._imbalance_mi"):
        result = compute_class_weights(y)
    assert result is None
    assert any("distinct integer values" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# X_EDGE_CASES_BEST_PRACTICES-4 (P2): residency_audit()'s monkeypatch/restore pair had no reentrancy
# guard -- two overlapping regions on different threads could silently corrupt each other's byte tally.
# ---------------------------------------------------------------------------


def test_regression_binned_numeric_agg_no_op_path_returns_same_object():
    """Direct pin: the no-candidate early-exit path returns the SAME X object (no copy), on a
    representative one of the 39 fixed constructors."""
    from mlframe.feature_selection.filters._binned_numeric_agg_fe import binned_numeric_agg_with_recipes

    rng = np.random.default_rng(0)
    n = 50
    # No numeric columns at all -> gcands/acands both empty -> hits the first no-op early exit.
    X = pd.DataFrame({"cat_only": pd.Categorical(rng.integers(0, 3, n).astype(str))})
    y = rng.standard_normal(n)
    X_out, appended, recipes = binned_numeric_agg_with_recipes(X, y)
    assert X_out is X
    assert appended == []
    assert recipes == []
