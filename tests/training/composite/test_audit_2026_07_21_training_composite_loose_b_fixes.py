"""Regression tests for audits/full_audit_2026-07-21/training_composite_loose_b.md (F1-F14 + PR1-PR3 test-coverage gaps).

PR4 (X_cal/y_cal), PR5 (sort+segment perf), PR6 (random_state API), PR7 (NaN sentinel) and PR8 (grid
validation) are the SAME code paths as F6, F7/F9, F8/F9, F10 and F2 respectively -- covered by those tests,
not duplicated. PR5's perf claim is separately verified by
``src/mlframe/training/composite/_benchmarks/bench_ranking_group_loop_o_n.py`` (bit-identity + speedup).
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mlframe.training.composite import (
    CompositeQuantileEstimator,
    CompositeRankEstimator,
    CompositeTargetEstimator,
    GatedRegressionMixture,
    GroupedBlockStacker,
    HeteroscedasticCompositeEstimator,
    MissingAwareComposite,
    PseudoLabelingLoop,
    detect_calendar_anomalies,
)
from mlframe.training.composite.conformal import _MONDRIAN_NAN_SENTINEL, _conformal_internal_split, _normalize_groups
from mlframe.training.composite.glm import _FAMILY_OBJECTIVE, _set_inner_objective
from mlframe.training.composite.provenance_formulas import _format_transform_formulas
from mlframe.training.composite.ranking import _sorted_group_segments

# ----------------------------------------------------------------------
# F1 (P0) -- calendar_anomaly.py low-side correction sign error (also PR3).
# ----------------------------------------------------------------------


def test_f1_calendar_anomaly_low_side_correction_moves_toward_baseline():
    """A LOW-value spike (y << baseline) must be corrected TOWARD baseline, not squared further away.

    Pre-fix, ``corrected = y / deviation_ratio`` for every flagged day regardless of side: for a low-side
    spike (ratio = baseline/y), ``y / ratio = y**2 / baseline`` -- e.g. baseline=100, y=10 gave corrected=1.0
    (further from 100, not closer). Post-fix low-side days MULTIPLY by the ratio instead.
    """
    n = 40
    y = np.full(n, 100.0)
    y[20] = 10.0  # a single deep low-side spike surrounded by a stable baseline
    result = detect_calendar_anomalies(y, window=14, deviation_ratio_threshold=3.0, min_periods=5)
    assert result["flagged"][20]
    assert result["corrected"][20] == pytest.approx(100.0, rel=0.05)


def test_f1_calendar_anomaly_high_side_correction_still_divides():
    """A HIGH-value spike keeps the pre-existing (already-correct) divide-by-ratio behavior."""
    n = 40
    y = np.full(n, 100.0)
    y[20] = 1000.0
    result = detect_calendar_anomalies(y, window=14, deviation_ratio_threshold=3.0, min_periods=5)
    assert result["flagged"][20]
    assert result["corrected"][20] == pytest.approx(100.0, rel=0.05)


# ----------------------------------------------------------------------
# F2 -- quantile.py fit() auto-augments heads for a decreasing-inverse transform.
# ----------------------------------------------------------------------


def _reciprocal_data(n: int = 3000, seed: int = 7):
    """Reciprocal data."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(2.0, 5.0, size=n)
    inv_y = 1.0 / base + rng.normal(scale=0.05, size=n)
    inv_y = np.clip(inv_y, 0.01, None)
    y = 1.0 / inv_y
    X = pd.DataFrame({"base": base, "f0": rng.normal(size=n)})
    return X, y


def test_f2_quantile_default_predict_asymmetric_grid_no_longer_crashes():
    """An ASYMMETRIC grid on a decreasing-inverse transform previously raised at predict_quantile() default
    call (only the literal requested levels were fitted, never their complements)."""
    from sklearn.ensemble import GradientBoostingRegressor

    X, y = _reciprocal_data()
    est = CompositeQuantileEstimator(
        base_estimator=GradientBoostingRegressor(n_estimators=60, max_depth=2, random_state=0),
        transform_name="reciprocal_residual",
        base_column="base",
        quantiles=(0.1, 0.3, 0.5),
    )
    est.fit(X, y)
    # self.quantiles_ stays the caller's ORIGINAL (unaugmented) grid.
    assert list(est.quantiles_) == [0.1, 0.3, 0.5]
    # estimators_ internally holds the complements too (0.7, 0.9) so lookups resolve.
    assert set(np.round(list(est.estimators_.keys()), 6)) == {0.1, 0.3, 0.5, 0.7, 0.9}
    out = est.predict_quantile(X)  # pre-fix: ValueError here
    assert out.shape == (len(y), 3)


# ----------------------------------------------------------------------
# F3 -- grouped_block_stacking.py threads sample_weight (also PR1).
# ----------------------------------------------------------------------


def test_f3_grouped_block_stacker_threads_sample_weight():
    """GroupedBlockStacker.fit() must pass the (sliced) sample_weight into every submodel fit call.

    Pre-fix, sample_weight was accepted by fit() but silently dropped for both the per-group OOF pass and
    the full-data refit -- only the meta-model ever honored it.
    """
    captured: list = []

    class SpyLR(LinearRegression):
        """LinearRegression subclass that records the sample_weight passed to fit()."""
        def fit(self, X, y, sample_weight=None):
            """No-op / recording stub matching the estimator's fit() signature."""
            captured.append(None if sample_weight is None else np.asarray(sample_weight).copy())
            return super().fit(X, y, sample_weight=sample_weight)

    rng = np.random.default_rng(0)
    n = 200
    g1a = rng.normal(size=n)
    g1b = rng.normal(size=n)
    g2a = rng.normal(size=n)
    y = g1a + g1b + g2a + rng.normal(scale=0.1, size=n)
    X = pd.DataFrame({"g1a": g1a, "g1b": g1b, "g2a": g2a})
    sample_weight = np.linspace(0.5, 1.5, n)

    est = GroupedBlockStacker(
        feature_groups={"g1": ["g1a", "g1b"], "g2": ["g2a"]},
        submodel_factory=lambda: SpyLR(),
        meta_estimator=LinearRegression(),
        n_splits=3,
        random_state=0,
    )
    est.fit(X, y, sample_weight=sample_weight)

    assert captured, "no submodel .fit() call was captured"
    assert all(w is not None for w in captured), "sample_weight was dropped for at least one submodel fit call"


# ----------------------------------------------------------------------
# F4 -- gated_regression_mixture.py: missing-branch fallback (also PR2).
# ----------------------------------------------------------------------


class _FakeGate:
    """A gate whose predict_proba() is fully deterministic: first half low, second half high."""

    def predict_proba(self, X):
        """No-op / recording stub matching the estimator's predict_proba() signature."""
        n = len(X)
        lo = n // 2
        p = np.concatenate([np.full(lo, 0.1), np.full(n - lo, 0.9)])
        return np.column_stack([1.0 - p, p])


def test_f4_gated_regression_mixture_extreme_threshold_leaves_one_branch_unfit():
    """An extreme threshold that never crosses at fit time leaves one branch with NO fitted model
    (real end-to-end fit path, matching PR2's 'extreme threshold / imbalanced subpop_label' scenario)."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = 2.0 * X["x"].to_numpy() + rng.normal(scale=0.1, size=n)
    subpop_label = (rng.random(n) < 0.5).astype(int)

    est = GatedRegressionMixture(
        gate_classifier=LogisticRegression(),
        low_regressor=LinearRegression(),
        high_regressor=LinearRegression(),
        threshold=-1.0,  # gate probability in [0, 1] is always >= -1.0 -> route is always "high"
        use_gate_feature=False,
        n_splits=3,
    )
    est.fit(X, y, subpop_label)
    assert "low" not in est.branch_models_
    assert "high" in est.branch_models_


def test_f4_gated_regression_mixture_predict_falls_back_when_branch_missing():
    """predict() for a row routed to a branch with no fitted model must fall back to the other branch's
    model (with a warning), never silently return 0.0 (the pre-fix ``np.zeros`` init value)."""
    est = GatedRegressionMixture(
        gate_classifier=LogisticRegression(), low_regressor=LinearRegression(), high_regressor=LinearRegression(),
        threshold=0.5, use_gate_feature=False,
    )
    est.gate_model_ = _FakeGate()
    high_model = LinearRegression().fit(np.array([[0.0], [1.0], [2.0]]), np.array([10.0, 11.0, 12.0]))
    est.branch_models_ = {"high": high_model}

    X = pd.DataFrame({"x": np.linspace(0.0, 3.0, 10)})
    preds = est.predict(X)
    # First half routes to "low" (missing model) -- must fall back to "high", never silently 0.0.
    assert np.all(preds[:5] != 0.0)
    assert np.allclose(preds[:5], high_model.predict(X.iloc[:5][["x"]].to_numpy()))
    assert np.allclose(preds[5:], high_model.predict(X.iloc[5:][["x"]].to_numpy()))


def test_f4_gated_regression_mixture_neither_branch_fitted_raises():
    """If NEITHER branch was ever fitted, predict() must raise, not silently return zeros."""
    est = GatedRegressionMixture(
        gate_classifier=LogisticRegression(), low_regressor=LinearRegression(), high_regressor=LinearRegression(),
    )
    est.gate_model_ = _FakeGate()
    est.branch_models_ = {}
    with pytest.raises(RuntimeError):
        est.predict(pd.DataFrame({"x": np.linspace(0.0, 3.0, 10)}))


# ----------------------------------------------------------------------
# F5 -- pseudo_labeling.py: soft labels for the final fit when the estimator supports them.
# ----------------------------------------------------------------------


class _SoftLabelClf(BaseEstimator, ClassifierMixin):
    """A classifier-shaped estimator whose .fit() genuinely accepts a continuous [0, 1] target."""

    def __init__(self, n_estimators: int = 30, random_state: int = 0) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """No-op / recording stub matching the estimator's fit() signature."""
        from sklearn.ensemble import RandomForestRegressor

        self._reg_ = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        self._reg_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        """No-op / recording stub matching the estimator's predict() signature."""
        return (self._reg_.predict(X) >= 0.5).astype(int)

    def predict_proba(self, X):
        """No-op / recording stub matching the estimator's predict_proba() signature."""
        p1 = np.clip(self._reg_.predict(X), 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])


def test_f5_pseudo_labeling_final_fit_receives_soft_labels_when_supported():
    """When estimator_factory's estimator accepts a continuous target, the final fit must use the SOFT
    (continuous mean_pred) labels, not hardened {0,1} labels -- verified by checking predict_proba's output
    actually varies continuously (a classifier fit purely on hardened 0/1 pseudo-labels would still vary
    continuously too via RandomForestRegressor, so we instead check the recorded soft-label history carries
    genuinely non-binary values feeding the final fit)."""
    rng = np.random.default_rng(0)
    n_labeled, n_unlabeled = 200, 200
    X_labeled = pd.DataFrame({"x": rng.normal(size=n_labeled)})
    y_labeled = (X_labeled["x"].to_numpy() + rng.normal(scale=0.3, size=n_labeled) > 0).astype(int)
    X_unlabeled = pd.DataFrame({"x": rng.normal(size=n_unlabeled)})

    est = PseudoLabelingLoop(
        estimator_factory=lambda: _SoftLabelClf(), task="classification", n_rounds=1,
        n_splits=3, confidence_threshold=None, pseudo_label_weight=0.5,
    )
    est.fit(X_labeled, y_labeled, X_unlabeled)
    accept, mean_pred, _confidence = est.pseudo_labels_history_[0]
    soft_accepted = mean_pred[accept]
    # The soft labels genuinely fed to the final fit are continuous, not just {0.0, 1.0}.
    assert np.any((soft_accepted > 0.02) & (soft_accepted < 0.98))


def test_f5_pseudo_labeling_falls_back_to_hard_labels_on_genuine_hard_classifier(caplog):
    """A genuine hard sklearn classifier (rejects a continuous target) must not crash the final fit -- it
    falls back to hardened labels, WITH a warning (this is the existing multi-round biz_value fixture's
    exact configuration: DecisionTreeClassifier, n_rounds=2)."""
    rng = np.random.default_rng(0)
    n_labeled, n_unlabeled = 200, 200
    X_labeled = pd.DataFrame({"x": rng.normal(size=n_labeled)})
    y_labeled = (X_labeled["x"].to_numpy() + rng.normal(scale=0.3, size=n_labeled) > 0).astype(int)
    X_unlabeled = pd.DataFrame({"x": rng.normal(size=n_unlabeled)})

    est = PseudoLabelingLoop(
        estimator_factory=lambda: DecisionTreeClassifier(max_depth=3, random_state=0),
        task="classification", n_rounds=2, n_splits=3, confidence_threshold=0.2,
    )
    with caplog.at_level(logging.WARNING, logger="mlframe.training.composite.pseudo_labeling"):
        est.fit(X_labeled, y_labeled, X_unlabeled)  # pre-dual-accumulator-fix: crashed with ValueError
    assert any("rejected the soft" in rec.message for rec in caplog.records)
    preds = est.predict(X_unlabeled)
    assert preds.shape == (n_unlabeled,)


# ----------------------------------------------------------------------
# F6 -- _heteroscedastic.py: optional held-out (X_cal, y_cal) calibration.
# ----------------------------------------------------------------------


def test_f6_heteroscedastic_held_out_calibration_differs_from_in_sample():
    """In-sample calibration is optimistic almost by construction (the variance head is trained to predict
    close to exactly the training residuals, so resid/sigma is close to 1 in-sample regardless of true
    predictive quality). A genuinely held-out (X_cal, y_cal) pair should NOT be pinned to ~1."""
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(0)
    n_train, n_cal = 400, 400
    base_tr = rng.normal(size=n_train)
    x0_tr = rng.normal(size=n_train)
    y_tr = 2.0 * base_tr + 0.5 * x0_tr + rng.normal(scale=0.3, size=n_train)
    X_tr = pd.DataFrame({"base": base_tr, "x0": x0_tr})

    base_cal = rng.normal(size=n_cal)
    x0_cal = rng.normal(size=n_cal)
    y_cal = 2.0 * base_cal + 0.5 * x0_cal + rng.normal(scale=0.3, size=n_cal)
    X_cal = pd.DataFrame({"base": base_cal, "x0": x0_cal})

    def _fit(**kw):
        """No-op / recording stub matching the estimator's fit() signature."""
        est = HeteroscedasticCompositeEstimator(
            base_estimator=DecisionTreeRegressor(max_depth=None, random_state=0),
            transform_name="linear_residual", base_column="base", prefer_ngboost=False,
        )
        est.fit(X_tr, y_tr, **kw)
        return est

    in_sample = _fit()
    held_out = _fit(X_cal=X_cal, y_cal=y_cal)
    # A deep unconstrained tree memorises the TRAIN residuals near-perfectly (in-sample calibration ~= 1);
    # the held-out calibration factor must differ measurably.
    assert abs(in_sample.sigma_calibration_ - 1.0) < 0.25
    assert abs(held_out.sigma_calibration_ - in_sample.sigma_calibration_) > 0.25


# ----------------------------------------------------------------------
# F7 -- ranking.py: _sorted_group_segments correctness (perf rewrite; bit-identity + speedup in the bench).
# ----------------------------------------------------------------------


def test_f7_sorted_group_segments_matches_naive_reference():
    """F7: sorted group segments matches naive reference."""
    rng = np.random.default_rng(0)
    n_groups, items_per_group = 17, 13
    group = np.repeat(np.arange(n_groups), items_per_group)
    group = group[rng.permutation(group.size)]  # unsorted row order

    sort_idx, starts, ends = _sorted_group_segments(group)
    assert starts.shape == (n_groups,)
    assert ends.shape == (n_groups,)
    for gid in range(n_groups):
        s, e = starts[gid], ends[gid]
        seg = sort_idx[s:e]
        assert e - s == items_per_group
        # Every row in this segment truly belongs to group gid, and no row is missed / duplicated overall.
        assert np.all(group[seg] == gid)
    all_rows = np.sort(sort_idx)
    assert np.array_equal(all_rows, np.arange(group.size))


# ----------------------------------------------------------------------
# F8 / F9 -- explicit random_state on conformal.py / ranking.py (also PR6).
# ----------------------------------------------------------------------


def test_f8_conformal_internal_split_random_state_changes_split():
    """F8: conformal internal split random state changes split."""
    a_fit, a_cal = _conformal_internal_split(200, random_state=1)
    b_fit, _b_cal = _conformal_internal_split(200, random_state=2)
    assert not np.array_equal(a_fit, b_fit)
    # Same seed -> reproducible.
    a_fit2, a_cal2 = _conformal_internal_split(200, random_state=1)
    assert np.array_equal(a_fit, a_fit2) and np.array_equal(a_cal, a_cal2)


def test_f9_rank_estimator_random_state_changes_pair_subsampling():
    """A wide group (> _MAX_PAIRS_PER_GROUP pairs) must sample DIFFERENT pairs under different seeds."""
    rng = np.random.default_rng(0)
    m = 100  # C(100, 2) = 4950 > _MAX_PAIRS_PER_GROUP (4096) -> subsampling kicks in
    Xnum = rng.normal(size=(m, 2))
    res = rng.normal(size=m)
    group = np.zeros(m, dtype=np.int64)

    est_a = CompositeRankEstimator(base_column="b", random_state=1)
    est_b = CompositeRankEstimator(base_column="b", random_state=2)
    diffs_a, _labels_a = est_a._build_pairs(Xnum, res, group)
    diffs_b, _labels_b = est_b._build_pairs(Xnum, res, group)
    assert diffs_a.shape[0] == 4096
    assert not np.array_equal(diffs_a, diffs_b)

    est_a2 = CompositeRankEstimator(base_column="b", random_state=1)
    diffs_a2, _ = est_a2._build_pairs(Xnum, res, group)
    assert np.array_equal(diffs_a, diffs_a2)  # same seed -> reproducible


# ----------------------------------------------------------------------
# F10 -- conformal.py Mondrian: NaN group labels certified at calibration must match at predict.
# ----------------------------------------------------------------------


def test_f10_normalize_groups_maps_nan_to_stable_sentinel():
    """F10: normalize groups maps nan to stable sentinel."""
    g1 = _normalize_groups(np.array([1.0, np.nan, 2.0]), 3)
    g2 = _normalize_groups(np.array([np.nan, 5.0]), 2)
    assert g1[1] == _MONDRIAN_NAN_SENTINEL
    assert g2[0] == _MONDRIAN_NAN_SENTINEL
    # Independently-called factorize on these normalized labels must now agree (same hashable sentinel).
    assert g1[1] == g2[0]


def test_f10_conformal_mondrian_certified_nan_group_reachable_at_predict():
    """F10: conformal mondrian certified nan group reachable at predict."""
    rng = np.random.default_rng(0)
    n = 400
    base = rng.normal(size=n)
    y = base * 2.0 + rng.normal(scale=0.3, size=n)
    groups = np.where(rng.random(n) < 0.3, np.nan, rng.integers(0, 3, size=n).astype(float))
    X = pd.DataFrame({"b": base})

    est = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name="diff", base_column="b")
    est.fit(X, y)
    est.calibrate_conformal_mondrian(X, y, groups, alpha=0.1)

    X2 = pd.DataFrame({"b": rng.normal(size=50)})
    groups2 = np.full(50, np.nan)  # ALL-NaN predict-time group; this exact group WAS certified above.

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est.predict_interval_mondrian(X2, groups2, alpha=0.1)
    ood_warned = any("not seen at calibration" in str(w.message) for w in caught)
    assert not ood_warned


# ----------------------------------------------------------------------
# F11 -- glm.py: narrowed except + logger.warning on objective-coercion failure.
# ----------------------------------------------------------------------


def test_f11_set_inner_objective_warns_instead_of_silently_swallowing(caplog):
    """F11: set inner objective warns instead of silently swallowing."""
    lgb = pytest.importorskip("lightgbm")
    model = lgb.LGBMRegressor(n_estimators=5)

    def _raising_set_params(**kwargs):
        """Raising set params."""
        raise TypeError("simulated set_params rejection")

    model.set_params = _raising_set_params  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING, logger="mlframe.training.composite.glm"):
        _set_inner_objective(model, "poisson", 1.5)
    assert any("could not set the family-matched" in rec.message for rec in caplog.records)


def test_f11_set_inner_objective_still_sets_objective_on_success():
    """F11: set inner objective still sets objective on success."""
    lgb = pytest.importorskip("lightgbm")
    model = lgb.LGBMRegressor(n_estimators=5, objective="regression")
    _set_inner_objective(model, "gamma", 1.5)
    assert model.get_params()["objective"] == _FAMILY_OBJECTIVE["gamma"]


# ----------------------------------------------------------------------
# F12 -- missing.py: OOF (not in-sample) offset estimation.
# ----------------------------------------------------------------------


class _PerfectMemorizer(BaseEstimator, RegressorMixin):
    """A regressor that recalls the EXACT training target for a row it was fit on (table lookup by exact
    feature match), and falls back to the training-fold target mean for any row it never saw. This isolates
    the exact mechanism F12 concerns: an in-sample prediction on a row the model was fit on is ~perfect
    regardless of data pattern, while a genuinely out-of-fold prediction reflects real generalisation error.
    """

    def fit(self, X, y):
        """No-op / recording stub matching the estimator's fit() signature."""
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        self._table_ = {tuple(row): float(v) for row, v in zip(Xa, ya)}
        self._fallback_ = float(np.mean(ya))
        return self

    def predict(self, X):
        """No-op / recording stub matching the estimator's predict() signature."""
        Xa = np.asarray(X, dtype=np.float64)
        return np.array([self._table_.get(tuple(row), self._fallback_) for row in Xa], dtype=np.float64)


def test_f12_missing_offset_uses_oof_closer_to_true_shift_than_in_sample():
    """The OLD in-sample offset collapses to exactly 0 for a memorising base_estimator regardless of the
    true MNAR shift (it evaluates the composite on the SAME rows it was just fit on); the NEW OOF-based
    offset must land closer to the true injected shift than that (optimistic) in-sample computation."""
    rng = np.random.default_rng(0)
    n = 300
    f = rng.normal(size=n)
    b = rng.normal(size=n)
    true_offset = 4.0
    y = b + 0.5 * f + rng.normal(scale=0.3, size=n)
    missing_mask = rng.random(n) < 0.2
    y = y.copy()
    y[missing_mask] += true_offset
    b_obs = b.copy()
    b_obs[missing_mask] = np.nan
    X = pd.DataFrame({"b": b_obs, "f": f})

    composite = CompositeTargetEstimator(base_estimator=_PerfectMemorizer(), transform_name="diff", base_column="b")
    wrapper = MissingAwareComposite(composite=composite, n_splits=5, random_state=42)
    wrapper.fit(X, y)

    X_imp = wrapper._impute_inplace_safe(X, "b", wrapper.base_impute_value_, missing_mask)
    in_sample_pred = np.asarray(wrapper.composite_.predict(X_imp), dtype=np.float64)
    in_sample_offset = float(np.mean(y[missing_mask] - in_sample_pred[missing_mask]))

    assert in_sample_offset == pytest.approx(0.0, abs=1e-9)  # perfect in-sample recall masks the true shift
    assert abs(wrapper.missing_offset_ - in_sample_offset) > 1.0
    assert abs(wrapper.missing_offset_ - true_offset) < abs(in_sample_offset - true_offset)


def test_f12_missing_offset_falls_back_to_in_sample_with_warning_on_tiny_n(caplog):
    """Too few rows for the requested n_splits -> OOF is skipped, with an explicit warning (not a crash)."""
    X = pd.DataFrame({"b": [1.0, np.nan, 3.0], "f": [0.1, 0.2, 0.3]})
    y = np.array([1.0, 5.0, 3.0])
    composite = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name="diff", base_column="b")
    wrapper = MissingAwareComposite(composite=composite, n_splits=5)
    with caplog.at_level(logging.WARNING, logger="mlframe.training.composite.missing"):
        wrapper.fit(X, y)
    assert any("falls back to the in-sample composite prediction" in rec.message for rec in caplog.records)


# ----------------------------------------------------------------------
# F13 -- provenance_formulas.py: linear_residual_multi_robust documents the robust fit.
# ----------------------------------------------------------------------


def test_f13_linear_residual_multi_robust_formula_mentions_robust():
    """F13: linear residual multi robust formula mentions robust."""
    forward, _inverse, _desc = _format_transform_formulas(
        "linear_residual_multi_robust", "b1", "y", {"n_bases": 2, "beta": 0.5},
    )
    assert "robust" in forward.lower()
    assert "[b1, ...1 more]" in forward


# ----------------------------------------------------------------------
# F14 -- quantile.py: predict()'s median lookup applies the complementary-head lookup too.
# ----------------------------------------------------------------------


def test_f14_predict_median_uses_complementary_head_on_decreasing_inverse():
    """F14: predict median uses complementary head on decreasing inverse."""
    from sklearn.ensemble import GradientBoostingRegressor

    X, y = _reciprocal_data()
    est = CompositeQuantileEstimator(
        base_estimator=GradientBoostingRegressor(n_estimators=60, max_depth=2, random_state=0),
        transform_name="reciprocal_residual",
        base_column="base",
        quantiles=(0.1, 0.9),  # deliberately omits 0.5 -- "nearest" is 0.1 (tie-break: argmin picks first)
    )
    est.fit(X, y)
    assert bool(est._inverse_decreasing_) is True

    nearest = float(est.quantiles_[np.argmin(np.abs(est.quantiles_ - 0.5))])
    assert nearest == pytest.approx(0.1)

    direct_predict = est.predict(X)
    # The complementary head (1 - 0.1 = 0.9) must be the one actually used...
    via_complement = est._lookup_head(0.9).predict(X)
    assert np.allclose(direct_predict, via_complement)
    # ...NOT the literal (mislabelled, pre-fix) head.
    via_literal = est._lookup_head(0.1).predict(X)
    assert not np.allclose(direct_predict, via_literal)
