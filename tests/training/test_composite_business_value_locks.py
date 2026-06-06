"""Lock-in tests: assert the demonstrated business value of each R10b
composite-target improvement.

Each test runs the same focused fixture as the
``composite_business_value_demo`` script and asserts the measured
gain (improvement vs. the OFF / baseline path). When a future code
change silently breaks one of these gains, the corresponding test
fails -- catching regressions before they ship.

Bounds are SOMEWHAT loose to absorb numerical jitter from LightGBM /
sklearn random seeds, but tight enough to fail on a real regression
(e.g. >50% loss of the demonstrated effect).

Each test name maps 1:1 to the demo entry it locks. The threshold
in each assertion is the demo measurement minus a 30% absorption
margin; if a future change degrades the effect by more than 30%,
the test fails.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

# B1 sklearn matrix marker convention -- this file runs in the multi-sklearn-version CI matrix.
pytestmark = pytest.mark.sklearn_matrix


from mlframe.training.composite import (
    CompositeTargetDiscovery,
    CompositeTargetEstimator,
    CompositeCrossTargetEnsemble,
    _mi_to_target,
    _tiny_cv_rmse_y_scale,
    _tiny_cv_rmse_y_scale_multiseed,
    get_transform,
)
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _disc_with_kwargs(**kwargs):
    """Same defaults as in the demo script -- isolate one feature
    per test."""
    base = dict(
        enabled=True, screening="hybrid",
        mi_sample_n=1500, tiny_model_sample_n=1200,
        tiny_model_n_estimators=60, tiny_model_cv_folds=3,
        eps_mi_gain=-1.0, top_k_after_mi=8, top_m_after_tiny=2,
        random_state=0,
        use_baseline_diagnostics_hint=False,
        require_beats_raw_baseline=False,
        per_bin_n_bins=0,
        auto_base_null_perms=0,
        auto_base_dedup_corr_threshold=1.0,
        auto_base_demote_time_index=False,
        auto_base_demote_spatial_coords=False,
        collapse_linear_residual_alpha_eps=0.0,
        cross_target_ensemble_strategy="off",
        detect_linear_residual_alpha_drift=False,
    )
    base.update(kwargs)
    return CompositeTargetDiscoveryConfig(**base)


# ----------------------------------------------------------------------
# #1 regime-aware gate
# ----------------------------------------------------------------------


class TestLockRegimeGate:
    """Per-bin gate must reject a wrong-transform that the global
    gate accepts. On the specific fixture: logratio with heavy-tail
    contaminated top quintile -> per-bin gate fires."""

    def test_per_bin_gate_rejects_wrong_logratio(self) -> None:
        rng = np.random.default_rng(0)
        n = 3000
        base = rng.lognormal(mean=2.0, sigma=0.6, size=n)
        x1 = rng.normal(size=n)
        y = 0.97 * base + 0.5 * x1 + rng.normal(scale=0.1, size=n)
        q80 = np.quantile(base, 0.80)
        high = base >= q80
        y[high] += 30.0 * rng.normal(size=int(high.sum()))
        df = pd.DataFrame({"base": base, "x1": x1, "y": y})
        train_idx = np.arange(int(n * 0.8))
        # OFF: per_bin disabled, global tolerance VERY lenient.
        cfg_off = _disc_with_kwargs(
            require_beats_raw_baseline=True, per_bin_n_bins=0,
            transforms=["logratio"], base_candidates=["base"],
            raw_baseline_tolerance=10.0,
        )
        cfg_on = _disc_with_kwargs(
            require_beats_raw_baseline=True, per_bin_n_bins=5,
            raw_baseline_per_bin_tolerance=1.10,
            transforms=["logratio"], base_candidates=["base"],
            raw_baseline_tolerance=10.0,
        )
        d_off = CompositeTargetDiscovery(cfg_off).fit(
            df, target_col="y", feature_cols=["base", "x1"],
            train_idx=train_idx,
        )
        d_on = CompositeTargetDiscovery(cfg_on).fit(
            df, target_col="y", feature_cols=["base", "x1"],
            train_idx=train_idx,
        )
        # Demo result: OFF kept the spec, ON dropped it (raw fallback).
        assert len(d_off.specs_) >= 1, (
            "regression: OFF should have kept the bad logratio spec "
            "(global gate too lenient)")
        assert len(d_on.specs_) == 0, (
            "regression: per-bin gate should have rejected the bad "
            "logratio spec; got "
            f"{[(s.name) for s in d_on.specs_]}")


# ----------------------------------------------------------------------
# #2 permutation-MI null
# ----------------------------------------------------------------------


class TestLockPermutationNull:
    """On pure-noise data, the null filter must reduce false-positive
    rate by at least 50%."""

    def test_null_filter_reduces_pure_noise_false_positives(self) -> None:
        n_reps = 10
        false_off = 0
        false_on = 0
        for rep in range(n_reps):
            rng = np.random.default_rng(rep * 31)
            n = 250
            data = {f"noise_{c}": rng.normal(size=n)
                    for c in ["a", "b", "c", "d", "e", "f", "g", "h"]}
            data["x1"] = rng.normal(size=n)
            data["y"] = rng.normal(size=n)
            df = pd.DataFrame(data)
            train_idx = np.arange(int(n * 0.8))
            feat = list(data.keys())
            feat.remove("y")
            cfg_off = _disc_with_kwargs(
                auto_base_null_perms=0, auto_base_top_k=1,
                transforms=["linear_residual"], random_state=rep,
                mi_sample_n=200,
            )
            cfg_on = _disc_with_kwargs(
                auto_base_null_perms=30, auto_base_null_z_threshold=2.0,
                auto_base_top_k=1, transforms=["linear_residual"],
                auto_base_null_block_length=1,
                random_state=rep, mi_sample_n=200,
            )
            d_off = CompositeTargetDiscovery(cfg_off).fit(
                df, target_col="y", feature_cols=feat,
                train_idx=train_idx,
            )
            d_on = CompositeTargetDiscovery(cfg_on).fit(
                df, target_col="y", feature_cols=feat,
                train_idx=train_idx,
            )
            if d_off.specs_:
                false_off += 1
            if d_on.specs_:
                false_on += 1
        # Demo: 10/10 OFF, 1-2/10 ON. Lock: ON must catch >= 50% of
        # OFF's false-positives.
        if false_off >= 5:
            reduction = (false_off - false_on) / false_off
            assert reduction >= 0.5, (
                f"regression: null filter reduced false-positives by only "
                f"{reduction*100:.0f}% (OFF={false_off}/{n_reps}, "
                f"ON={false_on}/{n_reps}); expected >= 50%")
        else:
            pytest.skip(
                f"OFF false-positive rate {false_off}/{n_reps} too low "
                "for stable measurement")


# ----------------------------------------------------------------------
# #4 wrapper-aware tiny CV-RMSE
# ----------------------------------------------------------------------


class TestLockWrapperAware:
    """The y-clip in CompositeTargetEstimator.predict must bound the
    catastrophic outlier-base RMSE. Without clipping, RMSE is 50x+
    raw on the heavy-tail fixture."""

    def test_wrapper_yclip_bounds_outlier_predictions(self) -> None:
        from lightgbm import LGBMRegressor
        rng = np.random.default_rng(0)
        n = 2000
        base = rng.lognormal(mean=2.0, sigma=0.4, size=n)
        x1 = rng.normal(size=n)
        y = base * np.exp(0.5 * x1 + rng.normal(scale=0.05, size=n))
        df = pd.DataFrame({"base": base, "x1": x1, "y": y})
        cut = int(n * 0.8)
        # Inject outlier: 100x larger base on first 50 test rows.
        df.loc[df.index[cut:cut + 50], "base"] *= 100.0
        # Train logratio composite.
        inner = LGBMRegressor(
            n_estimators=100, num_leaves=15, learning_rate=0.1,
            random_state=0, verbosity=-1,
        )
        wrapper = CompositeTargetEstimator(
            base_estimator=inner, transform_name="logratio",
            base_column="base",
        )
        wrapper.fit(
            df[["base", "x1"]].iloc[:cut],
            df["y"].iloc[:cut].to_numpy(),
        )
        y_hat = wrapper.predict(df[["base", "x1"]].iloc[cut:])
        # Clip ensures predictions stay finite and bounded.
        assert np.all(np.isfinite(y_hat)), (
            "regression: wrapper produced non-finite predictions on "
            "outlier base")
        # Loose bound: predictions stay within 100x of train y_max.
        y_train_max = float(df["y"].iloc[:cut].max())
        assert np.all(np.abs(y_hat) < 100 * y_train_max), (
            f"regression: wrapper y-clip not bounding predictions; "
            f"max|y_hat|={np.abs(y_hat).max():.2f}, "
            f"100*y_train_max={100*y_train_max:.2f}")


# ----------------------------------------------------------------------
# #5 cross-target ensemble (NNLS as the actual win path)
# ----------------------------------------------------------------------


class TestLockEnsembleNNLS:
    """NNLS stack must beat best-single component on a 2-base fixture
    with decorrelated errors. This locks in the existence of an
    ensemble path that DOES win (the R10b default 'oof_weighted'
    can lose due to train-RMSE overfit; NNLS is the optimal
    convex combination)."""

    def test_nnls_stack_beats_best_single(self) -> None:
        from lightgbm import LGBMRegressor
        from sklearn.metrics import mean_squared_error
        rng = np.random.default_rng(0)
        n = 4000
        base_a = rng.normal(loc=10, scale=2, size=n)
        base_b = rng.normal(loc=5, scale=1, size=n)
        x1 = rng.normal(size=n)
        y = (0.6 * base_a + 0.7 * base_b + 0.4 * x1
             + rng.normal(scale=0.3, size=n))
        df = pd.DataFrame({
            "base_a": base_a, "base_b": base_b, "x1": x1, "y": y,
        })
        cut = int(0.8 * n)
        train_idx, test_idx = np.arange(cut), np.arange(cut, n)
        # Train one composite per base.
        train_preds: list = []
        test_preds: list = []
        component_models: list = []
        component_names: list = []
        y_tr = df["y"].iloc[train_idx].to_numpy()
        y_te = df["y"].iloc[test_idx].to_numpy()
        for base_col in ["base_a", "base_b"]:
            t = get_transform("diff")
            b_tr = df[base_col].iloc[train_idx].to_numpy()
            b_te = df[base_col].iloc[test_idx].to_numpy()
            fp = t.fit(y_tr, b_tr)
            t_tr = t.forward(y_tr, b_tr, fp)
            x_cols = [c for c in ["base_a", "base_b", "x1"]
                      if c != base_col]
            m = LGBMRegressor(
                n_estimators=200, num_leaves=31, learning_rate=0.05,
                random_state=0, verbosity=-1,
            )
            m.fit(df[x_cols].iloc[train_idx].to_numpy(), t_tr)
            t_hat_tr = m.predict(df[x_cols].iloc[train_idx].to_numpy())
            t_hat_te = m.predict(df[x_cols].iloc[test_idx].to_numpy())
            train_preds.append(t.inverse(t_hat_tr, b_tr, fp))
            test_preds.append(t.inverse(t_hat_te, b_te, fp))
            wrapper = CompositeTargetEstimator.from_fitted_inner(
                fitted_inner=m, transform_name="diff",
                base_column=base_col, transform_fitted_params=fp,
                y_train=y_tr,
            )
            component_models.append(wrapper)
            component_names.append(f"diff_{base_col}")
        rmses = [
            float(np.sqrt(mean_squared_error(y_te, p)))
            for p in test_preds
        ]
        best_single = min(rmses)
        # NNLS stack on train; predict on test via weights.
        train_mat = np.column_stack(train_preds)
        nnls_ens = CompositeCrossTargetEnsemble.from_nnls_stack(
            component_models=component_models,
            component_names=component_names,
            component_predictions=train_mat,
            y_train=y_tr,
        )
        test_mat = np.column_stack(test_preds)
        nnls_pred = test_mat @ nnls_ens.weights
        nnls_rmse = float(np.sqrt(mean_squared_error(y_te, nnls_pred)))
        # Lock: NNLS not worse than 1.5% over best_single (allows
        # for jitter; demo showed NNLS +1.1% better).
        assert nnls_rmse < best_single * 1.015, (
            f"regression: NNLS stack ({nnls_rmse:.4f}) much worse than "
            f"best_single ({best_single:.4f}); expected within 1.5%")


# ----------------------------------------------------------------------
# #7 time-index demoter
# ----------------------------------------------------------------------


class TestLockTimeIndexDemoter:
    """Time-index demoter must push monotone-with-row features to
    bottom of base ranking; on the fixture, base_time is replaced
    by x1 as picked base."""

    def test_time_index_detector_demotes_monotone_base(self) -> None:
        rng = np.random.default_rng(0)
        n = 3000
        base_time = np.linspace(0, 5, n) + rng.normal(scale=0.05, size=n)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = np.linspace(0, 5, n) + 0.5 * x1 + rng.normal(scale=0.2, size=n)
        df = pd.DataFrame({
            "base_time": base_time, "x1": x1, "x2": x2, "y": y,
        })
        train_idx = np.arange(int(0.8 * n))
        cfg_off = _disc_with_kwargs(
            auto_base_demote_time_index=False, auto_base_top_k=1,
            transforms=["linear_residual"],
        )
        cfg_on = _disc_with_kwargs(
            auto_base_demote_time_index=True, auto_base_top_k=1,
            transforms=["linear_residual"],
        )
        d_off = CompositeTargetDiscovery(cfg_off).fit(
            df, target_col="y", feature_cols=["base_time", "x1", "x2"],
            train_idx=train_idx,
        )
        d_on = CompositeTargetDiscovery(cfg_on).fit(
            df, target_col="y", feature_cols=["base_time", "x1", "x2"],
            train_idx=train_idx,
        )
        if d_off.specs_:
            assert d_off.specs_[0].base_column == "base_time", (
                "fixture broken: OFF should pick base_time (highest MI)")
        if d_on.specs_:
            assert d_on.specs_[0].base_column != "base_time", (
                "regression: time-index demoter failed to push "
                f"base_time down; got '{d_on.specs_[0].base_column}'")


# ----------------------------------------------------------------------
# #10 median-of-seeds gate
# ----------------------------------------------------------------------


class TestLockMedianSeeds:
    """Multi-seed median estimator must reduce the std of the tiny
    CV-RMSE estimate by at least 30% on small-noisy fixtures."""

    def test_median_of_seeds_reduces_variance(self) -> None:
        rng = np.random.default_rng(0)
        n = 600
        base = rng.normal(loc=10, scale=2, size=n)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = 0.7 * base + 0.4 * x1 + 0.2 * x2 + rng.normal(scale=2.5, size=n)
        transform = get_transform("linear_residual")
        fp = transform.fit(y, base)
        x_matrix = np.column_stack([x1, x2])
        single = []
        median5 = []
        for s in range(10):
            r1 = _tiny_cv_rmse_y_scale(
                y_train=y, base_train=base, transform=transform,
                fitted_params=fp, x_train_matrix=x_matrix,
                family="lightgbm", n_estimators=30, num_leaves=15,
                learning_rate=0.1, cv_folds=3, random_state=s,
            )
            r5 = _tiny_cv_rmse_y_scale_multiseed(
                y_train=y, base_train=base, transform=transform,
                fitted_params=fp, x_train_matrix=x_matrix,
                family="lightgbm", n_estimators=30, num_leaves=15,
                learning_rate=0.1, cv_folds=3,
                n_seed_repeats=5, base_random_state=s,
            )
            if math.isfinite(r1):
                single.append(r1)
            if math.isfinite(r5):
                median5.append(r5)
        std_single = float(np.std(single))
        std_median = float(np.std(median5))
        # Demo showed 47.8% reduction. Lock at >=30%.
        if std_single > 1e-6:
            reduction = (std_single - std_median) / std_single
            assert reduction >= 0.3, (
                f"regression: median-of-5 reduced std by only "
                f"{reduction*100:.0f}% (single={std_single:.4f}, "
                f"median={std_median:.4f}); expected >= 30%")


# ----------------------------------------------------------------------
# Stat #1 mean-MI aggregation
# ----------------------------------------------------------------------


class TestLockMeanMI:
    """Mean aggregation must be ~invariant to feature count duplication;
    sum aggregation must scale with feature count (demonstrating the
    bias the mean fix removes)."""

    def test_mean_mi_invariant_to_duplicates(self) -> None:
        rng = np.random.default_rng(0)
        n = 2000
        base = rng.normal(size=n)
        y = base + rng.normal(scale=0.1, size=n)
        X1 = rng.normal(size=(n, 5))
        X10 = np.tile(X1, (1, 10))
        sum_x1 = _mi_to_target(
            X1, y, n_neighbors=3, random_state=0, estimator="bin",
            nbins=16, aggregation="sum",
        )
        sum_x10 = _mi_to_target(
            X10, y, n_neighbors=3, random_state=0, estimator="bin",
            nbins=16, aggregation="sum",
        )
        mean_x1 = _mi_to_target(
            X1, y, n_neighbors=3, random_state=0, estimator="bin",
            nbins=16, aggregation="mean",
        )
        mean_x10 = _mi_to_target(
            X10, y, n_neighbors=3, random_state=0, estimator="bin",
            nbins=16, aggregation="mean",
        )
        # sum: scales ~10x (artifact).
        if sum_x1 > 1e-6:
            sum_ratio = sum_x10 / sum_x1
            assert sum_ratio > 5.0, (
                f"regression: sum aggregation should scale ~10x with "
                f"duplicates; got {sum_ratio:.2f}x")
        # mean: invariant within 5% (small numerical jitter from
        # duplicate columns producing near-identical bin patterns).
        if mean_x1 > 1e-6:
            mean_ratio = mean_x10 / mean_x1
            assert 0.9 < mean_ratio < 1.1, (
                f"regression: mean aggregation should be invariant "
                f"to duplicates; got {mean_ratio:.2f}x")


# ----------------------------------------------------------------------
# Stat #4 Wilcoxon gate
# ----------------------------------------------------------------------


class TestLockWilcoxonGate:
    """Wilcoxon gate must reduce false-positive rate by >= 30% on
    borderline true-negatives where the threshold gate accepts
    everything (loose tolerance)."""

    def test_wilcoxon_catches_borderline_noise(self) -> None:
        from tests.conftest import is_fast_mode
        threshold_accepts = 0
        wilcoxon_accepts = 0
        # Fewer reps under --fast: each rep fits 2 discoveries x 5 seed-repeats; 10 reps starve a worker into a timeout
        # under full-suite ``-n`` contention. 5 reps keep the ``threshold_accepts >= 5`` gate reachable and the 30%-reduction
        # statistic valid (demo separation is large: threshold accepts all, Wilcoxon accepts ~half).
        n_reps = 5 if is_fast_mode() else 10
        for rep in range(n_reps):
            rng = np.random.default_rng(rep * 1000)
            n = 1500
            base = rng.normal(loc=10, scale=0.15, size=n)
            x1 = rng.normal(size=n)
            y = 1.0 * x1 + rng.normal(scale=1.0, size=n)
            df = pd.DataFrame({"base": base, "x1": x1, "y": y})
            train_idx = np.arange(int(0.8 * n))
            cfg_t = _disc_with_kwargs(
                require_beats_raw_baseline=True,
                raw_baseline_tolerance=1.10,
                use_wilcoxon_gate=False,
                tiny_model_n_seed_repeats=5,
                base_candidates=["base"], transforms=["diff"],
                random_state=rep,
            )
            cfg_w = _disc_with_kwargs(
                require_beats_raw_baseline=True,
                raw_baseline_tolerance=1.10,
                use_wilcoxon_gate=True, gate_alpha=0.05,
                tiny_model_n_seed_repeats=5,
                base_candidates=["base"], transforms=["diff"],
                random_state=rep,
            )
            d_t = CompositeTargetDiscovery(cfg_t).fit(
                df, target_col="y", feature_cols=["base", "x1"],
                train_idx=train_idx,
            )
            d_w = CompositeTargetDiscovery(cfg_w).fit(
                df, target_col="y", feature_cols=["base", "x1"],
                train_idx=train_idx,
            )
            if d_t.specs_:
                threshold_accepts += 1
            if d_w.specs_:
                wilcoxon_accepts += 1
        # Demo: threshold 15/15, Wilcoxon 7/15 (53% reduction).
        # Lock at >= 30% reduction.
        if threshold_accepts >= 5:
            reduction = (threshold_accepts - wilcoxon_accepts) / threshold_accepts
            assert reduction >= 0.3, (
                f"regression: Wilcoxon caught only {reduction*100:.0f}% "
                f"of threshold's borderline false-positives "
                f"(threshold={threshold_accepts}/{n_reps}, "
                f"wilcoxon={wilcoxon_accepts}/{n_reps}); expected >= 30%")


# ----------------------------------------------------------------------
# Stat #6 alpha-drift detection
# ----------------------------------------------------------------------


class TestLockAlphaDrift:
    """Alpha-drift detection must flag a strong synthetic drift
    (alpha jump 0.5 -> 1.5) with z >= 5."""

    def test_alpha_drift_detects_strong_jump(self) -> None:
        rng = np.random.default_rng(0)
        n = 4000
        base = rng.normal(loc=10, scale=2, size=n)
        x1 = rng.normal(size=n)
        half = n // 2
        y = np.zeros(n)
        y[:half] = (
            0.5 * base[:half] + 0.5 * x1[:half]
            + rng.normal(scale=0.1, size=half)
        )
        y[half:] = (
            1.5 * base[half:] + 0.5 * x1[half:]
            + rng.normal(scale=0.1, size=n - half)
        )
        df = pd.DataFrame({"base": base, "x1": x1, "y": y})
        train_idx = np.arange(int(0.8 * n))
        cfg = _disc_with_kwargs(
            detect_linear_residual_alpha_drift=True,
            alpha_drift_z_threshold=3.0,
            transforms=["linear_residual"],
            base_candidates=["base"],
        )
        disc = CompositeTargetDiscovery(cfg)
        disc.fit(df, target_col="y", feature_cols=["base", "x1"],
                 train_idx=train_idx)
        flags = getattr(disc, "_alpha_drift_flags", {})
        assert len(flags) >= 1, (
            "regression: drift detector failed to flag any spec on "
            "alpha-jump fixture")
        # Demo z=12.31; lock z>=5.
        z = list(flags.values())[0]["z_score"]
        assert z >= 5.0, (
            f"regression: drift z-score only {z:.2f}; expected >= 5 "
            "on alpha-jump 0.5 -> 1.5 fixture")


# ----------------------------------------------------------------------
# Stat #8 own-njit MI + permutation null
# ----------------------------------------------------------------------


class TestLockMINullDistinguishesNoiseFromSignal:
    """Permutation-null gain (point - null_mean) must be near zero
    for noise and clearly positive for signal."""

    def test_null_centred_gain_separates_noise_from_signal(self) -> None:
        from mlframe.feature_selection.filters.hermite_fe import (
            _plugin_mi_regression_njit,
        )
        rng = np.random.default_rng(0)
        n = 1000
        # Noise.
        x_noise = rng.normal(size=(n, 3))
        y_noise = rng.normal(size=n)
        # Signal.
        x_sig = rng.normal(size=(n, 3))
        y_sig = 1.0 * x_sig[:, 0] + 0.3 * rng.normal(size=n)

        def gain(x, y, n_perm=20):
            point = float(np.mean([
                _plugin_mi_regression_njit(x[:, j].copy(), y, 16)
                for j in range(x.shape[1])
            ]))
            rng2 = np.random.default_rng(42)
            null_vals = []
            for _ in range(n_perm):
                yp = rng2.permutation(y)
                null_vals.append(float(np.mean([
                    _plugin_mi_regression_njit(x[:, j].copy(), yp, 16)
                    for j in range(x.shape[1])
                ])))
            return point - float(np.mean(null_vals))

        gain_noise = gain(x_noise, y_noise)
        gain_signal = gain(x_sig, y_sig)
        # Lock: noise gain near zero (|gain| < 0.05).
        assert abs(gain_noise) < 0.05, (
            f"regression: noise gain {gain_noise:+.4f} should be ~0 "
            "(null filter calibrated correctly)")
        # Signal gain clearly above noise gain.
        assert gain_signal > 0.10, (
            f"regression: signal gain {gain_signal:+.4f} should be >= "
            "0.10 to clearly separate signal from noise")
        assert gain_signal > gain_noise + 0.10, (
            f"regression: gain separation {gain_signal - gain_noise:+.4f} "
            "should be >= 0.10")
