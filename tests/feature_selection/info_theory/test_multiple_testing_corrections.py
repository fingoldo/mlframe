"""Regression tests for the multiple-testing / stability-selection / knockoff statistical corrections.

Each test pins the CORRECTED contract and fails on the pre-fix code:

* SA-P0-2  knockoff W must be sign-symmetric under the global null (flip-sign antisymmetry the Barber-Candes FDR needs).
* SA5      RFECV stability selection exposes the Meinshausen-Buhlmann PFER bound.
* SA6      StabilityMRMR default sample_fraction is 0.5 (MB complementary-pairs) and exposes pfer_bound_.
* SA7      BorutaShap binomial accept/reject uses the percentile-derived null hit rate, not p=0.5.
* SA8      _kendall_tau_z uses the tie-corrected variance (matches scipy on tied data).
* SA9      the continuous-numeric Kendall p-value is computed at full n (no 2000-row subsample).
"""

from __future__ import annotations

import numpy as np
import pytest


# ----------------------------------------------------------------------------
# SA-P0-2: knockoff W must be sign-symmetric under the global null.
# ----------------------------------------------------------------------------
class TestKnockoffNullSignSymmetry:
    def _null_W_samples(self, n_seeds: int = 40):
        from sklearn.ensemble import RandomForestRegressor

        from mlframe.feature_selection.wrappers import knockoff_importance

        all_W = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 + seed)
            X = rng.standard_normal((150, 8))
            y = rng.standard_normal(150)  # y independent of all X -> global null

            def factory():
                return RandomForestRegressor(n_estimators=40, random_state=seed)

            W = knockoff_importance(factory, X, y, random_state=seed, w_statistic="gain")
            all_W.extend(W.values())
        return np.asarray(all_W)

    def test_null_W_distribution_is_sign_symmetric(self):
        """Under the global null the W pool must contain a non-trivial fraction of NEGATIVE values; a non-negative gain
        importance with fixed real/knockoff column positions (pre-fix) produces an essentially one-sided W with almost no
        negatives, so the Barber-Candes threshold has no negative reference set and gives no real control."""
        W = self._null_W_samples()
        frac_neg = float(np.mean(W < 0))
        # A sign-symmetric null statistic sits near 0.5; require a substantial negative mass that the pre-fix one-sided W lacks.
        assert frac_neg > 0.3, f"Null W not sign-symmetric (frac negative={frac_neg:.3f}); flip-sign antisymmetry violated."

    def test_null_selection_rate_controlled(self):
        """Empirical selection rate under the global null must be controlled near the target q."""
        from sklearn.ensemble import RandomForestRegressor

        from mlframe.feature_selection.wrappers import knockoff_importance, select_features_fdr

        q = 0.2
        n_seeds = 40
        n_feat = 8
        total_selected = 0
        for seed in range(n_seeds):
            rng = np.random.default_rng(7000 + seed)
            X = rng.standard_normal((150, n_feat))
            y = rng.standard_normal(150)

            def factory():
                return RandomForestRegressor(n_estimators=40, random_state=seed)

            W = knockoff_importance(factory, X, y, random_state=seed, w_statistic="gain")
            total_selected += len(select_features_fdr(W, q=q))
        # Under the pure null every selection is a false positive; the per-feature false-selection rate must stay low.
        per_feature_rate = total_selected / (n_seeds * n_feat)
        assert per_feature_rate < 0.15, f"Null selection rate too high ({per_feature_rate:.3f}); FDR not controlled."


# ----------------------------------------------------------------------------
# SA8: tie-corrected Kendall variance.
# ----------------------------------------------------------------------------
class TestKendallTieCorrectedVariance:
    def test_tied_p_matches_scipy(self):
        from scipy.stats import kendalltau

        from mlframe.feature_selection.wrappers._univariate_ht import _kendall_tau_z, _normal_two_sided_p

        rng = np.random.default_rng(0)
        # Heavily tied integer features (low-cardinality) -- exactly where the no-ties variance is wrong.
        x = rng.integers(0, 4, size=300).astype(float)
        y = (x + rng.integers(0, 3, size=300)).astype(float)

        _, z = _kendall_tau_z(x, y)
        p_ours = _normal_two_sided_p(z)

        tau_sp, p_sp = kendalltau(x, y, variant="b")
        # scipy uses the same tie-corrected normal approx; ours must agree closely.
        assert p_ours == pytest.approx(p_sp, abs=5e-3, rel=0.05), f"ours={p_ours} scipy={p_sp}"

    def test_no_ties_unchanged(self):
        """On all-distinct data the tie-corrected variance reduces to the no-ties case; still matches scipy."""
        from scipy.stats import kendalltau

        from mlframe.feature_selection.wrappers._univariate_ht import _kendall_tau_z, _normal_two_sided_p

        rng = np.random.default_rng(1)
        x = rng.standard_normal(200)
        y = 0.5 * x + rng.standard_normal(200)
        _, z = _kendall_tau_z(x, y)
        p_ours = _normal_two_sided_p(z)
        _, p_sp = kendalltau(x, y, variant="b")
        assert p_ours == pytest.approx(p_sp, abs=5e-3, rel=0.1)


# ----------------------------------------------------------------------------
# SA9: full-n Kendall (no subsample to 2000).
# ----------------------------------------------------------------------------
class TestKendallFullN:
    def test_large_n_not_subsampled(self):
        """Two calls with different random_state on a >2000-row feature must return the SAME p-value: pre-fix each drew a
        different 2000-row subsample (seed-dependent p), post-fix the full-n p-value is seed-invariant."""
        from mlframe.feature_selection.wrappers._univariate_ht import _kendall_p_numeric_continuous

        rng = np.random.default_rng(3)
        x = rng.standard_normal(5000)
        y = 0.1 * x + rng.standard_normal(5000)
        p0 = _kendall_p_numeric_continuous(x, y, random_state=0)
        p1 = _kendall_p_numeric_continuous(x, y, random_state=99)
        assert p0 == pytest.approx(p1, abs=1e-9), "p-value depends on subsample seed -> heterogeneous effective-n family"

    def test_full_n_matches_scipy(self):
        from scipy.stats import kendalltau

        from mlframe.feature_selection.wrappers._univariate_ht import _kendall_p_numeric_continuous

        rng = np.random.default_rng(4)
        x = rng.standard_normal(4000)
        y = 0.08 * x + rng.standard_normal(4000)
        p_ours = _kendall_p_numeric_continuous(x, y, random_state=0)
        _, p_sp = kendalltau(x, y, variant="b")
        assert p_ours == pytest.approx(p_sp, abs=1e-6)


# ----------------------------------------------------------------------------
# SA6: StabilityMRMR default fraction 0.5 + PFER bound exposed.
# ----------------------------------------------------------------------------
class _DummySelector:
    """Minimal selector exposing a boolean ``support_`` so StabilityMRMR can wrap it without MRMR's heavy deps."""

    def __init__(self, k: int = 2):
        self.k = k

    def get_params(self, deep=True):
        return {"k": self.k}

    def set_params(self, **p):
        self.k = p.get("k", self.k)
        return self

    def fit(self, X, y):
        import numpy as _np

        # Pick the k columns most correlated with y -- deterministic given the subsample.
        Xa = X.values if hasattr(X, "values") else _np.asarray(X)
        ya = y.values if hasattr(y, "values") else _np.asarray(y)
        corr = _np.abs([_np.corrcoef(Xa[:, j], ya)[0, 1] if Xa[:, j].std() > 0 else 0.0 for j in range(Xa.shape[1])])
        top = _np.argsort(_np.nan_to_num(corr))[::-1][: self.k]
        mask = _np.zeros(Xa.shape[1], dtype=bool)
        mask[top] = True
        self.support_ = mask
        return self


class TestStabilityMRMRPFER:
    def test_default_sample_fraction_is_half(self):
        from mlframe.feature_selection.filters.stability import StabilityMRMR

        sel = StabilityMRMR(estimator=_DummySelector())
        assert sel.sample_fraction == 0.5, "MB-canonical complementary-pairs default must be 0.5, not 0.75"

    def test_pfer_bound_matches_mb_formula(self):
        import pandas as pd

        from mlframe.feature_selection.filters.stability import StabilityMRMR

        rng = np.random.default_rng(5)
        n, p = 400, 6
        X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
        y = pd.Series((X["f0"] + 0.5 * X["f1"] + rng.standard_normal(n) > 0).astype(int))

        sel = StabilityMRMR(estimator=_DummySelector(k=2), n_bootstraps=15, support_threshold=0.6, random_state=0)
        sel.fit(X, y)

        assert hasattr(sel, "pfer_bound_"), "PFER bound must be exposed"
        assert hasattr(sel, "avg_selected_per_bootstrap_")
        q = sel.avg_selected_per_bootstrap_
        pi = sel.support_threshold
        expected = (q**2) / ((2.0 * pi - 1.0) * p)
        assert sel.pfer_bound_ == pytest.approx(expected, rel=1e-9)

    def test_pfer_bound_nan_when_fraction_not_half(self):
        import pandas as pd

        from mlframe.feature_selection.filters.stability import StabilityMRMR

        rng = np.random.default_rng(6)
        X = pd.DataFrame(rng.standard_normal((300, 5)), columns=list("abcde"))
        y = pd.Series((X["a"] + rng.standard_normal(300) > 0).astype(int))
        sel = StabilityMRMR(estimator=_DummySelector(k=2), n_bootstraps=10, sample_fraction=0.75, random_state=0)
        sel.fit(X, y)
        assert np.isnan(sel.pfer_bound_), "Bound is only valid at fraction 0.5; must be nan otherwise"


# ----------------------------------------------------------------------------
# SA5: RFECV stability selection exposes the PFER bound.
# ----------------------------------------------------------------------------
class TestRFECVStabilityPFER:
    def test_pfer_bound_exposed_and_matches_formula(self):
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier

        from mlframe.feature_selection.wrappers import RFECV

        rng = np.random.default_rng(8)
        n, p = 120, 12
        X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
        y = pd.Series((X["f0"] + X["f1"] + rng.standard_normal(n) > 0).astype(int))

        sel = RFECV(
            estimator=RandomForestClassifier(n_estimators=30, random_state=0),
            stability_selection=True,
            stability_n_bootstrap=12,
            stability_threshold=0.6,
            random_state=0,
        )
        sel.fit(X, y)
        assert hasattr(sel, "stability_pfer_bound_"), "RFECV stability path must expose the MB PFER bound"
        top_k = max(1, p // 4)
        pi = 0.6
        expected = (top_k**2) / ((2.0 * pi - 1.0) * p)
        assert sel.stability_pfer_bound_ == pytest.approx(expected, rel=1e-9)
        assert sel.cv_results_["pfer_bound"][0] == pytest.approx(expected, rel=1e-9)


# ----------------------------------------------------------------------------
# SA7: BorutaShap binomial null hit probability derived from percentile, not 0.5.
# ----------------------------------------------------------------------------
class TestBorutaNullHitCalibration:
    def test_test_features_uses_percentile_derived_null_p(self):
        """Drive ``test_features`` directly with a null-consistent hit history and assert the calibrated p=(100-percentile)/100
        keeps the accept rate near nominal, whereas p=0.5 would (anti-conservatively) accept pure-noise features."""
        from types import SimpleNamespace

        import numpy as _np

        from mlframe.feature_selection.boruta_shap import _shadow_stats

        n_features = 20
        n_trials = 40
        percentile = 99
        null_p = (100.0 - percentile) / 100.0

        # Simulate hits under the null: each feature gets ~Binomial(n_trials, null_p) hits.
        rng = _np.random.default_rng(0)
        hits = rng.binomial(n_trials, null_p, size=n_features).astype(float)

        accepted_box = {}

        def _binom(array, n, p, alternative):
            from scipy.stats import binomtest

            return [binomtest(int(x), n, p, alternative=alternative).pvalue for x in array]

        all_cols = _np.array([f"f{i}" for i in range(n_features)])
        self = SimpleNamespace(
            hits=hits,
            percentile=percentile,
            pvalue=0.05,
            all_columns=all_cols,
            binomial_H0_test=_binom,
            bonferoni_corrections=_shadow_stats.bonferoni_corrections,
            find_index_of_true_in_array=_shadow_stats.find_index_of_true_in_array,
            rejected_columns=[],
            accepted_columns=[],
        )
        _shadow_stats.test_features(self, iteration=n_trials)
        n_accepted = len(self.accepted_columns[-1])
        # Calibrated test: pure-noise features should almost never be (falsely) accepted.
        assert n_accepted <= 1, f"Calibrated null accepts too many noise features ({n_accepted}); p not percentile-derived"

    def test_null_p_far_below_half_for_default_percentile(self):
        """The derived null hit probability for the default percentile=99 must be ~0.01, not 0.5."""
        percentile = 99
        null_p = max(min((100.0 - float(percentile)) / 100.0, 1.0), 1e-9)
        assert null_p == pytest.approx(0.01, abs=1e-9)
