"""Sixteen defects across the smaller subsystems -- blending, signal, competition, integrations.

Two change numbers outright. The rank-percentile stacker offset the TEST percentile by exactly +0.5/n_oof
against the OOF one, because `rankdata` is one-based and `searchsorted` is zero-based -- so the meta-learner was
trained on one scale and applied to another. And a single interior NaN in a Hull moving average propagated
through `np.cumsum` and turned every later value into NaN: one missing tick at index 500 of a 100k-row series
voided 99.5% of the output.

Two more make a check inert. `drop_raw_after_embedding`'s safety comparison scored the raw categorical through
an IN-SAMPLE target-mean encoding and the derived columns as-is, so for the high-cardinality columns the module
exists to drop -- where the in-sample encoding nearly reproduces y -- the raw side always won and the gate could
never authorise the drop it was asked about. And a changepoint between two IDENTICAL constant segments got
effect size `inf`, so a spurious cut inside a constant run was always kept.

The rest: a get-or-create that created a duplicate on every call, an unseeded default RNG, a relative tolerance
applied absolutely, a `max_features` cap offset by the pre-seeded selection, a bare `assert` as control flow, an
`or`-default that swallowed a legitimate 0.0, a forwarded parameter that was not forwarded, and four docstrings
describing behaviour the code does not have.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest


class TestTheOofAndTestPercentilesShareOneScale:
    """A meta-learner trained on one scale and applied to another is a train-serve skew."""

    def _transform(self, oof, test):
        """The public entry point."""
        from mlframe.votenrank.rank_percentile_stacking import rank_percentile_transform

        return rank_percentile_transform(np.asarray(oof, dtype=np.float64), np.asarray(test, dtype=np.float64))

    def test_identical_input_gives_identical_percentiles(self):
        """The sharpest form: scoring the OOF set as if it were the test set must reproduce the OOF scale."""
        rng = np.random.default_rng(0)
        oof = rng.normal(size=200)
        oof_pct, test_pct = self._transform(oof, oof)
        assert float(np.max(np.abs(oof_pct - test_pct))) == 0.0

    def test_ties_are_handled_on_the_same_scale_too(self):
        """`rankdata(method="average")` and the left/right midpoint must agree on a tied value."""
        oof = np.repeat(np.arange(50.0), 4)
        oof_pct, test_pct = self._transform(oof, oof)
        assert float(np.max(np.abs(oof_pct - test_pct))) == 0.0

    def test_the_old_formula_was_offset_by_half_a_rank(self):
        """States the defect numerically, so the assertions above are not vacuous."""
        from scipy.stats import rankdata

        oof = np.arange(200.0)
        n = oof.size
        oof_pct = (rankdata(oof, method="average") - 0.5) / n
        left = np.searchsorted(np.sort(oof), oof, side="left")
        right = np.searchsorted(np.sort(oof), oof, side="right")
        old = ((left + right) / 2.0 + 0.5) / n
        assert float(np.max(np.abs(old - oof_pct))) == pytest.approx(0.5 / n)

    def test_percentiles_stay_in_range(self):
        """The clip must still hold at both ends."""
        rng = np.random.default_rng(1)
        oof = rng.normal(size=100)
        test = np.concatenate([[oof.min() - 10.0], oof, [oof.max() + 10.0]])
        _, test_pct = self._transform(oof, test)
        assert float(test_pct.min()) >= 0.0 and float(test_pct.max()) <= 1.0


class TestAnInteriorNanDoesNotVoidTheWholeSuffix:
    """A cumulative-sum moving average cannot skip a NaN, so it must say so rather than return NaN forever."""

    def test_an_interior_nan_raises_with_its_position(self):
        """One missing tick used to void every later value, silently."""
        from mlframe.signal.hull_moving_average import hull_moving_average

        x = np.arange(1000.0)
        x[500] = np.nan
        with pytest.raises(ValueError, match="interior NaN"):
            hull_moving_average(x, 20)

    def test_the_message_names_the_index(self):
        """A caller has to be able to act on it."""
        from mlframe.signal.hull_moving_average import hull_moving_average

        x = np.arange(1000.0)
        x[500] = np.nan
        with pytest.raises(ValueError, match="500"):
            hull_moving_average(x, 20)

    def test_a_leading_nan_run_is_still_tolerated(self):
        """That case was always handled and must keep working."""
        from mlframe.signal.hull_moving_average import hull_moving_average

        x = np.arange(1000.0)
        x[:30] = np.nan
        out = hull_moving_average(x, 20)
        assert np.isfinite(out[-1])

    def test_a_clean_series_is_unaffected(self):
        """The guard must not reject ordinary input."""
        from mlframe.signal.hull_moving_average import hull_moving_average

        out = hull_moving_average(np.arange(1000.0), 20)
        assert np.isfinite(out[-1]) and np.isnan(out[0])


class TestAConstantRunProducesNoChangepoint:
    """`inf` is the value that guarantees the cut SURVIVES the min-effect gate."""

    def test_a_cut_between_two_identical_constants_is_rejected(self):
        """Zero pooled variance AND zero mean difference means the two segments are the same value."""
        from mlframe.signal.changepoint_detection import detect_regime_changepoints

        y = np.full(200, 5.0)
        res = detect_regime_changepoints(y, penalty=0.0)
        ids = res["regime_id"] if isinstance(res, dict) else res
        assert len(np.unique(np.asarray(ids))) == 1, "a constant series was split into multiple regimes"

    def test_a_real_step_between_two_constants_is_still_found(self):
        """Zero pooled variance with a NON-zero mean difference is an infinitely clear change."""
        from mlframe.signal.changepoint_detection import detect_regime_changepoints

        y = np.concatenate([np.full(100, 1.0), np.full(100, 9.0)])
        res = detect_regime_changepoints(y, penalty=0.0)
        ids = np.asarray(res["regime_id"] if isinstance(res, dict) else res)
        assert len(np.unique(ids)) >= 2, "a clear step between two constants was not detected"


class TestTheRawSignalIsScoredOutOfFold:
    """An in-sample target encoding on a high-cardinality column nearly reproduces y."""

    def test_the_encoding_is_out_of_fold(self):
        """The whole point: the raw side must not get an advantage the derived side does not have."""
        import importlib

        # `import ... as m` binds the re-exported FUNCTION of the same name, not the submodule.
        m = importlib.import_module("mlframe.feature_selection.drop_raw_after_embedding")
        src = inspect.getsource(m._raw_column_signal)
        assert "OUT-OF-FOLD" in src
        assert 'groupby(col).transform("mean")' not in src

    def test_a_near_unique_column_no_longer_scores_near_perfectly(self):
        """~4 rows per group is the regime this module targets, and where in-sample encoding is worst."""
        import pandas as pd

        from mlframe.feature_selection.drop_raw_after_embedding import _raw_column_signal

        rng = np.random.default_rng(0)
        n = 4000
        y = (rng.random(n) > 0.5).astype(np.float64)  # binary: the signal helper is AUC-based
        ids = pd.Series([f"dev_{i // 4}" for i in range(n)])  # ~4 rows per distinct id, pure noise vs y
        df = pd.DataFrame({"device_id": ids})

        oof = _raw_column_signal(df, "device_id", y)
        in_sample = pd.Series(y, index=df.index).groupby(df["device_id"]).transform("mean").to_numpy()
        from mlframe.feature_selection.drop_raw_after_embedding import _univariate_signal

        insample_signal = _univariate_signal(in_sample, y)
        assert oof < insample_signal, f"out-of-fold signal {oof:.4f} is not below the in-sample {insample_signal:.4f}"

    def test_a_genuinely_predictive_column_still_scores(self):
        """The fix must not make the check blind in the other direction."""
        import pandas as pd

        from mlframe.feature_selection.drop_raw_after_embedding import _raw_column_signal

        rng = np.random.default_rng(1)
        n = 4000
        group = rng.integers(0, 2, size=n)
        y = np.where(rng.random(n) < 0.9, group, 1 - group).astype(np.float64)  # group predicts y ~90% of the time
        df = pd.DataFrame({"g": [f"g{v}" for v in group]})
        assert _raw_column_signal(df, "g", y) > 0.5


def test_the_mlflow_lookup_scopes_by_experiment_id():
    """`experiment_id` was accepted, forwarded to start_run, and ignored by the search -- so the "get" half
    looked in the currently-active experiment and a fresh run was created on every call."""
    from mlframe.integrations import mlflow as m

    src = inspect.getsource(m)
    assert "experiment_ids=[str(experiment_id)]" in src
    assert "if experiment_id:" in src


class TestTheShapleyDefaultIsReproducible:
    """Two calls on identical inputs returned different values, and anything pruning on them inherited that."""

    def test_the_default_rng_is_seeded(self):
        """`np.random.default_rng()` with no argument draws from OS entropy."""
        import mlframe.votenrank.shapley_blend as m

        src = inspect.getsource(m)
        assert "np.random.default_rng(_DEFAULT_SHAPLEY_SEED)" in src
        assert "rng = np.random.default_rng()" not in src

    def test_two_default_calls_agree(self):
        """The property, asserted end-to-end."""
        from mlframe.votenrank.shapley_blend import shapley_model_values

        rng = np.random.default_rng(0)
        n = 300
        y = (rng.random(n) > 0.5).astype(np.int64)
        preds = np.vstack([y * 0.8 + rng.random(n) * 0.2, rng.random(n), y * 0.6 + rng.random(n) * 0.4])

        v1, _ = shapley_model_values(preds, y, n_permutations=16)
        v2, _ = shapley_model_values(preds, y, n_permutations=16)
        np.testing.assert_allclose(v1, v2)

    def test_an_explicit_generator_still_overrides(self):
        """A caller who wants fresh randomness passes their own, which is explicit."""
        from mlframe.votenrank.shapley_blend import shapley_model_values

        rng = np.random.default_rng(0)
        n = 300
        y = (rng.random(n) > 0.5).astype(np.int64)
        preds = np.vstack([y * 0.8 + rng.random(n) * 0.2, rng.random(n), y * 0.6 + rng.random(n) * 0.4])

        v1, _ = shapley_model_values(preds, y, n_permutations=16, rng=np.random.default_rng(1))
        v2, _ = shapley_model_values(preds, y, n_permutations=16, rng=np.random.default_rng(1))
        np.testing.assert_allclose(v1, v2)


class TestTheLazyFacadeKeepsEveryName:
    """Making the re-exports lazy must not remove anything a consumer might import."""

    def test_every_declared_name_resolves(self):
        """A lazy hook that cannot resolve a name is worse than an eager import."""
        import mlframe.votenrank as v

        assert v.__all__, "the facade declares no surface"
        for name in v.__all__:
            assert getattr(v, name) is not None, name

    def test_the_one_name_consumers_actually_import_still_works(self):
        """`Leaderboard` is the single name reached through the package rather than by submodule path."""
        from mlframe.votenrank import Leaderboard

        assert isinstance(Leaderboard, type)

    def test_dir_lists_the_lazy_names(self):
        """`__getattr__` alone does not make them discoverable."""
        import mlframe.votenrank as v

        assert set(v.__all__) <= set(dir(v))

    def test_an_unknown_attribute_raises_attribute_error(self):
        """The hook must not swallow a typo into an import error or a None."""
        import mlframe.votenrank as v

        with pytest.raises(AttributeError, match="does_not_exist"):
            _ = v.does_not_exist


class TestTheDocumentedContractsMatchTheCode:
    """Six places promised behaviour the code does not implement."""

    def test_the_dichotomic_step_is_forwarded(self):
        """`RFECV(dichotomic_step=...)` was silently ignored under the scipy search methods."""
        from mlframe.feature_selection.wrappers import _helpers as m

        assert "step" in inspect.signature(m._suggest_scipy_local).parameters
        assert "step=step" in inspect.getsource(m._suggest_scipy_local)

    def test_the_ridge_tolerance_is_relative(self):
        """Documented as a relative drop, subtracted absolutely."""
        from mlframe.feature_selection import ridge_forward_prefilter as m

        src = inspect.getsource(m)
        assert "best_score - abs(best_score) * tol" in src
        assert "size_scores[size] >= best_score - tol" not in src

    def test_max_features_counts_the_whole_subset(self):
        """`max_features=5, initial_selected=[a, b, c]` could return 8 columns."""
        from mlframe.feature_selection import forward_select as m

        src = inspect.getsource(m)
        assert "cap = max_features if max_features is not None else len(all_candidates) + len(selected)" in src

    def test_the_simplex_solver_raises_instead_of_asserting(self):
        """Under `python -O` the bare assert vanished and the function returned None."""
        import mlframe.votenrank.constrained_weight_blend as m

        src = inspect.getsource(m)
        assert "assert best_weights is not None" not in src
        assert "raise ValueError(" in src

    def test_a_zero_subsample_fraction_is_honoured(self):
        """`0.0 or 0.75` silently drew 75% of the rows."""
        from mlframe.feature_selection.boruta_shap import _fit_explain as m

        src = inspect.getsource(m)
        assert '"stability_subsample_fraction", 0.75) or 0.75' not in src
        assert "_frac_cfg is None else float(_frac_cfg)" in src

    def test_the_pruning_summary_no_longer_promises_a_stop_rule(self):
        """The summary line said it stops on CV degradation; the module docstring and the code say otherwise."""
        from mlframe.feature_selection import zero_importance_pruning as m

        src = inspect.getsource(m)
        assert "stopping on CV degradation" not in src
        assert "does NOT stop on CV degradation" in src

    def test_the_shapley_stderr_is_documented_as_a_proxy(self):
        """`|value| / sqrt(n)` makes every model's value/stderr ratio identical, so it cannot discriminate."""
        import mlframe.votenrank.shapley_blend as m

        src = inspect.getsource(m)
        assert "ANALYTIC PROXY" in src
        # The CLAIM must be gone; the corrected docstring quotes the old wording to say it was never true.
        assert "holding ``stderr`` (per-model, two-branch running stats)" not in src

    def test_the_backend_precedence_is_documented_as_implemented(self):
        """The docstring said the env var is checked first; the argument wins."""
        import mlframe.votenrank.confidence_gated_blend as m

        src = inspect.getsource(m)
        assert "checked first" not in src

    def test_the_override_docstring_describes_the_unconditional_write(self):
        """It described an "already above threshold" guard the code has never had."""
        from mlframe.competition import known_label_override as m

        src = inspect.getsource(m)
        assert "isn't already >= positive threshold" not in src
        assert "INCLUDING rows already predicted" in src
