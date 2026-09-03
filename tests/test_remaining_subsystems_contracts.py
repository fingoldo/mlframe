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

import contextlib
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

    # `test_the_encoding_is_out_of_fold` used to sit here, asserting that the helper's source says
    # "OUT-OF-FOLD" and no longer contains an in-sample groupby-transform. The two siblings below drive the
    # consequence directly and are what an in-sample encoding would actually break: a near-unique column stops
    # scoring near-perfectly, and a genuinely predictive one still scores. A source phrase adds nothing to
    # that, and the phrase could be present while the encoding was in-sample anyway.

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

    import mlflow as _mlflow

    # Spy on the search rather than reading the source: the defect was that `experiment_id` was accepted,
    # forwarded to `start_run`, and IGNORED by the lookup, so the get half searched the currently-active
    # experiment, found nothing, and created a fresh run on every call -- the one thing a get-or-create must
    # not do. What matters is the argument the search actually receives.
    seen: list = []
    real_search = _mlflow.search_runs

    def _spy(*args, **kwargs):
        """Record the scoping arguments, then return no matches so the create half runs."""
        seen.append(kwargs)
        return []

    _mlflow.search_runs = _spy
    try:
        with contextlib.suppress(Exception):
            m.get_or_create_mlflow_run("probe-run", experiment_id="7")
    finally:
        _mlflow.search_runs = real_search

    assert seen, "the lookup never searched at all, so it can only ever create"
    assert seen[0].get("experiment_ids") == ["7"], f"the search was not scoped by the experiment_id it was given: {seen[0]!r}"


class TestTheShapleyDefaultIsReproducible:
    """Two calls on identical inputs returned different values, and anything pruning on them inherited that."""

    # `test_the_default_rng_is_seeded` used to sit here, asserting that the module's source contains
    # `default_rng(_DEFAULT_SHAPLEY_SEED)`. It was redundant with the behavioural sibling below, which drives
    # the actual contract -- two default calls agree -- and it had become wrong in a second way: it reached the
    # module via `import mlframe.votenrank.shapley_blend as m`, which resolves to the re-exported FUNCTION, not
    # the submodule, so `inspect.getsource` was reading the wrong object entirely.

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

        # ...and it is actually PASSED ON. A knob accepted at the boundary and dropped there gives the caller
        # the adaptive "auto" schedule while they believe they configured "midpoint", with nothing to
        # indicate it -- and both schedules return a valid suggestion, so no assertion on the result shows it.
        received: list = []
        real = m._suggest_dichotomic

        def _spy(*args, **kwargs):
            """Record the step the scipy-local branch forwards, then defer to the real suggester."""
            received.append(kwargs.get("step"))
            return real(*args, **kwargs)

        m._suggest_dichotomic = _spy
        try:
            # No `except Exception: pass` around this call. It was defensive padding -- the call returns a
            # suggestion cleanly on this input -- and a broad swallow here would turn a genuine fault inside
            # the branch into "the step was never forwarded", reporting the wrong defect. Let it raise.
            suggestion = m._suggest_scipy_local([1, 2, 3], {1: 0.5, 2: 0.6}, 3, epsilon=0.01, rng=np.random.default_rng(0), step="midpoint")
        finally:
            m._suggest_dichotomic = real

        assert suggestion in (1, 2, 3), f"the scipy-local branch returned a suggestion outside the candidate set: {suggestion!r}"
        assert received == ["midpoint"], f"the scipy-local branch did not forward the configured step: {received!r}"

    def test_the_ridge_tolerance_is_relative(self):
        """`tol` is a RELATIVE drop from the best score, not an absolute one.

        Driven on data where the two forms disagree outright. The signal is weak, so the best CV r2 is about
        +0.0033; a relative 1% floor is +0.00331 and only the 3-feature prefix clears it, while an absolute
        floor of best - 0.01 is -0.0067 and the 1-feature prefix -- scoring -0.0058, i.e. WORSE than predicting
        the mean -- clears that comfortably. Subtracting absolutely on a small-magnitude score turns `tol` into
        an enormous relative allowance, which is how a far smaller set than the operator asked for gets
        through. The gap widens with tol, so all three settings are checked.
        """
        import numpy as _np
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        from mlframe.feature_selection.ridge_forward_prefilter import ridge_coefficient_prefilter

        rng = _np.random.default_rng(0)
        n, p = 400, 8
        X = rng.normal(size=(n, p))
        y = 0.10 * X[:, 0] + 0.08 * X[:, 1] + 0.06 * X[:, 2] + 0.05 * X[:, 3] + rng.normal(0, 1.0, n)
        names = [f"f{i}" for i in range(p)]
        sizes = [1, 2, 3, 4, 6, 8]

        # Score each ridge-ranked prefix the way the prefilter does, so the fixture's premise is asserted
        # rather than assumed.
        order = _np.argsort(_np.abs(Ridge(alpha=1.0).fit(X, y).coef_))[::-1]
        scores = {s: float(_np.mean(cross_val_score(Ridge(alpha=1.0), X[:, order[:s]], y, cv=3, scoring="r2"))) for s in sizes}
        best = max(scores.values())
        assert 0.0 < best < 0.05, f"the fixture needs a small POSITIVE best score for the two floors to diverge; got {best:+.5f}"

        for tol in (0.01, 0.05, 0.2):
            relative_pick = next(s for s in sizes if scores[s] >= best - abs(best) * tol)
            absolute_pick = next(s for s in sizes if scores[s] >= best - tol)
            assert relative_pick != absolute_pick, f"tol={tol} does not separate the two rules on this fixture"
            chosen = ridge_coefficient_prefilter(X, y, names, candidate_sizes=sizes, cv=3, tol=tol)
            assert len(chosen) == relative_pick, f"tol={tol}: kept {len(chosen)} features; the relative floor wants {relative_pick}, an absolute one would give {absolute_pick}"

    def test_max_features_counts_the_whole_subset(self):
        """`max_features` bounds the FINAL subset, seeds included -- not the number of columns added to it.

        Driven: with three seeded columns and `max_features=5`, the result must be at most five columns in
        total. Counting only the additions returns eight, which is the defect -- and a caller sizing a model
        by `max_features` gets a subset over half again as wide as asked for, silently.
        """
        import numpy as _np
        import pandas as _pd
        from sklearn.linear_model import LinearRegression

        from mlframe.feature_selection.forward_select import forward_select

        rng = _np.random.default_rng(0)
        n = 300
        X = _pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(10)})
        y = X["f0"] + 0.8 * X["f1"] + 0.6 * X["f2"] + 0.4 * X["f3"] + 0.2 * X["f4"] + rng.normal(0, 0.1, n)

        seeds = ["f0", "f1", "f2"]
        chosen = forward_select(X, y.to_numpy(), LinearRegression, cv=3, max_features=5, initial_selected=seeds)

        assert len(chosen) <= 5, f"max_features=5 with 3 seeds returned {len(chosen)} columns: {chosen}"
        assert set(seeds) <= set(chosen), f"the seeded columns were dropped: {chosen}"

    def test_the_simplex_solver_raises_instead_of_returning_none(self):
        """Every restart failing must RAISE, naming the cause -- not hand back `None`.

        Driven rather than read: a loss function that returns NaN makes every restart fail the
        `loss < best_loss` test (any comparison against NaN is False), which is the exact state the guard
        exists for. It used to be a bare `assert`, so under `python -O` it vanished and the function returned
        `None`; the caller then crashed somewhere far from the cause. A `raise` holds under -O too, and the
        message says WHY -- non-finite losses throughout, typically a NaN in the predictions.
        """
        import numpy as _np
        import pytest as _pytest

        from mlframe.votenrank.constrained_weight_blend import constrained_weight_blend

        rng = _np.random.default_rng(0)
        n = 120
        y = rng.integers(0, 2, n).astype(float)
        preds = [_np.clip(y * 0.6 + rng.normal(0, s, n), 0.0, 1.0) for s in (0.2, 0.4)]

        def _always_nan(_a, _b):
            """A loss that never returns a finite value, so no restart can ever win."""
            return float("nan")

        with _pytest.raises(ValueError, match="none of the .* restarts produced a finite loss"):
            constrained_weight_blend(preds, y, loss_fn=_always_nan, n_restarts=3, random_state=0)

        # A finite loss still solves, so the guard is not simply refusing everything.
        def _mse(a, b):
            """Ordinary MSE, which the solver can minimise."""
            return float(_np.mean((_np.asarray(a) - _np.asarray(b)) ** 2))

        out = constrained_weight_blend(preds, y, loss_fn=_mse, n_restarts=3, random_state=0)
        assert out is not None, "a well-posed blend returned None"

    def test_a_zero_subsample_fraction_is_honoured(self):
        """`0.0 or 0.75` silently drew 75% of the rows."""
        from mlframe.feature_selection.boruta_shap import _fit_explain as m

        import ast

        from tests._source_ast import module_ast

        # Structural: `0.0 or 0.75` and an explicit None-test both yield 0.75 for every value EXCEPT a
        # deliberate zero, and a zero fraction means "subsample nothing" -- so the difference shows only for
        # the caller the old form silently overrode, on a path that needs a full stability run to reach.
        tree = module_ast(m)
        frac_reads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "stability_subsample_fraction"
        ]
        assert frac_reads, "the subsample fraction is no longer read from config; this test needs updating"
        or_operands = {id(inner) for node in ast.walk(tree) if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) for value in node.values for inner in ast.walk(value)}
        assert not any(id(r) in or_operands for r in frac_reads), "the fraction is read through an `or` default again, so a deliberate 0.0 draws 75% of the rows"

    def test_the_pruning_summary_no_longer_promises_a_stop_rule(self):
        """The summary must not claim a stop rule the pruner does not implement.

        Structural, and narrowly so: this is a LOG LINE's wording against the module's own documented
        behaviour, and the pruner returns the same columns either way -- the defect was an operator reading
        "stopping on CV degradation" in a summary and believing a safety rule was in force.
        """
        from mlframe.feature_selection import zero_importance_pruning as m
        from tests._source_ast import module_ast, string_literals

        emitted = " ".join(string_literals(module_ast(m)))
        assert "stopping on CV degradation" not in emitted, "the summary promises a stop rule the pruner does not implement"
        assert "does NOT stop on CV degradation" in emitted, "the summary no longer states that it keeps pruning regardless of CV"

    def test_the_shapley_stderr_cannot_discriminate_between_models(self):
        """`stderr` is the analytic proxy `|value| / sqrt(n)`, so every model's value/stderr ratio is IDENTICAL.

        Measured rather than read out of the docstring. This is the property that matters to a caller: a
        "keep the model if its value exceeds 2 stderr" rule compares the same number for every model, so it
        keeps all of them or none and can never rank one above another. A genuine per-model standard error
        would vary with that model's own coalition spread.
        """
        import numpy as _np

        from mlframe.votenrank.shapley_blend import shapley_model_values

        rng = _np.random.default_rng(0)
        n = 300
        y = rng.integers(0, 2, n).astype(float)
        # Three members of deliberately different quality, so a real stderr would differ between them.
        preds = _np.vstack([_np.clip(y * 0.6 + rng.normal(0, s, n), 0, 1) for s in (0.2, 0.35, 0.5)])

        values, info = shapley_model_values(preds, y, n_permutations=30, rng=_np.random.default_rng(1))
        values = _np.asarray(values, dtype=float)
        stderr = _np.asarray(info["stderr"], dtype=float)

        assert values.shape == (3,) and stderr.shape == (3,)
        assert _np.all(_np.isfinite(stderr)) and _np.all(stderr > 0.0), stderr
        assert len(set(_np.round(_np.abs(values), 6))) > 1, "the three members scored identically, so this fixture proves nothing"

        ratios = _np.abs(values) / stderr
        assert _np.allclose(ratios, ratios[0], rtol=1e-8), f"value/stderr differs between models, so stderr is no longer the |value|/sqrt(n) proxy: {ratios!r}"
        # ...and the shared ratio IS sqrt(n_permutations), which is what makes it a pure restatement of the value.
        assert _np.isclose(ratios[0], _np.sqrt(30.0), rtol=1e-8), f"the proxy is no longer |value|/sqrt(n): ratio={ratios[0]!r}"

    def test_the_backend_argument_wins_over_the_env_var(self, monkeypatch):
        """`force_backend` beats `MLFRAME_CONFIDENCE_BLEND_BACKEND`; the docstring had it the other way round.

        Every backend returns the same numbers, so which one ran is invisible in the result. Observed by
        making the numpy path raise: with `force_backend` set to something else the call must succeed, and
        with it left as None the env var takes over and the same call must hit numpy and raise. A caller who
        passes an explicit backend and happens to have the env var set was, per the old docstring, going to be
        silently overridden -- which only surfaces when a benchmark refuses to use the backend you asked for.
        """
        import importlib

        import numpy as _np
        import pytest as _pytest

        # NOT `import mlframe.votenrank.confidence_gated_blend as m`: the package binds the re-exported
        # FUNCTION under that name, so the plain import form hands back a callable with no `_blend_numpy` on
        # it. `import_module` reaches the submodule itself.
        m = importlib.import_module("mlframe.votenrank.confidence_gated_blend")

        def _boom(*_a, **_k):
            """Stand in for the numpy backend so its use is observable."""
            raise AssertionError("the numpy backend ran")

        monkeypatch.setattr(m, "_blend_numpy", _boom)
        monkeypatch.setenv("MLFRAME_CONFIDENCE_BLEND_BACKEND", "numpy")

        rng = _np.random.default_rng(0)
        n = 4_000  # above _DISPATCH_MIN_N, so the backend ladder is actually consulted (see below)
        args = (rng.random(n), rng.random(n), rng.random(n), 0.5, 0.3)

        # The env var alone routes to numpy -> the stand-in fires.
        with _pytest.raises(AssertionError, match="the numpy backend ran"):
            m.confidence_gated_blend(*args)

        # The explicit argument overrides it -> a different backend runs and the call completes.
        out = m.confidence_gated_blend(*args, force_backend="njit")
        assert out.shape == (n,), f"the forced backend did not produce a full result: {out.shape}"
        assert _np.all(_np.isfinite(out))

        # ...and the precedence is three-deep, not two: the SIZE guard sits above both. Below
        # `_DISPATCH_MIN_N` the function returns numpy before any backend is resolved, so `force_backend` is
        # ignored there. That is deliberate -- dispatch overhead dominates on a tiny input -- but it means an
        # explicit backend is silently not honoured, which a benchmark forcing a backend on small data needs
        # to know. Pinned so the guard cannot quietly move.
        small = (rng.random(64), rng.random(64), rng.random(64), 0.5, 0.3)
        with _pytest.raises(AssertionError, match="the numpy backend ran"):
            m.confidence_gated_blend(*small, force_backend="njit")
        assert m._DISPATCH_MIN_N == 2_000, f"the dispatch floor moved to {m._DISPATCH_MIN_N}; the note above needs updating"

    def test_the_override_writes_unconditionally_in_the_safe_direction(self):
        """A positive recovered label overwrites the prediction even when it is ALREADY above threshold.

        The docstring used to describe an "isn't already >= positive threshold" guard the code has never had.
        Driven instead of read: a row already predicted 0.98 and recovered as positive comes back as exactly
        the positive value, not left at 0.98. That distinction is invisible in aggregate metrics -- both are
        "confidently positive" -- and it is the whole reason the claim mattered.
        """
        import numpy as _np

        from mlframe.competition.known_label_override import known_label_override

        preds = _np.array([0.98, 0.10, 0.50, 0.99])
        # Row 0 is already far above threshold; row 1 is below it; row 3 is above and recovered as NEGATIVE.
        out = known_label_override(preds, {0: 1.0, 1: 1.0, 3: 0.0}, asymmetric_safe_direction="positive")

        assert out[0] == 1.0, f"an already-confident row was left at {out[0]!r} instead of being overwritten"
        assert out[1] == 1.0, f"a below-threshold row was not overridden: {out[1]!r}"
        assert out[2] == 0.50, "a row with no recovered label must be untouched"
        assert out[3] == 0.99, "a negative-direction recovered label must NOT overwrite, which is the asymmetry"
        assert preds[0] == 0.98, "the caller's array was mutated in place"
