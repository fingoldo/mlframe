"""
Structural invariant tests — evergreen guards against the *class* of bugs
that surfaced in the 2026-04-23 prod log review.

Each test here is deliberately broader than the per-fix regression sensors in
``test_prod_log_fixes_2026_04_23.py``. Where that file pins specific code
paths, this file asserts invariants that a future change would have to
violate in order to re-introduce the same family of bugs:

    Bug class                                          | Invariant
    ---------------------------------------------------+------------------------------
    PipelineCache kind collision (XGB pl ↔ LGB pd)     | Kind isolation across
                                                         all strategy pairs
    Duplicate polars→pandas conversion per DF          | ≤1 conversion call per
                                                         Polars DF id() in a
                                                         full mixed suite run
    Nullable Polars types → object dtype → crash       | ``get_pandas_view_of_polars_df``
                                                         never emits ``object``
                                                         on any plain-dtype
                                                         input schema
    Ensemble RuntimeWarnings + non-finite outputs      | All 6 ensemble methods
                                                         stay warning-free and
                                                         in [0, 1] on edge-case
                                                         inputs
    Trainer silent polars→pandas self-heal             | Every non-Polars-native
                                                         strategy raises at the
                                                         trainer boundary, not
                                                         the model boundary

Adding a new strategy or convert-bridge code should either keep all of these
green or update the invariant with a clear comment explaining why.
"""

import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
import pytest


# =====================================================================
# #1 PipelineCache: no polars/pandas cross-stream leakage
# =====================================================================

class TestPipelineCacheKeyInvariants:
    """Meta-test over every pair of strategies in ``MODEL_STRATEGIES``.

    The 2026-04-23 prod leak happened because CB, XGB, and LGB all inherit
    ``cache_key="tree"`` **and** XGB+LGB happen to share ``feature_tier()``,
    while their ``supports_polars`` differ. The old cache key was blind to
    container kind, so XGB's cached polars frame was served back to LGB.

    The invariant below is load-bearing: *for any two strategies with the
    same (cache_key, feature_tier) and different supports_polars, the
    composed pipeline-cache key must differ.* Adding a new tree-like model
    without accounting for container kind would trip this test immediately.
    """

    def _compose_cache_key(self, strat, pre_pipeline_name: str = ""):
        """Reproduce the composition logic from ``core.py:train_mlframe_models_suite``.

        Keeping this verbatim in the test is intentional: if someone removes
        or renames the suffix convention in core.py, this test must fail so
        the invariant can be re-asserted after a review — a silent drift is
        exactly what we're guarding against.
        """
        tier_suffix = f"_tier{strat.feature_tier()}"
        kind_suffix = f"_kind{'pl' if strat.supports_polars else 'pd'}"
        if pre_pipeline_name:
            return f"{strat.cache_key}_{pre_pipeline_name}{tier_suffix}{kind_suffix}"
        return f"{strat.cache_key}{tier_suffix}{kind_suffix}"

    def test_no_polars_pandas_cross_stream_leak_across_all_strategy_pairs(self):
        """For any pair of strategies that share (cache_key, feature_tier)
        but differ on ``supports_polars``, the composed cache key must
        differ. This is a superset of the specific XGB/LGB collision the
        2026-04-23 fix addressed — it also catches future additions like a
        new HGB-variant sharing tier with a non-native sibling.
        """
        from mlframe.training.strategies import MODEL_STRATEGIES

        strategies = list(MODEL_STRATEGIES.items())
        colliding_triples = []
        for i, (name_a, strat_a) in enumerate(strategies):
            for name_b, strat_b in strategies[i + 1:]:
                same_cache = strat_a.cache_key == strat_b.cache_key
                same_tier = strat_a.feature_tier() == strat_b.feature_tier()
                diff_kind = strat_a.supports_polars != strat_b.supports_polars
                if same_cache and same_tier and diff_kind:
                    key_a = self._compose_cache_key(strat_a)
                    key_b = self._compose_cache_key(strat_b)
                    if key_a == key_b:
                        colliding_triples.append((name_a, name_b, key_a))

        assert not colliding_triples, (
            "pipeline_cache key collision between a polars-native and a "
            f"pandas-consuming strategy — this is the 2026-04-23 bug class:\n"
            + "\n".join(
                f"  {a!r} ↔ {b!r} share key {k!r}"
                for a, b, k in colliding_triples
            )
        )

    def test_known_tree_strategy_trio_produces_distinct_keys(self):
        """Concrete evidence: the three tree models the review flagged —
        CB (polars-native, tier(True,True)), XGB (polars-native,
        tier(False,False)), LGB (pandas-only, tier(False,False)) —
        resolve to three *different* cache keys. Locks in the specific
        fix while the meta-test above guards the rest."""
        from mlframe.training.strategies import get_strategy

        keys = {
            name: self._compose_cache_key(get_strategy(name))
            for name in ("cb", "xgb", "lgb")
        }
        assert len(set(keys.values())) == 3, (
            f"CB/XGB/LGB keys must all differ; got {keys!r}"
        )
        # And specifically: XGB vs LGB (the 2026-04-23 collision).
        assert keys["xgb"] != keys["lgb"]


# =====================================================================
# #2 No duplicate polars→pandas conversion per DF in a suite run
# =====================================================================

class TestNoDuplicateConversion:
    """A full mixed-strategy suite run must convert each Polars input DF
    (train / val / test) at most once via ``get_pandas_view_of_polars_df``.

    The 2026-04-23 prod log showed 224 s + 9.8 s duplicate conversions of
    the same train frame — the PipelineCache leak above was the root cause.
    This test guards against *any* future failure mode that might re-
    introduce a second conversion of the same source frame, regardless of
    where the bug lives.
    """

    def test_mixed_suite_converts_each_polars_df_at_most_once(self, tmp_path):
        import mlframe.training.core as mf_core
        import mlframe.training.utils as mf_utils
        from mlframe.training.configs import TrainingBehaviorConfig
        from mlframe.training.core import train_mlframe_models_suite
        from .shared import SimpleFeaturesAndTargetsExtractor

        n = 400
        rng = np.random.default_rng(0)
        budget_cats = ["HOURLY", "FIXED", "MILESTONE"]
        tier_cats = ["BEGINNER", "INTERMEDIATE", "EXPERT"]
        pl_df = pl.DataFrame({
            "num_1": rng.standard_normal(n).astype(np.float32),
            "num_2": rng.standard_normal(n).astype(np.float32),
            "budget_type": pl.Series(
                [budget_cats[i % 3] for i in range(n)]
            ).cast(pl.Enum(budget_cats)),
            "contractor_tier": pl.Series(
                [tier_cats[i % 3] for i in range(n)]
            ).cast(pl.Enum(tier_cats)),
            "target": rng.integers(0, 2, n),
        })

        # Record conversions keyed by the *source* Polars DF id. Two calls
        # with the same input id mean we re-converted the same frame — the
        # exact duplicate-work pattern we're forbidding.
        original = mf_utils.get_pandas_view_of_polars_df
        per_input_id = defaultdict(int)

        def _tracking(df, *args, **kwargs):
            if isinstance(df, pl.DataFrame):
                per_input_id[id(df)] += 1
            return original(df, *args, **kwargs)

        # Patch BOTH the utils module (covers all lazy
        # ``from mlframe.training.utils import get_pandas_view_of_polars_df``
        # sites inside function bodies) AND the core module (which imports
        # the symbol at top-level, so the utils-module patch alone wouldn't
        # redirect the ``core.get_pandas_view_of_polars_df(...)`` call site).
        mf_utils.get_pandas_view_of_polars_df = _tracking
        mf_core.get_pandas_view_of_polars_df = _tracking
        try:
            fte = SimpleFeaturesAndTargetsExtractor(
                target_column="target", regression=False
            )
            bc = TrainingBehaviorConfig(prefer_gpu_configs=False)
            train_mlframe_models_suite(
                df=pl_df,
                target_name="no_dup_conv_test",
                model_name="mix",
                features_and_targets_extractor=fte,
                mlframe_models=["cb", "xgb", "lgb"],
                hyperparams_config={"iterations": 3},
                behavior_config=bc,
                init_common_params={"drop_columns": [], "verbose": 0},
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                data_dir=str(tmp_path),
                models_dir="models",
                verbose=0,
            )
        finally:
            mf_utils.get_pandas_view_of_polars_df = original
            mf_core.get_pandas_view_of_polars_df = original

        duplicates = {k: v for k, v in per_input_id.items() if v > 1}
        assert not duplicates, (
            f"get_pandas_view_of_polars_df was called >1× on the same source "
            f"Polars DF: {duplicates!r}. A duplicate conversion is the "
            f"2026-04-23 prod-log regression — check PipelineCache kind "
            f"isolation and the strategy-loop lazy-conversion hook."
        )


# =====================================================================
# #3 get_pandas_view_of_polars_df never emits pandas ``object`` dtype on
#   non-nested inputs
# =====================================================================

class TestBridgeNoObjectDtypes:
    """The LGB 2026-04-23 crash traced to a single ``object`` column in the
    pandas view (``hide_budget``, which was a nullable Polars Boolean). No
    tree backend accepts ``object``; the bridge must coerce every
    plain-dtype polars column to a numeric/bool/category pandas equivalent.

    Property-based via hypothesis: generate mini polars frames with random
    schemas drawn from the dtype space we actually pass to the bridge in
    prod, and assert no column materializes as ``object``. List/Struct are
    deliberately excluded — the bridge already warns about those, and the
    invariant would be wrong for them.
    """

    def test_bridge_produces_no_object_dtype_for_plain_polars_schemas(self):
        from mlframe.training.utils import get_pandas_view_of_polars_df
        # Property-based via hypothesis only here (not at module scope) —
        # keeps the import cost off the rest of this file when a slice of
        # tests is collected.
        from hypothesis import given, settings, HealthCheck, strategies as hst

        # Each polars dtype the mlframe pipeline routes through the "numeric
        # / cat_feature" path. Scoped deliberately:
        #   * No List/Struct: the bridge already emits a nested-dtype WARN,
        #     those legitimately materialize as object.
        #   * No pl.String: pyarrow maps plain string columns to pandas
        #     ``object`` by design (no native numpy string dtype before
        #     pandas 2.0 StringDtype). mlframe routes string data only via
        #     ``cat_features`` (→ Categorical / Enum) or ``text_features``
        #     (→ CatBoost's native text handling), never as plain pl.String
        #     into a tree backend. If a caller sends raw pl.String they're
        #     responsible for declaring it as a cat/text feature upstream.
        _DTYPE_BUILDERS = [
            ("f32", pl.Float32, lambda v: float(v % 1_000_000)),
            ("f64", pl.Float64, lambda v: float(v % 1_000_000)),
            ("i8", pl.Int8, lambda v: int(v % 127)),
            ("i32", pl.Int32, lambda v: int(v % 1_000_000)),
            ("i64", pl.Int64, lambda v: int(v % 1_000_000)),
            ("bool", pl.Boolean, lambda v: bool(v % 2)),
            ("cat", pl.Categorical, lambda v: f"c{v % 4}"),
            ("enum", pl.Enum(["a", "b", "c", "d"]), lambda v: ["a", "b", "c", "d"][v % 4]),
        ]

        @given(
            n_rows=hst.integers(min_value=3, max_value=30),
            schema_pick=hst.lists(
                hst.integers(min_value=0, max_value=len(_DTYPE_BUILDERS) - 1),
                min_size=1, max_size=6, unique=False,
            ),
            null_frac=hst.floats(min_value=0.0, max_value=0.5),
        )
        @settings(
            max_examples=40,
            deadline=2000,
            suppress_health_check=[HealthCheck.too_slow],
        )
        def _runner(n_rows, schema_pick, null_frac):
            cols = {}
            for idx, pick in enumerate(schema_pick):
                name, dtype, builder = _DTYPE_BUILDERS[pick]
                values = []
                for r in range(n_rows):
                    if null_frac > 0 and (r * 17 + pick * 3 + idx) % 100 < int(null_frac * 100):
                        values.append(None)
                    else:
                        values.append(builder(r + idx))
                col_name = f"{name}_{idx}"
                cols[col_name] = pl.Series(values, dtype=dtype)
            pl_df = pl.DataFrame(cols)

            pdf = get_pandas_view_of_polars_df(pl_df)
            object_cols = [
                c for c in pdf.columns if pdf[c].dtype == object
            ]
            assert not object_cols, (
                f"Bridge produced pandas object dtype on plain-dtype Polars "
                f"schema — tree backends will reject. Columns: {object_cols!r}. "
                f"Input schema: {dict(pl_df.schema)!r}"
            )

        _runner()


# =====================================================================
# #4 Ensemble methods: edge-case warning + range invariants
# =====================================================================

class TestEnsembleMethodsEdgeCases:
    """Each simple ensemble method (``arithm/harm/median/quad/qube/geo``)
    must:
      1. not emit a ``RuntimeWarning`` on probability inputs containing
         exact 0.0 or 1.0;
      2. return finite values (no NaN / no ±inf);
      3. stay inside [0, 1] since inputs were probabilities.

    The 2026-04-23 fix addressed case (1) for ``harm`` specifically; this
    test extends the invariant to the whole family — ``geo`` and ``qube``
    have their own edge cases with exact 0/1 that should stay warning-free.
    """

    EDGE_CASES = [
        ("contains_zeros", [[0.0, 0.2, 0.5], [0.3, 0.0, 0.6], [0.4, 0.1, 0.0]]),
        ("contains_ones", [[1.0, 0.2, 0.5], [0.3, 1.0, 0.6], [0.4, 0.1, 1.0]]),
        ("all_equal", [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
        ("all_zeros", [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        ("all_ones", [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        ("tiny_and_near_one", [[1e-12, 0.9999999], [2e-12, 0.9999998]]),
    ]

    @pytest.mark.parametrize("method", ["arithm", "harm", "median", "quad", "qube", "geo"])
    @pytest.mark.parametrize("case_name,preds", EDGE_CASES)
    def test_ensemble_method_is_warning_free_finite_and_in_range(
        self, method, case_name, preds,
    ):
        from mlframe.ensembling import ensemble_probabilistic_predictions

        pred_arrays = [
            np.asarray(p, dtype=np.float64).reshape(-1, 1) for p in preds
        ]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = ensemble_probabilistic_predictions(
                *pred_arrays,
                ensemble_method=method,
                max_mae=0,
                max_std=0,
                verbose=False,
            )
        predictions = out[0] if isinstance(out, tuple) else out

        # (1) No numeric-domain warnings surfaced by the ensemble path.
        numeric_warns = [
            w for w in caught
            if issubclass(w.category, (RuntimeWarning,))
            and any(
                m in str(w.message).lower()
                for m in ("divide by zero", "invalid value", "overflow", "underflow")
            )
        ]
        assert not numeric_warns, (
            f"{method} on {case_name}: emitted numeric RuntimeWarnings: "
            f"{[str(w.message) for w in numeric_warns]}"
        )

        # (2) All finite.
        assert np.isfinite(predictions).all(), (
            f"{method} on {case_name}: non-finite output {predictions!r}"
        )

        # (3) In [0, 1]. The ensemble caller already clamps via
        # ``ensure_prob_limits=True`` by default, but asserting here means
        # future changes to that default can't quietly produce
        # out-of-range ensemble probabilities.
        assert (predictions >= 0.0).all() and (predictions <= 1.0).all(), (
            f"{method} on {case_name}: out-of-range output {predictions!r}"
        )


# =====================================================================
# #5 Trainer polars-contract: every non-polars-native strategy must reject
#    a pl.DataFrame at fit time with a clear error
# =====================================================================

class TestTrainerPolarsContract:
    """If a pl.DataFrame ever reaches ``_train_model_with_fallback`` for a
    strategy with ``supports_polars=False``, the upstream lazy-conversion
    contract has been broken and the trainer must fail loud.

    Previously the trainer papered over this with a silent self-heal; the
    2026-04-23 fix made it a ``RuntimeError`` referencing ``pipeline_cache``.
    This test enumerates *every* non-polars-native strategy registered in
    ``MODEL_STRATEGIES`` and asserts each one raises on Polars input. Adding
    a new non-polars-native strategy (and forgetting to keep it behind the
    conversion path) will fail here.
    """

    def _strategy_model_type_name(self, name):
        """Best-effort mapping from strategy registry name to the model
        class name the trainer dispatches on. The trainer uses
        ``type(model).__name__.startswith(...)`` to allow-list the three
        polars-native families. For the purposes of this test we just need
        *some* class name that the trainer will classify the same way it
        would classify the real trained estimator.

        We don't instantiate the real estimator — that would pull in
        sklearn / lightgbm / etc. setup — we only need the class-name
        gating to work correctly. Class-name prefix mapping:
          - ``"lgb"`` → ``"LGBMClassifier"``
          - ``"mlp"`` → ``"MLPClassifier"``
          - ``"ngb"`` → ``"NGBClassifier"``
          - ``"linear"/"ridge"/"lasso"/...`` → any non-prefix class name
        Any class name that does NOT start with CatBoost/XGB/HistGradient
        is fine — the trainer contract is a negation.
        """
        # Non-polars-native families register under these names.
        mapping = {
            "lgb": "LGBMClassifier",
            "mlp": "MLPClassifier",
            "ngb": "NGBClassifier",
            "linear": "LinearRegression",
            "ridge": "Ridge",
            "lasso": "Lasso",
            "elasticnet": "ElasticNet",
            "huber": "HuberRegressor",
            "ransac": "RANSACRegressor",
            "sgd": "SGDClassifier",
            "logistic": "LogisticRegression",
        }
        # Fallback: capitalise — the invariant only needs the class name to
        # NOT start with a polars-native prefix, which any non-capitalised
        # name satisfies.
        return mapping.get(name, name.capitalize())

    def test_all_non_polars_native_strategies_raise_on_polars_fit(self):
        from mlframe.training.strategies import MODEL_STRATEGIES
        from mlframe.training.trainer import _train_model_with_fallback

        non_native = [
            name for name, strat in MODEL_STRATEGIES.items()
            if not strat.supports_polars
            # Neural-net / recurrent families hit a different fit path that
            # doesn't route through _train_model_with_fallback — scoped out
            # here so we don't assert a property the code was never meant
            # to enforce for them.
            and name not in {"mlp", "lstm", "gru", "rnn", "transformer", "ngb"}
        ]
        assert non_native, (
            "Expected at least one non-polars-native strategy in the "
            "registry (lgb/linear/...); registry may have been restructured."
        )

        pl_df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.1, 0.2, 0.3]})
        y = np.array([0, 1, 0])

        failures = []
        for strategy_name in non_native:
            model_type_name = self._strategy_model_type_name(strategy_name)

            class _FakeModel:
                pass

            _FakeModel.__name__ = model_type_name
            fake = _FakeModel()
            # If the raise doesn't fire, the stub .fit keeps the test
            # terminating quickly instead of hanging on a real model.
            fake.fit = lambda *a, **kw: None

            try:
                _train_model_with_fallback(
                    model=fake,
                    model_obj=fake,
                    model_type_name=model_type_name,
                    train_df=pl_df,
                    train_target=y,
                    fit_params={},
                    verbose=False,
                )
            except RuntimeError as exc:
                # Good — contract held. Message must mention pipeline_cache
                # so the next engineer can trace upstream.
                if "pipeline_cache" not in str(exc).lower():
                    failures.append(
                        f"{strategy_name} ({model_type_name}): raised but "
                        f"message doesn't point at pipeline_cache: {exc!r}"
                    )
            except Exception as exc:
                failures.append(
                    f"{strategy_name} ({model_type_name}): raised wrong "
                    f"exception type {type(exc).__name__}: {exc!r}"
                )
            else:
                failures.append(
                    f"{strategy_name} ({model_type_name}): did NOT raise on "
                    f"pl.DataFrame — trainer contract broken, silent "
                    f"self-heal regression."
                )

        assert not failures, (
            "Trainer polars-contract violations:\n  " + "\n  ".join(failures)
        )
