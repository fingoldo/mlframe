"""End-to-end integration test for composite-target discovery wired
into ``train_mlframe_models_suite`` (PR4).

The component-level tests (``test_composite.py``,
``test_composite_discovery.py``) cover the building blocks. This file
asserts the wiring contract:

- Default-OFF behaviour: a regression suite call with no composite
  config produces the same ``target_by_type`` keys it always did.
- Opt-in behaviour: enabling discovery causes new composite-target
  entries to appear in ``target_by_type``, the per-target loop trains
  a model on each, and the resulting ``models_dict`` carries them.
- ``metadata["schema_version"] == 2``.
- ``metadata["composite_target_specs"]`` populated under the
  ``regression`` key.
- ``MLFRAME_DISABLE_COMPOSITE=1`` env var disables discovery even
  when the config opts in.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

# Make np available to test bodies without re-imports.

# Defer heavy imports until inside tests so module-collection time
# stays low even when this file is collected alongside other slow
# integration tests.

pytest.importorskip("lightgbm")

_LEAN_OUTPUT_CONFIG_KWARGS = dict(
    save_charts=False,
    run_diagnostics=["cv_informativeness", "compare_cv_schemes", "group_leakage", "constant_group_leak", "subpopulation_drift"],
)
_LEAN_REPORTING_CONFIG_KWARGS = dict(
    show_perf_chart=False,
    show_fi=False,
    adversarial_validation=False,
    interaction_strength_charts=False,
    engineered_separability_charts=False,
    class_structure_charts=False,
    category_discriminability_charts=False,
    slice_finder=False,
    shap_panels=False,
    decision_curve=False,
    calibration_drift=False,
    target_acf=False,
    model_comparison=False,
)


def _tvt_dataset(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """TVT-style: y = 0.95*lag + structural signal + noise."""
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=10.0, scale=3.0, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.95 * base + 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"TVT_prev": base, "x1": x1, "x2": x2, "target": y})


def _apply_pre_pipeline_if_fitted(pre_pipeline, X):
    """Apply ``pre_pipeline.transform`` only when it's genuinely fitted, else pass ``X`` through unchanged.

    Mirrors ``predict_mlframe_models_suite``'s own guard (``_predict_main_suite.py``): some model
    entries legitimately carry an UNFITTED placeholder pre_pipeline (e.g. tree strategies that skip
    scaling), and calling ``.transform()`` on it raises ``NotFittedError``. The real suite predict
    driver checks fitted-state first and skips gracefully rather than crashing; a test that calls
    ``entry.model.predict()`` directly (bypassing that driver) must replay the same guard.
    """
    if pre_pipeline is None:
        return X
    from sklearn.utils.validation import check_is_fitted

    try:
        check_is_fitted(pre_pipeline)
    except Exception:
        return X
    return pre_pipeline.transform(X)


def _build_minimal_fte(target_col: str = "target"):
    """Construct the simple FTE used by the existing core tests."""
    from tests.training.shared import SimpleFeaturesAndTargetsExtractor

    return SimpleFeaturesAndTargetsExtractor(
        target_column=target_col,
        regression=True,
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def opted_in_suite(tmp_path_factory):
    """One suite trained with discovery on, shared by every test below that only reads what that run produced.

    Six tests used to fit nearly the same 400-row frame with nearly the same config and then inspect one metadata slot
    each; the file took ten minutes of the composite directory's wall time for assertions that never needed a run of
    their own. This config is the union of what they asked for: both transforms, the y-scale metric block on, and the
    cross-target ensemble on. Returns ``(models, metadata, df, records)``.
    """
    import logging

    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    from mlframe.training.core import train_mlframe_models_suite

    df = _tvt_dataset(n=400)
    cfg = CompositeTargetDiscoveryConfig(
        # A trained composite is the subject of these tests, so the fixture has to produce one: on a 400-row frame a spec
        # beats raw but cannot clear the default 2-SE significance floor, which is the right production call and would
        # leave every assertion vacuous.
        min_honest_gain_z=0.0,
        enabled=True,
        min_honest_gain_to_train=None,
        base_candidates=["TVT_prev"],
        transforms=["diff", "linear_residual"],
        mi_sample_n=200,
        top_k_after_mi=2,
        eps_mi_gain=-1.0,
        skip_wrap_pass_predict=False,  # the y-scale metric block is one of the units under test
        cross_target_ensemble_strategy="oof_weighted",
    )
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        """Keeps every record the run emits, for the tests that assert on a log line."""

        def emit(self, record):
            """Keep the record for the assertions."""
            records.append(record)

    handler = _Collect(level=logging.INFO)
    root = logging.getLogger("mlframe")
    root.addHandler(handler)
    previous_level = root.level
    root.setLevel(logging.INFO)
    try:
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="composite_shared",
            features_and_targets_extractor=_build_minimal_fte(),
            mlframe_models=["linear"],
            output_config={
                "data_dir": str(tmp_path_factory.mktemp("composite_shared")),
                "models_dir": "models",
                **_LEAN_OUTPUT_CONFIG_KWARGS,
            },
            reporting_config=_LEAN_REPORTING_CONFIG_KWARGS,
            verbose=0,
            composite_target_discovery_config=cfg,
        )
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)
    return models, metadata, df, records


class TestCompositeIntegration:
    """Groups tests covering composite integration."""
    def test_default_off_does_not_add_composite_targets(self, tmp_path) -> None:
        """No composite config -> ``metadata['composite_target_specs']``
        is an empty dict and ``models_dict`` carries only the original
        target."""
        from mlframe.training.core import train_mlframe_models_suite

        df = _tvt_dataset(n=400)
        _models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="composite_off",
            features_and_targets_extractor=_build_minimal_fte(),
            mlframe_models=["linear"],
            output_config={"data_dir": str(tmp_path / "data"), "models_dir": "models", **_LEAN_OUTPUT_CONFIG_KWARGS},
            reporting_config=_LEAN_REPORTING_CONFIG_KWARGS,
            verbose=0,
        )
        assert metadata.get("schema_version") == 2
        # Composite hooks present but empty.
        assert metadata.get("composite_target_specs") == {}
        assert metadata.get("composite_target_failures") == {}

    def test_opt_in_populates_composite_target_specs(self, opted_in_suite) -> None:
        """Enabling discovery puts at least one spec under ``regression / target`` and the composite target in the model dict."""
        from mlframe.training.configs import TargetTypes

        _models, metadata, _df, _records = opted_in_suite
        assert metadata.get("schema_version") == 2
        specs = metadata.get("composite_target_specs", {})
        assert "regression" in specs or TargetTypes.REGRESSION in specs
        regression_specs = specs.get("regression") or specs.get(TargetTypes.REGRESSION) or {}
        assert "target" in regression_specs, f"expected composite specs under regression/target, got {regression_specs}"
        spec_list = regression_specs["target"]
        assert len(spec_list) >= 1
        for s in spec_list:
            assert {"name", "target_col", "transform_name", "base_column", "fitted_params"}.issubset(s)
            assert s["target_col"] == "target"
            assert s["base_column"] == "TVT_prev"

    def test_composite_models_predict_in_y_scale_after_wrap(self, opted_in_suite) -> None:
        """A wrapped composite predicts on the y scale, not the T scale: T-scale residual predictions sit near zero."""
        from mlframe.training.composite import CompositeTargetEstimator

        models, _metadata, df, _records = opted_in_suite
        # Find the composite target entry.
        regression_models = (
            models.get("regression") or models.get(__import__("mlframe.training.configs", fromlist=["TargetTypes"]).TargetTypes.REGRESSION) or {}
        )
        # Whichever spec the gates ship is the subject here: the test is about the wrap, not about which transform won.
        # The spec names in the metadata are the authority, so a key is composite when the run recorded it as a spec.
        spec_names = {spec["name"] for by_target in _metadata.get("composite_target_specs", {}).values() for specs in by_target.values() for spec in specs}
        composite_keys = [k for k in regression_models if k in spec_names]
        assert composite_keys, f"expected at least one composite-target key in models[regression], got {list(regression_models.keys())}"
        composite_entries = regression_models[composite_keys[0]]
        assert composite_entries, "composite target should have at least one entry"

        # Check at least one entry's model is now a CompositeTargetEstimator
        # OR the entry itself is one. Both wrapping flavours are valid.
        wrapped_count = 0
        for entry in composite_entries:
            inner_model = getattr(entry, "model", None) or entry
            if isinstance(inner_model, CompositeTargetEstimator):
                wrapped_count += 1
        assert wrapped_count > 0, "no entries wrapped in CompositeTargetEstimator; predictions will still be in T-scale"

        # Verify predictions are in y-scale by predicting on a sample
        # row and checking the magnitude is within the y range, not
        # the T (residual) range.
        #
        # Route the raw user frame through the SAME (extensions -> pre_pipeline -> estimator)
        # chain the suite's own predict driver (predict_mlframe_models) applies: row-wise
        # extension columns (row_summary_*/row_extreme_*, default ON) are baked into every
        # model's fit-time feature set at the suite level, so calling entry.model.predict()
        # on the bare user columns raises "feature names ... unseen at fit time". Calling
        # entry.model.predict() directly (bypassing the suite driver) is exactly what this
        # test does, so it must replay that chain itself.
        from mlframe.training.core._predict_pre_pipeline import (
            _apply_extensions_pipeline, _apply_row_wise_extensions,
        )

        sample_X = df.drop(columns=["target"]).iloc[:5]
        _ext_pipeline = _metadata.get("extensions_pipeline")
        sample_X_ext = _apply_extensions_pipeline(sample_X, _ext_pipeline, verbose=0) if _ext_pipeline is not None else sample_X
        sample_X_ext = _apply_row_wise_extensions(sample_X_ext, _metadata.get("row_wise_extensions_config"), verbose=0)
        for entry in composite_entries:
            inner_model = getattr(entry, "model", None) or entry
            if not isinstance(inner_model, CompositeTargetEstimator):
                continue
            _pp = getattr(entry, "pre_pipeline", None)
            X_final = _apply_pre_pipeline_if_fitted(_pp, sample_X_ext)
            # The wrapper reads its base from the raw frame and takes the pre_pipeline output as ``inner_X`` (the contract
            # predict_from_models uses); feeding it the scaled frame as X inverts T at a standardised base.
            preds = inner_model.predict(sample_X_ext) if X_final is sample_X_ext else inner_model.predict(sample_X_ext, inner_X=X_final)
            assert np.all(np.isfinite(preds))
            # y-scale predictions track the true y of these rows; T-scale (residual) predictions sit near zero instead.
            y_rows = df["target"].iloc[:5].to_numpy()
            assert abs(float(np.mean(preds)) - float(np.mean(y_rows))) < 0.5 * float(
                df["target"].std()
            ), f"predictions {preds} do not track y {y_rows}; looks like T-scale (residual) instead of y-scale"

    def test_oof_holdout_gate_runs_without_crashing(self, tmp_path) -> None:
        """When ``oof_holdout_frac > 0``, the post-loop ensemble path
        must compute honest holdout predictions (re-fit clones on
        stack_train, predict on stack_holdout) and use them for
        weighting / the validation gate. Smoke test: just verify the
        suite completes successfully and produces an ensemble entry."""
        from mlframe.training.configs import CompositeTargetDiscoveryConfig
        from mlframe.training.core import train_mlframe_models_suite

        df = _tvt_dataset(n=600)
        cfg = CompositeTargetDiscoveryConfig(
            # These tests are about what a TRAINED composite target does (wrapping, persistence, serving), so they need
            # one to be trained: on a 400-row fixture a spec beats raw but cannot clear the default 2-SE significance
            # floor on its paired gain, which is the right production call and would leave every assertion vacuous.
            min_honest_gain_z=0.0,
            enabled=True,
            min_honest_gain_to_train=None,  # these tests check the suite wiring; the fixture's gains sit below the ship floor
            base_candidates=["TVT_prev"],
            transforms=["diff", "linear_residual"],
            mi_sample_n=300,
            top_k_after_mi=2,
            eps_mi_gain=-1.0,
            cross_target_ensemble_strategy="oof_weighted",
            oof_holdout_frac=0.2,  # 20% honest holdout
            oof_random_state=7,
        )
        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="composite_oof_gate",
            features_and_targets_extractor=_build_minimal_fte(),
            mlframe_models=["linear"],
            output_config={"data_dir": str(tmp_path / "data"), "models_dir": "models", **_LEAN_OUTPUT_CONFIG_KWARGS},
            reporting_config=_LEAN_REPORTING_CONFIG_KWARGS,
            verbose=0,
            composite_target_discovery_config=cfg,
        )
        # Either the ensemble entry exists OR the gate fired and
        # left a single best component instead. Both are valid
        # outcomes; the test just verifies the OOF code path
        # completes without crashing.
        regression = models.get("regression") or models.get(__import__("mlframe.training.configs", fromlist=["TargetTypes"]).TargetTypes.REGRESSION) or {}
        ensemble_keys = [k for k in regression if k.startswith("_CT_ENSEMBLE__")]
        # Ensemble entry may or may not exist depending on whether
        # the gate fired.
        # Validate: at least one composite-target entry exists either
        # way (post-wrap from PR5).
        composite_keys = [k for k in regression if "linear_residual" in k or "diff" in k]
        assert len(composite_keys + ensemble_keys) > 0

    def test_y_scale_metrics_populated_after_wrap(self, opted_in_suite) -> None:
        """The per-target loop reports composite metrics on the y scale beside the T-scale ones."""
        _models, metadata, df, _records = opted_in_suite
        y_metrics = metadata.get("composite_target_y_scale_metrics", {})
        assert y_metrics, "expected y-scale metrics to be populated"
        regression_metrics = y_metrics.get("regression") or y_metrics.get(
            __import__("mlframe.training.configs", fromlist=["TargetTypes"]).TargetTypes.REGRESSION,
        )
        assert regression_metrics
        # At least one composite entry, and it has train metrics.
        spec_names = {spec["name"] for by_target in metadata.get("composite_target_specs", {}).values() for specs in by_target.values() for spec in specs}
        composite_keys = [k for k in regression_metrics if k in spec_names]
        assert composite_keys, f"no y-scale metrics for any shipped spec; metrics keys {list(regression_metrics)}, specs {spec_names}"
        per_entry_metrics = regression_metrics[composite_keys[0]]
        assert per_entry_metrics  # at least one entry
        # First entry has at least train RMSE.
        train_metrics = per_entry_metrics[0].get("metrics", {}).get("train", {})
        assert "RMSE" in train_metrics
        # Measured against the target's own spread: an additive residual on TVT_prev reaches y-RMSE 0.29 with std(y) 2.95
        # (ratio 0.10). An inverse that returned T, or added the base twice, lands near the base level (~10), far above.
        y_std = float(np.std(np.asarray(df["target"], dtype=np.float64)))
        assert 0 < train_metrics["RMSE"] < 0.25 * y_std, f"train y-RMSE {train_metrics['RMSE']:.4g} vs std(y) {y_std:.4g}"

    def test_cross_target_ensemble_creates_aggregate_entry(self, opted_in_suite) -> None:
        """With a cross-target strategy on, the suite produces a ``_CT_ENSEMBLE__{target}`` entry that predicts on the y scale."""
        from mlframe.training.composite import (
            CompositeCrossTargetEnsemble,
            CompositeTargetEstimator,
        )

        models, metadata, df, _records = opted_in_suite
        # Strict: ensemble must be reachable via the enum key that
        # downstream consumers (predict_mlframe_models) iterate. A
        # fallback-string-OR-enum chain would mask the regression
        # users actually hit (silent absence of CT_ENSEMBLE entries
        # despite the gate firing) so we keep both checks separate
        # and assert BOTH succeed.
        from mlframe.training.configs import TargetTypes as _TT

        regression_models_via_enum = models.get(_TT.REGRESSION) or {}
        regression_models_via_str = models.get("regression") or {}
        # StrEnum invariant: both lookups must agree.
        assert (
            regression_models_via_enum.keys() == regression_models_via_str.keys()
        ), "models dict has divergent string vs enum keys -- StrEnum invariant violated; cross-target ensemble write path likely used wrong key type."
        regression_models = regression_models_via_enum
        # Look for the ensemble key.
        ensemble_keys = [k for k in regression_models if k.startswith("_CT_ENSEMBLE__")]
        assert ensemble_keys, f"expected _CT_ENSEMBLE__ entry, got keys={list(regression_models.keys())}"
        ens_entries = regression_models[ensemble_keys[0]]
        assert len(ens_entries) == 1
        ens_entry = ens_entries[0]
        # Keep the (possibly shim-wrapped) object used for the ACTUAL predict call separate
        # from ``ens_model`` (used only for isinstance/type checks below): calling .predict()
        # on the shim -- rather than unwrapping past it first -- lets the shim apply its own
        # pre_pipeline.transform() (scaler/imputer), which the raw inner estimator needs.
        ens_model_for_predict = getattr(ens_entry, "model", None)
        ens_model = ens_model_for_predict
        # Optional ``PrePipelinePredictShim`` wrap when cross-target components
        # needed an Imputer/StandardScaler pre-pipeline routed through predict;
        # unwrap one shim level so the isinstance check tests the real inner.
        from mlframe.training.composite.post_shim import PrePipelinePredictShim

        if isinstance(ens_model, PrePipelinePredictShim):
            ens_model = ens_model.model
        # The honest gate-chain can degrade the ensemble through up to THREE
        # fallback layers before emitting ``ens_model``:
        #   (a) ``CompositeCrossTargetEnsemble`` -- happy path, multi-component
        #       weighted ensemble.
        #   (b) ``CompositeTargetEstimator`` -- best-single-component fallback
        #       when the ensemble's OOF RMSE doesn't beat the solo best (added
        #       2026-05-27 in _phase_composite_post_xt_ensemble).
        #   (c) the raw inner estimator (e.g. ``Ridge``) -- bottom-of-chain
        #       fallback when even the best composite component fails to beat
        #       the dummy-floor baseline and only the raw#0 entry remains.
        # All three preserve the suite-level contract this test guards: a
        # ``_CT_ENSEMBLE__{target}`` aggregate entry that has callable
        # ``predict`` and emits finite y-scale predictions. Assert the real
        # contract (callable predict + downstream finite-prediction check)
        # rather than pin a specific fallback layer.
        assert ens_model_for_predict is not None and callable(
            getattr(ens_model_for_predict, "predict", None)
        ), f"_CT_ENSEMBLE__ aggregate entry missing predict(); got {type(ens_model_for_predict).__name__}"
        # Predict on a sample row. Row-wise extension columns (row_summary_*/row_extreme_*,
        # default ON) are baked into every model's fit-time feature set at the suite level, so
        # a raw user-column frame must be replayed through the suite's own extensions_pipeline
        # first -- mirroring predict_mlframe_models -- before reaching predict().
        from mlframe.training.core._predict_pre_pipeline import (
            _apply_extensions_pipeline, _apply_row_wise_extensions,
        )

        sample_X = df.drop(columns=["target"]).iloc[:5]
        _ext_pipeline = metadata.get("extensions_pipeline")
        sample_X_ext = _apply_extensions_pipeline(sample_X, _ext_pipeline, verbose=0) if _ext_pipeline is not None else sample_X
        sample_X_ext = _apply_row_wise_extensions(sample_X_ext, metadata.get("row_wise_extensions_config"), verbose=0)
        preds = ens_model_for_predict.predict(sample_X_ext)
        assert np.all(np.isfinite(preds))
        # y-scale magnitude check. The CompositeTargetEstimator /
        # CompositeCrossTargetEnsemble wraps apply a prediction-envelope
        # clip on the predict path; the raw-inner fallback (e.g. Ridge
        # directly) skips that safety net and on tiny synthetic frames
        # can produce predictions outside the in-sample y range. The
        # suite-level contract (sane finite emission) is already covered
        # by the isfinite check above; only enforce the tight 0.5x /
        # 1.5x bounds when a composite wrap is in place to provide the
        # clip.
        _has_composite_clip = isinstance(
            ens_model,
            (CompositeCrossTargetEnsemble, CompositeTargetEstimator),
        )
        if _has_composite_clip:
            y_range = (df["target"].min(), df["target"].max())
            assert preds.min() > 0.5 * y_range[0]
            assert preds.max() < 1.5 * y_range[1]
        # Metadata exports. The metadata schema depends on which fallback
        # layer fired:
        #   * Happy path / single-component composite wrap -> the metadata
        #     dict carries ``weights`` + ``component_names`` (the ensemble
        #     produced its standard introspection block).
        #   * single_best_fallback (raw-inner Ridge / lowest-RMSE component
        #     fallback) -> the dict is the marker ``{"strategy":
        #     "single_best_fallback"}``; weights/component_names are not
        #     applicable because there's no ensemble to introspect.
        # Pin the union of valid shapes so the sensor catches a missing
        # metadata block in EITHER path without bouncing on the path the
        # gate-chain happened to land on.
        ens_meta = metadata.get("composite_target_ensemble", {}).get("regression", {}).get("target")
        assert ens_meta is not None
        _has_ensemble_block = "weights" in ens_meta and "component_names" in ens_meta
        _has_fallback_marker = ens_meta.get("strategy") == "single_best_fallback"
        assert _has_ensemble_block or _has_fallback_marker, (
            f"ens_meta must carry either the ensemble introspection block "
            f"(weights + component_names) OR the single_best_fallback marker; "
            f"got keys={list(ens_meta)}"
        )

    def test_cross_target_ensemble_entry_banner_logged(self, opted_in_suite) -> None:
        """Whenever discovery is enabled the suite emits the ``[CompositeCrossTargetEnsemble] entry:`` banner.

        "The CompositeCrossTargetEnsemble line is absent from my output" is hard to diagnose without a line that always
        fires, so the banner is logged whether or not the gate then opens.
        """
        _models, _metadata, _df, records = opted_in_suite
        banners = [r for r in records if "[CompositeCrossTargetEnsemble] entry:" in r.getMessage()]
        assert banners, (
            "expected at least one entry banner from the cross-target ensemble gate so a missing ensemble is "
            f"diagnosable; got {[r.getMessage() for r in records[-20:]]}"
        )

    def test_composite_dummy_baseline_inverted_to_y_scale(self, tmp_path) -> None:
        """When the per-target loop computes dummy baselines on a
        composite target, the strongest dummy predictions live on the
        T-scale (e.g. ``median(T_train)``). The suite-end verdict block
        compares them against the wrapped composite model's y-scale
        RMSE, so the T-scale dummy must be inverted to y-scale via the
        spec's ``transform.inverse`` before comparison — otherwise the
        lift is apples-to-oranges and falsely fires
        ``MODELS_BARELY_BEAT_TRIVIAL``.

        This test locks the inversion contract:
        ``metadata['dummy_baselines'][regression][<composite_name>]
        ['y_scale_strongest_metrics']`` must be populated for both splits.
        For the additive ``linear_residual`` the inverted dummy's y-error
        equals its T-error row by row, so the y-scale RMSE must equal the
        T-scale one; the inverted dummy uses the base, so it is
        residual-sized rather than the raw target's spread.
        """
        from mlframe.training.configs import CompositeTargetDiscoveryConfig
        from mlframe.training.core import train_mlframe_models_suite

        df = _tvt_dataset(n=600)
        cfg = CompositeTargetDiscoveryConfig(
            # These tests are about what a TRAINED composite target does (wrapping, persistence, serving), so they need
            # one to be trained: on a 400-row fixture a spec beats raw but cannot clear the default 2-SE significance
            # floor on its paired gain, which is the right production call and would leave every assertion vacuous.
            min_honest_gain_z=0.0,
            enabled=True,
            min_honest_gain_to_train=None,  # these tests check the suite wiring; the fixture's gains sit below the ship floor
            base_candidates=["TVT_prev"],
            transforms=["linear_residual"],
            mi_sample_n=200,
            top_k_after_mi=1,
            eps_mi_gain=-1.0,
            cross_target_ensemble_strategy="off",
        )
        _models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="composite_yscale_dummy",
            features_and_targets_extractor=_build_minimal_fte(),
            mlframe_models=["linear"],
            output_config={"data_dir": str(tmp_path / "data"), "models_dir": "models", **_LEAN_OUTPUT_CONFIG_KWARGS},
            reporting_config=_LEAN_REPORTING_CONFIG_KWARGS,
            verbose=0,
            composite_target_discovery_config=cfg,
        )
        db = metadata.get("dummy_baselines", {}).get("regression", {})
        # Find the composite-target entry. Match both the legacy long form
        # ('__linear_residual__') and the new short alias ('-linres-') -
        # composite_transforms.py:1380 switched to short names 2026-05-16.
        composite_names = [n for n in db if "__linear_residual__" in n or "-linres-" in n]
        assert composite_names, f"expected a composite target dummy entry; got keys={list(db.keys())}"
        rep = db[composite_names[0]]
        ys = rep.get("y_scale_strongest_metrics")
        assert ys, (
            "expected y_scale_strongest_metrics populated for composite "
            "target (inverted via transform.inverse) so the suite-end "
            "verdict can compare apples-to-apples with model RMSE_y; "
            f"got: {rep.keys()}"
        )
        # Both splits must be scored: a missing split is the regression, not something to step over.
        assert {"val", "test"} <= set(ys), f"y-scale dummy metrics cover only {sorted(ys)}"
        t_scale = rep["data"][rep["strongest"]]
        y_std = float(np.std(np.asarray(df["target"], dtype=np.float64)))
        for split in ("val", "test"):
            assert np.isfinite(ys[split]["RMSE"]) and np.isfinite(ys[split]["MAE"])
            # linear_residual is additive in T (y = T + alpha*base + beta), so the inverted dummy's y-error equals its T-error
            # row by row: the y-scale RMSE must equal the T-scale one. The inverted dummy (median(T) + alpha*base) uses the
            # base, so it is residual-sized (measured 0.70-0.73), well under a raw constant's std(y) of 2.97.
            np.testing.assert_allclose(ys[split]["RMSE"], t_scale[f"{split}_RMSE"], rtol=1e-6, err_msg=f"{split}: y-scale dummy RMSE != T-scale")
            assert ys[split]["RMSE"] < 0.5 * y_std, f"{split}: inverted dummy RMSE {ys[split]['RMSE']:.4g} vs std(y) {y_std:.4g}"

    def test_env_var_kill_switch_disables_even_when_config_opts_in(self, tmp_path) -> None:
        """``MLFRAME_DISABLE_COMPOSITE=1`` must override the config."""
        from mlframe.training.configs import CompositeTargetDiscoveryConfig
        from mlframe.training.core import train_mlframe_models_suite

        df = _tvt_dataset(n=400)
        cfg = CompositeTargetDiscoveryConfig(
            # These tests are about what a TRAINED composite target does (wrapping, persistence, serving), so they need
            # one to be trained: on a 400-row fixture a spec beats raw but cannot clear the default 2-SE significance
            # floor on its paired gain, which is the right production call and would leave every assertion vacuous.
            min_honest_gain_z=0.0,
            enabled=True,
            min_honest_gain_to_train=None,  # these tests check the suite wiring; the fixture's gains sit below the ship floor
            base_candidates=["TVT_prev"],
            transforms=["diff"],
            mi_sample_n=200,
            eps_mi_gain=-1.0,
        )
        old = os.environ.get("MLFRAME_DISABLE_COMPOSITE", "")
        os.environ["MLFRAME_DISABLE_COMPOSITE"] = "1"
        try:
            _models, metadata = train_mlframe_models_suite(
                df=df,
                target_name="target",
                model_name="composite_killswitch",
                features_and_targets_extractor=_build_minimal_fte(),
                mlframe_models=["linear"],
                output_config={"data_dir": str(tmp_path / "data"), "models_dir": "models", **_LEAN_OUTPUT_CONFIG_KWARGS},
                reporting_config=_LEAN_REPORTING_CONFIG_KWARGS,
                verbose=0,
                composite_target_discovery_config=cfg,
            )
        finally:
            if old:
                os.environ["MLFRAME_DISABLE_COMPOSITE"] = old
            else:
                os.environ.pop("MLFRAME_DISABLE_COMPOSITE", None)
        # Kill switch -> empty specs.
        assert metadata.get("composite_target_specs") == {}
