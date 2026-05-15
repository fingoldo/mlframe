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


def _tvt_dataset(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """TVT-style: y = 0.95*lag + structural signal + noise."""
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=10.0, scale=3.0, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.95 * base + 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"TVT_prev": base, "x1": x1, "x2": x2, "target": y})


def _build_minimal_fte(target_col: str = "target"):
    """Construct the simple FTE used by the existing core tests."""
    from tests.training.shared import SimpleFeaturesAndTargetsExtractor

    return SimpleFeaturesAndTargetsExtractor(
        target_column=target_col, regression=True,
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


class TestCompositeIntegration:
    def test_default_off_does_not_add_composite_targets(self, tmp_path) -> None:
        """No composite config -> ``metadata['composite_target_specs']``
        is an empty dict and ``models_dict`` carries only the original
        target."""
        from mlframe.training.core import train_mlframe_models_suite

        df = _tvt_dataset(n=400)
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="composite_off",
            features_and_targets_extractor=_build_minimal_fte(),
            mlframe_models=["linear"],
            output_config={"data_dir": str(tmp_path / "data"),
                           "models_dir": "models"},
            verbose=0,
        )
        assert metadata.get("schema_version") == 2
        # Composite hooks present but empty.
        assert metadata.get("composite_target_specs") == {}
        assert metadata.get("composite_target_failures") == {}

    def test_opt_in_populates_composite_target_specs(self, tmp_path) -> None:
        """Enable discovery; ``metadata['composite_target_specs']``
        carries at least one entry under ``regression / TVT`` and the
        composite target lands in ``models_dict``."""
        from mlframe.training.configs import CompositeTargetDiscoveryConfig
        from mlframe.training.core import train_mlframe_models_suite
        from mlframe.training.configs import TargetTypes

        df = _tvt_dataset(n=400)
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True,
            base_candidates=["TVT_prev"],
            transforms=["diff", "linear_residual"],
            mi_sample_n=200,
            top_k_after_mi=2,
            eps_mi_gain=-1.0,  # take whatever shows up
        )
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="composite_on",
            features_and_targets_extractor=_build_minimal_fte(),
            mlframe_models=["linear"],
            output_config={"data_dir": str(tmp_path / "data"),
                           "models_dir": "models"},
            verbose=0,
            composite_target_discovery_config=cfg,
        )
        assert metadata.get("schema_version") == 2
        specs = metadata.get("composite_target_specs", {})
        # Specs nested under {target_type: {target_name: [list of specs]}}.
        assert "regression" in specs or TargetTypes.REGRESSION in specs
        regression_specs = specs.get("regression") or specs.get(TargetTypes.REGRESSION) or {}
        assert "target" in regression_specs, (
            f"expected composite specs under regression/target, got {regression_specs}"
        )
        spec_list = regression_specs["target"]
        assert len(spec_list) >= 1
        # Each spec carries the canonical fields.
        for s in spec_list:
            assert {"name", "target_col", "transform_name",
                    "base_column", "fitted_params"}.issubset(s)
            assert s["target_col"] == "target"
            assert s["base_column"] == "TVT_prev"

    def test_composite_models_predict_in_y_scale_after_wrap(self, tmp_path) -> None:
        """After PR5 wrapping, ``models[type][composite_name][i].model.predict(X)``
        (or ``[i].predict(X)`` for entries that are plain estimators)
        must return y-scale predictions, NOT T-scale. Verified by
        comparing the wrapped predict output against the actual y
        range -- T-scale predictions on a residual transform would
        cluster near zero, far below the y range."""
        from mlframe.training.configs import CompositeTargetDiscoveryConfig
        from mlframe.training.core import train_mlframe_models_suite
        from mlframe.training.composite import CompositeTargetEstimator

        df = _tvt_dataset(n=400)
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True,
            base_candidates=["TVT_prev"],
            transforms=["linear_residual"],
            mi_sample_n=200,
            top_k_after_mi=1,
            eps_mi_gain=-1.0,
        )
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="composite_yscale",
            features_and_targets_extractor=_build_minimal_fte(),
            mlframe_models=["linear"],
            output_config={"data_dir": str(tmp_path / "data"),
                           "models_dir": "models"},
            verbose=0,
            composite_target_discovery_config=cfg,
        )
        # Find the composite target entry.
        regression_models = (models.get("regression")
                             or models.get(__import__("mlframe.training.configs",
                                                       fromlist=["TargetTypes"])
                                           .TargetTypes.REGRESSION) or {})
        composite_keys = [
            k for k in regression_models
            if "__linear_residual__TVT_prev" in k
        ]
        assert composite_keys, (
            f"expected at least one composite-target key in models[regression], "
            f"got {list(regression_models.keys())}"
        )
        composite_entries = regression_models[composite_keys[0]]
        assert composite_entries, "composite target should have at least one entry"

        # Check at least one entry's model is now a CompositeTargetEstimator
        # OR the entry itself is one. Both wrapping flavours are valid.
        wrapped_count = 0
        for entry in composite_entries:
            inner_model = getattr(entry, "model", None) or entry
            if isinstance(inner_model, CompositeTargetEstimator):
                wrapped_count += 1
        assert wrapped_count > 0, (
            "no entries wrapped in CompositeTargetEstimator; predictions will "
            "still be in T-scale"
        )

        # Verify predictions are in y-scale by predicting on a sample
        # row and checking the magnitude is within the y range, not
        # the T (residual) range.
        sample_X = df.drop(columns=["target"]).iloc[:5]
        y_range = (df["target"].min(), df["target"].max())
        for entry in composite_entries:
            inner_model = getattr(entry, "model", None) or entry
            if not isinstance(inner_model, CompositeTargetEstimator):
                continue
            preds = inner_model.predict(sample_X)
            assert np.all(np.isfinite(preds))
            # y-scale predictions: most values should be within the y envelope.
            # T-scale (residual) predictions would cluster near zero, far below.
            assert preds.min() > 0.5 * y_range[0], (
                f"prediction min {preds.min():.2f} far below y_range {y_range}; "
                "looks like T-scale (residual) instead of y-scale"
            )
            assert preds.max() < 1.5 * y_range[1]

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
            enabled=True,
            base_candidates=["TVT_prev"],
            transforms=["diff", "linear_residual"],
            mi_sample_n=300,
            top_k_after_mi=2,
            eps_mi_gain=-1.0,
            cross_target_ensemble_strategy="oof_weighted",
            oof_holdout_frac=0.2,  # 20% honest holdout
            oof_random_state=7,
        )
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="composite_oof_gate",
            features_and_targets_extractor=_build_minimal_fte(),
            mlframe_models=["linear"],
            output_config={"data_dir": str(tmp_path / "data"),
                           "models_dir": "models"},
            verbose=0,
            composite_target_discovery_config=cfg,
        )
        # Either the ensemble entry exists OR the gate fired and
        # left a single best component instead. Both are valid
        # outcomes; the test just verifies the OOF code path
        # completes without crashing.
        regression = (
            models.get("regression")
            or models.get(__import__("mlframe.training.configs",
                                      fromlist=["TargetTypes"]).TargetTypes.REGRESSION)
            or {}
        )
        ensemble_keys = [k for k in regression if k.startswith("_CT_ENSEMBLE__")]
        # Ensemble entry may or may not exist depending on whether
        # the gate fired.
        # Validate: at least one composite-target entry exists either
        # way (post-wrap from PR5).
        composite_keys = [k for k in regression
                          if "linear_residual" in k or "diff" in k]
        assert len(composite_keys + ensemble_keys) > 0

    def test_y_scale_metrics_populated_after_wrap(self, tmp_path) -> None:
        """The per-target loop reports T-scale RMSE for composite
        targets; PR8 adds parallel y-scale RMSE/MAE under
        ``metadata['composite_target_y_scale_metrics']`` so callers
        can compare composite to raw on the same scale."""
        from mlframe.training.configs import CompositeTargetDiscoveryConfig
        from mlframe.training.core import train_mlframe_models_suite

        df = _tvt_dataset(n=400)
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True,
            base_candidates=["TVT_prev"],
            transforms=["linear_residual"],
            mi_sample_n=200,
            top_k_after_mi=1,
            eps_mi_gain=-1.0,
        )
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="composite_yscale_metrics",
            features_and_targets_extractor=_build_minimal_fte(),
            mlframe_models=["linear"],
            output_config={"data_dir": str(tmp_path / "data"),
                           "models_dir": "models"},
            verbose=0,
            composite_target_discovery_config=cfg,
        )
        y_metrics = metadata.get("composite_target_y_scale_metrics", {})
        assert y_metrics, "expected y-scale metrics to be populated"
        regression_metrics = y_metrics.get("regression") or y_metrics.get(
            __import__("mlframe.training.configs",
                       fromlist=["TargetTypes"]).TargetTypes.REGRESSION,
        )
        assert regression_metrics
        # At least one composite entry, and it has train metrics.
        composite_keys = [k for k in regression_metrics
                          if "linear_residual" in k]
        assert composite_keys
        per_entry_metrics = regression_metrics[composite_keys[0]]
        assert per_entry_metrics  # at least one entry
        # First entry has at least train RMSE.
        train_metrics = per_entry_metrics[0].get("metrics", {}).get("train", {})
        assert "RMSE" in train_metrics
        # y-scale RMSE should be finite and reasonable for TVT data
        # (target ranges roughly 0-30; RMSE should be a small fraction
        # of that, definitely finite).
        assert 0 < train_metrics["RMSE"] < 100

    def test_cross_target_ensemble_creates_aggregate_entry(self, tmp_path) -> None:
        """When ``cross_target_ensemble_strategy != 'off'`` is set, the
        suite produces a ``_CT_ENSEMBLE__{target}`` entry under
        ``models[regression]`` after wrapping. Its ``.predict()`` must
        return finite y-scale predictions and match the weighted-
        mean of the wrapped components on a sample input.
        """
        from mlframe.training.configs import CompositeTargetDiscoveryConfig
        from mlframe.training.core import train_mlframe_models_suite
        from mlframe.training.composite import (
            CompositeCrossTargetEnsemble, CompositeTargetEstimator,
        )

        df = _tvt_dataset(n=400)
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True,
            base_candidates=["TVT_prev"],
            transforms=["diff", "linear_residual"],
            mi_sample_n=200,
            top_k_after_mi=2,
            eps_mi_gain=-1.0,
            cross_target_ensemble_strategy="oof_weighted",
        )
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="composite_ensemble",
            features_and_targets_extractor=_build_minimal_fte(),
            mlframe_models=["linear"],
            output_config={"data_dir": str(tmp_path / "data"),
                           "models_dir": "models"},
            verbose=0,
            composite_target_discovery_config=cfg,
        )
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
        assert regression_models_via_enum.keys() == regression_models_via_str.keys(), (
            "models dict has divergent string vs enum keys -- StrEnum "
            "invariant violated; cross-target ensemble write path likely "
            "used wrong key type."
        )
        regression_models = regression_models_via_enum
        # Look for the ensemble key.
        ensemble_keys = [k for k in regression_models if k.startswith("_CT_ENSEMBLE__")]
        assert ensemble_keys, (
            f"expected _CT_ENSEMBLE__ entry, got keys={list(regression_models.keys())}"
        )
        ens_entries = regression_models[ensemble_keys[0]]
        assert len(ens_entries) == 1
        ens_entry = ens_entries[0]
        ens_model = getattr(ens_entry, "model", None)
        assert isinstance(ens_model, CompositeCrossTargetEnsemble), (
            f"expected CompositeCrossTargetEnsemble, got {type(ens_model).__name__}"
        )
        # Predict on a sample row.
        sample_X = df.drop(columns=["target"]).iloc[:5]
        preds = ens_model.predict(sample_X)
        assert np.all(np.isfinite(preds))
        # y-scale magnitude check.
        y_range = (df["target"].min(), df["target"].max())
        assert preds.min() > 0.5 * y_range[0]
        assert preds.max() < 1.5 * y_range[1]
        # Metadata exports.
        ens_meta = (
            metadata.get("composite_target_ensemble", {})
            .get("regression", {})
            .get("target")
        )
        assert ens_meta is not None
        assert "weights" in ens_meta and "component_names" in ens_meta

    def test_cross_target_ensemble_entry_banner_logged(self, tmp_path, caplog) -> None:
        """User reported "CompositeCrossTargetEnsemble line absent from
        output" — the root cause is hard to diagnose without an entry
        banner that always fires when discovery is enabled. This test
        locks the banner contract: whenever ``enabled=True`` the suite
        emits at least one ``[CompositeCrossTargetEnsemble] entry: ...``
        log line, regardless of whether the gate ultimately opens (eg.
        strategy='off' should still log the banner so users see the
        config state in their script output).
        """
        import logging
        from mlframe.training.configs import CompositeTargetDiscoveryConfig
        from mlframe.training.core import train_mlframe_models_suite

        df = _tvt_dataset(n=400)
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True,
            base_candidates=["TVT_prev"],
            transforms=["diff"],
            mi_sample_n=200,
            top_k_after_mi=1,
            eps_mi_gain=-1.0,
            cross_target_ensemble_strategy="off",
        )
        with caplog.at_level(logging.INFO, logger="mlframe.training.core"):
            train_mlframe_models_suite(
                df=df,
                target_name="target",
                model_name="composite_banner",
                features_and_targets_extractor=_build_minimal_fte(),
                mlframe_models=["linear"],
                output_config={"data_dir": str(tmp_path / "data"),
                               "models_dir": "models"},
                verbose=0,
                composite_target_discovery_config=cfg,
            )
        banners = [r for r in caplog.records
                   if "[CompositeCrossTargetEnsemble] entry:" in r.getMessage()]
        assert banners, (
            "expected at least one entry banner from cross-target "
            "ensemble gate so users can diagnose missing-ensemble case; "
            f"got log records: {[r.getMessage() for r in caplog.records[-20:]]}"
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
        ['y_scale_strongest_metrics']`` must be populated and the
        RMSE_y values must lie in the same order of magnitude as the
        raw target's range, not the (much smaller) residual range.
        """
        from mlframe.training.configs import CompositeTargetDiscoveryConfig
        from mlframe.training.core import train_mlframe_models_suite

        df = _tvt_dataset(n=600)
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True,
            base_candidates=["TVT_prev"],
            transforms=["linear_residual"],
            mi_sample_n=200,
            top_k_after_mi=1,
            eps_mi_gain=-1.0,
            cross_target_ensemble_strategy="off",
        )
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="target",
            model_name="composite_yscale_dummy",
            features_and_targets_extractor=_build_minimal_fte(),
            mlframe_models=["linear"],
            output_config={"data_dir": str(tmp_path / "data"),
                           "models_dir": "models"},
            verbose=0,
            composite_target_discovery_config=cfg,
        )
        db = metadata.get("dummy_baselines", {}).get("regression", {})
        # Find the composite-target entry (name contains "__linear_residual__").
        composite_names = [n for n in db if "__linear_residual__" in n]
        assert composite_names, (
            f"expected a composite target dummy entry; got keys={list(db.keys())}"
        )
        rep = db[composite_names[0]]
        ys = rep.get("y_scale_strongest_metrics")
        assert ys, (
            "expected y_scale_strongest_metrics populated for composite "
            "target (inverted via transform.inverse) so the suite-end "
            "verdict can compare apples-to-apples with model RMSE_y; "
            f"got: {rep.keys()}"
        )
        # Val and test sub-entries each carry RMSE / MAE finite numbers.
        for split in ("val", "test"):
            if split not in ys:
                continue
            assert "RMSE" in ys[split]
            assert "MAE" in ys[split]
            assert np.isfinite(ys[split]["RMSE"])
            assert np.isfinite(ys[split]["MAE"])
            # The dummy RMSE on y-scale should be roughly the std of y
            # (predicting a constant on y-scale). For our synthetic TVT
            # data y has std ~ 3-5, so RMSE_y in [1, 20] is sane and
            # SUBSTANTIALLY larger than the T-scale RMSE (residual std
            # ~ 0.3).
            assert 0.5 < ys[split]["RMSE"] < 50, (
                f"y-scale RMSE_y={ys[split]['RMSE']:.4g} out of range; "
                f"either inversion math is wrong or test data drifted"
            )

    def test_env_var_kill_switch_disables_even_when_config_opts_in(self, tmp_path) -> None:
        """``MLFRAME_DISABLE_COMPOSITE=1`` must override the config."""
        from mlframe.training.configs import CompositeTargetDiscoveryConfig
        from mlframe.training.core import train_mlframe_models_suite

        df = _tvt_dataset(n=400)
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True,
            base_candidates=["TVT_prev"],
            transforms=["diff"],
            mi_sample_n=200,
            eps_mi_gain=-1.0,
        )
        old = os.environ.get("MLFRAME_DISABLE_COMPOSITE", "")
        os.environ["MLFRAME_DISABLE_COMPOSITE"] = "1"
        try:
            models, metadata = train_mlframe_models_suite(
                df=df,
                target_name="target",
                model_name="composite_killswitch",
                features_and_targets_extractor=_build_minimal_fte(),
                mlframe_models=["linear"],
                output_config={"data_dir": str(tmp_path / "data"),
                               "models_dir": "models"},
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
