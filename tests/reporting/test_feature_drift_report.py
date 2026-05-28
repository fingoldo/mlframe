"""Regression coverage for ``compute_feature_distribution_drift``.

Sensor lands 2026-05-22 to complement the existing
``label_distribution_drift`` -- catches the feature-side shift that broke
the TVT-2026-05-21 MLP path (Ridge tolerates 14-sigma TVT_prev drift fine
via linear extrapolation; MLP collapses).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training.feature_drift_report import (
    DEFAULT_FEATURE_DRIFT_WARN_THRESHOLD_Z,
    WEIGHTED_DRIFT_NEURAL_OVERRIDE_THRESHOLD,
    compute_feature_distribution_drift,
    translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs,
)


def _make_frames(*, drift_z: float, n: int = 1000, seed: int = 0):
    """Build train/val/test pandas frames where ``f_shift`` has its test mean
    drifted by exactly ``drift_z`` train-stds. f_stable matches across splits."""
    rng = np.random.default_rng(seed)
    train = pd.DataFrame({
        "f_stable": rng.normal(0.0, 1.0, n),
        "f_shift": rng.normal(0.0, 1.0, n),
    })
    val = pd.DataFrame({
        "f_stable": rng.normal(0.0, 1.0, n // 2),
        "f_shift": rng.normal(0.0, 1.0, n // 2),
    })
    # Inject a deterministic mean shift into f_shift on the test slice. The
    # underlying noise std stays ~1 so train_std=1 and the z is exactly drift_z.
    test = pd.DataFrame({
        "f_stable": rng.normal(0.0, 1.0, n // 2),
        "f_shift": rng.normal(drift_z, 1.0, n // 2),
    })
    return train, val, test


class TestFeatureDriftSensor:
    def test_no_drift_clean_iid_splits(self):
        train, val, test = _make_frames(drift_z=0.0)
        rep = compute_feature_distribution_drift(train, val, test)
        assert rep["n_numeric_features"] == 2
        assert rep["drift_candidates"] == []

    def test_moderate_drift_logged_info_not_warn(self, caplog):
        """8-sigma drift -- moderate by absolute scale, NOT escalated to WARN
        because (a) drift does NOT prove harm and (b) per-model FS may drop
        the feature. INFO is the right level."""
        train, val, test = _make_frames(drift_z=8.0)
        with caplog.at_level(logging.INFO):
            rep = compute_feature_distribution_drift(train, val, test)
        cands = rep["drift_candidates"]
        names = [c for c, _z in cands]
        assert "f_shift" in names
        assert "f_stable" not in names
        # The log line surfaces at INFO level with the top-drifter list.
        info_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any("[feature-distribution-drift]" in m for m in info_msgs), (
            f"INFO log missing on moderate drift; got info_msgs={info_msgs}"
        )
        # And NO WARN should fire at this magnitude without FI weighting.
        warn_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("[feature-distribution-drift]" in m for m in warn_msgs), (
            f"WARN level fired on moderate drift without FI weighting; "
            f"design says only escalate at >=10x sigma OR weighted>=1.0. warn_msgs={warn_msgs}"
        )

    def test_extreme_drift_escalates_to_warn(self, caplog):
        """>= 10x threshold (so >= 30 sigma at default) escalates to WARN."""
        train, val, test = _make_frames(drift_z=35.0)
        with caplog.at_level(logging.WARNING):
            rep = compute_feature_distribution_drift(train, val, test)
        assert any(c == "f_shift" for c, _z in rep["drift_candidates"])
        msgs = " | ".join(rec.getMessage() for rec in caplog.records)
        assert "[feature-distribution-drift]" in msgs

    def test_fi_weighted_aggregate_grounds_harm_signal(self, caplog):
        """Per-feature z-score alone isn't a grounded harm signal -- a 5-sigma
        drift on an irrelevant feature is harmless. With FI weighting we get
        an aggregate that DOES correlate with model harm: high z * high FI
        = the important feature is drifting.

        Scenario: only ONE feature drifts strongly (8 sigma), and we tell the
        sensor that feature has FI=1.0 (dominant). The weighted score should
        be near the feature's z, escalating to WARN."""
        train, val, test = _make_frames(drift_z=8.0)
        with caplog.at_level(logging.WARNING):
            rep = compute_feature_distribution_drift(
                train, val, test,
                feature_importance={"f_shift": 1.0, "f_stable": 0.0},
            )
        ws = rep["weighted_drift_score"]
        assert ws is not None and ws > 5.0, (
            f"FI-weighted aggregate should be ~ z of the drifting dominant feature; got {ws}"
        )
        # And WARN fires because weighted_drift_score >= 1.0.
        msgs = " | ".join(rec.getMessage() for rec in caplog.records)
        assert "[feature-distribution-drift]" in msgs
        assert "weighted_drift=" in msgs

    def test_recommend_neural_overrides_above_threshold_when_sweep_populated(self, monkeypatch):
        """When weighted_drift_score crosses WEIGHTED_DRIFT_NEURAL_OVERRIDE_THRESHOLD
        (3.0) AND the bench has populated ROBUST_MLP_OVERRIDES_UNDER_DRIFT, the
        report must surface that override dict so the per-target model-selection
        can merge it into MLPConfig.

        Skip-neural is too blunt (loses stacking diversity). The empirical
        approach: change the MLP HPT under drift, don't drop the model."""
        import mlframe.training.feature_drift_report as fdr
        monkeypatch.setattr(
            fdr, "ROBUST_MLP_OVERRIDES_UNDER_DRIFT",
            {"alpha": 1.0, "hidden_layer_sizes": (16,)},
        )
        train, val, test = _make_frames(drift_z=8.0)
        rep = compute_feature_distribution_drift(
            train, val, test,
            feature_importance={"f_shift": 1.0, "f_stable": 0.0},
        )
        assert rep["recommend_neural_overrides"] == {
            "alpha": 1.0, "hidden_layer_sizes": (16,),
        }, f"expected override dict from sweep, got {rep['recommend_neural_overrides']}"

    def test_recommend_neural_overrides_silent_below_threshold(self, monkeypatch):
        """2-sigma drift on a dominant feature -> weighted score ~= 2 < 3.0.
        Override recommendation must NOT fire because the empirical paired
        experiment showed weighted_drift_score < 3.0 has high false-positive
        rate vs MLP_excess_harm > 0.1."""
        import mlframe.training.feature_drift_report as fdr
        monkeypatch.setattr(
            fdr, "ROBUST_MLP_OVERRIDES_UNDER_DRIFT", {"alpha": 1.0},
        )
        train, val, test = _make_frames(drift_z=2.0)
        rep = compute_feature_distribution_drift(
            train, val, test,
            feature_importance={"f_shift": 1.0, "f_stable": 0.0},
        )
        assert rep["recommend_neural_overrides"] is None

    def test_recommend_neural_overrides_requires_fi(self, monkeypatch):
        """Without feature_importance the weighted score is None, so the
        recommendation MUST stay None -- the per-feature z-score alone is
        not a grounded harm signal (drift on FI=0 features is harmless)."""
        import mlframe.training.feature_drift_report as fdr
        monkeypatch.setattr(
            fdr, "ROBUST_MLP_OVERRIDES_UNDER_DRIFT", {"alpha": 1.0},
        )
        train, val, test = _make_frames(drift_z=35.0)  # extreme drift
        rep = compute_feature_distribution_drift(train, val, test)  # no FI
        assert rep["weighted_drift_score"] is None
        assert rep["recommend_neural_overrides"] is None

    def test_recommend_neural_overrides_none_when_sweep_unpopulated(self, monkeypatch):
        """If ROBUST_MLP_OVERRIDES_UNDER_DRIFT is {} (sweep not yet wired into
        the constant), the report must not invent a recommendation -- it must
        stay None so downstream code does no-op merge."""
        import mlframe.training.feature_drift_report as fdr
        monkeypatch.setattr(fdr, "ROBUST_MLP_OVERRIDES_UNDER_DRIFT", {})
        train, val, test = _make_frames(drift_z=8.0)
        rep = compute_feature_distribution_drift(
            train, val, test,
            feature_importance={"f_shift": 1.0, "f_stable": 0.0},
        )
        assert rep["recommend_neural_overrides"] is None

    def test_recommend_neural_overrides_carries_sweep_constant(self):
        """When ROBUST_MLP_OVERRIDES_UNDER_DRIFT is populated (2026-05-22
        sweep result), the report surfaces those keys at the actual default."""
        from mlframe.training.feature_drift_report import ROBUST_MLP_OVERRIDES_UNDER_DRIFT
        if not ROBUST_MLP_OVERRIDES_UNDER_DRIFT:
            import pytest
            pytest.skip("sweep constant not yet populated in this build")
        train, val, test = _make_frames(drift_z=8.0)
        rep = compute_feature_distribution_drift(
            train, val, test,
            feature_importance={"f_shift": 1.0, "f_stable": 0.0},
        )
        assert rep["recommend_neural_overrides"] == dict(ROBUST_MLP_OVERRIDES_UNDER_DRIFT)

    def test_fi_weighted_aggregate_NOT_alarmed_when_drift_on_unimportant_feature(self, caplog):
        """Inverse scenario: f_shift drifts 8 sigma but its FI is 0; f_stable
        has FI=1 but no drift. Weighted score should be ~0 -- the system is
        safe even though one feature's z is high."""
        train, val, test = _make_frames(drift_z=8.0)
        with caplog.at_level(logging.WARNING):
            rep = compute_feature_distribution_drift(
                train, val, test,
                feature_importance={"f_shift": 0.0, "f_stable": 1.0},
            )
        ws = rep["weighted_drift_score"]
        assert ws is not None and ws < 0.5, (
            f"With drift on FI=0 feature, weighted score should stay low; got {ws}"
        )
        # No WARN should fire -- the harm signal is grounded and doesn't escalate.
        warn_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("[feature-distribution-drift]" in m for m in warn_msgs), (
            f"WARN should NOT fire when the drift is on an unimportant feature. "
            f"warn_msgs={warn_msgs}"
        )

    def test_threshold_respected(self):
        train, val, test = _make_frames(drift_z=2.5)
        # Default threshold is 3.0; 2.5-sigma drift must NOT fire.
        rep_default = compute_feature_distribution_drift(train, val, test)
        assert rep_default["drift_candidates"] == []
        # Tighter threshold (2.0) should catch it.
        rep_tight = compute_feature_distribution_drift(
            train, val, test, warn_threshold_z=2.0,
        )
        assert any(c == "f_shift" for c, _z in rep_tight["drift_candidates"])

    def test_constant_feature_skipped_via_nan_z(self):
        """A feature with zero train-std produces NaN z (no drift signal can
        be computed). The sensor must NOT crash and the feature must NOT be
        flagged as a drift candidate."""
        n = 500
        rng = np.random.default_rng(1)
        train = pd.DataFrame({"const": np.full(n, 5.0), "f": rng.normal(0, 1, n)})
        val = pd.DataFrame({"const": np.full(n // 2, 5.0), "f": rng.normal(0, 1, n // 2)})
        test = pd.DataFrame({"const": np.full(n // 2, 5.0), "f": rng.normal(0, 1, n // 2)})
        rep = compute_feature_distribution_drift(train, val, test)
        const_entry = rep["per_feature"]["const"]
        assert const_entry["train_std"] == 0.0
        assert np.isnan(const_entry["val_z"])
        assert np.isnan(const_entry["test_z"])
        assert not any(c == "const" for c, _z in rep["drift_candidates"])

    def test_polars_input_handled(self):
        pl = pytest.importorskip("polars")
        n = 500
        rng = np.random.default_rng(2)
        train = pl.DataFrame({"f": rng.normal(0, 1, n).astype(np.float32)})
        val = pl.DataFrame({"f": rng.normal(0, 1, n // 2).astype(np.float32)})
        test = pl.DataFrame({"f": rng.normal(5, 1, n // 2).astype(np.float32)})  # 5-sigma drift
        rep = compute_feature_distribution_drift(train, val, test)
        assert any(c == "f" for c, _z in rep["drift_candidates"]), (
            f"Polars frame with clear drift not flagged: {rep}"
        )

    def test_no_val_frame_falls_back_to_test_only(self):
        train, _, test = _make_frames(drift_z=6.0)
        rep = compute_feature_distribution_drift(train, val_df=None, test_df=test)
        # val_z is NaN, test_z is high; f_shift still in candidates.
        per = rep["per_feature"]["f_shift"]
        assert np.isnan(per["val_z"])
        assert abs(per["test_z"]) > 5.0
        assert any(c == "f_shift" for c, _z in rep["drift_candidates"])


class TestSklearnToMlframeMlpKwargsTranslator:
    """Coverage for the sklearn-MLP-shape -> mlframe-mlp_kwargs-shape translator.

    The bench (``profiling/bench_mlp_robustness_sweep.py``) produces overrides
    in sklearn naming (alpha / hidden_layer_sizes / activation); the mlframe
    MLP wrapper consumes a nested ``mlp_kwargs`` shape with different field
    names. The translator must produce that nested shape faithfully so the
    wire-in in ``_phase_train_one_target`` can deep-merge it."""

    def test_empty_input_returns_empty(self):
        assert translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs({}) == {}

    def test_alpha_routes_to_adamw_weight_decay(self):
        out = translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs({"alpha": 0.5})
        assert "model_params" in out
        # AdamW is the optimizer that natively decouples weight_decay from
        # the gradient step; pure Adam treats weight_decay as L2 in the
        # loss (different math, less effective regularizer).
        mp = out["model_params"]
        assert mp["optimizer"].__name__ == "AdamW"
        assert mp["optimizer_kwargs"] == {"weight_decay": 0.5}

    def test_hidden_layer_sizes_single_layer(self):
        out = translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs({
            "hidden_layer_sizes": (16,),
        })
        np_ = out["network_params"]
        assert np_["nlayers"] == 1
        assert np_["first_layer_num_neurons"] == 16
        assert np_["min_layer_neurons"] == 16
        assert np_["consec_layers_neurons_ratio"] == 1.0

    def test_hidden_layer_sizes_two_layers_decode_ratio(self):
        out = translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs({
            "hidden_layer_sizes": (32, 16),
        })
        np_ = out["network_params"]
        assert np_["nlayers"] == 2
        assert np_["first_layer_num_neurons"] == 32
        assert np_["min_layer_neurons"] == 16
        assert np_["consec_layers_neurons_ratio"] == 2.0

    def test_activation_relu_maps_to_torch_relu(self):
        import torch
        out = translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs({
            "activation": "relu",
        })
        assert out["network_params"]["activation_function"] is torch.nn.ReLU

    def test_activation_tanh_maps_to_torch_tanh(self):
        import torch
        out = translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs({
            "activation": "tanh",
        })
        assert out["network_params"]["activation_function"] is torch.nn.Tanh

    def test_activation_identity_uses_nn_identity_with_zero_dropout(self):
        """The bench winner uses ``activation='identity'`` -- a linear head.
        mlframe ``generate_mlp`` rejects ``nlayers < 1`` (ValueError), so we
        instead use ``torch.nn.Identity`` as the per-layer activation and
        zero out dropout sources. Layer composition is then Linear -> Identity
        -> Linear -> Identity ..., mathematically collapsing to a single linear
        transform of the input -- equivalent to sklearn's identity activation."""
        import torch
        out = translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs({
            "activation": "identity",
        })
        assert out["network_params"]["activation_function"] is torch.nn.Identity
        assert out["network_params"]["dropout_prob"] == 0.0
        assert out["network_params"]["inputs_dropout_prob"] == 0.0

    def test_unknown_activation_recorded_as_untranslated(self):
        out = translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs({
            "activation": "gelu",
        })
        assert any("activation=gelu" in s for s in out.get("__untranslated__", []))

    def test_full_sweep_winner_translates_cleanly(self):
        """End-to-end: paste the 2026-05-22 multi-metric sweep winner through
        the translator and confirm the output is the shape the consumer site
        assumes."""
        import torch
        from mlframe.training.feature_drift_report import ROBUST_MLP_OVERRIDES_UNDER_DRIFT
        if not ROBUST_MLP_OVERRIDES_UNDER_DRIFT:
            pytest.skip("sweep constant not yet populated")
        out = translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs(
            ROBUST_MLP_OVERRIDES_UNDER_DRIFT,
        )
        # The winner is {alpha=1e-4, hidden=(32,16), activation='identity'}
        # (R^2 / RMSE / MAE all agree on this config under min-max cross-DGP).
        # Translation: AdamW weight_decay=1e-4, identity activation collapses
        # the stack to a single linear transform -- the translator MUST set
        # ``nlayers=1`` (not 2 from hidden topology) so generate_mlp builds an
        # honest ``Linear(in, out) -> Identity`` instead of a 25->32->16->1
        # Identity stack that mathematically collapses but optimises poorly
        # and catastrophically OOD-extrapolates under covariate shift (prod
        # TVT 2026-05-22: stacked Identity went to ~-17 sigma on test split,
        # R^2=-326 while Ridge R^2=1.00 on the same data).
        import torch
        assert out["model_params"]["optimizer_kwargs"]["weight_decay"] == pytest.approx(1e-4)
        assert out["network_params"]["activation_function"] is torch.nn.Identity
        assert out["network_params"]["first_layer_num_neurons"] == 32
        assert out["network_params"]["nlayers"] == 1  # identity collapse
        assert out["network_params"]["dropout_prob"] == 0.0
        assert "__untranslated__" not in out


class TestFeatureDriftAutoActionWireIn:
    """Integration coverage for the per-target wire-in in
    ``_phase_train_one_target_body.py``: when the diagnostics phase stamps
    a ``recommend_neural_overrides`` payload into ``metadata``, the body
    must (a) translate it via ``translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs``,
    (b) deep-merge into ``hyperparams_config.mlp_kwargs`` for THIS target,
    (c) NOT touch the original ``hyperparams_config`` (other targets must
    keep their mlp_kwargs), and (d) stamp ``feature_drift_auto_action``
    into metadata for observability.

    These tests exercise the metadata-driven merge logic directly without
    spinning up the full suite -- the wire-in path is a pure dict
    transform plus a ``hyperparams_config.model_copy(update=...)`` call.
    Failing here means the production code is misshaping the override
    before MLP construction reads it."""

    @staticmethod
    def _stamped_metadata(sklearn_override: dict, target_type: str = "regression",
                         cur_target_name: str = "y") -> dict:
        """Mirror the metadata shape ``run_per_target_diagnostics`` stamps."""
        return {
            "feature_distribution_drift": {
                target_type: {
                    cur_target_name: {
                        "recommend_neural_overrides": sklearn_override,
                        "weighted_drift_score": 8.4,
                    },
                },
            },
        }

    def test_merge_produces_correct_mlp_kwargs_shape(self):
        """Replicates the deep-merge the wire-in performs and asserts the
        output shape matches what get_training_configs reads."""
        from mlframe.training.feature_drift_report import (
            ROBUST_MLP_OVERRIDES_UNDER_DRIFT,
            translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs,
        )
        if not ROBUST_MLP_OVERRIDES_UNDER_DRIFT:
            pytest.skip("sweep constant not populated")

        _orig_mlp_kwargs = {
            "network_params": {"use_layernorm": False, "first_layer_num_neurons": 128},
            "model_params": {"learning_rate": 3e-3},
        }
        _mlframe_override = translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs(
            ROBUST_MLP_OVERRIDES_UNDER_DRIFT,
        )
        _mlframe_override.pop("__untranslated__", None)
        _merged = dict(_orig_mlp_kwargs)
        for _slot in ("model_params", "network_params"):
            if _slot in _mlframe_override:
                _merged.setdefault(_slot, {})
                _merged[_slot] = dict({**_merged[_slot], **_mlframe_override[_slot]})

        # The original use_layernorm=False is preserved (drift override does
        # not touch it); first_layer_num_neurons is OVERRIDDEN by the bench
        # pick (32, was 128); activation_function becomes torch.nn.Identity
        # (linear-collapse path) and dropout sources are zeroed.
        import torch
        assert _merged["network_params"]["use_layernorm"] is False
        assert _merged["network_params"]["first_layer_num_neurons"] == 32
        # identity-activation collapses the stack to a single linear transform
        # (see TestSklearnToMlframeMlpKwargsTranslator commentary above)
        assert _merged["network_params"]["nlayers"] == 1
        assert _merged["network_params"]["activation_function"] is torch.nn.Identity
        assert _merged["network_params"]["dropout_prob"] == 0.0
        # learning_rate stays untouched by the override; weight_decay is
        # added from the bench pick's alpha translation.
        assert _merged["model_params"]["learning_rate"] == 3e-3
        assert _merged["model_params"]["optimizer_kwargs"]["weight_decay"] == pytest.approx(1e-4)

    def test_per_target_type_threshold_table(self):
        """Per-type threshold table:
          regression -> 3.0 (grounded universally by paired study, precision=1.000)
          classification -> 3.0 + linear-shape gate (interaction-rich
            classification targets show negative correlation; gate enforces
            the override only fires when init_score_baseline.delta_vs_raw_pct
            says the target is linear-shape).
        """
        from mlframe.training.feature_drift_report import (
            WEIGHTED_DRIFT_NEURAL_OVERRIDE_THRESHOLDS,
            CLASSIFICATION_LINEAR_SHAPE_MAX_DELTA_VS_RAW_PCT,
        )
        assert WEIGHTED_DRIFT_NEURAL_OVERRIDE_THRESHOLDS["regression"] == 3.0
        assert WEIGHTED_DRIFT_NEURAL_OVERRIDE_THRESHOLDS["classification"] == 3.0
        assert CLASSIFICATION_LINEAR_SHAPE_MAX_DELTA_VS_RAW_PCT == 10.0

    def test_regression_fires_without_shape_signal(self):
        """Regression doesn't need the shape gate -- Ridge wins universally
        on drifted regression DGPs per the empirical study, so the sensor
        recommends the override on any regression target with drift >= 3.0
        regardless of linear_shape_delta_vs_raw_pct."""
        from mlframe.training.feature_drift_report import ROBUST_MLP_OVERRIDES_UNDER_DRIFT
        train, val, test = _make_frames(drift_z=8.0)
        rep = compute_feature_distribution_drift(
            train, val, test,
            feature_importance={"f_shift": 1.0, "f_stable": 0.0},
            target_type="regression",
            # Shape signal absent / arbitrary -- regression ignores it.
            linear_shape_delta_vs_raw_pct=None,
        )
        assert rep["recommend_neural_overrides"] == dict(ROBUST_MLP_OVERRIDES_UNDER_DRIFT)

    def test_classification_fires_on_linear_shape(self):
        """Classification override applies when the shape signal says linear
        (|delta_vs_raw_pct| <= CLASSIFICATION_LINEAR_SHAPE_MAX_DELTA_VS_RAW_PCT,
        meaning LogReg captures essentially what LightGBM does)."""
        from mlframe.training.feature_drift_report import (
            ROBUST_MLP_OVERRIDES_UNDER_DRIFT_CLASSIFICATION,
        )
        train, val, test = _make_frames(drift_z=8.0)
        rep = compute_feature_distribution_drift(
            train, val, test,
            feature_importance={"f_shift": 1.0, "f_stable": 0.0},
            target_type="binary_classification",
            linear_shape_delta_vs_raw_pct=5.0,  # linear within 10% of LGBM
        )
        assert rep["recommend_neural_overrides"] == dict(ROBUST_MLP_OVERRIDES_UNDER_DRIFT_CLASSIFICATION)

    def test_classification_skips_on_nonlinear_shape(self):
        """Interaction / sinusoidal-rich classification targets show
        delta_vs_raw_pct > 10% -- nonlinear-shape. The override would HURT
        these targets (paired study r=-0.227 on interaction_binary) so the
        recommendation must be None."""
        train, val, test = _make_frames(drift_z=8.0)
        rep = compute_feature_distribution_drift(
            train, val, test,
            feature_importance={"f_shift": 1.0, "f_stable": 0.0},
            target_type="binary_classification",
            linear_shape_delta_vs_raw_pct=45.0,  # LogReg far worse than LGBM
        )
        assert rep["recommend_neural_overrides"] is None

    def test_classification_skips_when_shape_signal_missing(self):
        """Conservative default: without the shape signal from
        baseline_diagnostics, classification stays gated off. This covers
        suites that didn't enable baseline_diagnostics."""
        train, val, test = _make_frames(drift_z=8.0)
        rep = compute_feature_distribution_drift(
            train, val, test,
            feature_importance={"f_shift": 1.0, "f_stable": 0.0},
            target_type="binary_classification",
            linear_shape_delta_vs_raw_pct=None,
        )
        assert rep["recommend_neural_overrides"] is None

    def test_recommend_payload_absent_when_no_fi(self):
        """The recommendation requires feature_importance to score. Without
        FI, no override should be produced even on extreme drift -- the
        per-feature z-score alone is not a grounded harm signal."""
        train, val, test = _make_frames(drift_z=35.0)
        rep = compute_feature_distribution_drift(train, val, test)
        assert rep["weighted_drift_score"] is None
        assert rep["recommend_neural_overrides"] is None


class TestColValueCounts:
    """``_col_value_counts`` converts a per-column value_counts to a plain
    {value: int_count} dict via .tolist()+zip (bulk C int conversion). Pin
    the exact counts + that values are Python ``int`` so the PSI consumer's
    arithmetic stays correct and the dict-build optimisation can't regress."""

    def test_pandas_counts_exact_and_python_int(self):
        from mlframe.training.feature_drift_report import _col_value_counts
        df = pd.DataFrame({"c": ["a", "a", "b", "c", "c", "c"]})
        out = _col_value_counts(df, "c")
        assert out == {"a": 2, "b": 1, "c": 3}
        assert all(type(v) is int for v in out.values()), "counts must be Python int"

    def test_pandas_keeps_nan_bucket(self):
        """``value_counts(dropna=False)`` keeps NaN as its own bucket -- a new
        all-NaN column in serving is exactly the drift PSI must surface."""
        from mlframe.training.feature_drift_report import _col_value_counts
        df = pd.DataFrame({"c": [1.0, 1.0, np.nan, 2.0, np.nan]})
        out = _col_value_counts(df, "c")
        assert out[1.0] == 2 and out[2.0] == 1
        nan_counts = [v for k, v in out.items() if isinstance(k, float) and np.isnan(k)]
        assert nan_counts == [2], "NaN bucket count must be preserved"

    def test_polars_counts_exact_and_python_int(self):
        pl = pytest.importorskip("polars")
        from mlframe.training.feature_drift_report import _col_value_counts
        df = pl.DataFrame({"c": ["x", "y", "y", "z", "z", "z"]})
        out = _col_value_counts(df, "c")
        assert out == {"x": 1, "y": 2, "z": 3}
        assert all(type(v) is int for v in out.values())
