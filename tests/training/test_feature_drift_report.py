"""Unit tests for `training.feature_drift_report` public API.

Pre-existing `test_feature_drift_auto_action_e2e.py` exercises the auto-action
wiring via a full suite call. This file pins the unit-level contracts of the
two sklearn-MLP override translators and `compute_feature_distribution_drift`
(numeric z-score) + `compute_categorical_drift_psi` directly.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mlframe.training.feature_drift_report import (
    compute_categorical_drift_psi,
    compute_feature_distribution_drift,
    translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs,
    translate_sklearn_mlp_overrides_to_recurrent_config_kwargs,
)


# ----- translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs --------------


def test_translate_mlp_empty_overrides_returns_empty_dict():
    """Translate mlp empty overrides returns empty dict."""
    assert translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs({}) == {}


def test_translate_mlp_alpha_maps_to_weight_decay():
    """Translate mlp alpha maps to weight decay."""
    pytest.importorskip("torch")
    out = translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs({"alpha": 1e-4})
    assert "model_params" in out
    assert "optimizer_kwargs" in out["model_params"]
    assert out["model_params"]["optimizer_kwargs"]["weight_decay"] == pytest.approx(1e-4)


def test_translate_mlp_hidden_layer_sizes_populates_network_params():
    """Translate mlp hidden layer sizes populates network params."""
    out = translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs({"hidden_layer_sizes": (64, 32, 16)})
    assert "network_params" in out
    np_ = out["network_params"]
    assert np_["nlayers"] == 3
    assert np_["first_layer_num_neurons"] == 64
    assert np_["min_layer_neurons"] == 16
    assert np_["consec_layers_neurons_ratio"] == pytest.approx(64 / 16)


def test_translate_mlp_activation_relu_routes_to_torch_relu():
    """Translate mlp activation relu routes to torch relu."""
    pytest.importorskip("torch")
    import torch

    out = translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs({"activation": "relu"})
    assert out["network_params"]["activation_function"] is torch.nn.ReLU


def test_translate_mlp_activation_identity_collapses_to_linear():
    # Per docstring: identity activation MUST set nlayers=1 and zero the
    # dropout sources to keep the network honestly linear (prod TVT
    # 2026-05-22 regression).
    """Translate mlp activation identity collapses to linear."""
    pytest.importorskip("torch")
    import torch

    out = translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs({"activation": "identity"})
    np_ = out["network_params"]
    assert np_["activation_function"] is torch.nn.Identity
    assert np_["nlayers"] == 1
    assert np_["dropout_prob"] == 0.0
    assert np_["inputs_dropout_prob"] == 0.0


def test_translate_mlp_passes_through_unknown_keys_under_model_params():
    """Translate mlp passes through unknown keys under model params."""
    out = translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs({"custom_knob": 42, "lr": 0.001})
    assert out["model_params"]["custom_knob"] == 42
    assert out["model_params"]["lr"] == 0.001


def test_translate_mlp_unknown_activation_recorded_in_untranslated():
    """Translate mlp unknown activation recorded in untranslated."""
    pytest.importorskip("torch")
    out = translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs({"activation": "not_a_real_activation"})
    assert "__untranslated__" in out
    assert any("not_a_real_activation" in s for s in out["__untranslated__"])


# ----- translate_sklearn_mlp_overrides_to_recurrent_config_kwargs ---------


def test_translate_recurrent_empty_overrides_returns_empty_dict():
    """Translate recurrent empty overrides returns empty dict."""
    assert translate_sklearn_mlp_overrides_to_recurrent_config_kwargs({}) == {}


def test_translate_recurrent_alpha_maps_to_weight_decay():
    """Translate recurrent alpha maps to weight decay."""
    out = translate_sklearn_mlp_overrides_to_recurrent_config_kwargs({"alpha": 1e-3})
    assert out["weight_decay"] == pytest.approx(1e-3)


def test_translate_recurrent_hidden_layer_sizes_to_mlp_hidden_sizes():
    """Translate recurrent hidden layer sizes to mlp hidden sizes."""
    out = translate_sklearn_mlp_overrides_to_recurrent_config_kwargs({"hidden_layer_sizes": (128, 64)})
    assert out["mlp_hidden_sizes"] == (128, 64)


def test_translate_recurrent_activation_recorded_as_untranslated():
    # Recurrent cells have hard-coded gate activations; the translator
    # documents this and records the skip rather than silently fabricating.
    """Translate recurrent activation recorded as untranslated."""
    out = translate_sklearn_mlp_overrides_to_recurrent_config_kwargs({"activation": "relu"})
    assert "__untranslated__" in out
    assert any("activation" in s for s in out["__untranslated__"])


# ----- compute_feature_distribution_drift (numeric z) --------------------


def _make_pandas_frame(seed: int, n_rows: int, shift_x: float = 0.0):
    """Make pandas frame."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x": rng.normal(loc=shift_x, scale=1.0, size=n_rows),
            "y": rng.normal(loc=0.0, scale=2.0, size=n_rows),
            "cat": rng.choice(["a", "b", "c"], size=n_rows),
        }
    )
    return df


def test_numeric_drift_no_shift_keeps_z_small():
    """Numeric drift no shift keeps z small."""
    train = _make_pandas_frame(0, 2000)
    val = _make_pandas_frame(1, 500)
    test = _make_pandas_frame(2, 500)
    out = compute_feature_distribution_drift(train, val, test)
    assert "per_feature" in out
    assert "x" in out["per_feature"]
    # |z| < 3 sigma for both x and y on matched DGP.
    for col in ("x", "y"):
        entry = out["per_feature"][col]
        assert abs(entry["val_z"]) < 3.0
        assert abs(entry["test_z"]) < 3.0
    assert out["drift_candidates"] == []


def test_numeric_drift_shifted_mean_appears_in_candidates():
    """Numeric drift shifted mean appears in candidates."""
    train = _make_pandas_frame(3, 2000, shift_x=0.0)
    # 5-sigma shifted x on val and test.
    val = _make_pandas_frame(4, 500, shift_x=5.0)
    test = _make_pandas_frame(5, 500, shift_x=5.0)
    out = compute_feature_distribution_drift(train, val, test, warn_threshold_z=3.0)
    candidates = dict(out["drift_candidates"])
    assert "x" in candidates
    # |z| should be ~5 sigma; pin a conservative floor (>=3.0).
    assert candidates["x"] >= 3.0
    # And "y" should NOT be flagged because its DGP matched.
    assert "y" not in candidates


def test_numeric_drift_constant_train_column_nan_z():
    # A constant feature has no drift signal; the function must yield
    # NaN z without crashing.
    """Numeric drift constant train column nan z."""
    train = pd.DataFrame({"const": np.full(500, 7.0), "v": np.linspace(0, 1, 500)})
    val = pd.DataFrame({"const": np.full(100, 7.0), "v": np.linspace(0.5, 1.5, 100)})
    test = pd.DataFrame({"const": np.full(100, 7.0), "v": np.linspace(0.5, 1.5, 100)})
    out = compute_feature_distribution_drift(train, val, test)
    entry = out["per_feature"]["const"]
    assert entry["train_std"] == 0.0
    assert math.isnan(entry["val_z"])
    assert math.isnan(entry["test_z"])


def test_numeric_drift_handles_none_val_or_test():
    """Numeric drift handles none val or test."""
    train = _make_pandas_frame(6, 500)
    out = compute_feature_distribution_drift(train, None, None)
    for col in ("x", "y"):
        entry = out["per_feature"][col]
        assert math.isnan(entry["val_z"])
        assert math.isnan(entry["test_z"])


def test_numeric_drift_weighted_score_uses_feature_importance():
    """Numeric drift weighted score uses feature importance."""
    train = _make_pandas_frame(7, 2000, shift_x=0.0)
    val = _make_pandas_frame(8, 500, shift_x=5.0)
    test = _make_pandas_frame(9, 500, shift_x=5.0)
    # Heavy weight on x (the shifted col) -> high weighted_drift_score.
    out_heavy = compute_feature_distribution_drift(
        train,
        val,
        test,
        feature_importance={"x": 1.0, "y": 0.01},
    )
    # Heavy weight on y (the matched col) -> low weighted_drift_score.
    out_light = compute_feature_distribution_drift(
        train,
        val,
        test,
        feature_importance={"x": 0.01, "y": 1.0},
    )
    assert out_heavy["weighted_drift_score"] is not None
    assert out_light["weighted_drift_score"] is not None
    # Drift on the important feature must score strictly higher.
    assert out_heavy["weighted_drift_score"] > out_light["weighted_drift_score"]


# ----- compute_categorical_drift_psi -------------------------------------


def test_categorical_psi_no_drift_when_train_val_test_match_distribution():
    """Categorical psi no drift when train val test match distribution."""
    rng = np.random.default_rng(20)
    train = pd.DataFrame({"cat": rng.choice(["a", "b", "c"], size=2000, p=[0.5, 0.3, 0.2])})
    val = pd.DataFrame({"cat": rng.choice(["a", "b", "c"], size=500, p=[0.5, 0.3, 0.2])})
    test = pd.DataFrame({"cat": rng.choice(["a", "b", "c"], size=500, p=[0.5, 0.3, 0.2])})
    out = compute_categorical_drift_psi(train, val, test)
    assert "per_feature" in out
    assert "cat" in out["per_feature"]
    entry = out["per_feature"]["cat"]
    # Matched DGP -> PSI well below moderate threshold (0.20).
    assert entry["val_psi"] < 0.10
    assert entry["test_psi"] < 0.10
    # No drift candidates flagged.
    assert out["drift_candidates"] == []


def test_categorical_psi_shifted_distribution_flagged():
    """Categorical psi shifted distribution flagged."""
    train = pd.DataFrame({"cat": ["a"] * 700 + ["b"] * 200 + ["c"] * 100})
    # val/test: nearly all "c" (heavily shifted).
    val = pd.DataFrame({"cat": ["c"] * 450 + ["a"] * 50})
    test = pd.DataFrame({"cat": ["c"] * 450 + ["a"] * 50})
    out = compute_categorical_drift_psi(train, val, test)
    entry = out["per_feature"]["cat"]
    # Shifted distribution -> PSI well above moderate threshold.
    assert entry["val_psi"] > 0.25
    assert entry["test_psi"] > 0.25
    # Candidate list contains the column.
    assert any(c == "cat" for c, _ in out["drift_candidates"])


def test_categorical_psi_new_category_in_val_treated_as_positive_psi():
    # A category present only in val (never seen in train) is exactly the
    # silent-prod-failure case PSI is meant to surface. Confirm the
    # implementation routes it to a finite POSITIVE PSI (not nan, not 0).
    """Categorical psi new category in val treated as positive psi."""
    train = pd.DataFrame({"cat": ["a"] * 500 + ["b"] * 500})
    val = pd.DataFrame({"cat": ["new_cat"] * 200 + ["a"] * 100})  # all-new
    test = pd.DataFrame({"cat": ["a"] * 100 + ["b"] * 100})
    out = compute_categorical_drift_psi(train, val, test)
    entry = out["per_feature"]["cat"]
    assert math.isfinite(entry["val_psi"])
    # New category contributes >0 PSI.
    assert entry["val_psi"] > 0.0


def test_categorical_psi_no_categorical_columns_returns_empty_per_feature():
    # Numeric-only frame -> no cat columns -> empty per_feature dict.
    """Categorical psi no categorical columns returns empty per feature."""
    train = pd.DataFrame({"x": np.arange(100, dtype=float)})
    val = pd.DataFrame({"x": np.arange(50, dtype=float)})
    test = pd.DataFrame({"x": np.arange(50, dtype=float)})
    out = compute_categorical_drift_psi(train, val, test)
    assert out["per_feature"] == {}
    assert out["n_categorical_features"] == 0


# ----- target-invariant z-stats cache (bit-identity) ---------------------


def _drift_frames(seed: int = 0):
    """Drift frames."""
    rng = np.random.default_rng(seed)
    n = 4000
    data = {f"num_{j}": rng.standard_normal(n) + 0.01 * j for j in range(8)}
    data["cat"] = rng.integers(0, 15, size=n).astype(str)
    train = pd.DataFrame(data)
    val = pd.DataFrame({c: rng.permutation(train[c].to_numpy()) for c in train.columns})
    test = pd.DataFrame({c: rng.permutation(train[c].to_numpy()) for c in train.columns})
    return train, val, test


def _strip_target_specific(report: dict) -> dict:
    """The target-invariant slice of a drift report (everything but the
    FI-weighted aggregate / override / threshold scalar)."""
    return {
        "per_feature": report["per_feature"],
        "drift_candidates": report["drift_candidates"],
        "categorical_psi": report["categorical_psi"],
        "n_numeric_features": report["n_numeric_features"],
    }


def test_drift_zstats_cache_is_bit_identical_to_fresh_recompute():
    """A 3-target run on shared frames: the cached invariant (per-feature z,
    candidates, categorical PSI) for targets 2..N must EQUAL a from-scratch
    recompute. Pins cached == fresh -- the cache must never change the output."""
    import mlframe.training.feature_drift_report as fdr

    train, val, test = _drift_frames(seed=1)
    num_cols = [c for c in train.columns if c.startswith("num_")]
    fis = [{c: float(abs(hash((c, t)) % 97)) for c in num_cols} for t in range(3)]

    # Fresh: cache cleared before EVERY call -> always a from-scratch compute.
    fresh_reports = []
    for t in range(3):
        fdr._DRIFT_INVARIANT_CACHE.clear()
        fresh_reports.append(
            compute_feature_distribution_drift(
                train,
                val,
                test,
                feature_importance=fis[t],
                target_type="regression",
            )
        )

    # Cached: cleared ONCE, then targets 2..3 hit the cache.
    fdr._DRIFT_INVARIANT_CACHE.clear()
    cached_reports = [
        compute_feature_distribution_drift(
            train,
            val,
            test,
            feature_importance=fis[t],
            target_type="regression",
        )
        for t in range(3)
    ]
    # At least one cached entry was stored (cacheable frames).
    assert len(fdr._DRIFT_INVARIANT_CACHE) >= 1

    for t in range(3):
        f = _strip_target_specific(fresh_reports[t])
        c = _strip_target_specific(cached_reports[t])
        assert f["n_numeric_features"] == c["n_numeric_features"]
        assert f["drift_candidates"] == c["drift_candidates"]
        # Per-feature dicts: every train_mean/std/val_z/test_z bit-identical.
        assert set(f["per_feature"]) == set(c["per_feature"])
        for col, ent in f["per_feature"].items():
            cent = c["per_feature"][col]
            for k in ("train_mean", "train_std", "val_z", "test_z"):
                fv, cv = ent[k], cent[k]
                if isinstance(fv, float) and math.isnan(fv):
                    assert math.isnan(cv)
                else:
                    assert fv == cv, (col, k, fv, cv)
        # Categorical PSI bit-identical.
        fp = f["categorical_psi"]["per_feature"]
        cp = c["categorical_psi"]["per_feature"]
        assert set(fp) == set(cp)
        for col, ent in fp.items():
            for k, v in ent.items():
                cv = cp[col][k]
                if isinstance(v, float) and math.isnan(v):
                    assert math.isnan(cv)
                else:
                    assert v == cv, (col, k, v, cv)
        # Target-specific fields are still computed fresh per target.
        assert fresh_reports[t]["weighted_drift_score"] == cached_reports[t]["weighted_drift_score"]
        assert fresh_reports[t]["recommend_neural_overrides"] == cached_reports[t]["recommend_neural_overrides"]


def test_drift_cache_isolates_distinct_frames():
    """Different frame CONTENT must not collide: a second frame set yields its
    own stats, never the first set's cached entry."""
    import mlframe.training.feature_drift_report as fdr

    fdr._DRIFT_INVARIANT_CACHE.clear()
    train_a, val_a, test_a = _drift_frames(seed=2)
    train_b, val_b, test_b = _drift_frames(seed=99)
    rep_a = compute_feature_distribution_drift(train_a, val_a, test_a, target_type="regression")
    rep_b = compute_feature_distribution_drift(train_b, val_b, test_b, target_type="regression")
    # Distinct content -> distinct per-feature train means (not a stale hit).
    means_a = {k: v["train_mean"] for k, v in rep_a["per_feature"].items()}
    means_b = {k: v["train_mean"] for k, v in rep_b["per_feature"].items()}
    assert means_a != means_b
    # And a fresh recompute of B equals the cached B (self-consistency).
    fdr._DRIFT_INVARIANT_CACHE.clear()
    rep_b_fresh = compute_feature_distribution_drift(train_b, val_b, test_b, target_type="regression")
    assert means_b == {k: v["train_mean"] for k, v in rep_b_fresh["per_feature"].items()}
