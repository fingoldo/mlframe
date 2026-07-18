"""Wave-2 predict-path parity Fix 3: replay the chosen ensemble flavour at predict time.

Pre-fix ``predict_mlframe_models_suite`` / ``predict_from_models`` hard-coded ``np.mean(np.stack(all_probs))`` to
combine per-model probability arrays, ignoring the flavour that won ``compare_ensembles`` at training time. Now
finalize_suite stamps ``metadata['ensembles_chosen'][target_type][target_name] = flavour`` and the predict path
reads the dict, calling the right combine (arithm / harm / median / geo / quad / rrf / qube) per target slot.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.core.predict import (
    _combine_probs,
    _resolve_chosen_flavour,
)


def test_combine_probs_arithm_matches_mean():
    """``arithm`` / empty flavour falls back to arithmetic mean (back-compat with pre-fix predict)."""
    probs = [np.array([[0.1, 0.9], [0.2, 0.8]]), np.array([[0.3, 0.7], [0.4, 0.6]])]
    expected = np.mean(np.stack(probs), axis=0)
    np.testing.assert_allclose(_combine_probs(probs, None), expected)
    np.testing.assert_allclose(_combine_probs(probs, "arithm"), expected)
    np.testing.assert_allclose(_combine_probs(probs, "mean"), expected)


def test_combine_probs_harm_differs_from_mean():
    """``harm`` invokes the harmonic mean -- visibly different from arithmetic on non-degenerate probs."""
    probs = [np.array([[0.1, 0.9]]), np.array([[0.3, 0.7]])]
    arith = _combine_probs(probs, "arithm")
    harm = _combine_probs(probs, "harm")
    # Harmonic mean of {0.1, 0.3} = 2 / (1/0.1 + 1/0.3) = 0.15 (NOT 0.2 = arithmetic).
    np.testing.assert_allclose(harm[0, 0], 2.0 / (1.0 / 0.1 + 1.0 / 0.3), rtol=1e-5)
    # Sanity: harm should differ from arithm.
    assert not np.allclose(harm, arith), "harmonic mean should differ from arithmetic mean on this input"


def test_combine_probs_geo_quad_median():
    """``geo``, ``quad``, ``median`` produce distinct values; verify each matches its known formula on a 1x1 sample."""
    probs = [np.array([[0.2]]), np.array([[0.6]]), np.array([[0.4]])]
    np.testing.assert_allclose(_combine_probs(probs, "geo"), np.exp(np.mean(np.log([0.2, 0.6, 0.4]))))
    np.testing.assert_allclose(_combine_probs(probs, "quad"), np.sqrt(np.mean([0.04, 0.36, 0.16])))
    np.testing.assert_allclose(_combine_probs(probs, "median"), 0.4)


def test_combine_probs_unknown_flavour_falls_back_to_mean():
    """Unknown / typo'd flavour names fall back to arithmetic mean rather than crashing the predict path. This
    preserves predict for older saved suites whose metadata predates the ensembles_chosen key."""
    probs = [np.array([[0.1, 0.9]]), np.array([[0.3, 0.7]])]
    expected = np.mean(np.stack(probs), axis=0)
    np.testing.assert_allclose(_combine_probs(probs, "unknown_xyz"), expected)


def test_resolve_chosen_flavour_nested_lookup():
    """Nested ``{target_type: {target_name: flavour}}`` lookup returns the per-target choice."""
    metadata = {
        "ensembles_chosen": {
            "regression": {"y": "harm"},
            "binary": {"label": "geo"},
        }
    }
    assert _resolve_chosen_flavour(metadata, "regression", "y") == "harm"
    assert _resolve_chosen_flavour(metadata, "binary", "label") == "geo"


def test_resolve_chosen_flavour_single_target_fallback():
    """When the suite has one target and the lookup misses, fall back to the unique flavour."""
    metadata = {"ensembles_chosen": {"regression": {"y": "harm"}}}
    # Suite-wide lookup (no tt, tname) returns the unique value when all leaves agree.
    assert _resolve_chosen_flavour(metadata, None, None) == "harm"


def test_resolve_chosen_flavour_absent_returns_none():
    """Backward-compat: older saved metadata without ``ensembles_chosen`` returns None so callers fall back to mean."""
    assert _resolve_chosen_flavour({}, "regression", "y") is None
    assert _resolve_chosen_flavour({"ensembles_chosen": {}}, "regression", "y") is None


def test_predict_uses_chosen_flavour_for_per_target_probs():
    """End-to-end: feed predict_from_models a metadata key picking ``harm`` and verify the per-target probability
    matches the harmonic mean of the inputs, not the arithmetic mean."""
    pytest.importorskip("lightgbm")
    import numpy as np
    import polars as pl
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.core.predict import predict_from_models
    from mlframe.training.configs import (
        BaselineDiagnosticsConfig,
        CompositeTargetDiscoveryConfig,
        DummyBaselinesConfig,
        OutputConfig,
        ReportingConfig,
    )
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

    rng = np.random.default_rng(0)
    n = 2_000
    # Binary classification so probs are 2-D and harm vs arithm meaningfully differ.
    df = pl.DataFrame(
        {
            "x0": rng.normal(size=n).astype("float32"),
            "x1": rng.normal(size=n).astype("float32"),
            "y": rng.integers(0, 2, n).astype("int8"),
        }
    )
    fte = SimpleFeaturesAndTargetsExtractor(classification_targets=["y"], classification_exact_values={"y": 1})
    models, metadata = train_mlframe_models_suite(
        df=df,
        target_name="y",
        model_name="flav",
        features_and_targets_extractor=fte,
        mlframe_models=["lgb", "xgb"],
        verbose=0,
        output_config=OutputConfig(data_dir="", models_dir="", save_charts=False),
        composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(plot_outputs="matplotlib[png]", plot_inline_display=False),
    )
    # Force the chosen flavour to ``harm`` for the only target slot so we can compare predict output against the
    # known harmonic combine of the contributing model probabilities.
    target_slot = next(iter(next(iter(models.values())).keys()))
    target_type = next(iter(models.keys()))
    metadata["ensembles_chosen"] = {str(target_type): {str(target_slot): "harm"}}

    results = predict_from_models(
        df=df,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=fte,
        return_probabilities=True,
        verbose=0,
    )
    per_t = results.get("per_target_probabilities") or {}
    assert per_t, f"predict_from_models did not emit per_target_probabilities; got keys={list(results.keys())}"
    # Per-target key includes "{tt}_{tname}".
    key = f"{target_type}_{target_slot}"
    assert key in per_t, f"missing per-target key {key!r}; keys: {list(per_t.keys())}"
    combined_harm = per_t[key]

    # Recompute the harmonic mean over the same per-model probs the predict path used.
    per_model_probs = [v for k, v in results["probabilities"].items() if k != "ensemble"]
    assert len(per_model_probs) >= 2, "expected >=2 contributing models for the flavour selection to be observable"
    expected_harm = len(per_model_probs) / np.sum(1.0 / np.clip(np.stack(per_model_probs), 1e-12, None), axis=0)
    expected_arith = np.mean(np.stack(per_model_probs), axis=0)

    np.testing.assert_allclose(combined_harm, expected_harm, rtol=1e-5, atol=1e-7)
    # And the chosen-flavour replay must NOT be equal to arithmetic mean (sanity: harm and arith must visibly differ).
    assert not np.allclose(
        combined_harm, expected_arith
    ), "harm-flavoured per-target probability is identical to arithmetic mean -- replay likely fell through to the default code path."
