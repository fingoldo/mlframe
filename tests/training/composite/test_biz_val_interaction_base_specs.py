"""Interaction bases become composite targets that discovery selects, the suite trains, and predict and serving rebuild.

On y = f0 * f1 + 3 sin(2 f2) no single feature is a good base: the best old spec cut the honest-holdout RMSE by 0.08-0.13
(of ~0.53). The step found ``f0__mul__f1`` but only reported it. Screened as a base, the product lets T = y - f0 * f1 carry
just the sin(f2) part, and the honest gain doubles. The column exists in no user frame: every consumer recomputes it from
its parents, so the trained composite predicts on the raw features, reloads from disk and serves from a spec.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite._synthetic_bases import parse_synthetic, synthetic_column

pytest.importorskip("lightgbm")


def _frame(n: int, seed: int) -> pd.DataFrame:
    """Six positive features; y is the f0 x f1 interaction plus a smooth term in f2 plus noise."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.uniform(1.0, 5.0, n) for i in range(6)})
    X["y"] = X["f0"] * X["f1"] + 3.0 * np.sin(2.0 * X["f2"]) + rng.normal(0.0, 0.3, n)
    return X


def _best_honest_gain(interaction: bool, seed: int) -> tuple[float, list]:
    """The largest honest-holdout RMSE gain among the kept specs, and the specs' base columns."""
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    X = _frame(4000, seed)
    cfg = CompositeTargetDiscoveryConfig(enabled=True, random_state=0, interaction_base_discovery_enabled=interaction)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = CompositeTargetDiscovery(cfg).fit(X, "y", [f"f{i}" for i in range(6)], np.arange(len(X)))
    gains = [float(getattr(s, "honest_holdout_rmse_gain", 0.0) or 0.0) for s in d.specs_]
    return max(gains, default=0.0), [s.base_column for s in d.specs_]


def test_the_resolver_rebuilds_a_product_from_its_parents_and_defers_to_a_real_column():
    """``a__mul__b`` is a*b when absent (parents may contain ``__``); a real column of that name is left alone."""
    df = pd.DataFrame({"a__x": [1.0, 2.0], "b": [3.0, 4.0]})
    np.testing.assert_allclose(synthetic_column(df, "a__x__mul__b"), [3.0, 8.0])
    assert parse_synthetic("a__x__sub__b", df.columns) == ("a__x", "sub", "b")
    assert synthetic_column(df.assign(**{"a__x__mul__b": [9.0, 9.0]}), "a__x__mul__b") is None
    assert synthetic_column(df, "a__x__div__b") is None, "div needs a fitted floor and is not resolvable"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_an_interaction_base_doubles_the_honest_gain_on_an_interaction_target(seed: int):
    """With interaction bases screened, a spec on the product base is kept and beats the best spec found without them."""
    with_ib, bases = _best_honest_gain(True, seed)
    without, _ = _best_honest_gain(False, seed)
    assert any("__mul__" in str(b) for b in bases), bases
    assert with_ib > 1.5 * without, (with_ib, without)


def test_the_suite_trains_predicts_reloads_and_serves_a_synthetic_base_composite(tmp_path):
    """The composite on the product base trains in the suite, predicts on raw features, reloads and serves identically."""
    from mlframe.training.composite import CompositeTargetEstimator
    from mlframe.training.composite.serving import export_serving_spec, load_serving_spec, serving_base_from_columns
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.core._predict_main_from_models import predict_from_models
    from mlframe.training.core.predict import load_mlframe_suite

    from .test_composite_integration import _LEAN_OUTPUT_CONFIG_KWARGS, _LEAN_REPORTING_CONFIG_KWARGS, _build_minimal_fte

    df = _frame(3000, 0).rename(columns={"y": "target"})
    cfg = CompositeTargetDiscoveryConfig(enabled=True, random_state=0, max_total_composite_targets=1, min_honest_gain_to_train=None,
                                         transforms=["diff", "linear_residual"])
    models, metadata = train_mlframe_models_suite(
        df=df, target_name="target", model_name="ib", features_and_targets_extractor=_build_minimal_fte(), mlframe_models=["linear"],
        output_config={"data_dir": str(tmp_path), "models_dir": "models", **_LEAN_OUTPUT_CONFIG_KWARGS},
        reporting_config=_LEAN_REPORTING_CONFIG_KWARGS, verbose=0, composite_target_discovery_config=cfg,
    )
    specs = [s for by_t in metadata["composite_target_specs"].values() for ss in by_t.values() for s in ss]
    bases = [[s["base_column"], *(s.get("extra_base_columns") or ())] for s in specs]
    synth = [b for bs in bases for b in bs if "__mul__" in str(b)]
    assert specs and synth, bases
    key = specs[0]["name"]
    wrappers = [e.model for by in models.values() for e in by.get(key, []) if isinstance(getattr(e, "model", None), CompositeTargetEstimator)]
    assert wrappers, f"the synthetic-base composite {key} did not train"

    fresh = _frame(400, 7)
    X = fresh.drop(columns=["y"])
    assert synth[0] not in X.columns
    w = wrappers[0]
    seen = {}
    real_unclipped = type(w)._predict_unclipped

    def capture(self, frame, *a, **k):
        if self is w and k.get("t_hat_override") is None:
            out = real_unclipped(self, frame, *a, **k)
            seen["frame"], seen["y"] = frame, out[0]
            return out
        return real_unclipped(self, frame, *a, **k)

    type(w)._predict_unclipped = capture
    try:
        p_mem = predict_from_models(X, models, metadata, verbose=0)
    finally:
        type(w)._predict_unclipped = real_unclipped
    comp = np.asarray(p_mem["predictions"][next(k for k in p_mem["predictions"] if key in k)], dtype=float)
    assert np.all(np.isfinite(comp))
    rmse_comp = float(np.sqrt(np.mean((comp - fresh["y"].to_numpy()) ** 2)))
    raw_key = next(k for k in p_mem["predictions"] if key not in k and "ENSEMBLE" not in k)
    rmse_raw = float(np.sqrt(np.mean((np.asarray(p_mem["predictions"][raw_key], dtype=float) - fresh["y"].to_numpy()) ** 2)))
    assert rmse_comp < rmse_raw, (rmse_comp, rmse_raw)

    loaded_models, loaded_md = load_mlframe_suite(os.path.join(str(tmp_path), "models", "target", "ib"))
    p_disk = predict_from_models(X, loaded_models, loaded_md, verbose=0)
    np.testing.assert_allclose(np.asarray(p_disk["predictions"][next(k for k in p_disk["predictions"] if key in k)], dtype=float), comp, rtol=1e-9)

    # Serving: the spec records the product's recipe, and the pure-numpy predict rebuilds it from the raw columns.
    spec = export_serving_spec(w)
    assert synth[0] in spec.get("base_recipes", {}), spec.get("base_recipes")
    stage = seen["frame"]
    from mlframe.training.composite.estimator._routing import inner_input
    from mlframe.training.composite.transforms import get_transform

    t = np.asarray(w.estimator_.predict(inner_input(w, stage, get_transform(w.transform_name))), dtype=float)
    base = serving_base_from_columns(spec, {c: np.asarray(stage[c]) for c in stage.columns})
    np.testing.assert_allclose(load_serving_spec(spec)(base, t), w.predict(stage), rtol=1e-9, atol=1e-9)
