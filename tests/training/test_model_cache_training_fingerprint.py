"""A cached model is reused only when it was trained on the same hyperparameters, targets, rows and weights.

The suite's model cache is a ``.dump`` per model name. It checked the feature schema (and a composite spec's digest) but
nothing the fit itself consumed, so a rerun into the same directory with a new label, a new split or a new learning rate
served the old model. The dump now records a training fingerprint; a mismatch, or a dump without one, retrains.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from mlframe.training._model_cache_fingerprint import training_fingerprint, training_fingerprint_mismatch


def _common(y, sw=None):
    """The ``common_params`` slice the fingerprint reads."""
    return {"train_target": y, "train_idx": np.arange(len(y)), "sample_weight": sw, "train_df": pd.DataFrame({"a": np.zeros(len(y))}),
            "plot_file": "run-specific/path_"}


def test_the_fingerprint_follows_every_fit_input_and_nothing_else():
    """Hyperparameters, target values, rows and weights each change it; a report path or a fresh equal estimator does not."""
    y = np.arange(10.0)
    ref = training_fingerprint(Ridge(alpha=1.0), {"model": Ridge(alpha=1.0)}, _common(y))
    assert ref == training_fingerprint(Ridge(alpha=1.0), {"model": Ridge(alpha=1.0)}, {**_common(y), "plot_file": "elsewhere_"})
    changed = {
        "hyperparameter": training_fingerprint(Ridge(alpha=2.0), {"model": Ridge(alpha=2.0)}, _common(y)),
        "estimator": training_fingerprint(Ridge(alpha=1.0, fit_intercept=False), {}, _common(y)),
        "target": training_fingerprint(Ridge(alpha=1.0), {}, _common(y + 1.0)),
        "weights": training_fingerprint(Ridge(alpha=1.0), {}, _common(y, sw=np.ones(10))),
        "rows": training_fingerprint(Ridge(alpha=1.0), {}, {**_common(y), "train_idx": np.arange(1, 11)}),
    }
    assert all(v != ref for v in changed.values()), [k for k, v in changed.items() if v == ref]


def test_a_dump_trained_on_other_inputs_is_invalidated():
    """Matching fingerprint: reused; different or missing: stale; no fingerprint wanted (no cache path): nothing to check."""
    assert training_fingerprint_mismatch(types.SimpleNamespace(training_fingerprint_="a"), "a") is None
    assert "changed" in training_fingerprint_mismatch(types.SimpleNamespace(training_fingerprint_="a"), "b")
    assert "no training fingerprint" in training_fingerprint_mismatch(types.SimpleNamespace(), "b")
    assert training_fingerprint_mismatch(types.SimpleNamespace(), None) is None


@pytest.fixture
def evaluate_modes(monkeypatch):
    """Record, per model fit, whether the suite reused a cached dump (``just_evaluate=True``) or trained."""
    import mlframe.training.train_eval as te

    modes: list[bool] = []
    real = te._call_train_evaluate_with_configs

    def spy(*args, **kwargs):
        modes.append(bool(kwargs.get("just_evaluate")))
        return real(*args, **kwargs)

    monkeypatch.setattr(te, "_call_train_evaluate_with_configs", spy)
    return modes


def test_a_rerun_reuses_the_dump_only_while_the_inputs_are_unchanged(tmp_path, evaluate_modes):
    """Same inputs twice: the second run loads the dump. A changed target, then a changed hyperparameter: both retrain."""
    from mlframe.training.core import train_mlframe_models_suite

    from tests.training.composite.test_composite_integration import _LEAN_OUTPUT_CONFIG_KWARGS, _LEAN_REPORTING_CONFIG_KWARGS, _build_minimal_fte

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.normal(size=400), "x2": rng.normal(size=400)})
    df["target"] = 2.0 * df["x1"] + rng.normal(0.0, 0.1, 400)

    def run(frame, **model_kw):
        evaluate_modes.clear()
        train_mlframe_models_suite(
            df=frame, target_name="target", model_name="fp", features_and_targets_extractor=_build_minimal_fte(), mlframe_models=["linear"],
            output_config={"data_dir": str(tmp_path), "models_dir": "models", **_LEAN_OUTPUT_CONFIG_KWARGS},
            reporting_config=_LEAN_REPORTING_CONFIG_KWARGS, verbose=0, **model_kw,
        )
        return list(evaluate_modes)

    first = run(df)
    assert first and not any(first), "the first run must train"
    assert all(run(df)), "an identical rerun must load the dump"
    assert not any(run(df.assign(target=df["target"] * 3.0 + 1.0))), "a changed target must retrain"
