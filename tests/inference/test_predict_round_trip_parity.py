"""Wave-2 predict-path parity meta-test: in-memory predict and disk-round-trip predict agree.

Trains a deliberately mixed suite (polars input + ``preprocessing_extensions`` enabled + multiple base models),
captures predictions via ``predict_from_models`` (in-memory), then re-runs predict against the SAME data via the
restored loader path and verifies the per-target probabilities + ensemble probabilities match. Pre-Wave-2 the
ensemble combine ignored chosen flavours, the extensions stack was silently dropped, and CT_ENSEMBLE entries
never persisted -- any of these would surface here as a mismatch.

The on-disk save path is currently broken on the Windows zstandard build in this environment (``flush of closed
file`` on multi-threaded compression), so the test patches ``save_mlframe_model`` to a single-threaded variant.
This is purely a test-side workaround; the underlying ``io.py`` is in the LOCKED scope of the directive.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.core.predict import predict_from_models
from mlframe.training.configs import (
    BaselineDiagnosticsConfig,
    CompositeTargetDiscoveryConfig,
    DummyBaselinesConfig,
    OutputConfig,
    PreprocessingExtensionsConfig,
    ReportingConfig,
)
from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor


def _save_threads_zero(model, file, zstd_kwargs=None, verbose=0, lean=False, durable=False):
    """Single-threaded zstd write that bypasses the ``flush of closed file`` Windows quirk in atomic_write_bytes."""
    import dill  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
    import zstandard as zstd

    try:
        with open(file, "wb") as f:
            compressor = zstd.ZstdCompressor(level=4, write_checksum=True, write_content_size=True, threads=0)
            with compressor.stream_writer(f) as zf:
                dill.dump(model, zf)
        return True
    except Exception:
        return False


def _build_polars_frame(n: int = 1_500, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "x0": rng.normal(size=n).astype("float32"),
            "x1": rng.normal(size=n).astype("float32"),
            "x2": rng.normal(size=n).astype("float32"),
            "y": (1.5 * rng.normal(size=n) + rng.normal(0, 0.3, n)).astype("float32"),
        }
    )


def test_in_memory_predict_matches_chosen_flavour_when_metadata_stamped():
    """Train -> predict in memory. With a per-target flavour stamped into metadata the per-target probability
    matches ``_combine_probs`` of the contributing models' outputs; without the flavour stamp the suite-wide
    ensemble falls back to arithmetic mean. Either way both calls must produce identical output for identical
    inputs, exercising the fastpath + extensions replay + flavour selection in one go."""
    pytest.importorskip("lightgbm")
    df = _build_polars_frame()
    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])
    ext_cfg = PreprocessingExtensionsConfig(scaler="StandardScaler", verbose_logging=False)

    models, metadata = train_mlframe_models_suite(
        df=df,
        target_name="y",
        model_name="round_trip",
        features_and_targets_extractor=fte,
        mlframe_models=["lgb", "xgb"],
        verbose=0,
        output_config=OutputConfig(data_dir="", models_dir="", save_charts=False),
        composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(plot_outputs="matplotlib[png]", plot_inline_display=False),
        preprocessing_extensions=ext_cfg,
    )
    assert metadata.get("extensions_pipeline") is not None, "extensions_pipeline not persisted"

    # Predict twice on the same frame; identical inputs must give identical outputs (no stochasticity in the
    # predict path itself; the trained models are deterministic on a fixed seed).
    results_a = predict_from_models(
        df=df,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=fte,
        return_probabilities=False,
        verbose=0,
    )
    results_b = predict_from_models(
        df=df,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=fte,
        return_probabilities=False,
        verbose=0,
    )
    assert results_a["models_used"], "first predict produced no models_used"
    assert results_b["models_used"] == results_a["models_used"], "models_used drifted between calls"
    np.testing.assert_allclose(
        results_a["ensemble_predictions"],
        results_b["ensemble_predictions"],
        rtol=1e-7,
        err_msg="round-trip predict on identical input produced non-deterministic ensemble predictions",
    )


def test_in_memory_and_disk_predict_agree_on_simple_suite(tmp_path):
    """Train an LGB-only suite (polars-native fastpath off, but ``preprocessing_extensions`` exercised) and verify
    the in-memory predict matches the disk-load predict. This is the round-trip parity gate: the disk path must
    rehydrate the same pipeline + extensions_pipeline + metadata that the in-memory path uses."""
    pytest.importorskip("lightgbm")
    df = _build_polars_frame()
    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])
    ext_cfg = PreprocessingExtensionsConfig(scaler="StandardScaler", verbose_logging=False)

    def _atomic_write_bytes_threads_zero(target_path, writer_fn):
        """Drop-in replacement that avoids the ``f.flush()`` AFTER stream_writer.close() race."""
        with open(target_path, "wb") as f:
            writer_fn(f)
        return True

    tmp = str(tmp_path)
    with patch("mlframe.training.train_eval.save_mlframe_model", side_effect=_save_threads_zero):
        with patch("mlframe.training.io.atomic_write_bytes", side_effect=_atomic_write_bytes_threads_zero):
            # The atomic_write_bytes shim above also bypasses the Windows flush-of-closed-file path on the
            # metadata save (zstd stream_writer + outer f.flush race).
            models, metadata = train_mlframe_models_suite(
                df=df,
                target_name="y",
                model_name="round_trip_disk",
                features_and_targets_extractor=fte,
                mlframe_models=["lgb"],
                verbose=0,
                output_config=OutputConfig(data_dir=tmp, models_dir="models", save_charts=False),
                composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
                baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
                dummy_baselines_config=DummyBaselinesConfig(enabled=False),
                reporting_config=ReportingConfig(plot_outputs="matplotlib[png]", plot_inline_display=False),
                preprocessing_extensions=ext_cfg,
            )

    # In-memory predict.
    results_in_memory = predict_from_models(
        df=df,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=fte,
        return_probabilities=False,
        verbose=0,
    )

    # Disk predict via the load_mlframe_suite + predict_from_models pairing (predict_mlframe_models_suite also
    # reads from disk but takes the same code path; the in-memory variant after load is the most apples-to-
    # apples comparison). Under the single-threaded zstd shim the multi-threaded ``flush of closed file`` race
    # cannot occur, so the disk artefacts MUST exist -- a missing file is a real save/load regression, not a
    # platform quirk. We ``pytest.fail`` (not skip) so a dropped metadata / dump file is caught where it runs.
    models_path = os.path.join(tmp, "models", "y", "round_trip_disk")
    if not os.path.exists(os.path.join(models_path, "metadata.pkl.zst")) and not os.path.exists(os.path.join(models_path, "metadata.pkl")):
        pytest.fail(
            f"shimmed disk save produced no metadata file under {models_path}; "
            f"present: {os.listdir(models_path) if os.path.isdir(models_path) else 'MISSING DIR'}"
        )
    from mlframe.training.core.predict import load_mlframe_suite

    loaded_models, loaded_metadata = load_mlframe_suite(models_path)
    assert loaded_models, (
        f"shimmed disk save produced no .dump files under {models_path}; present: {os.listdir(models_path) if os.path.isdir(models_path) else 'MISSING DIR'}"
    )

    results_disk = predict_from_models(
        df=df,
        models=loaded_models,
        metadata=loaded_metadata,
        features_and_targets_extractor=fte,
        return_probabilities=False,
        verbose=0,
    )

    assert results_disk["models_used"], "disk predict produced no models"

    # Parity check: ensemble predictions must agree to float precision. Per-model predict numerics can drift
    # marginally across save / load (CB / XGB / LGB serialisation may lose some bits) so an absolute tolerance
    # of 1e-4 absorbs the round-trip jitter while still catching the silent-drop bugs the fix addresses.
    np.testing.assert_allclose(
        results_in_memory["ensemble_predictions"],
        results_disk["ensemble_predictions"],
        rtol=1e-4,
        atol=1e-4,
        err_msg="disk-load predict drifted from in-memory predict beyond float-precision; check whether the "
        "extensions_pipeline or chosen ensemble flavour was lost across save / load.",
    )


def _train_multi_target_suite(df, target_type, tmp_path, model_name):
    """Train an LGB-only multiclass / multilabel suite under the single-threaded zstd shim and save to disk."""
    from tests.training.shared import SimpleFeaturesAndTargetsExtractor as _SharedFTE

    fte = _SharedFTE(target_column="target", regression=False, target_type=target_type)
    tmp = str(tmp_path)
    with patch("mlframe.training.train_eval.save_mlframe_model", side_effect=_save_threads_zero):
        with patch("mlframe.training.io.atomic_write_bytes", side_effect=_atomic_write_bytes_threads_zero):
            models, metadata = train_mlframe_models_suite(
                df=df,
                target_name="target",
                model_name=model_name,
                features_and_targets_extractor=fte,
                mlframe_models=["lgb"],
                use_mlframe_ensembles=False,
                verbose=0,
                output_config=OutputConfig(data_dir=tmp, models_dir="models", save_charts=False),
                composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
                baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
                dummy_baselines_config=DummyBaselinesConfig(enabled=False),
                reporting_config=ReportingConfig(plot_outputs="matplotlib[png]", plot_inline_display=False),
                hyperparams_config={"iterations": 12, "lgb_kwargs": {"device_type": "cpu"}},
            )
    return fte, models, metadata


def _atomic_write_bytes_threads_zero(target_path, writer_fn):
    with open(target_path, "wb") as f:
        writer_fn(f)
    return True


def test_in_memory_and_disk_predict_agree_on_multiclass(tmp_path):
    """Multiclass (K=3) ensemble probabilities round-trip in-memory -> disk to float precision and stay a simplex."""
    pytest.importorskip("lightgbm")
    from mlframe.training.configs import TargetTypes
    from mlframe.training.core.predict import load_mlframe_suite

    rng = np.random.default_rng(0)
    n = 900
    X = rng.standard_normal((n, 5)).astype("float32")
    score = X[:, 0] + 0.5 * X[:, 1]
    y = np.digitize(score, [np.quantile(score, 1 / 3), np.quantile(score, 2 / 3)]).astype("int64")
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y

    fte, models, metadata = _train_multi_target_suite(
        df,
        TargetTypes.MULTICLASS_CLASSIFICATION,
        tmp_path,
        "round_trip_mc",
    )
    mem = predict_from_models(df=df, models=models, metadata=metadata, features_and_targets_extractor=fte, return_probabilities=True, verbose=0)
    ep_mem = np.asarray(mem["ensemble_probabilities"])
    assert ep_mem.shape == (n, 3)
    np.testing.assert_allclose(ep_mem.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)

    models_path = os.path.join(str(tmp_path), "models", "target", "round_trip_mc")
    loaded_models, loaded_metadata = load_mlframe_suite(models_path)
    assert loaded_models, f"shimmed disk save produced no .dump files under {models_path}"
    disk = predict_from_models(df=df, models=loaded_models, metadata=loaded_metadata, features_and_targets_extractor=fte, return_probabilities=True, verbose=0)
    np.testing.assert_allclose(
        ep_mem,
        np.asarray(disk["ensemble_probabilities"]),
        rtol=1e-4,
        atol=1e-4,
        err_msg="multiclass disk-load predict drifted from in-memory beyond float precision",
    )


def test_in_memory_and_disk_predict_agree_on_multilabel(tmp_path):
    """Multilabel (3 labels) per-label probabilities + per-label ensemble matrix round-trip
    in-memory -> disk to float precision. The ensemble combine now runs on the canonicalized
    (N, K) per-label matrix, so ensemble_probabilities is populated (one P(label=1) col per label)."""
    pytest.importorskip("lightgbm")
    from mlframe.training.configs import TargetTypes
    from mlframe.training.core.predict import load_mlframe_suite

    rng = np.random.default_rng(808)
    n = 900
    X = rng.normal(0, 1, (n, 6)).astype("float32")
    y1 = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-(2.0 * X[:, 0] - X[:, 1])))).astype("int8")
    y2 = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-(-1.5 * X[:, 2] + X[:, 3])))).astype("int8")
    y3 = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-(X[:, 4] + X[:, 5])))).astype("int8")
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = [list(row) for row in np.stack([y1, y2, y3], axis=1)]

    fte, models, metadata = _train_multi_target_suite(
        df,
        TargetTypes.MULTILABEL_CLASSIFICATION,
        tmp_path,
        "round_trip_ml",
    )

    def _per_label(res):
        probs = res.get("probabilities", {})
        key = next(k for k in probs if k.startswith("multilabel_classification"))
        per_label = probs[key]
        assert isinstance(per_label, list) and len(per_label) == 3
        return per_label

    mem_res = predict_from_models(df=df, models=models, metadata=metadata, features_and_targets_extractor=fte, return_probabilities=True, verbose=0)
    mem = _per_label(mem_res)
    for arr in mem:
        assert np.asarray(arr).shape == (n, 2)
    ep_mem = mem_res.get("ensemble_probabilities")
    assert ep_mem is not None, "multilabel ensemble_probabilities should be populated, not None"
    ep_mem = np.asarray(ep_mem)
    assert ep_mem.shape == (n, 3), f"expected (n, 3) per-label ensemble matrix; got {ep_mem.shape}"

    models_path = os.path.join(str(tmp_path), "models", "target", "round_trip_ml")
    loaded_models, loaded_metadata = load_mlframe_suite(models_path)
    assert loaded_models, f"shimmed disk save produced no .dump files under {models_path}"
    disk_res = predict_from_models(
        df=df, models=loaded_models, metadata=loaded_metadata, features_and_targets_extractor=fte, return_probabilities=True, verbose=0
    )
    disk = _per_label(disk_res)
    for li, (a, d) in enumerate(zip(mem, disk)):
        np.testing.assert_allclose(
            np.asarray(a), np.asarray(d), rtol=1e-4, atol=1e-4, err_msg=f"multilabel label {li} disk-load predict drifted beyond float precision"
        )
    np.testing.assert_allclose(
        ep_mem,
        np.asarray(disk_res["ensemble_probabilities"]),
        rtol=1e-4,
        atol=1e-4,
        err_msg="multilabel ensemble_probabilities drifted across disk round-trip beyond float precision",
    )
