"""predict_from_models parity for multiclass (K=3) and multilabel (3 labels) suite output.

Covers the P0 inference path for K>2 and multi-output targets, which the regression/binary parity
tests do not exercise: per-class probability shape, per-row sum-to-1 (multiclass), determinism across
two predict calls on identical input, and float-precision round-trip after save/load.

Multilabel ensemble combine: ``predict_from_models`` builds the per-model per-label probabilities
correctly (a list of (N, 2) arrays) AND now canonicalizes them to an (N, K) per-label matrix for the
ensemble path, so ``ensemble_probabilities`` is populated with one column of P(label=1) per label.
The per-model ``results["probabilities"][model]`` stays the raw list for consumers that need it.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.core.predict import predict_from_models, load_mlframe_suite
from mlframe.training.configs import (
    BaselineDiagnosticsConfig,
    CompositeTargetDiscoveryConfig,
    DummyBaselinesConfig,
    OutputConfig,
    ReportingConfig,
    TargetTypes,
)
from tests.training.shared import SimpleFeaturesAndTargetsExtractor


def _save_threads_zero(model, file, zstd_kwargs=None, verbose=0, lean=False, durable=False):
    """Single-threaded zstd write that bypasses the multi-threaded ``flush of closed file`` quirk."""
    import dill  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
    import zstandard as zstd

    with open(file, "wb") as f:
        compressor = zstd.ZstdCompressor(level=4, write_checksum=True, write_content_size=True, threads=0)
        with compressor.stream_writer(f) as zf:
            dill.dump(model, zf)
    return True


def _atomic_write_bytes_threads_zero(target_path, writer_fn):
    """Returns ``True`` (after 1 setup step)."""
    with open(target_path, "wb") as f:
        writer_fn(f)
    return True


def _make_multiclass_df(n: int = 900, seed: int = 0) -> pd.DataFrame:
    """Builds seeded synthetic test data; returns ``df``."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5)).astype("float32")
    score = X[:, 0] + 0.5 * X[:, 1]
    y = np.digitize(score, [np.quantile(score, 1 / 3), np.quantile(score, 2 / 3)]).astype("int64")
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    return df


def _make_multilabel_df(n: int = 900, seed: int = 808) -> pd.DataFrame:
    """Builds seeded synthetic test data; returns ``df``."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 6)).astype("float32")
    y1 = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-(2.0 * X[:, 0] - X[:, 1])))).astype("int8")
    y2 = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-(-1.5 * X[:, 2] + X[:, 3])))).astype("int8")
    y3 = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-(X[:, 4] + X[:, 5])))).astype("int8")
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = [list(row) for row in np.stack([y1, y2, y3], axis=1)]
    return df


def _train_suite(df, target_type, tmp_path, model_name):
    """Returns ``(fte, models, metadata)`` (after 3 setup steps)."""
    fte = SimpleFeaturesAndTargetsExtractor(
        target_column="target",
        regression=False,
        target_type=target_type,
    )
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


def _predict(df, models, metadata, fte):
    """Returns ``predict_from_models(df=df, models=models, metadata=metadata, features_and_targets_extra...``."""
    return predict_from_models(
        df=df,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=fte,
        return_probabilities=True,
        verbose=0,
    )


# ---------------------------------------------------------------------------
# Multiclass (K=3)
# ---------------------------------------------------------------------------


def test_multiclass_ensemble_probability_shape_and_simplex():
    """Multiclass ensemble_probabilities is (n, 3) with each row a valid probability simplex."""
    pytest.importorskip("lightgbm")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        df = _make_multiclass_df()
        fte, models, metadata = _train_suite(df, TargetTypes.MULTICLASS_CLASSIFICATION, td, "mc_shape")
        res = _predict(df, models, metadata, fte)

    ep = res.get("ensemble_probabilities")
    assert ep is not None, "multiclass ensemble_probabilities should not be None"
    ep = np.asarray(ep)
    assert ep.shape == (len(df), 3), f"expected (n, 3); got {ep.shape}"
    np.testing.assert_allclose(ep.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)
    assert (ep >= -1e-9).all() and (ep <= 1.0 + 1e-9).all(), "probabilities outside [0, 1]"


def test_multiclass_predict_twice_and_after_load_parity(tmp_path):
    """Two predict calls on identical input agree; disk-loaded suite agrees to float precision."""
    pytest.importorskip("lightgbm")
    df = _make_multiclass_df()
    fte, models, metadata = _train_suite(df, TargetTypes.MULTICLASS_CLASSIFICATION, tmp_path, "mc_parity")

    res_a = _predict(df, models, metadata, fte)
    res_b = _predict(df, models, metadata, fte)
    np.testing.assert_allclose(
        np.asarray(res_a["ensemble_probabilities"]),
        np.asarray(res_b["ensemble_probabilities"]),
        rtol=1e-7,
        err_msg="multiclass predict is non-deterministic on identical input",
    )

    models_path = os.path.join(str(tmp_path), "models", "target", "mc_parity")
    assert os.path.isdir(models_path), f"shimmed disk save produced no model dir: {models_path}"
    loaded_models, loaded_metadata = load_mlframe_suite(models_path)
    assert loaded_models, f"disk save produced no .dump files under {models_path}"
    res_disk = _predict(df, loaded_models, loaded_metadata, fte)
    np.testing.assert_allclose(
        np.asarray(res_a["ensemble_probabilities"]),
        np.asarray(res_disk["ensemble_probabilities"]),
        rtol=1e-4,
        atol=1e-4,
        err_msg="multiclass disk-load predict drifted from in-memory beyond float precision",
    )


# ---------------------------------------------------------------------------
# Multilabel (3 labels)
# ---------------------------------------------------------------------------


def _multilabel_per_model_probs(res):
    """Return the per-model multilabel probabilities as the canonical (N, K) P(label=1) matrix.

    ``_wrap_predict_result`` (``cb/_cb_pool.py``) now canonicalises ANY list-form
    ``predict_proba`` result -- including ``MultiOutputClassifier``'s per-label list of
    (N, 2) arrays -- to a single ``(N, K)`` ndarray before it ever reaches
    ``predict_from_models``, specifically so a blind ``np.asarray`` on the list elsewhere
    can't silently mis-stack it into ``(K, N, 2)``. So ``results["probabilities"][model_name]``
    is this canonical ``(N, K)`` matrix (one P(label=1) column per label), not the raw list.
    """
    probs = res.get("probabilities", {})
    key = next((k for k in probs if k.startswith("multilabel_classification")), None)
    assert key is not None, f"no multilabel probability key in {list(probs)}"
    per_label = np.asarray(probs[key])
    assert per_label.ndim == 2, f"expected the canonical (N, K) multilabel probability matrix; got shape {per_label.shape}"
    return per_label


def test_multilabel_per_label_probability_shapes():
    """Per-model multilabel probabilities are the canonical (n, 3) P(label=1) matrix."""
    pytest.importorskip("lightgbm")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        df = _make_multilabel_df()
        fte, models, metadata = _train_suite(df, TargetTypes.MULTILABEL_CLASSIFICATION, td, "ml_shape")
        res = _predict(df, models, metadata, fte)

    per_label = _multilabel_per_model_probs(res)
    assert per_label.shape == (len(df), 3), f"expected (n, 3); got {per_label.shape}"
    assert np.all((per_label >= 0.0) & (per_label <= 1.0)), "P(label=1) values must lie in [0, 1]"

    # Ensemble combine runs on the same canonicalized per-label matrix: one P(label=1) column per label.
    ep = res.get("ensemble_probabilities")
    assert ep is not None, "multilabel ensemble_probabilities should be populated, not None"
    ep = np.asarray(ep)
    assert ep.shape == (len(df), 3), f"expected (n, 3) per-label ensemble matrix; got {ep.shape}"
    np.testing.assert_allclose(
        ep,
        per_label,
        rtol=1e-6,
        atol=1e-6,
        err_msg="single-model ensemble_probabilities should equal that model's own per-label matrix",
    )


def test_multilabel_predict_twice_and_after_load_parity(tmp_path):
    """Per-label multilabel probabilities + the per-label ensemble matrix are deterministic across
    predict-twice and round-trip to disk to float precision."""
    pytest.importorskip("lightgbm")
    df = _make_multilabel_df()
    fte, models, metadata = _train_suite(df, TargetTypes.MULTILABEL_CLASSIFICATION, tmp_path, "ml_parity")

    res_a = _predict(df, models, metadata, fte)
    res_b = _predict(df, models, metadata, fte)
    pl_a = _multilabel_per_model_probs(res_a)
    pl_b = _multilabel_per_model_probs(res_b)
    np.testing.assert_allclose(
        pl_a,
        pl_b,
        rtol=1e-7,
        err_msg="multilabel per-label matrix predict is non-deterministic on identical input",
    )
    ep_a = res_a.get("ensemble_probabilities")
    assert ep_a is not None, "multilabel ensemble_probabilities should be populated, not None"
    ep_a = np.asarray(ep_a)
    assert ep_a.shape == (len(df), 3), f"expected (n, 3) per-label ensemble matrix; got {ep_a.shape}"
    np.testing.assert_allclose(
        ep_a,
        np.asarray(res_b["ensemble_probabilities"]),
        rtol=1e-7,
        err_msg="multilabel ensemble_probabilities is non-deterministic on identical input",
    )

    models_path = os.path.join(str(tmp_path), "models", "target", "ml_parity")
    assert os.path.isdir(models_path), f"shimmed disk save produced no model dir: {models_path}"
    loaded_models, loaded_metadata = load_mlframe_suite(models_path)
    assert loaded_models, f"disk save produced no .dump files under {models_path}"
    res_disk = _predict(df, loaded_models, loaded_metadata, fte)
    pl_disk = _multilabel_per_model_probs(res_disk)
    assert pl_disk.shape == pl_a.shape
    np.testing.assert_allclose(
        pl_a,
        pl_disk,
        rtol=1e-4,
        atol=1e-4,
        err_msg="multilabel per-label matrix disk-load predict drifted beyond float precision",
    )
    np.testing.assert_allclose(
        ep_a,
        np.asarray(res_disk["ensemble_probabilities"]),
        rtol=1e-4,
        atol=1e-4,
        err_msg="multilabel ensemble_probabilities drifted across disk round-trip beyond float precision",
    )
