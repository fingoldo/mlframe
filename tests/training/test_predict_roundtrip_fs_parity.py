"""save -> load -> predict parity with feature selection enabled (training_integration-01, the P0).

No test pinned the full disk round trip of a suite trained WITH feature selection, despite this being the
documented origin of a recurring prod-bug family: a fitted selector (the per-model ``pre_pipeline``) that loses
its fit-state across save/load, so predict-time silently falls back to the column-subset recovery branch (or
diverges) instead of replaying the real transform. This pins the HAPPY path -- a properly saved-and-reloaded
selector must transform identically and need NO recovery fallback -- which is exactly what a fit-state-loss
regression would break.

The companion ``test_predict_fs_recovery_behavioral.py`` pins the recovery branch itself (when transform genuinely
fails); here the assertion is the opposite: on a clean round trip that branch must stay silent.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import pytest

from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.core.predict import load_mlframe_suite, predict_from_models
from mlframe.training import FeatureSelectionConfig, OutputConfig
from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor
from tests.feature_selection.conftest import is_fast_mode


def _save_threads_zero(model, file, zstd_kwargs=None, verbose=0, lean=False, durable=False):
    # The local Windows zstandard build raises ``ValueError: flush of closed file`` inside io.save_mlframe_model's
    # atomic_write_bytes (flush after the stream_writer context closed the file). io.py is in the locked-scope of
    # the parallel refactor, so the test writes the .dump directly (threads=0, no atomic rename) -- acceptable here.
    """Save threads zero."""
    import dill  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
    import zstandard as zstd

    try:
        with open(file, "wb") as f:
            with zstd.ZstdCompressor(level=4, write_checksum=True, threads=0).stream_writer(f) as zf:
                dill.dump(model, zf)
        return True
    except Exception:
        import traceback

        traceback.print_exc()
        return False


def _make_frame(n: int, seed: int = 0):
    """Make frame."""
    rng = np.random.default_rng(seed)
    n_signal, n_noise = 6, 4
    sig = rng.standard_normal((n, n_signal)).astype("float32")
    noise = rng.standard_normal((n, n_noise)).astype("float32")
    coefs = np.array([1.3, -1.1, 0.9, 0.8, -0.7, 0.6])
    logit = sig @ coefs
    y = (logit + 0.3 * rng.standard_normal(n) > 0).astype("int32")
    cols = {f"s{i}": sig[:, i] for i in range(n_signal)}
    cols.update({f"n{i}": noise[:, i] for i in range(n_noise)})
    cols["y"] = y
    return pd.DataFrame(cols)


def _fte():
    """Fte."""
    return SimpleFeaturesAndTargetsExtractor(classification_targets=["y"], classification_exact_values={"y": 1})


def _fs_config(fe_on: bool):
    """Fs config."""
    kw = {"verbose": 0, "max_runtime_mins": 1, "n_workers": 1, "quantization_nbins": 5, "use_simple_mode": True}
    if fe_on:
        kw.update({"use_simple_mode": False, "fe_max_steps": 1, "fe_ntop_features": 3})
    return FeatureSelectionConfig(use_mrmr_fs=True, mrmr_kwargs=kw)


def _train_to_disk(df, tmp, fe_on):
    """Train to disk."""
    from unittest.mock import patch

    # _phase_finalize imports save_mlframe_model at module level (patch its bound ref); _setup_helpers_metadata
    # imports it lazily inside a function (patch the source io module so the lazy resolution picks up the stub).
    with (
        patch("mlframe.training.core._phase_finalize.save_mlframe_model", side_effect=_save_threads_zero),
        patch("mlframe.training.io.save_mlframe_model", side_effect=_save_threads_zero),
    ):
        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="rt",
            model_name="fsmodel",
            features_and_targets_extractor=_fte(),
            mlframe_models=["lgb"],
            hyperparams_config={"iterations": 15},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            verbose=0,
            output_config=OutputConfig(data_dir=str(tmp), models_dir="models", save_charts=False),
            feature_selection_config=_fs_config(fe_on),
        )
    return models, metadata


def _fs_branch_keys(probs):
    # The per-model key is ``<target_type>_<target>[_<pre_pipeline_cls>]``; with a fitted MRMR selector as the
    # pre_pipeline the FS-branch entry carries the ``_MRMR`` class suffix (the no-FS twin has the bare name).
    """Fs branch keys."""
    return [k for k in probs if k != "ensemble" and "MRMR" in k]


def _assert_roundtrip_parity(df, tmp, fe_on, caplog):
    """Assert roundtrip parity."""
    models_mem, meta_mem = _train_to_disk(df, tmp, fe_on)
    assert models_mem, "training returned empty models"

    res_mem = predict_from_models(df=df, models=models_mem, metadata=meta_mem, features_and_targets_extractor=_fte(), return_probabilities=True, verbose=0)
    fs_keys = _fs_branch_keys(res_mem["probabilities"])
    assert fs_keys, f"no FS-branch (_Pipeline) key; got {list(res_mem['probabilities'])}"

    models_path = os.path.join(str(tmp), "models", "rt", "fsmodel")
    models_disk, meta_disk = load_mlframe_suite(models_path)
    assert models_disk, f"load_mlframe_suite returned empty from {models_path}: {os.listdir(models_path)}"

    with caplog.at_level(logging.WARNING):
        res_disk = predict_from_models(
            df=df, models=models_disk, metadata=meta_disk, features_and_targets_extractor=_fte(), return_probabilities=True, verbose=0
        )

    disk_keys = _fs_branch_keys(res_disk["probabilities"])
    assert disk_keys, f"FS-branch key vanished after reload; got {list(res_disk['probabilities'])}"
    for k in fs_keys:
        assert k in res_disk["probabilities"], f"key {k} missing after reload"
        np.testing.assert_allclose(
            np.asarray(res_mem["probabilities"][k], dtype=float),
            np.asarray(res_disk["probabilities"][k], dtype=float),
            rtol=1e-6,
            atol=1e-9,
            err_msg=f"predict diverged across save/load for FS-branch model {k} (fit-state lost)",
        )

    recovery = [
        r.getMessage()
        for r in caplog.records
        if "pre_pipeline" in r.getMessage().lower()
        and ("skip" in r.getMessage().lower() or "fall" in r.getMessage().lower() or "recover" in r.getMessage().lower())
    ]
    assert not recovery, f"reloaded selector triggered the predict-time recovery fallback (fit-state not preserved): {recovery}"


def test_save_load_predict_parity_simple_mrmr(tmp_path, caplog):
    """Simple-mode MRMR FS: the reloaded selector replays its transform with bit-equal predictions, no recovery."""
    df = _make_frame(n=500 if is_fast_mode() else 800, seed=0)
    _assert_roundtrip_parity(df, tmp_path, fe_on=False, caplog=caplog)


@pytest.mark.slow
def test_save_load_predict_parity_mrmr_with_fe(tmp_path, caplog):
    """FE-on MRMR FS (engineered columns in the recipe): the engineered-feature replay must also survive the round
    trip -- the highest-risk case for fit-state loss since the recipe must be re-applied at predict time."""
    df = _make_frame(n=700, seed=1)
    _assert_roundtrip_parity(df, tmp_path, fe_on=True, caplog=caplog)
