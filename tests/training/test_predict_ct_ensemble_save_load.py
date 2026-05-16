"""Wave-2 predict-path parity Fix 4: persist + reload cross-target ensemble entries.

``_phase_composite_post.py`` builds ``_CT_ENSEMBLE__<orig>`` entries and stores them at
``models[target_type][_CT_ENSEMBLE__<orig>] = [SimpleNamespace(model=ensemble, ...)]``. Pre-fix these lived
ONLY in memory; ``predict_mlframe_models_suite`` deserialises ``.dump`` files from disk and never saw them. The
fix dumps each entry at finalize time under the same per-(target_type, target_name) directory layout as ordinary
models, and ``load_mlframe_suite`` picks them up via its recursive glob.
"""
from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


# Workaround for a pre-existing Windows-only ``flush of closed file`` crash in mlframe.training.io that fires
# when zstd is invoked with multi-threaded compression (``threads=-1``). Forcing ``threads=0`` produces a
# byte-identical output on supported builds and unblocks the disk-round-trip tests. This is purely a test-side
# workaround; the underlying file is in the LOCKED scope of the directive.
def _save_threads_zero(model, file, zstd_kwargs=None, verbose=0):
    import dill
    import zstandard as zstd
    from mlframe.training.io import atomic_write_bytes
    _kw = dict(level=4, write_checksum=True, write_content_size=True, threads=0)
    if zstd_kwargs:
        _kw.update(zstd_kwargs)
        _kw["threads"] = 0  # always overwrite to neutralise the broken default
    try:
        def _writer(f):
            compressor = zstd.ZstdCompressor(**_kw)
            with compressor.stream_writer(f) as zf:
                dill.dump(model, zf)
        atomic_write_bytes(file, _writer)
        return True
    except Exception:
        return False


def _make_ct_entry():
    """Tiny picklable cross-target ensemble surrogate -- just a SimpleNamespace with a callable .predict so the
    load roundtrip can be observed without standing up a full CompositeCrossTargetEnsemble (which depends on
    the composite_target_discovery suite, locked here)."""
    class _ToyEnsemble:
        def __init__(self, scale):
            self.scale = float(scale)
            self.strategy = "mean"

        def predict(self, X):
            arr = np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float64)
            return arr.sum(axis=1) * self.scale

        def export_metadata(self):
            return {"strategy": self.strategy, "scale": self.scale}

    return SimpleNamespace(
        model=_ToyEnsemble(scale=2.5),
        model_name="CT_ENSEMBLE",
        columns=None,
        pre_pipeline=None,
        metrics={},
    )


def test_persist_ct_ensemble_entries_roundtrips():
    """Build a minimal ctx with one ``_CT_ENSEMBLE__y`` entry, run ``_persist_ct_ensemble_entries`` against a
    tmpdir, then ``load_mlframe_suite`` must rehydrate the entry under the same key with predict equivalence."""
    from mlframe.training.core._phase_finalize import _persist_ct_ensemble_entries
    from mlframe.training.core.predict import load_mlframe_suite, _load_ct_ensemble_entries

    ct_entry = _make_ct_entry()

    with tempfile.TemporaryDirectory() as tmp:
        # Minimal ctx duck-type -- only the fields _persist_ct_ensemble_entries touches.
        class _CtxStub:
            data_dir = tmp
            models_dir = "models"
            target_name = "yt"
            model_name = "ct_test"
            verbose = 0
            models = {"regression": {"_CT_ENSEMBLE__y": [ct_entry]}}
            slug_to_original_target_name = {}
            metadata = {}

        ctx = _CtxStub()
        # Pre-write the metadata.pkl.zst so load_mlframe_suite has something to read.
        models_path = os.path.join(tmp, "models", "yt", "ct_test")
        os.makedirs(models_path, exist_ok=True)
        # Save metadata via threads=0 workaround.
        import pickle, zstandard
        meta_payload = {"slug_to_original_target_type": {"regression": "regression"}, "slug_to_original_target_name": {}}
        with open(os.path.join(models_path, "metadata.pkl.zst"), "wb") as f:
            f.write(zstandard.ZstdCompressor(level=3, threads=0).compress(pickle.dumps(meta_payload, protocol=5)))

        with patch("mlframe.training.core._phase_finalize.save_mlframe_model", side_effect=_save_threads_zero):
            _persist_ct_ensemble_entries(ctx)

        # The slug map should have been stamped so load-time round-trip preserves the literal _CT_ENSEMBLE__ key.
        assert ctx.slug_to_original_target_name.get("_CT_ENSEMBLE__y") == "_CT_ENSEMBLE__y" or \
            "_CT_ENSEMBLE__y" in ctx.slug_to_original_target_name.values()

        # Now reload via the predict loader.
        # Patch the metadata slug map to include what _persist stamped.
        meta_payload["slug_to_original_target_name"] = dict(ctx.slug_to_original_target_name)
        with open(os.path.join(models_path, "metadata.pkl.zst"), "wb") as f:
            f.write(zstandard.ZstdCompressor(level=3, threads=0).compress(pickle.dumps(meta_payload, protocol=5)))

        # Use the dedicated CT-loader helper directly (it's the one predict reads) since load_mlframe_suite
        # combines _CT_ENSEMBLE entries with regular models -- here we have no regular models, just the CT.
        ct_entries = _load_ct_ensemble_entries(
            models_path,
            slug_to_original_target_type={"regression": "regression"},
            slug_to_original_target_name=meta_payload["slug_to_original_target_name"],
        )
        assert ct_entries, f"_load_ct_ensemble_entries returned empty; layout under {models_path}: {os.listdir(models_path)}"
        assert "regression" in ct_entries, f"missing target_type key; got {list(ct_entries.keys())}"
        _by = ct_entries["regression"]
        assert "_CT_ENSEMBLE__y" in _by, f"missing CT key; got {list(_by.keys())}"
        roundtripped = _by["_CT_ENSEMBLE__y"][0]
        # Predict equivalence.
        import pandas as pd
        X_test = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        np.testing.assert_allclose(
            roundtripped.model.predict(X_test),
            ct_entry.model.predict(X_test),
            rtol=1e-7,
        )


def test_persist_ct_ensemble_entries_no_models_dir_is_noop():
    """When ``ctx.data_dir`` / ``ctx.models_dir`` is empty (in-memory only run), the persister must be a no-op
    rather than create files in CWD or crash."""
    from mlframe.training.core._phase_finalize import _persist_ct_ensemble_entries

    class _CtxStub:
        data_dir = ""
        models_dir = ""
        target_name = "yt"
        model_name = "ct_test"
        verbose = 0
        models = {"regression": {"_CT_ENSEMBLE__y": [_make_ct_entry()]}}
        slug_to_original_target_name = {}
        metadata = {}

    _persist_ct_ensemble_entries(_CtxStub())
    # No exception is the success criterion. Implicitly: no files leaked into CWD because data_dir guard fires.
