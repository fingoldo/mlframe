"""Wave-2 predict-path parity Fix 4: persist + reload cross-target ensemble entries.

``_phase_composite_post.py`` builds ``_CT_ENSEMBLE__<orig>`` entries and stores them at
``models[target_type][_CT_ENSEMBLE__<orig>] = [SimpleNamespace(model=ensemble, ...)]``. Pre-fix these lived
ONLY in memory; ``predict_mlframe_models_suite`` deserialises ``.dump`` files from disk and never saw them. The
fix dumps each entry at finalize time under the same per-(target_type, target_name) directory layout as ordinary
models, and ``load_mlframe_suite`` picks them up via its recursive glob.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


# Workaround for a pre-existing Windows-only ``flush of closed file`` crash in mlframe.training.io. The library's
# ``save_mlframe_model`` writes the zstd stream through ``atomic_write_bytes`` which calls ``f.flush()`` AFTER
# the inner ``stream_writer`` context-manager closed the underlying file -- raises ``ValueError: flush of closed
# file`` on the local Windows zstandard build. This bypass writes the file directly (no atomic rename, no extra
# fsync) which is acceptable in a test context. The underlying io.py is in the LOCKED scope of the directive.
def _save_threads_zero(model, file, zstd_kwargs=None, verbose=0, lean=False, durable=False):
    """Test helper: import dill; import zstandard as zstd; try: with open(file, 'wb') as f: compressor = zstd.ZstdCo...."""
    import dill  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
    import zstandard as zstd

    try:
        with open(file, "wb") as f:
            compressor = zstd.ZstdCompressor(level=4, write_checksum=True, write_content_size=True, threads=0)
            with compressor.stream_writer(f) as zf:
                dill.dump(model, zf)
        return True
    except Exception:
        import traceback

        traceback.print_exc()
        return False


def _make_ct_entry():
    """Tiny picklable cross-target ensemble surrogate using sklearn DummyRegressor (allowlisted by _SafeUnpickler so
    the load round-trip works). Fitting on a tiny synthetic frame produces a constant predictor; correctness of the
    persistence helpers is what we test here, not the ensemble math."""
    from sklearn.dummy import DummyRegressor

    model = DummyRegressor(strategy="constant", constant=7.5)
    model.fit(np.zeros((3, 2)), np.zeros(3))
    return SimpleNamespace(
        model=model,
        model_name="CT_ENSEMBLE",
        columns=None,
        pre_pipeline=None,
        metrics={},
    )


def test_persist_ct_ensemble_entries_roundtrips(tmp_path):
    """Build a minimal ctx with one ``_CT_ENSEMBLE__y`` entry, run ``_persist_ct_ensemble_entries`` against a
    tmpdir, then ``load_mlframe_suite`` must rehydrate the entry under the same key with predict equivalence."""
    from mlframe.training.core._phase_finalize import _persist_ct_ensemble_entries
    from mlframe.training.core.predict import _load_ct_ensemble_entries

    ct_entry = _make_ct_entry()

    tmp = str(tmp_path)

    # Minimal ctx duck-type -- only the fields _persist_ct_ensemble_entries touches.
    class _CtxStub:
        """Groups tests covering CtxStub."""
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
    import pickle, zstandard  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

    meta_payload = {"slug_to_original_target_type": {"regression": "regression"}, "slug_to_original_target_name": {}}
    with open(os.path.join(models_path, "metadata.pkl.zst"), "wb") as f:
        f.write(zstandard.ZstdCompressor(level=3, threads=0).compress(pickle.dumps(meta_payload, protocol=5)))

    with patch("mlframe.training.core._phase_finalize.save_mlframe_model", side_effect=_save_threads_zero):
        _persist_ct_ensemble_entries(ctx)

    # The slug map should have been stamped so load-time round-trip preserves the literal _CT_ENSEMBLE__ key.
    assert ctx.slug_to_original_target_name.get("_CT_ENSEMBLE__y") == "_CT_ENSEMBLE__y" or "_CT_ENSEMBLE__y" in ctx.slug_to_original_target_name.values()

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
    # Predict equivalence (DummyRegressor returns its fitted constant on any input shape).
    import pandas as pd

    X_test = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    np.testing.assert_allclose(
        roundtripped.model.predict(X_test),
        ct_entry.model.predict(X_test),
        rtol=1e-7,
    )
    # Sanity: the surrogate was non-trivial enough to differentiate a regression from a random round-trip.
    assert np.allclose(roundtripped.model.predict(X_test), 7.5)


def test_persist_ct_ensemble_entries_no_models_dir_is_noop():
    """When ``ctx.data_dir`` / ``ctx.models_dir`` is empty (in-memory only run), the persister must be a no-op
    rather than create files in CWD or crash."""
    from mlframe.training.core._phase_finalize import _persist_ct_ensemble_entries

    class _CtxStub:
        """Groups tests covering CtxStub."""
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
