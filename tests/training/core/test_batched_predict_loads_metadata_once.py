"""`predict_batch_rows` must not re-read and re-unpickle the bundle metadata for every batch."""

from __future__ import annotations

import numpy as np
import pandas as pd

import mlframe.training.core._predict_main_suite as pms


def test_the_bundle_is_loaded_once_and_shared_by_every_batch(monkeypatch, tmp_path):
    loads = []
    monkeypatch.setattr(pms, "_load_suite_metadata", lambda *a, **kw: loads.append(a) or {"pipeline": None})

    seen = []
    real = pms.predict_mlframe_models_suite

    def spy(df, *a, **kw):
        if kw.get("predict_batch_rows") is None:  # the per-batch recursion, not the entry call
            seen.append(kw.get("_preloaded_metadata"))
            return {"predictions": {"m": np.zeros(len(df))}, "probabilities": {}}
        return real(df, *a, **kw)

    monkeypatch.setattr(pms, "predict_mlframe_models_suite", spy)
    spy(pd.DataFrame({"x": np.arange(25.0)}), str(tmp_path), predict_batch_rows=10, verbose=0)

    assert len(seen) == 3, "three batches expected"
    assert len(loads) == 1, f"the metadata must be unpickled once for the whole call, not per batch: {len(loads)} loads"
    assert all(m is seen[0] and m is not None for m in seen), "every batch must be handed the one loaded bundle"
