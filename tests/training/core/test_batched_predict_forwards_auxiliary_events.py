"""`predict_batch_rows` must not change what the predict call sees.

Both batched recursions dropped `auxiliary_events_df`, so a caller passing it together with `predict_batch_rows`
silently lost the latent_interaction_svd / nearest_past_join replay for every batch, while the un-batched call on the
same data got the real embeddings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mlframe.training.core._predict_main_from_models as pmm
import mlframe.training.core._predict_main_suite as pms


def _record_kwargs(monkeypatch, module, fn_name):
    seen = []
    real = getattr(module, fn_name)

    def spy(df, *a, **kw):
        if kw.get("predict_batch_rows") is None:  # the per-batch recursion, not the entry call
            seen.append(kw.get("auxiliary_events_df"))
            return {"predictions": {"m": np.zeros(len(df))}, "probabilities": {}}
        return real(df, *a, **kw)

    monkeypatch.setattr(module, fn_name, spy)
    return spy, seen


def test_in_memory_batches_receive_the_events_table(monkeypatch):
    spy, seen = _record_kwargs(monkeypatch, pmm, "predict_from_models")
    events = pd.DataFrame({"entity": [1, 2], "t": [0.0, 1.0]})
    spy(pd.DataFrame({"x": np.arange(25.0)}), {}, {}, predict_batch_rows=10, auxiliary_events_df=events, verbose=0)
    assert len(seen) == 3 and all(e is events for e in seen)


def test_disk_batches_receive_the_events_table(monkeypatch, tmp_path):
    # The bundle is loaded once, before the batches are dispatched, so a run over an empty directory needs it stubbed.
    monkeypatch.setattr(pms, "_load_suite_metadata", lambda *a, **kw: {"pipeline": None})
    spy, seen = _record_kwargs(monkeypatch, pms, "predict_mlframe_models_suite")
    events = pd.DataFrame({"entity": [1], "t": [0.0]})
    spy(pd.DataFrame({"x": np.arange(25.0)}), str(tmp_path), predict_batch_rows=10, auxiliary_events_df=events, verbose=0)
    assert len(seen) == 3 and all(e is events for e in seen)
