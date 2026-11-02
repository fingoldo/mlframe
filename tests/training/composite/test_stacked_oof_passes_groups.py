"""Pass-2 stacked discovery builds its OOF columns with the group ids pass 1 honours (tracker 2026-11-02 wave 15, LEAK15-6)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.training.composite.discovery import _stacked
from mlframe.training.composite.ensemble import feature_stacking


def test_oof_receives_the_train_slice_of_the_group_ids(monkeypatch):
    """Shuffled-row OOF let each fold model memorise a group; the ids must reach composite_oof_predictions, aligned to the train rows."""
    seen = {}

    def fake_oof(factory, X, y, **kw):
        """Record the groups argument and stop the pass."""
        seen["groups"] = kw.get("groups")
        raise RuntimeError("stop after recording")

    monkeypatch.setattr(feature_stacking, "composite_oof_predictions", fake_oof)
    n = 40
    df = pd.DataFrame({"a": np.arange(n, dtype=float), "y": np.arange(n, dtype=float)})
    spec = SimpleNamespace(name="s1")
    fake = SimpleNamespace(
        fit=lambda *a, **k: None, specs_=[spec], tiny_rerank_scores_={"s1": 0.1}, config=SimpleNamespace(random_state=0),
        _group_ids_for_rerank=np.repeat(np.arange(8), 5),
    )
    train_idx = np.arange(10, 30)
    try:
        _stacked.fit_stacked(fake, df, "y", ["a"], train_idx)
    except Exception:
        pass
    assert seen["groups"] is not None
    np.testing.assert_array_equal(seen["groups"], np.repeat(np.arange(8), 5)[train_idx])
