"""Audit 2026-11-20 finding 12-F38: the round-2 hybrid bench's downstream AUC must not crash on an empty feature selection."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_BENCH = Path(__file__).resolve().parents[2] / "src/mlframe/feature_selection/_benchmarks/fs_hybrid/round2_hybrid_bench.py"


def _load_bench():
    """Import the bench module by path (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("round2_hybrid_bench", _BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_downstream_with_no_selected_columns_returns_nan_triple():
    """An empty ``cols`` gives NaN for all three models instead of letting lgb.fit raise."""
    mod = _load_bench()
    rng = np.random.default_rng(0)
    Xtr = pd.DataFrame(rng.standard_normal((60, 3)), columns=list("abc"))
    Xte = pd.DataFrame(rng.standard_normal((20, 3)), columns=list("abc"))
    ytr = (Xtr["a"] > 0).astype(int).to_numpy()
    yte = (Xte["a"] > 0).astype(int).to_numpy()
    out = mod.downstream(Xtr, Xte, ytr, yte, [])
    assert set(out) == {"lgbm", "logit", "knn"}
    assert all(np.isnan(v) for v in out.values())
