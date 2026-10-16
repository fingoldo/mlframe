"""External validation of FE pairs scored in RAM-budget chunks gives the same per-pair validation MI as one batch and as the per-candidate loop."""

import numpy as np
import pandas as pd
import pytest

import mlframe.feature_selection.filters._feature_engineering_pairs._pairs_emit as emit


def _frame(n=3000, seed=0):
    """Frame."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(8)})
    y = ((X["f0"] * X["f1"] + X["f2"] / (1 + X["f3"].abs()) + 0.3 * rng.standard_normal(n)) > 0).astype(np.int64)
    return X, y


def _run(monkeypatch, budget_bytes):
    """Fit once with the external-validation RAM budget capped at ``budget_bytes`` (None = real budget); returns every pair's validation MI."""
    seen = {}
    real = emit._emit_pair_features_step2_read_here_once

    def _wrapped(*args, **kwargs):
        """Wrapped."""
        names = real.__code__.co_varnames[: real.__code__.co_argcount]
        bound = dict(zip(names, args))
        bound.update(kwargs)
        if budget_bytes is not None:
            hoist = bound["_can_hoist_shared_buffer"]
            bound["_can_hoist_shared_buffer"] = lambda nbytes, n_workers=1: (False, 0, 0) if nbytes > budget_bytes else hoist(nbytes, n_workers=n_workers)
        out = real(**bound)
        seen.update({str(k): v for k, v in bound["valid_pairs_perf"].items()})
        return out

    monkeypatch.setattr(emit, "_emit_pair_features_step2_read_here_once", _wrapped)
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _frame()
    sel = MRMR(verbose=0, random_state=0, n_jobs=1, fit_cache_max=0).fit(X, y)  # no replay of the identical earlier fit
    monkeypatch.setattr(emit, "_emit_pair_features_step2_read_here_once", real)
    return seen, list(sel.get_feature_names_out())


@pytest.mark.parametrize("budget_cols", [1, 3])
def test_chunked_extval_matches_one_batch_and_loop(monkeypatch, budget_cols):
    """Chunked extval matches one batch and loop."""
    full, names_full = _run(monkeypatch, None)
    assert full, "the frame must reach FE external validation"
    n_bin = 8  # upper bound on binary ops per factor: forces several chunks without excluding a chunk of one factor
    chunked, names_chunked = _run(monkeypatch, 3000 * budget_cols * n_bin * 8)
    loop, names_loop = _run(monkeypatch, 0)
    assert chunked.keys() == full.keys() == loop.keys()
    for k in full:
        assert chunked[k] == full[k], k
        assert loop[k] == pytest.approx(full[k], abs=1e-12), k
    assert names_chunked == names_full == names_loop
