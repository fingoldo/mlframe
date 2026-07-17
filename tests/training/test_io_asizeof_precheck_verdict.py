"""Pins the iter144 SAVE/LOAD perf verdict for the pre-pickle ``pympler.asizeof`` size precheck.

The precheck flips ``lean=True`` upfront when a SimpleNamespace bundle's in-memory size exceeds a threshold,
to skip the fat-then-lean double dump. iter144 re-profiled this lead and REJECTED any optimization: asizeof is
already ~3% of the save wall. The real risk surfaced by the bench is the opposite -- asizeof grossly
under-estimates models whose bulk lives in Cython / numpy buffers it cannot introspect (a fitted RandomForest
reports ~0.4 MB while its pickle is ~155 MB). This test pins that property so nobody "improves" the precheck by
treating asizeof's estimate as a real byte count (which would silently break the eager/lean gate decision).
"""

import numpy as np
import pytest
from types import SimpleNamespace


def test_asizeof_underestimates_buffer_backed_model_vs_pickle():
    """pympler.asizeof badly underestimates a sklearn model's true pickled size (blind to Cython tree buffers) -- the precheck must not trust it as a byte count."""
    pa = pytest.importorskip("pympler.asizeof")
    import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    X = rng.standard_normal((5_000, 20))
    y = (rng.standard_normal(5_000) > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=0).fit(X, y)
    bundle = SimpleNamespace(model=rf, meta={"k": "v"})

    est = pa.asizeof(bundle)
    serialized = len(pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL))

    # The whole point: asizeof cannot see the Cython tree buffers, so its estimate is a small fraction of the
    # true serialized size. If this ever flips (asizeof learns to walk the buffers), the precheck threshold
    # semantics must be re-derived -- this test failing is the signal to do that, not to "fix" the test.
    assert est < serialized * 0.5, f"asizeof est {est} vs serialized {serialized}: precheck must not be treated as a real byte count"


def test_asizeof_shallow_ndarray_bundle_is_accurate_and_cheap():
    """For a plain ndarray-backed bundle (no Cython buffers), asizeof's estimate stays close to the raw nbytes."""
    pa = pytest.importorskip("pympler.asizeof")
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(1_000_000).astype(np.float32)  # 4 MB raw
    bundle = SimpleNamespace(preds=arr, meta={"a": 1})
    est = pa.asizeof(bundle)
    # For a plain ndarray the estimate IS close to the raw nbytes -- this is the case the precheck is calibrated
    # for (SimpleNamespace per-split arrays), which is why the lean flip is gated to SimpleNamespace payloads.
    assert est >= arr.nbytes
    assert est < arr.nbytes * 1.2
