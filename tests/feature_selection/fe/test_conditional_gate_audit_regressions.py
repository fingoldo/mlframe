"""Regression tests for the conditional-gate family: target encoding in the prototype scans, numpy-int subsample, silent resident fallback."""
import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._conditional_gate_fe import best_existing_op_mi, cheap_conditional_gate_scan
from mlframe.feature_selection.filters._conditional_gate_fe_proto import scan_row_argmax
from mlframe.utils.log_throttle import reset_throttle_counts


def test_proto_scan_does_not_truncate_a_fractional_target():
    """A float target in [0, 1) must keep its classes: the int cast collapsed it to one class and the scan found nothing."""
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = 0.3 * np.argmax(X.to_numpy(), axis=1)
    hits = scan_row_argmax(X, y, ["a", "b", "c"])
    assert len(hits) == 1 and hits[0]["mi"] > 0.5


def test_gate_scan_honours_a_numpy_integer_subsample_n():
    """subsample_n given as a numpy integer subsamples exactly like the same python int instead of being ignored."""
    rng = np.random.default_rng(1)
    n = 3000
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = ((np.where(X.c > 0.2, X.a, X.b) + rng.normal(0, 0.5, n)) > 0).astype(int)
    full = cheap_conditional_gate_scan(X, y, subsample_n=0)
    as_int = cheap_conditional_gate_scan(X, y, subsample_n=900)
    as_np = cheap_conditional_gate_scan(X, y, subsample_n=np.int64(900))
    assert [(h.mode, h.cols, h.tau, h.feat_mi) for h in as_np] == [(h.mode, h.cols, h.tau, h.feat_mi) for h in as_int]
    assert [h.feat_mi for h in as_np] != [h.feat_mi for h in full]


def test_baseline_resident_failure_is_logged_not_silent(monkeypatch, caplog):
    """A fault in the resident baseline route falls back to the host and says so at WARNING."""
    import sys
    import types

    import mlframe.feature_selection.filters._resident_candidate_mi_ktc as ktc

    def _boom(*a, **k):
        """Stand-in resident route that faults."""
        raise RuntimeError("device fault")

    reset_throttle_counts()
    monkeypatch.setattr(ktc, "rescand_use_resident", lambda n, k: True)
    monkeypatch.setitem(sys.modules, "mlframe.feature_selection.filters._resident_candidate_mi", types.SimpleNamespace(best_existing_op_mi_resident=_boom))
    rng = np.random.default_rng(2)
    arrs = {c: rng.normal(size=500) for c in "abc"}
    yi = (arrs["a"] > 0).astype(np.int64)
    with caplog.at_level(logging.WARNING):
        val = best_existing_op_mi(arrs, ("a", "b", "c"), yi, 12)
    assert val >= 0.0
    assert any("gate baseline" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
