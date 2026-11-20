"""Regression tests for the CMI-greedy step helpers: the noise floor must follow the live conditioning support, the resident fingerprint must not collide, and a failed gate probe must be logged."""
from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _mi_greedy_cmi_fe_steps as steps
from mlframe.utils.log_throttle import reset_throttle_counts


def test_noise_floor_is_conditioned_on_the_grown_support(monkeypatch):
    """After a winner folds into Z the recomputed floor must be scored against that Z, not the empty support it was created with."""
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe as top

    monkeypatch.setattr(steps, "_cmi_gpu_enabled", lambda **_k: False)
    monkeypatch.setattr(top, "_cmi_gpu_enabled", lambda **_k: False, raising=False)
    monkeypatch.setattr("mlframe.feature_selection.filters._gpu_strict_fe.fe_gpu_strict_resident_enabled", lambda *a, **k: False)
    calls: list[tuple[str, str]] = []
    real_yz, real_marg = steps.precompute_cmi_yz_terms, steps.precompute_marginal_y_terms

    def spy_yz(*a, **k):
        """Record the caller of the conditional y/z precompute."""
        calls.append(("yz", sys._getframe(1).f_code.co_name))
        return real_yz(*a, **k)

    def spy_marg(*a, **k):
        """Record the caller of the marginal y precompute."""
        calls.append(("marg", sys._getframe(1).f_code.co_name))
        return real_marg(*a, **k)

    monkeypatch.setattr(steps, "precompute_cmi_yz_terms", spy_yz)
    monkeypatch.setattr(steps, "precompute_marginal_y_terms", spy_marg)
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n), "c": rng.standard_normal(n)})
    y = ((X["a"] + X["b"]) > 0).astype(np.int64).to_numpy()
    top.greedy_cmi_fe_construct_with_recipes(X, y, top_k=3, min_cmi_gain=0.0, include_binary=False)
    floor_calls = [kind for kind, who in calls if who == "_noise_floor_for_current_z"]
    assert floor_calls[0] == "marg"
    assert "yz" in floor_calls, "the floor recomputed after Z grew was still scored against the empty support"


def test_fingerprint_weights_do_not_cancel_small_code_changes():
    """Two code vectors differing by +2 at one row and -1 at another row must not share a weighted-sum fingerprint (they did with arithmetic-progression weights)."""
    n = 1000
    w = steps._fingerprint_weights(np, n)
    rng = np.random.default_rng(1)
    a = rng.integers(0, 10, n).astype(np.int64)
    b = a.copy()
    a[10], a[11] = 3, 3
    b[10], b[11] = 5, 2
    with np.errstate(over="ignore"):
        fa = int(np.sum((a + 1) * w, dtype=np.int64))
        fb = int(np.sum((b + 1) * w, dtype=np.int64))
    assert fa != fb


def test_failed_strict_gate_probe_is_logged_at_warning(monkeypatch, caplog):
    """A failing strict-GPU probe sends the CMI to the host path and says so at WARNING."""
    import mlframe.feature_selection.filters._fe_gpu_strict as strict

    def boom(**_k):
        """Raise like a broken probe."""
        raise RuntimeError("probe broke")

    reset_throttle_counts()
    monkeypatch.delenv("MLFRAME_CMI_GPU", raising=False)
    monkeypatch.setattr(strict, "fe_gpu_strict_enabled", boom)
    with caplog.at_level(logging.WARNING, logger=steps.logger.name):
        assert steps._cmi_gpu_enabled(n=10, p=1) is False
    assert any("probe failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
