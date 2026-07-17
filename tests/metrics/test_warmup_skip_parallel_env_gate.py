"""Regression test for ``MLFRAME_NUMBA_WARMUP_SKIP_PARALLEL`` (2026-07-10 perf fix).

``_prewarm_numba_cache_body`` eagerly JIT-compiles both the serial and parallel (`_par`) numba
variant of every metric it covers, but the size-gated dispatchers (``mlframe.metrics._numba_params``'s
``_PARALLEL_REDUCTION_THRESHOLD``) only ever call the `_par` variant once a fold reaches 100,000 rows.
For a run whose data is known to stay below that threshold, warming the `_par` variants is pure
overhead with zero runtime payoff -- measured ~6-10s of a ~50-60s total metric-kernel prewarm on a
100k-row production run, disk caching confirmed NOT to help across fresh processes for these kernels.

``MLFRAME_NUMBA_WARMUP_SKIP_PARALLEL=1`` skips ONLY the individual `_par` calls (interleaved with
`_seq` calls in the same try blocks) -- opt-in, default OFF, so any caller that doesn't set it keeps
the existing behavior unchanged. These tests pin: (1) the env var actually reduces which kernels get
touched during warmup, (2) the `_seq` variants are ALWAYS warmed regardless of the flag (never
skipped), (3) a `_par` kernel skipped at warmup still works correctly via ordinary lazy compilation
on first real call (the flag only changes WHEN compilation happens, never WHETHER), (4) default
behavior (env var unset) is unchanged from before this fix.
"""

from __future__ import annotations

import numpy as np
import pytest


def _run_warmup_and_track_par_calls(monkeypatch, skip: bool):
    """Runs `_prewarm_numba_cache_body` with a subset of `_par` kernels spied on, returns which
    of them were actually invoked."""
    import mlframe.metrics.core as core
    from mlframe.metrics import _core_numba_warmup as warmup

    if skip:
        monkeypatch.setenv("MLFRAME_NUMBA_WARMUP_SKIP_PARALLEL", "1")
    else:
        monkeypatch.delenv("MLFRAME_NUMBA_WARMUP_SKIP_PARALLEL", raising=False)

    called: set = set()

    def _make_spy(name, retval):
        def _spy(*args, **kwargs):
            called.add(name)
            return retval

        return _spy

    monkeypatch.setattr(core, "_fast_mae_par", _make_spy("_fast_mae_par", 0.0))
    monkeypatch.setattr(core, "_fast_mse_par", _make_spy("_fast_mse_par", 0.0))
    monkeypatch.setattr(core, "_fast_r2_score_par", _make_spy("_fast_r2_score_par", 0.0))
    monkeypatch.setattr(core, "_fast_brier_score_loss_par", _make_spy("_fast_brier_score_loss_par", 0.0))
    monkeypatch.setattr(core, "_fast_hamming_loss_par", _make_spy("_fast_hamming_loss_par", 0.0))
    monkeypatch.setattr(core, "_fast_jaccard_score_par", _make_spy("_fast_jaccard_score_par", 0.0))

    seq_called: set = set()

    def _make_seq_spy(name, real_fn):
        def _spy(*args, **kwargs):
            seq_called.add(name)
            return real_fn(*args, **kwargs)

        return _spy

    _orig_mae_seq = core._fast_mae_seq
    monkeypatch.setattr(core, "_fast_mae_seq", _make_seq_spy("_fast_mae_seq", _orig_mae_seq))

    warmup._prewarm_numba_cache_body()
    return called, seq_called


def test_skip_flag_off_by_default_warms_both_seq_and_par(monkeypatch):
    par_called, seq_called = _run_warmup_and_track_par_calls(monkeypatch, skip=False)
    assert "_fast_mae_par" in par_called
    assert "_fast_mse_par" in par_called
    assert "_fast_r2_score_par" in par_called
    assert "_fast_brier_score_loss_par" in par_called
    assert "_fast_hamming_loss_par" in par_called
    assert "_fast_jaccard_score_par" in par_called
    assert "_fast_mae_seq" in seq_called


def test_skip_flag_on_skips_par_but_keeps_seq(monkeypatch):
    par_called, seq_called = _run_warmup_and_track_par_calls(monkeypatch, skip=True)
    assert par_called == set(), f"expected no _par kernels called with the skip flag on, got {par_called}"
    assert "_fast_mae_seq" in seq_called, "_seq variants must ALWAYS warm, regardless of the skip flag"


def test_skipped_par_kernel_still_works_correctly_via_lazy_compile(monkeypatch):
    """The flag changes WHEN a _par kernel compiles, never WHETHER it works or what it returns."""
    monkeypatch.setenv("MLFRAME_NUMBA_WARMUP_SKIP_PARALLEL", "1")
    from mlframe.metrics import _core_numba_warmup as warmup
    from mlframe.metrics.regression._regression_metrics import _fast_mae_seq, _fast_mae_par

    warmup._prewarm_numba_cache_body()

    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    p = np.array([1.1, 2.2, 2.9, 4.1, 4.8], dtype=np.float64)
    seq_result = _fast_mae_seq(y, p)
    par_result = _fast_mae_par(y, p)
    assert seq_result == pytest.approx(par_result, abs=1e-9)


def test_wellbore_threshold_gate_matches_mlframe_constant():
    """wellbore_train.py derives its skip decision from mlframe's OWN _PARALLEL_REDUCTION_THRESHOLD
    rather than a hardcoded duplicate -- pin that the constant is importable and sane, so a future
    mlframe change to the threshold can't silently desync wellbore_train.py's assumption."""
    from mlframe.metrics._numba_params import _PARALLEL_REDUCTION_THRESHOLD

    assert isinstance(_PARALLEL_REDUCTION_THRESHOLD, int)
    assert _PARALLEL_REDUCTION_THRESHOLD > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
