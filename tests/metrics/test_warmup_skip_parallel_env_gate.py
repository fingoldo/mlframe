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
        """Helper: Make spy."""
        def _spy(*args, **kwargs):
            """Helper: Spy."""
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
        """Helper: Make seq spy."""
        def _spy(*args, **kwargs):
            """Helper: Spy."""
            seq_called.add(name)
            return real_fn(*args, **kwargs)

        return _spy

    _orig_mae_seq = core._fast_mae_seq
    monkeypatch.setattr(core, "_fast_mae_seq", _make_seq_spy("_fast_mae_seq", _orig_mae_seq))

    # Guarded wrapper, not the raw _prewarm_numba_cache_body() (2026-08-21): the body's own
    # "Warm dummy_baselines kernels" block calls back into prewarm_numba_cache() (mutual
    # forward/reverse recursion, see that function's docstring) -- calling the unguarded body
    # directly means the re-entrancy guard's flag was never set, so that reentry re-runs the
    # WHOLE body a second time, confirmed on CI via _core_numba_warmup.py's own diagnostic.
    warmup.prewarm_numba_cache()
    return called, seq_called


_PAR_KERNELS = (
    "_fast_mae_par",
    "_fast_mse_par",
    "_fast_r2_score_par",
    "_fast_brier_score_loss_par",
    "_fast_hamming_loss_par",
    "_fast_jaccard_score_par",
)

# The two tests below used to be `@pytest.mark.skip`, so the env var's core contract -- which kernels actually
# get warmed -- had NO coverage at all: a refactor that stopped honouring the flag, or honoured it for the `_seq`
# kernels too, would have passed CI unchanged. The recorded blocker was real (monkeypatching an njit dispatcher
# is not observed by the warmup body, and `nopython_signatures` cannot distinguish "skipped here" from "already
# warm from another test in this worker"), but a FRESH SUBPROCESS has no prior compilation state at all, which is
# exactly the reset mechanism the skip reason said was missing.
_PROBE = """
import json
import mlframe.metrics._core_numba_warmup as warmup
import mlframe.metrics.core as core

warmup.prewarm_numba_cache()
names = {names!r}
print("RESULT " + json.dumps({{n: len(getattr(getattr(core, n, None), "nopython_signatures", [])) for n in names}}))
"""


def _warm_in_subprocess(skip: str) -> dict:
    """Run the warmup in a clean interpreter with the flag set to ``skip``; return per-kernel signature counts."""
    import json
    import os
    import subprocess
    import sys

    names = [*_PAR_KERNELS, "_fast_mae_seq"]
    env = dict(os.environ, MLFRAME_NUMBA_WARMUP_SKIP_PARALLEL=skip)
    proc = subprocess.run([sys.executable, "-c", _PROBE.format(names=names)], capture_output=True, text=True, env=env, timeout=900, check=False)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line is not None, f"probe produced no result (exit {proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    return json.loads(line[len("RESULT ") :])


@pytest.mark.slow
def test_skip_flag_off_by_default_warms_both_seq_and_par():
    """Every `_par` kernel must carry a compiled signature after a default warmup. Measured: all six do."""
    counts = _warm_in_subprocess("0")
    for name in _PAR_KERNELS:
        assert counts[name] >= 1, f"{name} was not warmed with the flag off: {counts}"
    assert counts["_fast_mae_seq"] >= 1, counts


@pytest.mark.slow
def test_skip_flag_on_skips_par_but_keeps_seq():
    """With the flag on, measured: all six `_par` kernels have zero signatures and `_fast_mae_seq` still has one."""
    counts = _warm_in_subprocess("1")
    warmed = [n for n in _PAR_KERNELS if counts[n]]
    assert warmed == [], f"expected no _par kernels warmed with the skip flag on, got {warmed}: {counts}"
    assert counts["_fast_mae_seq"] >= 1, "_seq variants must ALWAYS warm, regardless of the skip flag"


@pytest.mark.slow
def test_the_flag_is_what_makes_the_difference():
    """Both runs in one assertion, so neither can pass by the kernels simply never warming on this host."""
    off, on = _warm_in_subprocess("0"), _warm_in_subprocess("1")
    assert all(off[n] >= 1 and on[n] == 0 for n in _PAR_KERNELS), (off, on)


def test_skipped_par_kernel_still_works_correctly_via_lazy_compile(monkeypatch):
    """The flag changes WHEN a _par kernel compiles, never WHETHER it works or what it returns."""
    monkeypatch.setenv("MLFRAME_NUMBA_WARMUP_SKIP_PARALLEL", "1")
    from mlframe.metrics import _core_numba_warmup as warmup
    from mlframe.metrics.regression._regression_metrics import _fast_mae_seq, _fast_mae_par

    # Guarded wrapper, not the raw body function -- see _run_warmup_and_track_par_calls's identical note above.
    warmup.prewarm_numba_cache()

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
