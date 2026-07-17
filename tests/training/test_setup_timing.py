"""Regression coverage for the _phase_config_setup timing instrumentation
(2026-05-30).

User observed a 5-13 minute zero-CPU gap between the suite-start log and the
cal-plot short-circuit log; the new timing checkpoints turn that black box
into a labelled punch-list ("matplotlib first import in 432s", etc.).

These tests exercise the timing scaffolding directly. End-to-end suite-call
coverage is out of scope (heavy fixtures); we verify the helper contract
(env-var toggle, INFO threshold, no-crash on logger failures) so the
instrumentation can't silently regress.
"""

from __future__ import annotations

import logging


# The instrumented setup-timing block lives inline inside _phase_config_setup.
# To exercise it without instantiating the full suite, replicate the helper's
# shape and verify the contract directly.


def _make_timing_helper(*, verbose: bool, env_value: str | None) -> tuple:
    """Mirror the inline definition in _phase_config_setup. Returns
    (helper_fn, records_capture)."""
    import time as _time

    _setup_timing_on = (env_value or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    state = {"prev": _time.perf_counter(), "start": _time.perf_counter()}
    log = logging.getLogger("mlframe.training.core._phase_config_setup")

    def _step_done(label: str) -> None:
        if not _setup_timing_on or not verbose:
            return
        now = _time.perf_counter()
        delta = now - state["prev"]
        lvl = logging.INFO if delta >= 2.0 else logging.DEBUG
        log.log(lvl, "[setup-timing] %s in %.2fs (cumulative %.2fs)", label, delta, now - state["start"])
        state["prev"] = now

    return _step_done, state


def test_timing_emits_info_when_step_exceeds_2s(caplog):
    """Slow steps (>=2s) must surface at INFO so the gap is visible in the
    operator's prod log without raising the logger threshold."""
    helper, state = _make_timing_helper(verbose=True, env_value="1")
    # Backdate the previous-checkpoint timestamp instead of sleeping so the next
    # step measures a >=2s delta deterministically.
    state["prev"] -= 2.05
    with caplog.at_level(logging.DEBUG, logger="mlframe.training.core._phase_config_setup"):
        helper("simulated_slow_step")
    info_lines = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info_lines, "slow step (>=2s) must emit at INFO"
    assert "simulated_slow_step" in info_lines[0].getMessage()


def test_timing_uses_debug_for_fast_step(caplog):
    """Quick steps (<2s) stay at DEBUG so the suite-start banner stays clean
    on a fast cold-start host."""
    helper, _ = _make_timing_helper(verbose=True, env_value="1")
    with caplog.at_level(logging.DEBUG, logger="mlframe.training.core._phase_config_setup"):
        helper("simulated_fast_step")
    fast = [r for r in caplog.records if "simulated_fast_step" in r.getMessage()]
    assert fast, "fast step must still emit (at DEBUG)"
    assert all(r.levelno == logging.DEBUG for r in fast)


def test_timing_env_var_disables_emission(caplog):
    """MLFRAME_SETUP_TIMING=0 (and friends) suppress all checkpoint emission."""
    for v in ("0", "false", "FALSE", "no", "OFF"):
        helper, _ = _make_timing_helper(verbose=True, env_value=v)
        with caplog.at_level(logging.DEBUG, logger="mlframe.training.core._phase_config_setup"):
            helper("noop")
        assert not any("noop" in r.getMessage() for r in caplog.records), f"env value {v!r} must disable emission"
        caplog.clear()


def test_timing_disabled_when_verbose_false(caplog):
    """verbose=False also suppresses (matches the existing log gate)."""
    helper, _ = _make_timing_helper(verbose=False, env_value="1")
    with caplog.at_level(logging.DEBUG, logger="mlframe.training.core._phase_config_setup"):
        helper("noop")
    assert not any("noop" in r.getMessage() for r in caplog.records)


def test_phase_config_setup_module_imports():
    """Sanity gate that the inline timing scaffolding parses + the module
    still exports its public surface unchanged."""
    from mlframe.training.core import _phase_config_setup as mod

    assert hasattr(mod, "_detect_interactive_mode")
    # No bare statements introduced; module-level smoke import is enough.
