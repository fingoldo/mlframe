"""``mlframe._patch_colorama_reinit_storm`` makes colorama.init() idempotent per call signature,
without changing the FIRST call's real (colorized) behavior, and rebinds numba.core.errors' own
independent name -- not just colorama.initialise's -- since that is the actual call site
NumbaWarning construction goes through.
"""

from __future__ import annotations

import pytest

import mlframe

pytest.importorskip("colorama")
pytest.importorskip("numba")


def _fresh_patch_state(monkeypatch):
    """Undo a prior patch application (idempotent across test runs in one process) by restoring
    colorama.initialise.init / numba.core.errors.init to an unpatched stand-in, so each test exercises
    a genuine first application of the patch."""
    import colorama.initialise as ci
    import numba.core.errors as ne

    real_init = getattr(ci.init, "__wrapped__", ci.init)
    monkeypatch.setattr(ci, "init", real_init, raising=True)
    monkeypatch.setattr(ne, "init", real_init, raising=True)
    return real_init


def test_second_call_with_same_signature_skips_real_init(monkeypatch):
    """After patching, a second colorama.initialise.init() call with the SAME (autoreset, convert,
    strip, wrap) signature must not re-invoke the real (expensive) colorama logic."""
    real_init = _fresh_patch_state(monkeypatch)
    calls = {"n": 0}

    def _counting(*a, **k):
        """Records a call and delegates to the real (pre-patch) init()."""
        calls["n"] += 1
        return real_init(*a, **k)

    import colorama.initialise as ci

    monkeypatch.setattr(ci, "init", _counting, raising=True)
    monkeypatch.delenv("MLFRAME_KEEP_COLORAMA_REINIT", raising=False)
    mlframe._patch_colorama_reinit_storm()

    ci.init()
    ci.init()
    ci.init()
    assert calls["n"] == 1, f"expected the real init to run exactly once, ran {calls['n']} times"


def test_numba_errors_init_is_rebound_not_just_colorama_initialise(monkeypatch):
    """The fix must rebind numba.core.errors.init directly -- rebinding only
    colorama.initialise.init leaves NumbaWarning construction going through the ORIGINAL function
    (numba imported it by name at module load time, an independent reference)."""
    _fresh_patch_state(monkeypatch)
    monkeypatch.delenv("MLFRAME_KEEP_COLORAMA_REINIT", raising=False)
    mlframe._patch_colorama_reinit_storm()

    import colorama.initialise as ci
    import numba.core.errors as ne

    assert getattr(ne.init, "_mlframe_idempotent", False), "numba.core.errors.init was not rebound to the idempotent wrapper"
    assert ne.init is ci.init, "numba.core.errors.init and colorama.initialise.init should be the SAME patched callable"


def test_warning_message_still_correctly_formatted_after_patch(monkeypatch):
    """The FIRST call's real colorama behavior is unaffected: NumbaWarning messages still get their
    highlight escape codes, not silently dropped by the patch."""
    _fresh_patch_state(monkeypatch)
    monkeypatch.delenv("MLFRAME_KEEP_COLORAMA_REINIT", raising=False)
    mlframe._patch_colorama_reinit_storm()

    from numba.core.errors import NumbaPerformanceWarning

    w = NumbaPerformanceWarning("distinctive test payload xyz123")
    assert "distinctive test payload xyz123" in str(w)


def test_different_signatures_each_run_the_real_init_once(monkeypatch):
    """Two DIFFERENT (autoreset, convert, strip, wrap) signatures are each real inits (not
    conflated into a single cached call) -- only a REPEATED signature should be skipped."""
    real_init = _fresh_patch_state(monkeypatch)
    calls = {"n": 0}

    def _counting(*a, **k):
        """Records a call and delegates to the real (pre-patch) init()."""
        calls["n"] += 1
        return real_init(*a, **k)

    import colorama.initialise as ci

    monkeypatch.setattr(ci, "init", _counting, raising=True)
    monkeypatch.delenv("MLFRAME_KEEP_COLORAMA_REINIT", raising=False)
    mlframe._patch_colorama_reinit_storm()

    ci.init(autoreset=False)
    ci.init(autoreset=True)
    ci.init(autoreset=False)  # repeat of the first signature -> skipped
    assert calls["n"] == 2, f"expected 2 real inits (one per distinct signature), got {calls['n']}"


def test_opt_out_env_var_leaves_colorama_untouched(monkeypatch):
    """MLFRAME_KEEP_COLORAMA_REINIT=1 must skip the patch entirely."""
    _fresh_patch_state(monkeypatch)
    import colorama.initialise as ci

    before = ci.init
    monkeypatch.setenv("MLFRAME_KEEP_COLORAMA_REINIT", "1")
    mlframe._patch_colorama_reinit_storm()
    assert ci.init is before, "opt-out env var should leave colorama.initialise.init unpatched"


def test_second_patch_application_is_a_no_op(monkeypatch):
    """Calling _patch_colorama_reinit_storm() twice must not double-wrap (idempotent at the
    patch-application level, not just the wrapped call level)."""
    _fresh_patch_state(monkeypatch)
    monkeypatch.delenv("MLFRAME_KEEP_COLORAMA_REINIT", raising=False)
    mlframe._patch_colorama_reinit_storm()
    import colorama.initialise as ci

    once_wrapped = ci.init
    mlframe._patch_colorama_reinit_storm()
    assert ci.init is once_wrapped, "a second patch application should not re-wrap an already-wrapped init"
