"""``mlframe-tune-kernels ensure`` and the fit-time kernel-tuning policy.

Fakes stand in for specs, the cache and the sweep, so nothing is measured; the point is which kernels are selected, what the exit codes say, and what a fit may or may not
leave behind in the environment.
"""

from __future__ import annotations

import logging
import os
import subprocess
from types import SimpleNamespace

import pytest

from mlframe.system import kernel_tuning_cache as cli
from mlframe.system.kernel_tuning_cache import _ensure, _policy

SWITCH = "PYUTILZ_KERNEL_DISABLE_SWEEP"


def _spec(name, gpu=False):
    """A stand-in TunerSpec with just what ensure reads."""
    return SimpleNamespace(kernel_name=name, gpu_capable=gpu, variant_fns=(), extra_fns=(), salt=0)


class _Cache:
    """A cache where `tuned` kernels are current and `stale` ones hold another code version."""

    def __init__(self, tuned=(), stale=()):
        self.tuned, self.stale = set(tuned), set(stale)

    def has(self, name):
        """Whether any tuning is stored for the kernel."""
        return name in self.tuned or name in self.stale

    def code_version_stale(self, name, code_version):
        """Whether the stored tuning was made for other source."""
        return name in self.stale

    def get_regions(self, name):
        """The stored regions of a tuned kernel (three per kernel here)."""
        return [{"backend_choice": "cpu"}] * 3 if name in self.tuned else []


@pytest.fixture
def fake_world(monkeypatch, tmp_path):
    """Specs a/b/c plus a GPU kernel g (b, c and g need work), fake code versions, a controllable cache and CUDA flag; records which kernels `tune_spec` was asked for."""
    specs = {"a": _spec("a"), "b": _spec("b"), "c": _spec("c"), "g": _spec("g", gpu=True)}
    cache = _Cache(tuned={"a"}, stale={"c"})
    tuned_now: list = []
    state = {"cuda": True, "fail": set(), "timeout": set()}
    limits: list = []
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))  # the tuning lock lives next to the cache: keep it out of the real one
    monkeypatch.setattr(_ensure, "KernelTuningCache", lambda: cache)
    monkeypatch.setattr(_ensure, "spec_code_version", lambda spec: "cv")
    monkeypatch.setattr(_ensure, "_cuda_present", lambda: state["cuda"])

    def _sweep(spec, timeout_s):
        """Stand-in for the per-kernel child process: records the call, optionally times out or fails, otherwise marks the kernel tuned."""
        limits.append((spec.kernel_name, timeout_s))
        if spec.kernel_name in state["timeout"]:
            return "timeout", "exceeded the limit"
        if spec.kernel_name in state["fail"]:
            return "failed", "the sweep process exited with code 1"
        tuned_now.append(spec.kernel_name)
        cache.tuned.add(spec.kernel_name)
        cache.stale.discard(spec.kernel_name)
        return "ok", ""

    monkeypatch.setattr(_ensure, "_run_sweep", _sweep)
    return SimpleNamespace(specs=specs, cache=cache, tuned=tuned_now, state=state, limits=limits)


def test_status_distinguishes_fresh_stale_and_missing():
    """The same test tune_spec(skip_existing=True) applies: tuned and unchanged is fresh, tuned for other source is stale, no entry is missing."""
    cache = _Cache(tuned={"a"}, stale={"c"})
    assert _ensure.spec_status(_spec("a"), cache, "cv") == "fresh"
    assert _ensure.spec_status(_spec("c"), cache, "cv") == "stale"
    assert _ensure.spec_status(_spec("b"), cache, "cv") == "missing"


def test_ensure_tunes_only_missing_and_stale_kernels(fake_world):
    """a is current and is left alone; b (missing), c (stale) and the GPU kernel g are tuned."""
    assert _ensure.cmd_ensure(fake_world.specs) == 0
    assert sorted(fake_world.tuned) == ["b", "c", "g"]


def test_ensure_is_a_no_op_when_everything_is_current(fake_world):
    """The cheap path deploy scripts rely on: nothing to do, nothing swept, exit 0."""
    fake_world.cache.tuned |= {"b", "c", "g"}
    fake_world.cache.stale.clear()
    assert _ensure.cmd_ensure(fake_world.specs) == 0
    assert fake_world.tuned == []


def test_check_reports_without_sweeping_and_fails_when_work_is_pending(fake_world, capsys):
    """--check never sweeps; exit 1 says something is out of date, --advisory turns that into exit 0."""
    assert _ensure.cmd_ensure(fake_world.specs, check=True) == 1
    assert fake_world.tuned == []
    out = capsys.readouterr().out
    assert "missing b" in out and "stale   c" in out and "need tuning" in out
    assert _ensure.cmd_ensure(fake_world.specs, check=True, advisory=True) == 0


def test_if_cuda_exits_quietly_on_a_host_without_cuda(fake_world, capsys):
    """Hooks run on machines without a GPU: --if-cuda must do and print nothing there."""
    fake_world.state["cuda"] = False
    assert _ensure.cmd_ensure(fake_world.specs, if_cuda=True) == 0
    assert fake_world.tuned == [] and capsys.readouterr().out == ""


def test_a_gpu_kernel_is_never_counted_or_tuned_without_cuda(fake_world):
    """Without CUDA only the CPU kernels are in play, so a missing GPU tuning is not work to do."""
    fake_world.state["cuda"] = False
    assert _ensure.cmd_ensure(fake_world.specs) == 0
    assert sorted(fake_world.tuned) == ["b", "c"]
    assert "g" not in _ensure.survey(fake_world.specs, cache=fake_world.cache, cuda=False)


def test_only_restricts_the_selection(fake_world):
    """--only gpu tunes just the GPU kernel; --only cpu just the CPU ones."""
    _ensure.cmd_ensure(fake_world.specs, only="gpu")
    assert fake_world.tuned == ["g"]
    fake_world.tuned.clear()
    fake_world.cache.tuned.discard("g")
    _ensure.cmd_ensure(fake_world.specs, only="cpu")
    assert sorted(fake_world.tuned) == ["b", "c"]


def test_a_failing_sweep_does_not_stop_the_others_and_is_reported(fake_world, capsys):
    """One kernel's sweep raising must not block the rest; the exit code is 2 and the failure is named."""
    fake_world.state["fail"] = {"b"}
    assert _ensure.cmd_ensure(fake_world.specs) == _ensure.EXIT_SWEEP_FAILED
    assert sorted(fake_world.tuned) == ["c", "g"]
    assert "b" in capsys.readouterr().err


def test_the_time_budget_stops_new_sweeps_and_reports_the_rest(fake_world, capsys, monkeypatch):
    """With the budget already spent nothing new starts; the untuned kernels are listed and the exit code says the budget ran out (not that a sweep failed)."""
    clock = iter([0.0] + [10_000.0] * 20)
    monkeypatch.setattr(_ensure.time, "monotonic", lambda: next(clock))
    assert _ensure.cmd_ensure(fake_world.specs, max_minutes=1) == _ensure.EXIT_BUDGET_SPENT
    assert fake_world.tuned == []
    assert "not tuned" in capsys.readouterr().err


def test_ensure_is_reachable_from_the_command_line(monkeypatch, fake_world):
    """The subcommand is registered and its flags reach cmd_ensure."""
    monkeypatch.setattr(cli, "discover_specs", lambda package="mlframe": fake_world.specs)
    assert cli.main(["ensure", "--check", "--advisory"]) == 0
    assert cli.main(["ensure", "--check"]) == 1


# --- fit-time policy -------------------------------------------------------------------------------------------------


@pytest.fixture
def clean_policy(monkeypatch):
    """No leftover switch, no AUTOTUNE setting, a quiet cold-cache probe."""
    monkeypatch.delenv(SWITCH, raising=False)
    monkeypatch.delenv("MLFRAME_AUTOTUNE", raising=False)
    monkeypatch.setattr(_policy, "cold_kernel_names", lambda: [])
    monkeypatch.setattr(_policy, "_DEPTH", 0)
    monkeypatch.setattr(_policy, "_WE_SET_SWITCH", False)
    monkeypatch.setattr(_policy, "_WARNED", False)


@pytest.mark.usefixtures("clean_policy")
def test_a_fit_runs_with_background_sweeps_disabled_and_the_environment_is_restored():
    """Inside the policy the pyutilz sweep switch is on; afterwards it is gone again, also when the fit raises."""
    assert SWITCH not in os.environ
    with _policy.kernel_tuning_fit_policy():
        assert os.environ[SWITCH] == "1"
    assert SWITCH not in os.environ
    with pytest.raises(ValueError):
        with _policy.kernel_tuning_fit_policy():
            raise ValueError("fit failed")
    assert SWITCH not in os.environ


@pytest.mark.usefixtures("clean_policy")
def test_the_callers_own_setting_is_never_overridden_or_removed(monkeypatch):
    """An explicit PYUTILZ_KERNEL_DISABLE_SWEEP (any value) belongs to the caller: kept inside, kept after."""
    monkeypatch.setenv(SWITCH, "0")
    with _policy.kernel_tuning_fit_policy():
        assert os.environ[SWITCH] == "0"
    assert os.environ[SWITCH] == "0"


@pytest.mark.usefixtures("clean_policy")
def test_autotune_on_restores_the_old_behaviour(monkeypatch):
    """MLFRAME_AUTOTUNE=on leaves the sweep switch alone, so the dispatchers may start their background sweeps."""
    monkeypatch.setenv("MLFRAME_AUTOTUNE", "on")
    assert _policy.autotune_mode() == "on"
    with _policy.kernel_tuning_fit_policy():
        assert SWITCH not in os.environ


@pytest.mark.usefixtures("clean_policy")
def test_overlapping_fits_keep_the_switch_until_the_last_one_leaves():
    """Two fits in flight: the first to finish must not re-enable sweeps under the second."""
    outer = _policy.kernel_tuning_fit_policy()
    inner = _policy.kernel_tuning_fit_policy()
    outer.__enter__()
    inner.__enter__()
    outer.__exit__(None, None, None)
    assert os.environ[SWITCH] == "1"
    inner.__exit__(None, None, None)
    assert SWITCH not in os.environ


@pytest.mark.usefixtures("clean_policy")
def test_a_cold_cache_is_announced_once_with_the_command_to_run(monkeypatch, caplog):
    """The warning names the count, an example kernel and the command to run, and appears once per process."""
    monkeypatch.setattr(_policy, "cold_kernel_names", lambda: ["k1", "k2", "k3", "k4"])
    with caplog.at_level(logging.WARNING, logger=_policy.logger.name):
        for _ in range(3):
            with _policy.kernel_tuning_fit_policy():
                pass
    msgs = [r.getMessage() for r in caplog.records if "kernel tuning cache is cold" in r.getMessage()]
    assert len(msgs) == 1, msgs
    assert "4 kernel(s)" in msgs[0] and "mlframe-tune-kernels ensure" in msgs[0] and "k1" in msgs[0]


@pytest.mark.usefixtures("clean_policy")
def test_a_failing_cold_cache_probe_never_breaks_a_fit(monkeypatch):
    """The notice is advice: if the probe itself raises, the fit still runs."""

    def _boom():
        """Stand-in for a registry that cannot be read."""
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(_policy, "cold_kernel_names", _boom)
    ran = []
    with _policy.kernel_tuning_fit_policy():
        ran.append(True)
    assert ran == [True]


@pytest.mark.usefixtures("clean_policy")
def test_hygienic_fit_wraps_the_fit_in_the_policy():
    """Every selector fit that goes through hygienic_fit runs with sweeps disabled and leaves the environment as it found it."""
    from mlframe.utils.misc import hygienic_fit

    seen = {}

    class _Sel:
        """A selector whose fit only reports the environment it ran in."""

        @hygienic_fit
        def fit(self, X):
            """Record the sweep switch seen during the fit."""
            seen["switch"] = os.environ.get(SWITCH)
            return self

    _Sel().fit([[1.0]])
    assert seen["switch"] == "1"
    assert SWITCH not in os.environ


# --- one tuning run per machine ---------------------------------------------------------------------------------------


def test_a_second_ensure_does_not_start_while_one_is_running(fake_world, capsys, monkeypatch):
    """The lock is held by a live process: ensure reports it and tunes nothing (sweeps that overlap measure each other), exit 0."""
    lock = _ensure._lock_path()
    lock.write_text(str(os.getpid() + 1), encoding="utf-8")
    monkeypatch.setattr(_ensure, "_pid_alive", lambda pid: True)
    assert _ensure.cmd_ensure(fake_world.specs) == 0
    assert fake_world.tuned == []
    assert "already in progress" in capsys.readouterr().out
    assert lock.exists(), "a run that did not start must not remove the other run's lock"


def test_a_lock_left_by_a_dead_process_is_taken_over(fake_world, monkeypatch):
    """A crashed run must not block tuning forever: its lock is replaced and the work is done."""
    _ensure._lock_path().write_text("999999", encoding="utf-8")
    monkeypatch.setattr(_ensure, "_pid_alive", lambda pid: False)
    assert _ensure.cmd_ensure(fake_world.specs) == 0
    assert sorted(fake_world.tuned) == ["b", "c", "g"]
    assert not _ensure._lock_path().exists(), "the lock is released when the run ends"


def test_the_lock_is_released_even_when_a_sweep_raises(fake_world):
    """A failing sweep must not leave the machine locked."""
    fake_world.state["fail"] = {"b"}
    _ensure.cmd_ensure(fake_world.specs)
    assert not _ensure._lock_path().exists()


# --- one child process per kernel -------------------------------------------------------------------------------------


def test_a_kernel_that_hits_its_time_limit_is_left_untuned_and_the_rest_still_run(fake_world, capsys):
    """A hung sweep is stopped, named, and does not stop the other kernels; the exit code says time ran out, not that something broke."""
    fake_world.state["timeout"] = {"b"}
    assert _ensure.cmd_ensure(fake_world.specs, per_kernel_minutes=2) == _ensure.EXIT_BUDGET_SPENT
    assert sorted(fake_world.tuned) == ["c", "g"]
    err = capsys.readouterr().err
    assert "STOPPED" in err and "b" in err
    assert dict(fake_world.limits)["b"] == 120.0


def test_a_failure_outranks_a_timeout_in_the_exit_code(fake_world):
    """If one sweep failed and another timed out, the failure (which blocks a push) is what the exit code reports."""
    fake_world.state["fail"] = {"b"}
    fake_world.state["timeout"] = {"c"}
    assert _ensure.cmd_ensure(fake_world.specs) == _ensure.EXIT_SWEEP_FAILED


def test_no_single_sweep_may_run_past_the_overall_budget(fake_world, monkeypatch):
    """The per-kernel limit is capped by what is left of --max-minutes."""
    clock = iter([0.0, 0.0, 30.0, 30.0, 30.0, 60.0, 60.0, 60.0, 90.0, 90.0, 90.0, 90.0])
    monkeypatch.setattr(_ensure.time, "monotonic", lambda: next(clock, 90.0))
    _ensure.cmd_ensure(fake_world.specs, max_minutes=2, per_kernel_minutes=15)
    assert fake_world.limits and all(limit <= 120.0 for _name, limit in fake_world.limits)


def test_run_sweep_maps_the_child_process_outcome(monkeypatch):
    """The real helper: exit 0 is ok, a non-zero exit is failed, subprocess.TimeoutExpired is timeout; the command targets the kernel by name."""
    seen = {}

    def fake_run(cmd, timeout, check):
        """Return a canned completed process for the given kernel."""
        seen["cmd"], seen["timeout"] = cmd, timeout
        return SimpleNamespace(returncode=seen.get("rc", 0))

    monkeypatch.setattr(_ensure.subprocess, "run", fake_run)
    assert _ensure._run_sweep(_spec("k"), 60.0) == ("ok", "")
    assert seen["cmd"][-2:] == ["refresh", "k"] and seen["timeout"] == 60.0
    seen["rc"] = 1
    outcome, detail = _ensure._run_sweep(_spec("k"), 60.0)
    assert outcome == "failed" and "code 1" in detail

    def hung(cmd, timeout, check):
        """A sweep that never finishes."""
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(_ensure.subprocess, "run", hung)
    outcome, detail = _ensure._run_sweep(_spec("k"), 900.0)
    assert outcome == "timeout" and "15 min" in detail
