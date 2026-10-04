"""A GPU-less run (opt-out or a failed probe) must not persist, or be keyed by, a ``no-gpu`` hardware fingerprint that later GPU runs would resolve."""
from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("pyutilz.performance.kernel_tuning.cache")

_GPU_FP = "cpu_testcpu_gpu_testgpu_cc8.9"
_NO_GPU_FP = "cpu_testcpu_no-gpu"


@pytest.fixture
def fp_env(tmp_path, monkeypatch):
    """Isolated fingerprint cache directory with a fake CPU slug, GPU summary probe and GPU-usability predicate."""
    import mlframe.feature_selection.filters._gpu_policy as policy
    import mlframe.feature_selection.filters._kernel_tuning  # noqa: F401  (installs the guard)
    from pyutilz.performance.kernel_tuning import cache as kc
    from pyutilz.performance.kernel_tuning.cache import cache_base as cb

    for var in ("PYUTILZ_HW_FP_REFRESH", "MLFRAME_DISABLE_GPU", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(kc, "_cpu_model_slug", lambda: "testcpu")
    monkeypatch.setattr(cb, "_current_device_id", lambda: 0)
    state = {"summary": {"name": "testgpu", "cc_major": 8, "cc_minor": 9}, "usable": True}

    def probe(_device_id=0):
        """Fake ``gpu_capability_summary``: the configured summary, ``None``, or a raised failure."""
        if isinstance(state["summary"], Exception):
            raise state["summary"]
        return state["summary"]

    monkeypatch.setattr(kc, "gpu_capability_summary", probe)
    monkeypatch.setattr(policy, "cuda_available_for_run", lambda: state["usable"] and not policy.gpu_globally_disabled())

    def reset():
        """Drop both fingerprint memos so the next call recomputes."""
        kc.hw_fingerprint.cache_clear()
        cb._gpu_summary_cached.cache_clear()

    reset()
    state["path"] = tmp_path / ".hw_fingerprint.json"
    state["reset"] = reset
    state["fp"] = lambda: (reset(), kc.hw_fingerprint())[1]
    yield state
    reset()


def _persisted(path):
    """Fingerprint stored on disk, or ``None`` when the file does not exist."""
    return json.loads(path.read_text(encoding="utf-8"))["fingerprint"] if path.exists() else None


def _write(path, fingerprint):
    """Persist ``fingerprint`` the way pyutilz does."""
    path.write_text(json.dumps({"schema_version": 1, "fingerprint": fingerprint, "ts_utc": "2026-01-01T00:00:00+00:00"}), encoding="utf-8")
    os.utime(path, None)


def test_gpu_run_persists_its_fingerprint(fp_env):
    """A normal GPU run resolves and persists the GPU fingerprint."""
    assert fp_env["fp"]() == _GPU_FP
    assert _persisted(fp_env["path"]) == _GPU_FP


def test_opted_out_run_is_keyed_no_gpu_but_does_not_persist_it(fp_env, monkeypatch):
    """An opted-out run keys the CPU tuning directory in memory and leaves the disk fingerprint alone."""
    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    assert fp_env["fp"]() == _NO_GPU_FP
    assert _persisted(fp_env["path"]) is None


def test_opted_out_run_ignores_a_persisted_gpu_fingerprint(fp_env, monkeypatch):
    """An opted-out run does not resolve the GPU tuning directory it is not going to use, and does not rewrite the file."""
    _write(fp_env["path"], _GPU_FP)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert fp_env["fp"]() == _NO_GPU_FP
    assert _persisted(fp_env["path"]) == _GPU_FP


def test_failed_probe_on_a_gpu_host_is_not_persisted(fp_env):
    """A transient probe failure on a host with a usable GPU yields a no-gpu key for that process only."""
    fp_env["summary"] = RuntimeError("driver busy")
    assert fp_env["fp"]() == _NO_GPU_FP
    assert _persisted(fp_env["path"]) is None


def test_persisted_no_gpu_fingerprint_is_ignored_when_a_gpu_is_usable(fp_env):
    """A poisoned no-gpu file does not make a GPU run pick the CPU-only tuning directory, and the file is healed."""
    _write(fp_env["path"], _NO_GPU_FP)
    assert fp_env["fp"]() == _GPU_FP
    assert _persisted(fp_env["path"]) == _GPU_FP


def test_genuine_cpu_only_host_still_persists_no_gpu(fp_env):
    """On a host with no usable CUDA device and no opt-out the no-gpu fingerprint is cached as before."""
    fp_env["summary"] = None
    fp_env["usable"] = False
    assert fp_env["fp"]() == _NO_GPU_FP
    assert _persisted(fp_env["path"]) == _NO_GPU_FP


def test_fingerprint_for_opted_out_run_only_rewrites_gpu_keys():
    """The opted-out rewrite maps a GPU key to its CPU-only twin and leaves other keys untouched."""
    from mlframe.feature_selection.filters._hw_fingerprint_guard import fingerprint_for_opted_out_run

    assert fingerprint_for_opted_out_run("cpu_x_gpu_rtx-3090_cc8.6") == "cpu_x_no-gpu"
    assert fingerprint_for_opted_out_run("cpu_x_no-gpu") == "cpu_x_no-gpu"
