"""End to end against pyutilz: a failed or opted-out GPU probe must not poison the persisted hardware fingerprint that keys every kernel-tuning lookup."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("pyutilz.performance.kernel_tuning.cache")

_SUMMARY = {"name": "testgpu", "cc_major": 8, "cc_minor": 9}


@pytest.fixture
def fp_env(tmp_path, monkeypatch):
    """Isolated fingerprint cache directory with a fake CPU slug, GPU summary probe and device-count query."""
    import mlframe.feature_selection.filters._kernel_tuning  # noqa: F401  (registers the mlframe GPU opt-out with pyutilz)
    from pyutilz.performance.kernel_tuning import cache as kc
    from pyutilz.performance.kernel_tuning.cache import cache_base as cb

    if not hasattr(kc, "register_gpu_opt_out"):
        pytest.skip("installed pyutilz predates the probe-state fingerprint")
    for var in ("PYUTILZ_HW_FP_REFRESH", "PYUTILZ_DISABLE_GPU", "MLFRAME_DISABLE_GPU", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
    state = {"summary": dict(_SUMMARY), "count": 1, "path": tmp_path / ".hw_fingerprint.json"}

    def probe(_device_id=0):
        """Fake ``gpu_capability_summary``: the configured summary, ``None``, or a raised failure."""
        if isinstance(state["summary"], Exception):
            raise state["summary"]
        return state["summary"]

    monkeypatch.setattr(kc, "_cpu_model_slug", lambda: "testcpu")
    monkeypatch.setattr(kc, "gpu_capability_summary", probe)
    monkeypatch.setattr(cb, "_gpu_device_count", lambda: state["count"])
    monkeypatch.setattr(cb, "_current_device_id", lambda: 0)

    def fresh():
        """Fingerprint of a new process: both memos dropped first."""
        kc.hw_fingerprint.cache_clear()
        cb._gpu_summary_cached.cache_clear()
        return kc.hw_fingerprint()

    state["fp"] = fresh
    kc.hw_fingerprint.cache_clear()
    cb._gpu_summary_cached.cache_clear()
    yield state
    kc.hw_fingerprint.cache_clear()
    cb._gpu_summary_cached.cache_clear()


def _entries(path):
    """Persisted fingerprint entries, ``{}`` when the file does not exist."""
    return json.loads(path.read_text(encoding="utf-8"))["entries"] if path.exists() else {}


def test_gpu_run_persists_its_gpu_fingerprint(fp_env):
    """A normal GPU run resolves and persists the GPU hardware key."""
    fp = fp_env["fp"]()
    assert "_gpu_testgpu_cc8.9_" in fp
    assert "_gpu_testgpu_cc8.9_" in _entries(fp_env["path"])["default"]["fingerprint"]


def test_failed_probe_does_not_poison_the_cache_for_a_later_gpu_run(fp_env):
    """A transient probe failure keys a non-persistent unknown fingerprint; the next process with a working GPU resolves the GPU key."""
    fp_env["summary"] = RuntimeError("driver busy")
    failed = fp_env["fp"]()
    assert "no-gpu" not in failed
    assert _entries(fp_env["path"]) == {}
    fp_env["summary"] = dict(_SUMMARY)
    assert "_gpu_testgpu_cc8.9_" in fp_env["fp"]()


@pytest.mark.parametrize("opt_out_env, value", [("MLFRAME_DISABLE_GPU", "1"), ("CUDA_VISIBLE_DEVICES", "")])
def test_opted_out_run_keys_cpu_only_without_reading_or_writing_the_gpu_entry(fp_env, monkeypatch, opt_out_env, value):
    """MLFRAME_DISABLE_GPU=1 and CUDA_VISIBLE_DEVICES='' both key the CPU tuning directory and leave a persisted GPU entry untouched."""
    gpu_fp = fp_env["fp"]()
    before = _entries(fp_env["path"])
    monkeypatch.setenv(opt_out_env, value)
    assert "_no-gpu_" in fp_env["fp"]()
    assert _entries(fp_env["path"]) == before
    monkeypatch.delenv(opt_out_env)
    assert fp_env["fp"]() == gpu_fp


def test_genuine_cpu_only_host_still_persists_no_gpu(fp_env):
    """Zero devices reported without error is a definite answer and is cached."""
    fp_env["summary"] = None
    fp_env["count"] = 0
    fp = fp_env["fp"]()
    assert "_no-gpu_" in fp
    assert "_no-gpu_" in _entries(fp_env["path"])["default"]["fingerprint"]
