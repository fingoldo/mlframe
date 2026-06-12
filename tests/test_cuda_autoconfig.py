"""``mlframe._autoconfigure_cuda_home`` points numba.cuda at the pip-installed NVVM
only when nothing else has configured CUDA -- and never overrides a real install."""
from __future__ import annotations

import os
import sysconfig

import pytest

import mlframe


def _make_fake_pip_nvvm(root):
    nvvm = root / "nvidia" / "cuda_nvcc" / "nvvm"
    (nvvm / "bin").mkdir(parents=True)
    (nvvm / "libdevice").mkdir(parents=True)
    (nvvm / "bin" / "nvvm64_40_0.dll").write_bytes(b"\x00")
    (nvvm / "libdevice" / "libdevice.10.bc").write_bytes(b"\x00")
    return str((nvvm.parent).resolve())


def _clear_cuda_env(monkeypatch):
    for k in list(os.environ):
        if k == "CUDA_HOME" or k == "CUDA_PATH" or k.startswith("CUDA_PATH_V") or k == "MLFRAME_NO_CUDA_AUTOCONFIG":
            monkeypatch.delenv(k, raising=False)


def test_sets_cuda_home_to_pip_nvvm_when_env_clean(monkeypatch, tmp_path):
    _clear_cuda_env(monkeypatch)
    cuda_nvcc = _make_fake_pip_nvvm(tmp_path)
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {"purelib": str(tmp_path), "platlib": str(tmp_path)})
    mlframe._autoconfigure_cuda_home()
    assert os.environ.get("CUDA_HOME") == cuda_nvcc
    assert os.environ.get("CUDA_PATH") == cuda_nvcc


def test_skips_when_cuda_path_already_set(monkeypatch, tmp_path):
    _clear_cuda_env(monkeypatch)
    monkeypatch.setenv("CUDA_PATH", r"C:\Real\CUDA\v12.3")
    _make_fake_pip_nvvm(tmp_path)
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {"purelib": str(tmp_path), "platlib": str(tmp_path)})
    mlframe._autoconfigure_cuda_home()
    assert os.environ.get("CUDA_PATH") == r"C:\Real\CUDA\v12.3"  # untouched
    assert "CUDA_HOME" not in os.environ


def test_skips_when_versioned_system_installer_var_present(monkeypatch, tmp_path):
    _clear_cuda_env(monkeypatch)
    monkeypatch.setenv("CUDA_PATH_V12_3", r"C:\Real\CUDA\v12.3")
    _make_fake_pip_nvvm(tmp_path)
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {"purelib": str(tmp_path), "platlib": str(tmp_path)})
    mlframe._autoconfigure_cuda_home()
    assert "CUDA_HOME" not in os.environ
    assert "CUDA_PATH" not in os.environ


def test_skips_on_opt_out(monkeypatch, tmp_path):
    _clear_cuda_env(monkeypatch)
    monkeypatch.setenv("MLFRAME_NO_CUDA_AUTOCONFIG", "1")
    _make_fake_pip_nvvm(tmp_path)
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {"purelib": str(tmp_path), "platlib": str(tmp_path)})
    mlframe._autoconfigure_cuda_home()
    assert "CUDA_HOME" not in os.environ


def test_no_op_when_no_pip_nvvm(monkeypatch, tmp_path):
    _clear_cuda_env(monkeypatch)
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {"purelib": str(tmp_path), "platlib": str(tmp_path)})
    mlframe._autoconfigure_cuda_home()  # tmp_path has no nvidia/ tree
    assert "CUDA_HOME" not in os.environ
