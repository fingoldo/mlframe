"""``mlframe._autoconfigure_cuda_home`` points numba.cuda at the pip-installed NVVM
only when nothing else has configured CUDA -- and never overrides a real install."""

from __future__ import annotations

import os
import sysconfig


import mlframe


def _make_fake_pip_nvvm(root):
    """Test helper: nvvm = root / 'nvidia' / 'cuda_nvcc' / 'nvvm'; (nvvm / 'bin').mkdir(parents=True); (nvvm / 'libdevice').mkdir(parents=True)."""
    nvvm = root / "nvidia" / "cuda_nvcc" / "nvvm"
    (nvvm / "bin").mkdir(parents=True)
    (nvvm / "libdevice").mkdir(parents=True)
    (nvvm / "bin" / "nvvm64_40_0.dll").write_bytes(b"\x00")
    (nvvm / "libdevice" / "libdevice.10.bc").write_bytes(b"\x00")
    return str((nvvm.parent).resolve())


def _clear_cuda_env(monkeypatch):
    """Test helper: for k in list(os.environ): if k == 'CUDA_HOME' or k == 'C...."""
    for k in list(os.environ):
        if k == "CUDA_HOME" or k == "CUDA_PATH" or k.startswith("CUDA_PATH_V") or k == "MLFRAME_NO_CUDA_AUTOCONFIG":
            monkeypatch.delenv(k, raising=False)


def test_sets_cuda_home_to_pip_nvvm_when_env_clean(monkeypatch, tmp_path):
    """Sets cuda home to pip nvvm when env clean."""
    _clear_cuda_env(monkeypatch)
    cuda_nvcc = _make_fake_pip_nvvm(tmp_path)
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {"purelib": str(tmp_path), "platlib": str(tmp_path)})
    mlframe._autoconfigure_cuda_home()
    assert os.environ.get("CUDA_HOME") == cuda_nvcc
    assert os.environ.get("CUDA_PATH") == cuda_nvcc


def test_skips_when_cuda_path_already_set(monkeypatch, tmp_path):
    """Skips when cuda path already set."""
    _clear_cuda_env(monkeypatch)
    monkeypatch.setenv("CUDA_PATH", r"C:\Real\CUDA\v12.3")
    _make_fake_pip_nvvm(tmp_path)
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {"purelib": str(tmp_path), "platlib": str(tmp_path)})
    mlframe._autoconfigure_cuda_home()
    assert os.environ.get("CUDA_PATH") == r"C:\Real\CUDA\v12.3"  # untouched
    assert "CUDA_HOME" not in os.environ


def test_skips_when_versioned_system_installer_var_present(monkeypatch, tmp_path):
    """Skips when versioned system installer var present."""
    _clear_cuda_env(monkeypatch)
    monkeypatch.setenv("CUDA_PATH_V12_3", r"C:\Real\CUDA\v12.3")
    _make_fake_pip_nvvm(tmp_path)
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {"purelib": str(tmp_path), "platlib": str(tmp_path)})
    mlframe._autoconfigure_cuda_home()
    assert "CUDA_HOME" not in os.environ
    assert "CUDA_PATH" not in os.environ


def test_skips_on_opt_out(monkeypatch, tmp_path):
    """Skips on opt out."""
    _clear_cuda_env(monkeypatch)
    monkeypatch.setenv("MLFRAME_NO_CUDA_AUTOCONFIG", "1")
    _make_fake_pip_nvvm(tmp_path)
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {"purelib": str(tmp_path), "platlib": str(tmp_path)})
    mlframe._autoconfigure_cuda_home()
    assert "CUDA_HOME" not in os.environ


def test_no_op_when_no_pip_nvvm(monkeypatch, tmp_path):
    """No op when no pip nvvm."""
    _clear_cuda_env(monkeypatch)
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {"purelib": str(tmp_path), "platlib": str(tmp_path)})
    mlframe._autoconfigure_cuda_home()  # tmp_path has no nvidia/ tree
    assert "CUDA_HOME" not in os.environ


def _make_fake_complete_toolkit(root):
    """A complete CUDA toolkit dir: nvvm (codegen) AND cudart (runtime, needed by get_supported_ccs)."""
    tk = root / "CUDA" / "v12.9"
    (tk / "bin").mkdir(parents=True)
    (tk / "nvvm" / "bin").mkdir(parents=True)
    (tk / "bin" / "cudart64_12.dll").write_bytes(b"\x00")
    (tk / "nvvm" / "bin" / "nvvm64_40_0.dll").write_bytes(b"\x00")
    return str(tk.resolve())


def test_repairs_incomplete_cuda_path_to_complete_toolkit(monkeypatch, tmp_path):
    """A stale CUDA_PATH pointing at the nvvm-only cuda_nvcc wheel (no cudart) is redirected to a
    complete CUDA_PATH_V* system toolkit so numba.cuda kernels can actually compile."""
    _clear_cuda_env(monkeypatch)
    cuda_nvcc = _make_fake_pip_nvvm(tmp_path)  # nvvm-only, no cudart
    complete = _make_fake_complete_toolkit(tmp_path)
    monkeypatch.setenv("CUDA_PATH", cuda_nvcc)
    monkeypatch.setenv("CUDA_PATH_V12_9", complete)
    mlframe._autoconfigure_cuda_home()
    assert os.environ.get("CUDA_PATH") == complete
    assert os.environ.get("CUDA_HOME") == complete


def test_no_repair_when_cuda_path_already_complete(monkeypatch, tmp_path):
    """A CUDA_PATH that already has cudart is a real install -- left untouched."""
    _clear_cuda_env(monkeypatch)
    complete = _make_fake_complete_toolkit(tmp_path)
    monkeypatch.setenv("CUDA_PATH", complete)
    mlframe._autoconfigure_cuda_home()
    assert os.environ.get("CUDA_PATH") == complete  # untouched
    assert "CUDA_HOME" not in os.environ
