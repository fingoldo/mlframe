"""The subset-rank GPU crossover comes from a registered kernel-tuning entry; the env override and the module mark keep their roles."""

from __future__ import annotations

import sys

from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_subsetrank as sr
from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_subsetrank_ktc as ktc


class _Spec:
    """Stand-in for the registered entry: the GPU from a given subset count on."""

    def __init__(self, from_n: int) -> None:
        """Remember the mark."""
        self.from_n = from_n

    def choose(self, **dims):
        """GPU at or above the mark."""
        return "gpu" if dims["n_subsets"] >= self.from_n else "cpu"


def test_the_crossover_is_the_first_grid_point_the_entry_sends_to_the_gpu(monkeypatch):
    """One monotone threshold from the tuned entry."""
    monkeypatch.delenv("MLFRAME_SHAP_SUBSETRANK_GPU_MIN_SUBSETS", raising=False)
    monkeypatch.setattr(ktc, "_SPEC", _Spec(100_000))
    assert sr._gpu_min_subsets() == 262_143


def test_no_grid_point_on_the_gpu_means_never(monkeypatch):
    """A host where the GPU never wins keeps the CPU for every subset count."""
    monkeypatch.delenv("MLFRAME_SHAP_SUBSETRANK_GPU_MIN_SUBSETS", raising=False)
    monkeypatch.setattr(ktc, "_SPEC", _Spec(10**12))
    assert sr._gpu_min_subsets() == sys.maxsize


def test_without_an_entry_the_module_mark_decides(monkeypatch):
    """No registry: the measured default."""
    monkeypatch.delenv("MLFRAME_SHAP_SUBSETRANK_GPU_MIN_SUBSETS", raising=False)
    monkeypatch.setattr(ktc, "_SPEC", None)
    assert sr._gpu_min_subsets() == sr._DEFAULT_GPU_MIN_SUBSETS


def test_the_env_override_wins_over_the_entry(monkeypatch):
    """The A/B override beats the tuned entry."""
    monkeypatch.setenv("MLFRAME_SHAP_SUBSETRANK_GPU_MIN_SUBSETS", "7")
    monkeypatch.setattr(ktc, "_SPEC", _Spec(100_000))
    assert sr._gpu_min_subsets() == 7
