"""The honest null decides its root split analytically from 20k rows, and an explicit env override still wins."""
from mlframe.feature_selection.filters import _analytic_mi_null as A
from mlframe.feature_selection.filters._honest_mdlp_null import _honest_analytic_min_n


def test_floor_is_20k_and_env_override_wins(monkeypatch):
    """Default 25k is lowered to 15k for the honest null only; MLFRAME_MI_ANALYTIC_NULL_MIN_N=50000 keeps the permutation path."""
    monkeypatch.delenv("MLFRAME_MI_ANALYTIC_NULL_MIN_N", raising=False)
    assert _honest_analytic_min_n(A.analytic_null_min_n, A.analytic_null_min_n_lowered) == 15_000
    assert A.analytic_null_min_n() == 25_000
    monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL_MIN_N", "50000")
    assert _honest_analytic_min_n(A.analytic_null_min_n, A.analytic_null_min_n_lowered) == 50_000


def test_explicit_cpu_pin_keeps_the_host_kernel(monkeypatch):
    """MLFRAME_FE_GPU_STRICT=0 is the documented CPU pin: the device honest null stays off even with a GPU present."""
    from mlframe.feature_selection.filters._honest_mdlp_null import _device_honest_null_enabled

    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "0")
    assert not _device_honest_null_enabled()
