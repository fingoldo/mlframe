"""batch_numaggs computes the same means as the serial path (audit 2026-11-20, finding 18-F5)."""
import inspect

from mlframe.models.ensembling import base


def test_batch_path_disables_whitening_like_the_serial_path():
    """The batch call must pass whiten_means=False, as the serial call does; the default changes the quadratic/qubic/geometric/harmonic means."""
    src = inspect.getsource(base)
    batch_call = [ln for ln in src.splitlines() if "compute_numerical_aggregates_numba(arr, geomean_log_mode=False, directional_only=False" in ln]
    assert len(batch_call) == 2
    assert all("whiten_means=False" in ln for ln in batch_call)
