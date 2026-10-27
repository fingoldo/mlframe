"""The occupied-cell counts of the analytic CMI null are the same whether the codes arrive on the host or already resident on the device."""
import numpy as np
import pytest

from mlframe.feature_selection.filters._mi_greedy_cmi_fe import joint_cardinalities_cupy


@pytest.mark.parametrize("device", ["x", "z", "y", "xyz"])
def test_device_inputs_give_the_host_counts(device):
    """A device-resident support (z) or target (y) used to raise and drop the caller to the host path; the counts must equal the host ones."""
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(0)
    n = 5000
    x, y, z = rng.integers(0, 8, n), rng.integers(0, 2, n), rng.integers(0, 6, n)
    try:
        expected = joint_cardinalities_cupy(x, y, z)
    except Exception as exc:  # no usable device
        pytest.skip(f"needs a CUDA device: {exc}")
    args = {"x": x, "y": y, "z": z}
    for name in device:
        args[name] = cp.asarray(args[name])
    assert joint_cardinalities_cupy(args["x"], args["y"], args["z"]) == expected
