"""The device stratification bins of a continuous target equal the host branch's strata, so the subsample is unchanged."""

from __future__ import annotations


import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters import _fe_subsample as S


def _host_bins(y: np.ndarray):
    """The host branch's strata, verbatim."""
    finite = np.isfinite(y)
    edges = np.unique(np.quantile(y[finite], np.linspace(0.0, 1.0, S._FE_STRATIFY_REG_BINS + 1)))
    ids = np.digitize(y, edges[1:-1], right=False)
    ids[~finite] = edges.shape[0]
    uniq = np.unique(ids)
    remap = np.zeros(int(ids.max()) + 1, dtype=np.int64)
    remap[uniq] = np.arange(uniq.shape[0], dtype=np.int64)
    return remap[ids].astype(np.int64), int(uniq.shape[0])


@pytest.mark.parametrize("kind", ["normal", "ties", "nan_inf", "heavy"])
def test_device_strata_equal_the_host_strata(monkeypatch, kind):
    """Same codes and stratum count, for continuous, tie-heavy, non-finite-bearing and heavy-tailed targets."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    rng = np.random.default_rng(1)
    y = {"normal": rng.standard_normal(300_000), "ties": np.round(rng.standard_normal(300_000), 1), "heavy": rng.standard_normal(300_000) ** 3, "nan_inf": rng.standard_normal(300_000)}[kind]
    if kind == "nan_inf":
        y[::97] = np.nan
        y[5::1013] = np.inf
    got = S._device_regression_bins(y, np.isfinite(y))
    assert got is not None
    codes, n_bins = got
    ref_codes, ref_bins = _host_bins(y)
    assert n_bins == ref_bins
    np.testing.assert_array_equal(codes, ref_codes)


def test_the_strict_off_path_declines(monkeypatch):
    """Without the strict-resident flag the helper returns None and the host branch runs."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "0")
    y = np.random.default_rng(0).standard_normal(300_000)
    assert S._device_regression_bins(y, np.isfinite(y)) is None
