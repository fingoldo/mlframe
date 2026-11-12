"""The k-fold OOF MI scorer gives the host scores on the device (train-edge binning, Miller-Madow MI with the observed supports), up to round-off."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._orthogonal_three_gate_mi_fe import score_features_by_kfold_oof_mi


@pytest.mark.parametrize("n_classes", [2, 3])
def test_device_equals_host(monkeypatch, n_classes):
    """Signal, constant, tied and heavy-tailed raw and engineered columns."""
    rng = np.random.default_rng(1)
    n = 20000
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    raw = pd.DataFrame({"a": a, "b": b, "c": np.full(n, 2.0), "d": np.round(rng.standard_normal(n))})
    eng = pd.DataFrame({"a__sq": a * a, "b__cauchy": rng.standard_cauchy(n), "a__step": (a > 0.5).astype(float), "d__x": np.round(rng.standard_normal(n) * 2)})
    y = np.digitize(a * a + 0.3 * rng.standard_normal(n), np.quantile(a * a, np.linspace(0, 1, n_classes + 1)[1:-1]))
    monkeypatch.setenv("MLFRAME_KFOLD_OOF_MI_DEVICE", "0")
    want = score_features_by_kfold_oof_mi(raw, eng, y, n_folds=5, seed=0, nbins=10)
    monkeypatch.setenv("MLFRAME_KFOLD_OOF_MI_DEVICE", "1")
    from mlframe.feature_selection.filters import _kfold_oof_mi_gpu as G

    monkeypatch.setattr(G, "kfold_oof_mi_device_enabled", lambda *a: True)
    got = score_features_by_kfold_oof_mi(raw, eng, y, n_folds=5, seed=0, nbins=10)
    assert list(got["engineered_col"]) == list(want["engineered_col"])
    for c in got.columns:
        if got[c].dtype.kind == "f":
            np.testing.assert_allclose(got[c].to_numpy(), want[c].to_numpy(), rtol=1e-9, atol=1e-12)


def test_fold_beyond_the_vram_cushion_is_not_copied_to_the_device(monkeypatch):
    """A fold the VRAM cushion refuses returns None before any copy: the device upload is never called, and the host runs the fold."""
    from mlframe.feature_selection.filters import _fe_gpu_vram
    from mlframe.feature_selection.filters import _kfold_oof_mi_gpu as G

    rng = np.random.default_rng(0)
    arr = rng.standard_normal((400, 6)).astype(np.float32)
    y = rng.integers(0, 2, 400).astype(np.int64)
    train, test = np.arange(300), np.arange(300, 400)

    def no_room(*args, **kwargs):
        """The cushion refuses this fold."""
        return False

    copies: list = []
    real_asarray = cp.asarray

    def recording_asarray(*args, **kwargs):
        """Records every device copy; the fold must produce none."""
        copies.append(1)
        return real_asarray(*args, **kwargs)

    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", no_room)
    monkeypatch.setattr(cp, "asarray", recording_asarray)
    assert G.fold_mi_device(arr, train, test, y[test], 8) is None
    assert not copies, "the fold was copied to the device before the admission check"
