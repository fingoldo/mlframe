"""The matrix-vector coarse sweep of the adaptive Fourier detector emits the same extra-basis columns and metadata as the per-frequency loop."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_extra_basis_fe_helpers as H
from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe_generate import generate_extra_basis_features


def test_same_output_as_the_loop(monkeypatch):
    """Periodic, chirp-like and noise columns with the adaptive and chirp detectors on."""
    rng = np.random.default_rng(4)
    n = 6000
    a, b, c = rng.uniform(-3, 3, n), rng.uniform(-2, 2, n), rng.standard_normal(n)
    y = np.sin(2.7 * a) + 0.5 * np.cos(np.sign(b) * b * b * 3.1) + 0.3 * rng.standard_normal(n)
    X = pd.DataFrame({"a": a, "b": b, "c": c, "d": rng.standard_normal(n)})
    kw = dict(extra_bases=("fourier",), y=y, fourier_adaptive=True, fourier_chirp=True, max_adaptive_cols=100)
    monkeypatch.setattr(H, "_COARSE_GEMV", False)
    want_cols, want_meta = generate_extra_basis_features(X, **kw)
    monkeypatch.setattr(H, "_COARSE_GEMV", True)
    got_cols, got_meta = generate_extra_basis_features(X, **kw)
    assert list(got_cols.columns) == list(want_cols.columns)
    assert got_cols.equals(want_cols)
    assert repr(got_meta) == repr(want_meta)
    assert any(m.get("adaptive") for m in got_meta.values() if isinstance(m, dict))
