"""RESIDENT UPLOAD (wave 10): ``_binned_mi_cupy`` must upload the fit-constant ``y``/``y_codes`` target
ONCE across repeated calls with the SAME content -- e.g. successive legs of one source column via
``_heldout_incremental_mi`` (up to 6 legs/column), or successive source columns sharing the same
train/held-out mask via ``_select_wavelet_legs`` -- instead of a fresh ``cp.asarray`` every call. ``feat``
(the candidate leg/joint code) genuinely varies per call and stays a raw upload. Covers BOTH the
``discrete=True`` and ``discrete=False`` code paths, and both the ``y_codes``-given and ``y``-given
sub-branches, since the fix touches all four upload sites.

Only reachable when the STRICT/``MLFRAME_CMI_GPU`` gate is on (``_binnedmi_gpu_enabled``), so this test
calls ``_binned_mi_cupy`` directly rather than through the gated ``_binned_mi`` wrapper.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cupy")
import mlframe.feature_selection.filters.hermite_fe  # noqa: F401  (resolve import cycle first)

from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands
from mlframe.feature_selection.filters._wavelet_basis_fe import _binned_mi, _binned_mi_cupy


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear cache."""
    clear_fe_resident_operands()
    yield
    clear_fe_resident_operands()


def _prefix_binned_mi_cupy(feat, y, nbins, y_codes, discrete=False):
    """Exact reconstruction of the pre-fix ``_binned_mi_cupy`` body (raw cp.asarray for every y/y_codes
    site, no resident cache) -- the ground truth the fixed function must stay bit-identical to."""
    import cupy as cp

    from mlframe.feature_selection.filters._fe_batched_mi import binned_mi_from_codes_gpu
    from mlframe.feature_selection.filters._gpu_resident_select import (
        _radix_select_interior_edges,
        fe_gpu_radix_edges_enabled,
    )

    df = cp.asarray(np.asarray(feat, dtype=np.float64).ravel())
    if discrete:
        fb = (df - df.min()).astype(cp.int64)
        if y_codes is not None:
            yb = cp.asarray(np.asarray(y_codes).ravel()).astype(cp.int64, copy=False)
        else:
            dy = cp.asarray(np.asarray(y).ravel()).astype(cp.int64, copy=False)
            yb = dy - dy.min()
        return float(max(float(binned_mi_from_codes_gpu(fb[:, None], yb)[0]), 0.0))

    def _interior_edges(v):
        """Interior edges."""
        try:
            if fe_gpu_radix_edges_enabled():
                e = _radix_select_interior_edges(v.reshape(-1, 1), nbins)
                if e is not None:
                    return e.ravel()
        except Exception:  # nosec B110 -- best-effort cleanup/optional step; failure here never masks this test's own assertions
            pass
        return cp.quantile(v, cp.linspace(0.0, 1.0, nbins + 1)[1:-1])

    uf = cp.unique(df)
    if int(uf.size) <= nbins:
        fb = cp.searchsorted(uf, df)
    else:
        fb = cp.digitize(df, _interior_edges(df))
    if y_codes is not None:
        yb = cp.asarray(np.asarray(y_codes).ravel())
    else:
        dy = cp.asarray(np.asarray(y).ravel()) if not isinstance(y, cp.ndarray) else y.ravel()
        uy = cp.unique(dy)
        if int(uy.size) <= 20:
            yb = cp.searchsorted(uy, dy)
        else:
            yb = cp.digitize(dy.astype(cp.float64), _interior_edges(dy.astype(cp.float64)))
    fb = fb.astype(cp.int64, copy=False)
    yb = yb.astype(cp.int64, copy=False)
    mi = float(binned_mi_from_codes_gpu(fb[:, None], yb)[0])
    return float(max(mi, 0.0))


def _joint_and_y(n=3000, seed=0, n_classes=4):
    """Joint and y."""
    rng = np.random.default_rng(seed)
    joint = rng.integers(0, 30, size=n).astype(np.float64)  # mimics xc*3+legcode joint codes
    y = rng.integers(0, n_classes, size=n).astype(np.int64)
    return joint, y


class TestDiscreteYGivenBranch:
    """discrete=True, y_codes=None -- the actually-reachable production path (_heldout_incremental_mi_from_prep)."""

    def test_uploads_y_once_across_legs(self):
        """Uploads y once across legs."""
        import cupy as cp

        joint1, y = _joint_and_y(seed=1)
        joint2, _ = _joint_and_y(seed=2)  # a DIFFERENT leg's joint code, mirrors a second leg of the same column
        y2 = y.copy()

        upload_calls = {"n": 0}
        orig_asarray = cp.asarray

        def _counting(arr, *a, **kw):
            """Helper that counting."""
            if getattr(arr, "dtype", None) == np.int64 and getattr(arr, "shape", None) == y.shape:
                upload_calls["n"] += 1
            return orig_asarray(arr, *a, **kw)

        cp.asarray = _counting
        try:
            mi1 = _binned_mi_cupy(joint1, y, 30, None, discrete=True)
            mi2 = _binned_mi_cupy(joint2, y2, 30, None, discrete=True)
        finally:
            cp.asarray = orig_asarray

        assert upload_calls["n"] == 1, f"y-shaped int64 cp.asarray called {upload_calls['n']} times across 2 legs (expected 1)"
        assert mi1 >= 0.0 and mi2 >= 0.0

    def test_bit_identical_to_prefix_raw_path(self):
        """Bit identical to prefix raw path."""
        joint, y = _joint_and_y(seed=3)
        new_mi = _binned_mi_cupy(joint, y, 30, None, discrete=True)
        clear_fe_resident_operands()
        old_mi = _prefix_binned_mi_cupy(joint, y, 30, None, discrete=True)
        assert new_mi == old_mi, f"new={new_mi!r} old={old_mi!r}"


class TestDiscreteYCodesGivenBranch:
    """discrete=True, y_codes given -- exercised for coverage of all four upload sites."""

    def test_bit_identical_to_prefix_raw_path(self):
        """Bit identical to prefix raw path."""
        joint, y = _joint_and_y(seed=4)
        y_codes = y.copy()
        new_mi = _binned_mi_cupy(joint, y, 30, y_codes, discrete=True)
        clear_fe_resident_operands()
        old_mi = _prefix_binned_mi_cupy(joint, y, 30, y_codes, discrete=True)
        assert new_mi == old_mi


class TestNonDiscreteYCodesGivenBranch:
    """discrete=False, y_codes given -- the actually-reachable production path (_select_wavelet_legs)."""

    def test_uploads_y_codes_once_across_legs(self):
        """Uploads y codes once across legs."""
        import cupy as cp

        rng = np.random.default_rng(5)
        n = 3000
        leg1 = rng.choice([-1.0, 0.0, 1.0], size=n)
        leg2 = rng.choice([-1.0, 0.0, 1.0], size=n)
        y_codes = rng.integers(0, 6, size=n).astype(np.int64)
        y_codes2 = y_codes.copy()

        upload_calls = {"n": 0}
        orig_asarray = cp.asarray

        def _counting(arr, *a, **kw):
            """Helper that counting."""
            if getattr(arr, "dtype", None) == np.int64 and getattr(arr, "shape", None) == y_codes.shape:
                upload_calls["n"] += 1
            return orig_asarray(arr, *a, **kw)

        cp.asarray = _counting
        try:
            mi1 = _binned_mi_cupy(leg1, None, 10, y_codes, discrete=False)
            mi2 = _binned_mi_cupy(leg2, None, 10, y_codes2, discrete=False)
        finally:
            cp.asarray = orig_asarray

        assert upload_calls["n"] == 1, f"y_codes-shaped int64 cp.asarray called {upload_calls['n']} times across 2 legs (expected 1)"
        assert mi1 >= 0.0 and mi2 >= 0.0

    def test_bit_identical_to_prefix_raw_path(self):
        """Bit identical to prefix raw path."""
        rng = np.random.default_rng(6)
        n = 3000
        leg = rng.choice([-1.0, 0.0, 1.0], size=n)
        y_codes = rng.integers(0, 6, size=n).astype(np.int64)
        new_mi = _binned_mi_cupy(leg, None, 10, y_codes, discrete=False)
        clear_fe_resident_operands()
        old_mi = _prefix_binned_mi_cupy(leg, None, 10, y_codes, discrete=False)
        assert new_mi == old_mi


class TestNonDiscreteYGivenBranch:
    """discrete=False, y given (no y_codes) -- exercised for coverage of all four upload sites."""

    def test_bit_identical_to_prefix_raw_path(self):
        """Bit identical to prefix raw path."""
        rng = np.random.default_rng(7)
        n = 2500
        leg = rng.choice([-1.0, 0.0, 1.0], size=n)
        y = rng.integers(0, 5, size=n).astype(np.int64)
        new_mi = _binned_mi_cupy(leg, y, 10, None, discrete=False)
        clear_fe_resident_operands()
        old_mi = _prefix_binned_mi_cupy(leg, y, 10, None, discrete=False)
        assert new_mi == old_mi


def test_gpu_matches_cpu_reference():
    """Sanity cross-check: the resident-cached GPU path still agrees with the CPU plug-in MI reference."""
    joint, y = _joint_and_y(seed=8)
    gpu_mi = _binned_mi_cupy(joint, y, 30, None, discrete=True)
    cpu_mi = _binned_mi(joint, y, nbins=30, discrete=True)
    assert abs(gpu_mi - cpu_mi) < 1e-9, f"gpu={gpu_mi} cpu={cpu_mi}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
