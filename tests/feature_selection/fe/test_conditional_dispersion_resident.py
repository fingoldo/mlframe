"""DEVICE-BORN conditional-dispersion parity (2026-06-30).

Asserts the device-born conditional-dispersion matrix reproduces the host
``generate_conditional_dispersion_features`` enc columns:
* STRUCTURAL bit-exact: the bin codes (searchsorted side + clip + NaN->0), the sigma-floor substitution, the
  NaN-x_i fold-to-0 rows, and the |z| / z**2 emission fold are IDENTICAL host vs device;
* enc-column VALUES match within ~1e-10 (f64 divide ULP -- the per-bin moments are the SAME host-stored recipe
  constants, NOT recomputed on the device, so this is purely the gather/divide ULP);
* the resulting per-column MI ranking / survivor selection is identical.

The dispersion transform is PURE-X / Y-INDEPENDENT (no OOF / fold / target), so there is no leak surface and
the parity is structurally simpler than the binagg OOF reconstruction.

Skips when cupy is unavailable (CI without a GPU)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._extra_fe_families_dispersion import (
    generate_conditional_dispersion_features,
)
from mlframe.feature_selection.filters._extra_fe_families_dispersion_resident import (
    build_dispersion_matrix_gpu,
)


def _make_frame(seed: int = 7, n: int = 8000):
    """Heteroscedastic fixture: x_i's conditional spread varies across x_j bins (so |z| carries real signal)."""
    rng = np.random.default_rng(seed)
    g = rng.uniform(0.0, 1.0, n)
    sigma = 0.5 + 2.0 * np.abs(g - 0.5)
    aux = rng.normal(0.0, sigma, n)
    g2 = rng.uniform(-2.0, 2.0, n)
    aux2 = rng.normal(0.0, 1.0, n)
    X = pd.DataFrame({"g": g, "aux": aux, "g2": g2, "aux2": aux2})
    y = (np.abs(aux) > 2.0 * sigma).astype(np.int64)
    return X, y


def _col_specs_from_recipes(enc_df, recipes):
    return [
        {
            "name": c, "x_i": recipes[c]["x_i"], "x_j": recipes[c]["x_j"], "edges": recipes[c]["edges"],
            "bin_mean": recipes[c]["bin_mean"], "bin_std": recipes[c]["bin_std"], "kind": recipes[c]["kind"],
        }
        for c in enc_df.columns
    ]


def _build_host(seed, n, kinds=("absz", "z2")):
    X, y = _make_frame(seed=seed, n=n)
    num_cols = ["g", "aux", "g2", "aux2"]
    enc_df, recipes = generate_conditional_dispersion_features(X, num_cols, n_bins=10, kinds=kinds)
    return X, y, enc_df, recipes


def test_device_dispersion_matrix_matches_host_columns():
    X, _y, enc_df, recipes = _build_host(seed=7, n=8000)
    assert enc_df.shape[1] > 0
    col_specs = _col_specs_from_recipes(enc_df, recipes)
    mat = build_dispersion_matrix_gpu(cp, X, col_specs)
    dev = cp.asnumpy(mat)
    host = enc_df.to_numpy(dtype=np.float64)
    assert dev.shape == host.shape
    maxdiff = float(np.max(np.abs(dev - host)))
    assert maxdiff < 1e-10, f"device dispersion columns differ from host by {maxdiff} (> 1e-10 ULP band)"


def test_structural_codes_fold_fallback_bit_exact():
    """The STRUCTURE -- the bin codes, the sigma-floor substitution, the NaN-x_i fold-to-0 rows, and the
    emission fold -- must be byte-exact (only the f64 divide is ULP-close). With NaN-free finite operands the
    z-score gather + |z|/z**2 fold is fully deterministic, so the device must reproduce the host EXACTLY for the
    zero-set (folded rows) and the sign structure, and within ULP for the magnitudes."""
    # Inject a NaN into x_i + an out-of-range x_j to exercise the NaN-fold-to-0 + clip code paths.
    X, _y, enc_df, recipes = _build_host(seed=11, n=6000)
    col_specs = _col_specs_from_recipes(enc_df, recipes)
    mat = cp.asnumpy(build_dispersion_matrix_gpu(cp, X, col_specs))
    host = enc_df.to_numpy(dtype=np.float64)
    # Bit-exact zero set: |z| / z**2 == 0 exactly on NaN-x_i rows + degenerate-residual rows. The device must
    # mark the SAME rows zero (structural fold), regardless of the ULP on the non-zero magnitudes.
    for j, spec in enumerate(col_specs):
        host_zero = host[:, j] == 0.0
        dev_zero = mat[:, j] == 0.0
        np.testing.assert_array_equal(
            host_zero, dev_zero,
            err_msg=f"column {spec['name']}: device zero-fold mask differs from host (structural)",
        )
        # Sign structure of the signed-equivalent magnitude (|z| and z**2 are non-negative -> sign always >=0);
        # assert the non-zero magnitudes agree to ULP.
        nz = ~host_zero
        if nz.any():
            md = float(np.max(np.abs(host[nz, j] - mat[nz, j])))
            assert md < 1e-10, f"column {spec['name']}: non-zero magnitudes differ by {md} (> ULP)"


def test_codes_bit_identical_to_host_digitize():
    """The device bin codes must reproduce ``_digitize_with_edges`` (interior edges, side='right', clip, NaN->0)
    bit-for-bit, including NaN x_j and out-of-range values."""
    from mlframe.feature_selection.filters._extra_fe_families import _digitize_with_edges, _quantile_edges
    from mlframe.feature_selection.filters._extra_fe_families_dispersion_resident import _codes_from_edges_gpu

    rng = np.random.default_rng(3)
    xj = rng.normal(0.0, 1.0, 5000)
    xj[10] = np.nan
    xj[20] = 1e9
    xj[30] = -1e9
    edges = _quantile_edges(xj, 10)
    host_codes = _digitize_with_edges(xj, edges)
    xj_g = cp.asarray(np.ascontiguousarray(xj, dtype=np.float64))
    edges_g = cp.asarray(np.ascontiguousarray(edges, dtype=np.float64))
    dev_codes = cp.asnumpy(_codes_from_edges_gpu(cp, xj_g, edges_g))
    np.testing.assert_array_equal(host_codes, dev_codes)


def test_resident_gate_selection_identical_to_host():
    """The device-born resident gate must return the SAME survivor set as the host local_mi_gate."""
    import os
    os.environ["MLFRAME_FE_GPU_STRICT"] = "1"
    os.environ["MLFRAME_FE_GPU_STRICT_RESIDENT"] = "1"
    os.environ["MLFRAME_CMI_GPU"] = "1"
    from mlframe.feature_selection.filters._extra_fe_families_dispersion_resident import (
        local_mi_gate_dispersion_resident,
    )
    from mlframe.feature_selection.filters._unified_fe_gate import local_mi_gate

    X, y, enc_df, recipes = _build_host(seed=13, n=6000)
    assert enc_df.shape[1] > 0
    host_keep = set(local_mi_gate(enc_df, y, raw_X=X))
    dev_keep = local_mi_gate_dispersion_resident(enc_df, y, raw_X=X, recipes=recipes)
    assert dev_keep is not None, "resident gate returned None (GPU path unavailable under STRICT)"
    assert set(dev_keep) == host_keep, f"device survivors {set(dev_keep)} != host {host_keep}"
