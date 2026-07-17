"""Identity regression for the CMI-greedy y/z hoist (CPX10).

cmi_from_binned_fixed_yz (fed by precompute_cmi_yz_terms) must be bit-identical
to _cmi_from_binned for a present Z -- the greedy loop now routes every
candidate through the hoisted path. Fuzz a range of cardinalities/sizes.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._mi_greedy_cmi_fe import (
    _cmi_from_binned,
    cmi_from_binned_fixed_yz,
    precompute_cmi_yz_terms,
)


@pytest.mark.parametrize("seed", range(8))
def test_fixed_yz_bit_identical_to_cmi_from_binned(seed, monkeypatch):
    # CPU-vs-CPU bit-identity contract: ``_cmi_gpu_enabled()`` reroutes BOTH helpers to the GPU twin under
    # EITHER MLFRAME_CMI_GPU==1 OR MLFRAME_FE_GPU_STRICT, and the GPU fp-reduction order then breaks the ``==``
    # by ~1e-15 (not a refactor bug). Force the CPU path for the duration so a suite-global GPU env cannot
    # reroute this exactness fuzz.
    monkeypatch.setenv("MLFRAME_CMI_GPU", "0")
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    rng = np.random.default_rng(seed)
    n = int(rng.integers(200, 3000))
    x = rng.integers(0, int(rng.integers(2, 12)), n).astype(np.int64)
    y = rng.integers(0, int(rng.integers(2, 6)), n).astype(np.int64)
    z = rng.integers(0, int(rng.integers(2, 15)), n).astype(np.int64)

    expected = _cmi_from_binned(x, y, z)
    yi, zi, h_yz, h_z, k_yz, k_z, nn = precompute_cmi_yz_terms(y, z)
    got = cmi_from_binned_fixed_yz(x, yi, zi, h_yz, h_z, k_yz, k_z, nn)
    assert got == expected
