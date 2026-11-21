"""The bounded candidate-code pool returns the codes and fingerprints of the one-shot binner, from the device share and from the host spill."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _cand_code_pool as P
from mlframe.feature_selection.filters._bin_fingerprint_gpu import pool_fingerprints
from mlframe.feature_selection.filters._mi_greedy_cmi_fe_binning import _qbin_float_dtype
from mlframe.feature_selection.filters._qbin_block_gpu import resident_qbin_codes_block


@pytest.mark.parametrize("device_bytes", [0, 10**9])
def test_pool_matches_one_shot_block_codes(monkeypatch, device_bytes):
    """Tiny blocks (3 columns) and a device share of 0 (everything on the host) or ample give the same widened codes and fingerprints."""
    monkeypatch.setattr(P, "_DEVICE_POOL_BYTES", device_bytes)
    rng = np.random.default_rng(0)
    cols = [rng.standard_normal(3000).astype(np.float32) for _ in range(10)]
    cols[4][::7] = np.nan
    names = [f"c{i}" for i in range(10)]
    ref = resident_qbin_codes_block(cols, 16, _qbin_float_dtype())
    ref_fp = pool_fingerprints(cp, ref)
    got = P.bin_candidate_pool(cp, names, cols, 16, _qbin_float_dtype(), resident_qbin_codes_block, pool_fingerprints, 3 * 8 * 3000)
    assert got is not None
    pool, fps = got
    assert [fps[n] for n in names] == ref_fp
    assert len(pool) == 10 and pool.get("nope") is None
    for n, r in zip(names, ref):
        w = pool[n]
        assert w.dtype == cp.int64
        np.testing.assert_array_equal(cp.asnumpy(w), cp.asnumpy(r))
