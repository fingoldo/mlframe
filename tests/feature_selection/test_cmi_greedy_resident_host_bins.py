"""On the resident GPU path the lazy host projection of a candidate's bin codes must work, not raise a NameError that drops the whole CMI-greedy step."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _mi_greedy_cmi_fe_steps as steps


def test_host_bins_is_usable_after_the_resident_fingerprint_path(monkeypatch):
    """The resident path leaves ``cand_bins`` unset in the host branch; the closure must still return the host codes of a candidate."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT_RESIDENT", "1")
    rng = np.random.default_rng(3)
    n = 4000
    engineered = pd.DataFrame({"a": rng.standard_normal(n).astype(np.float32), "b": rng.standard_normal(n).astype(np.float32)})
    st = SimpleNamespace(cand_fp={}, _resident_fp_ok=False)
    cand_bins_dev: dict = {}
    host_bins, cand_bins_dev, _rng = steps._greedy_cmi_fe_constr_step1_content_hash_already(
        ["a", "b"], engineered, 8, st, lambda codes: int(np.asarray(codes).sum()), 0, cand_bins_dev
    )
    assert st._resident_fp_ok is True, "the resident branch was not taken, so this test would prove nothing"
    got = host_bins("a")
    assert got.shape == (n,)
    np.testing.assert_array_equal(got, cp.asnumpy(cand_bins_dev["a"]))
