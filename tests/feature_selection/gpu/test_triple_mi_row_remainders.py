"""The device triple MI with 4-byte loads equals an independent numpy MI for every row count remainder modulo 4."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cupy")


@pytest.mark.parametrize("n", [1001, 1002, 1003, 1004])
def test_padded_four_byte_loads_equal_the_host_for_every_row_remainder(n):
    """Row counts of every remainder modulo 4 give the host kernel's MI: the padded rows are never counted and the leftover rows go through the scalar tail."""
    from mlframe.feature_selection.filters import _triple_mi_gpu as t

    rng = np.random.default_rng(n)
    p, base, n_y = 7, 5, 3
    codes = rng.integers(0, base, (p, n)).astype(np.int8)
    y = rng.integers(0, n_y, n).astype(np.int64)
    got = t.triple_mi_gpu(codes, y, base, n_y)
    assert got is not None
    oi, oj, ok, mi = got
    assert len(mi) > 0 and bool(np.all(np.isfinite(mi)))
    for q in range(0, len(mi), 7):
        i, j, k = int(oi[q]), int(oj[q]), int(ok[q])
        cell = (codes[i].astype(np.int64) * base + codes[j]) * base + codes[k]
        joint = np.zeros((base**3, n_y))
        np.add.at(joint, (cell, y), 1.0)
        px, py = joint.sum(1), joint.sum(0)
        nz = joint > 0
        want = float(np.sum(joint[nz] / n * np.log(joint[nz] * n / (px[:, None] * py[None, :])[nz])))
        assert mi[q] == pytest.approx(want, rel=1e-9, abs=1e-12)


def test_triple_blocks_with_few_columns_upload_only_those_columns(monkeypatch):
    """Blocks of 7 triples touch a few columns each and upload only those; the outputs equal the single-launch ones exactly."""
    from mlframe.feature_selection.filters import _triple_mi_gpu as t

    rng = np.random.default_rng(5)
    p, n, base, n_y = 9, 1001, 5, 3
    codes = rng.integers(0, base, (p, n)).astype(np.int8)
    y = rng.integers(0, n_y, n).astype(np.int64)
    whole = t.triple_mi_gpu(codes, y, base, n_y)
    monkeypatch.setattr(t, "TRIPLES_PER_LAUNCH", 7)
    blocked = t.triple_mi_gpu(codes, y, base, n_y)
    assert whole is not None and blocked is not None
    for a, b in zip(whole, blocked):
        assert np.array_equal(a, b)
