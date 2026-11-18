"""The one-launch |corr| kernel equals the cupy-operation chain it replaced, case by case, and is faster per call."""

import time

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_abs_corr_gpu as mod
from tests._cuda_require import require_cuda_device

pytestmark = pytest.mark.gpu


def _target(n, seed=1, nan_frac=0.0):
    """A float64 target with an optional share of NaN rows, plus its finiteness mask."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    if nan_frac:
        y[rng.random(n) < nan_frac] = np.nan
    return y, np.isfinite(y)


def _cases():
    """(name, column, target, finite mask) rows: plain, offset, non-finite column and target, odd length, constant, short, correlated."""
    rng = np.random.default_rng(0)
    out = []
    for n in (50_000, 40_001, 9):
        y, f = _target(n)
        for dt in (np.float32, np.float64):
            out.append((f"plain-{n}-{dt.__name__}", rng.standard_normal(n).astype(dt), y, f))
    y, f = _target(30_000)
    out.append(("offset1e7", (1e7 + rng.standard_normal(30_000)).astype(np.float64), y + 1e7, f))
    col = rng.standard_normal(30_000).astype(np.float32)
    col[rng.random(30_000) < 0.1] = np.nan
    col[5] = np.inf
    yn, fn = _target(30_000, nan_frac=0.2)
    out.append(("nonfinite-both", col, yn, fn))
    out.append(("constant", np.full(30_000, 3.0, dtype=np.float32), y, f))
    out.append(("correlated", (2.0 * y + 0.01 * rng.standard_normal(30_000)).astype(np.float32), y, f))
    return out


@pytest.mark.parametrize("name,col,y,fin", _cases(), ids=[c[0] for c in _cases()])
def test_kernel_equals_the_cupy_chain(name, col, y, fin):
    """Same value to 1e-11 relative (the degenerate cases return exactly 0.0 on both sides); the correlated case is near 1, so the check is not vacuous."""
    require_cuda_device()
    d = cp.asarray(col)
    got = mod.abs_corr_finite_gpu(d, y, fin, 8)
    want = mod._abs_corr_finite_cupy(d, y, fin, 8)
    tolerance = 0.0 if want == 0.0 else 1e-11 * max(1.0, abs(want))
    assert abs(got - want) <= tolerance, (got, want)
    assert name != "correlated" or want > 0.99


def test_kernel_is_faster_per_call_than_the_chain():
    """Interleaved medians over 32 columns of 40000 rows: the kernel path costs under half of the cupy chain per call."""
    require_cuda_device()
    rng = np.random.default_rng(3)
    y, f = _target(40_000)
    cols = [cp.asarray(rng.standard_normal(40_000).astype(np.float32)) for _ in range(32)]
    for c in cols[:3]:
        mod.abs_corr_finite_gpu(c, y, f, 8)
        mod._abs_corr_finite_cupy(c, y, f, 8)
    new, old = [], []
    for _ in range(5):
        t = time.perf_counter()
        for c in cols:
            mod.abs_corr_finite_gpu(c, y, f, 8)
        new.append((time.perf_counter() - t) / len(cols))
        t = time.perf_counter()
        for c in cols:
            mod._abs_corr_finite_cupy(c, y, f, 8)
        old.append((time.perf_counter() - t) / len(cols))
    assert float(np.median(new)) < 0.5 * float(np.median(old)), (np.median(new), np.median(old))


@pytest.mark.parametrize("dt", [np.float32, np.float64])
def test_batch_row_equals_the_single_column_call(dt):
    """Every row of a (K, n) batch gets exactly the value of the single-column call on that row (plain, constant, non-finite, correlated and near-constant columns together)."""
    rng = np.random.default_rng(7)
    n = 30_001
    y, f = _target(n, nan_frac=0.1)
    rows = [rng.standard_normal(n), np.full(n, 2.0), 3.0 * y + 0.1 * rng.standard_normal(n), 1e7 + rng.standard_normal(n), rng.standard_normal(n)]
    rows[4][rng.random(n) < 0.2] = np.nan
    batch = cp.asarray(np.stack(rows).astype(dt))
    got = mod.abs_corr_finite_batch_gpu(batch, y, f)
    want = [mod.abs_corr_finite_gpu(batch[i], y, f) for i in range(len(rows))]
    assert got == want, "batch and single-column values differ"
    assert got[1] == 0.0 and got[2] > 0.9 and 0.0 < got[0] < 0.1, "the batch must hold both degenerate and informative columns for the comparison to mean anything"


def test_flips_uses_one_batched_call_and_falls_back_per_config():
    """``_score_one_pair_flips`` asks the resolver for every config at once; when the batched capability declines (``None``) the per-config loop fills the same map."""
    from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_score_helpers import _score_one_pair_flips

    calls = {"many": 0, "one": 0}

    def resolve(_i):
        """A resolver whose host column is absent."""
        return None

    def one(i, y, yf):
        """The per-column device capability: 0.1 times the column index."""
        calls["one"] += 1
        return 0.1 * i

    resolve.abs_corr = one

    class Safe:
        """A host |corr| helper that carries the target pair the device capability reads."""

        target = (np.ones(4), np.ones(4, dtype=bool))

        def __call__(self, v):
            """The host fallback value (never reached in this test)."""
            return -1.0

    perf = {("a", "b", 2): 0, ("a", "b", 5): 0, ("a", "b", 7): 0}

    def many(cols, y, yf):
        """The batched capability: ten times the column index, counted."""
        calls["many"] += 1
        return [10.0 * c for c in cols]

    resolve.abs_corr_many = many
    m = {}
    _score_one_pair_flips(perf, resolve, Safe(), m)
    assert m == {("a", "b", 2): 20.0, ("a", "b", 5): 50.0, ("a", "b", 7): 70.0} and calls == {"many": 1, "one": 0}
    resolve.abs_corr_many = lambda cols, y, yf: None
    m2 = {}
    _score_one_pair_flips(perf, resolve, Safe(), m2)
    assert m2 == {("a", "b", 2): pytest.approx(0.2), ("a", "b", 5): pytest.approx(0.5), ("a", "b", 7): pytest.approx(0.7)} and calls["one"] == 3
