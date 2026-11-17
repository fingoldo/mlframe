"""The device coder of the gain statistics gives the codes of ``_fe_gain_stats.quantile_codes`` for every column kind, and the screens that use it keep their verdicts."""

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")
from mlframe.feature_selection.filters import _gain_stats_codes_gpu as mod
from mlframe.feature_selection.filters._fe_gain_stats import quantile_codes
from tests.feature_selection.gpu._gpu_support import has_device

pytestmark = pytest.mark.gpu


def _columns(dt, n=20_001):
    """Continuous, heavily tied, constant, binary-like and non-finite columns of dtype ``dt``."""
    rng = np.random.default_rng(5)
    cols = [rng.standard_normal(n).astype(dt) for _ in range(6)]
    cols.append(np.round(rng.standard_normal(n) * 30).astype(dt))  # ties at many edges
    cols.append(np.full(n, 3.0, dtype=dt))  # every edge equal
    cols.append((rng.random(n) < 0.2).astype(dt))
    nonfinite = rng.standard_normal(n).astype(dt)
    nonfinite[::17] = np.nan
    nonfinite[5], nonfinite[9] = np.inf, -np.inf
    cols.append(nonfinite)
    cols.append(np.exp(4 * rng.standard_normal(n)).astype(dt))  # heavy tail
    return cols


@pytest.mark.parametrize("dt", [np.float32, np.float64])
@pytest.mark.parametrize("bins", [2, 10, 60])
def test_device_codes_equal_the_host_quantile_codes(dt, bins):
    """Every column kind, float32 and float64, a small and a large bin count: the codes are exactly those of ``quantile_codes``."""
    if not has_device(cp):
        pytest.skip("no usable device")
    cols = _columns(dt)
    got = mod.quantile_codes_columns_device(cols, bins)
    assert got is not None
    want = np.stack([quantile_codes(c, bins) for c in cols])
    assert got[0].dtype == np.int64 and np.array_equal(got[0], want), f"{int((got[0] != want).sum())} codes differ"


def test_two_bin_counts_come_from_one_call_and_match_separate_calls():
    """A list of bin counts returns one matrix per count, equal to the single-count calls."""
    if not has_device(cp):
        pytest.skip("no usable device")
    cols = _columns(np.float32)
    both = mod.quantile_codes_columns_device(cols, [10, 60])
    assert len(both) == 2
    same = [np.array_equal(m, np.stack([quantile_codes(c, b) for c in cols])) for b, m in zip((10, 60), both)]
    assert len(same) == 2 and all(same)


def test_edges_are_bit_identical_to_np_quantile():
    """The sorted-block edges are numpy's own numbers, not an approximation: equal to ``np.quantile`` of every column."""
    if not has_device(cp):
        pytest.skip("no usable device")
    rng = np.random.default_rng(1)
    for dt in (np.float32, np.float64):
        x = rng.standard_normal((7, 4001)).astype(dt)
        for bins in (3, 10, 60):
            edges = cp.asnumpy(mod._numpy_edges(cp, cp.sort(cp.asarray(x), axis=1), bins))
            want = np.stack([np.quantile(r, np.linspace(0.0, 1.0, bins + 1)[1:-1]) for r in x])
            assert np.array_equal(edges, want), (dt.__name__, bins)


def test_unsupported_requests_decline_and_the_dispatcher_falls_back():
    """One bin, more bins than a byte holds and an empty list decline on the device; ``quantile_codes_columns`` then returns the host codes."""
    cols = _columns(np.float32, 3000)[:3]
    assert mod.quantile_codes_columns_device(cols, 1) is None
    assert mod.quantile_codes_columns_device(cols, 300) is None
    assert mod.quantile_codes_columns_device([], 10) is None
    got = mod.quantile_codes_columns(cols, [1])
    assert np.array_equal(got[0], np.stack([quantile_codes(c, 1) for c in cols]))


def test_dispatcher_uses_the_host_when_the_tuner_says_host(monkeypatch):
    """With the tuner answering 'host' no device call is made and the codes are the host form."""
    cols = _columns(np.float32, 3000)[:4]
    monkeypatch.setattr(mod, "quantile_codes_on_device", lambda n_rows: False)
    monkeypatch.setattr(mod, "quantile_codes_columns_device", lambda *a, **k: pytest.fail("device called"))
    got = mod.quantile_codes_columns(cols, 10)
    assert np.array_equal(got[0], np.stack([quantile_codes(c, 10) for c in cols]))


def test_screens_keep_their_verdicts_whether_the_device_or_the_host_codes_them(monkeypatch):
    """``target_has_structure`` and ``pair_interaction_screen`` return the same answers and the same pair statistics on the device and on the host coder."""
    if not has_device(cp):
        pytest.skip("no usable device")
    from mlframe.feature_selection.filters._pair_residual_screen import pair_interaction_screen
    from mlframe.feature_selection.filters._target_mix_gate import target_has_structure

    rng = np.random.default_rng(2)
    n = 6000
    X = pd.DataFrame(rng.standard_normal((n, 12)).astype(np.float32), columns=[f"c{i}" for i in range(12)])
    y_pair = X["c1"].to_numpy() * X["c4"].to_numpy() + 0.2 * rng.standard_normal(n)
    y_noise = rng.standard_normal(n)
    answers = {}
    for label, forced in (("device", True), ("host", False)):
        monkeypatch.setattr(mod, "quantile_codes_on_device", lambda n_rows, f=forced: f)
        answers[label] = (target_has_structure(X, y_pair), target_has_structure(X, y_noise), pair_interaction_screen(X, y_pair))
    assert answers["device"][0] == answers["host"][0] and answers["device"][1] == answers["host"][1]
    assert answers["device"][2] == answers["host"][2], "pair statistics must be identical"
    assert answers["device"][0] is True, "the pair target must be found, otherwise the comparison proves nothing"


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("dt", [np.float32, np.float64])
def test_device_main_effect_statistics_equal_the_host_loop(dt, seed):
    """The held-out-gain t statistic of every column computed on the card equals the host loop (to 1e-9 relative), for continuous and heavily tied columns; a signal column is the strongest."""
    if not has_device(cp):
        pytest.skip("no usable device")
    from mlframe.feature_selection.filters._pair_residual_screen import main_effect_bins

    rng = np.random.default_rng(seed)
    n = 30_000
    cols = [rng.standard_normal(n).astype(dt) for _ in range(30)]
    cols[3] = np.round(cols[3] * 30).astype(dt)
    t = rng.standard_normal(n) + 0.5 * cols[7].astype(np.float64) ** 2
    even = np.arange(n) % 2 == 0
    base = float(t[even].mean())
    bins = main_effect_bins(int(even.sum()))
    got = mod.main_effect_z_device(cols, bins, t, even, base)
    assert got is not None

    def host(c):
        """The loop body of ``target_has_structure`` before the device statistic."""
        q = quantile_codes(c, bins)
        cnt = np.bincount(q[even], minlength=bins)
        mean = np.where(cnt > 0, np.bincount(q[even], weights=t[even], minlength=bins) / np.maximum(cnt, 1), base)
        d = (t[~even] - base) ** 2 - (t[~even] - mean[q[~even]]) ** 2
        sd = float(d.std())
        return float(d.mean()) / (sd / np.sqrt(len(d))) if sd > 0.0 else 0.0

    want = np.array([host(c) for c in cols])
    assert np.allclose(got, want, rtol=1e-9, atol=1e-9), float(np.abs(got - want).max())
    assert int(np.argmax(got)) == int(np.argmax(want)) == 7


def test_main_effect_statistics_decline_on_non_finite_columns_and_small_inputs():
    """A column with a non-finite value, one bin, and a parity with fewer than two rows send the caller to its host loop."""
    cols = _columns(np.float32, 3000)
    even = np.arange(3000) % 2 == 0
    t = np.ones(3000)
    assert mod.main_effect_z_device(cols, 10, t, even, 1.0) is None  # the last-but-one column holds NaN
    assert mod.main_effect_z_device(cols[:3], 1, t, even, 1.0) is None
    assert mod.main_effect_z_device(cols[:3], 10, t, np.zeros(3000, dtype=bool), 1.0) is None
