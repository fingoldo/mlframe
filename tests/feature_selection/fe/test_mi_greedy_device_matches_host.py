"""The device MI-greedy scoring keeps the host's candidates, gives their MI to the selection-equivalence tolerance, and selects the same winners."""

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _mi_greedy_device as dev
from mlframe.feature_selection.filters import _mi_greedy_fe as host
from mlframe.feature_selection.filters._orthogonal_univariate_fe import mi_classif_batch_chunked


def _frame(n, seed):
    """Five float32 columns (one bounded, so trig candidates exist, one with a constant stretch) and a target that needs a product."""
    r = np.random.default_rng(seed)
    X = pd.DataFrame(r.standard_normal((n, 5)).astype(np.float32), columns=[f"x{i}" for i in range(5)])
    X["x3"] = r.uniform(-1, 1, n).astype(np.float32)
    y = ((X["x0"] * X["x1"] + 0.5 * X["x2"] + r.standard_normal(n)) > 0).astype(np.int64).to_numpy()
    return X, y


@pytest.mark.parametrize("n", [3000, 20000])
def test_device_scores_match_the_host_candidates(n):
    """Same kept names in the same order; MI within the quantile-boundary tolerance."""
    X, y = _frame(n, 0)
    cands = list(host.iter_candidates(X, cols=list(X.columns)))
    got = dev.score_candidates_device(X, cands, y, 10)
    assert got is not None
    names, parsed, mi = got
    engineered, parsed_host = host.generate_mi_greedy_features(X, cands)
    assert names == list(engineered.columns) and parsed == parsed_host
    want = np.asarray(mi_classif_batch_chunked(engineered, y, nbins=10), dtype=np.float64)
    np.testing.assert_allclose(mi, want, rtol=2e-2, atol=2e-4)


def test_construct_selects_the_same_winners_on_the_device(monkeypatch):
    """``greedy_mi_fe_construct`` with the device chosen returns the host's winners with identical column values."""
    X, y = _frame(20000, 1)
    monkeypatch.setattr(dev, "use_device", lambda n_rows: False)
    x_host, s_host = host.greedy_mi_fe_construct(X, y, seed_cols_count=5, top_k=5)
    monkeypatch.setattr(dev, "use_device", lambda n_rows: True)
    x_dev, s_dev = host.greedy_mi_fe_construct(X, y, seed_cols_count=5, top_k=5)
    assert list(x_dev.columns) == list(x_host.columns)
    pd.testing.assert_frame_equal(x_dev, x_host)
    assert len(s_dev) == len(s_host)


def test_the_selection_view_marks_the_top_candidates_and_absorbs_rounding_noise():
    """The sweep compares what the greedy step takes: the count and the set of the best ten, with a tie band at the cut so values that differ by rounding noise land on the same side."""
    from mlframe.feature_selection.filters import _mi_greedy_device as D

    rng = np.random.default_rng(0)
    mi = rng.random(40) * 0.01
    base = D._selection_view(mi)
    assert base[0] == 40 and base[1:41].sum() == 10 and base[41:].sum() == 0
    assert (base[1:41].astype(bool) == (mi >= np.sort(mi)[-10])).all()
    noisy = mi + rng.standard_normal(40) * 1e-12
    assert np.array_equal(D._selection_view(noisy), base)
    shifted = mi.copy()
    shifted[np.argmin(mi)] = mi.max() * 2
    assert not np.array_equal(D._selection_view(shifted), base), "a candidate that enters the top ten must change the view"
    assert D._selection_view(mi[:-1])[0] == 39, "a different number of kept candidates must change the view"


def test_the_sweep_runs_at_the_strict_tolerance_and_the_device_agrees_with_the_host():
    """The sweep callables agree on the selection view of the sweep input (this is the equivalence the tuner enforces at 1e-6)."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters import _mi_greedy_device as D

    args = D._sweep_inputs({"n_samples": 30_000})
    host_view, device_view = D._host(*args), D._device(*args)
    assert np.allclose(host_view, device_view, rtol=1e-6, atol=1e-6), (host_view, device_view)
