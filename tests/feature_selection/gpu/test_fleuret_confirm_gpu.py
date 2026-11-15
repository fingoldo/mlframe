"""The device Fleuret recheck returns ``(nfailed, nchecked)`` equal to ``get_fleuret_criteria_confidence`` for every candidate of a window, over random frames and gains."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from mlframe.feature_selection.filters import _fleuret_confirm_gpu as mod
from mlframe.feature_selection.filters._confirm_predictor_helpers import _compact_confidence_inputs
from mlframe.feature_selection.filters._numba_utils import get_fleuret_criteria_confidence
from tests.feature_selection.gpu._gpu_support import has_device

pytestmark = pytest.mark.gpu


def _frame(n, nb_list, seed):
    """A code matrix: the target depends on the first two columns, the others are noise of the declared bin counts."""
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, b, n) for b in nb_list]
    y = ((cols[0] + cols[1] + rng.integers(0, 2, n)) % 3).astype(np.int64)
    data = np.column_stack([*cols, y]).astype(np.int8)
    return data, np.asarray([*nb_list, 3], dtype=np.int64)


def _host(data, nbins, x, y, selected, nperm, gain, max_failed, nexisting, extra_x, seed, order=1, use_su=False, use_mm=False, use_jmim=False, discount=False, perm_offset=0):
    """The production serial kernel on the compact inputs, as ``fleuret_conditional_recheck`` calls it (``x`` is a column or a tuple of columns)."""
    xt = (x,) if isinstance(x, int) else tuple(x)
    cd, cnb, cx, cy, csel, cent = _compact_confidence_inputs(data.copy(), nbins, xt, (y,), selected, None)
    return get_fleuret_criteria_confidence(
        data_copy=cd, factors_nbins=cnb, x=cx, y=cy, selected_vars=csel, bootstrapped_gain=float(gain), npermutations=nperm, max_failed=max_failed, nexisting=nexisting,
        mrmr_relevance_algo="fleuret", mrmr_redundancy_algo="fleuret", max_veteranes_interactions_order=order, cached_cond_MIs=cent[1], entropy_cache=cent[0],
        extra_x_shuffling=extra_x, base_seed=np.uint64(seed), perm_offset=perm_offset, use_su=use_su, use_mm=use_mm, use_jmim=use_jmim, jmim_discount_only=discount,
    )


@pytest.mark.parametrize("extra_x", [True, False])
@pytest.mark.parametrize("nexisting", [0, 1, 2])
def test_batch_counts_equal_the_host_kernel(extra_x, nexisting):
    """Four candidates, three selected variables, gains spread around the scale of the statistic: every candidate's counts equal the host kernel's."""
    if not has_device(cp):
        pytest.skip("no usable device")
    data, nbins = _frame(20_000, [6, 5, 7, 4, 8, 6, 9], seed=nexisting + 3)
    y_col = data.shape[1] - 1
    xs, selected = [3, 4, 5, 6], [0, 1, 2]
    for gain in (0.0005, 0.003, 0.02):
        seeds = [123456789 + 17 * i for i in range(len(xs))]
        got = mod.fleuret_confirm_batch(data, nbins, [(x,) for x in xs], y_col, selected, 12, [gain] * len(xs), 3, [nexisting] * len(xs), extra_x, seeds, 1000)
        assert got is not None
        for i, x in enumerate(xs):
            want = _host(data, nbins, x, y_col, selected, 12, gain, 3, nexisting, extra_x, seeds[i])
            assert got[i] == tuple(want), (gain, x, got[i], want)


def test_counts_span_pass_and_fail_so_the_comparison_means_something():
    """Across the gains of the test above both outcomes occur (some candidates fail every permutation, some none), otherwise equal counts would prove nothing."""
    if not has_device(cp):
        pytest.skip("no usable device")
    data, nbins = _frame(20_000, [6, 5, 7, 4, 8, 6, 9], seed=3)
    y_col = data.shape[1] - 1
    seen = set()
    for gain in (0.0, 0.0005, 0.003, 0.02, 0.5):
        got = mod.fleuret_confirm_batch(data, nbins, [(3,), (4,)], y_col, [0, 1, 2], 12, [gain, gain], 3, [1, 1], True, [11, 12], 1000)
        seen.update(g[0] for g in got)
    assert 0 in seen and max(seen) >= 3, seen


def test_a_high_cardinality_selected_variable_contributes_zero_like_the_host():
    """A selected variable over the confirmation cap (50 bins) is skipped on the device exactly as ``evaluate_gain`` does: its term is 0.0, which then bounds the minimum."""
    if not has_device(cp):
        pytest.skip("no usable device")
    data, nbins = _frame(20_000, [6, 5, 60, 4], seed=9)
    y_col = data.shape[1] - 1
    for gain in (-0.5, 0.0, 0.001):
        got = mod.fleuret_confirm_batch(data, nbins, [(3,)], y_col, [0, 1, 2], 10, [gain], 3, [1], True, [77])
        assert got is not None
        assert got[0] == tuple(_host(data, nbins, 3, y_col, [0, 1, 2], 10, gain, 3, 1, True, 77)), gain


def test_unsupported_requests_decline():
    """No selected variable, a joint table over the shared-memory budget and a code outside its bin count return None so the host kernel runs."""
    data, nbins = _frame(5_000, [6, 5, 7, 4], seed=1)
    y_col = data.shape[1] - 1
    assert mod.fleuret_confirm_batch(data, nbins, [(3,)], y_col, [], 5, [0.1], 3, [0], True, [1], 1000) is None
    bad = data.copy()
    bad[0, 3] = 99
    assert mod.fleuret_confirm_batch(bad, nbins, [(3,)], y_col, [0], 5, [0.1], 3, [0], True, [1], 1000) is None


def test_screen_selection_is_the_same_on_the_device_and_on_the_host(monkeypatch):
    """A real ``MRMR.fit`` selects the same features whether the Fleuret rechecks of the prefetch window run on the device or on the host kernels, and the device really ran."""
    if not has_device(cp):
        pytest.skip("no usable device")
    import pandas as pd

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(4)
    n = 24_000
    X = pd.DataFrame(rng.standard_normal((n, 14)).astype(np.float32), columns=[f"c{i}" for i in range(14)])
    y = (X["c0"].to_numpy() + 0.8 * X["c1"].to_numpy() - 0.6 * X["c2"].to_numpy() + 0.5 * rng.standard_normal(n) > 0).astype(np.int64)
    calls = []
    real = mod.fleuret_confirm_batch
    monkeypatch.setattr(mod, "fleuret_confirm_batch", lambda *a, **k: calls.append(1) or real(*a, **k))
    chosen = {}
    for label, on_device in (("device", True), ("host", False)):
        monkeypatch.setattr(mod, "fleuret_confirm_on_device", lambda n_rows, on=on_device: on)
        MRMR._FIT_CACHE.clear()
        sel = MRMR(verbose=0, random_seed=0, n_jobs=1, fe_max_steps=0, full_npermutations=3).fit(X, y)
        chosen[label] = list(sel.get_feature_names_out())
    assert chosen["device"] == chosen["host"], chosen
    assert calls, "the device recheck never ran, the comparison proves nothing"


def _frame_small(n=12_000, seed=5):
    """Narrow bin counts so the pair candidates and the order-2 conditioning sets stay inside the shared-memory table."""
    return _frame(n, [4, 4, 5, 3, 4, 5, 3], seed)


@pytest.mark.parametrize("statistic,flags", [("cmi", {}), ("mm", {"use_mm": True}), ("su", {"use_su": True}), ("jmim", {"use_jmim": True}), ("jmim", {"use_jmim": True, "discount": True})])
def test_every_statistic_equals_the_host_kernel(statistic, flags):
    """The Miller-Madow, symmetric-uncertainty and JMIM variants, each from the same nine device statistics, give the host kernel's counts."""
    if not has_device(cp):
        pytest.skip("no usable device")
    data, nbins = _frame_small()
    y_col = data.shape[1] - 1
    xs, selected = [3, 4, 5, 6], [0, 1, 2]
    seen = set()
    for gain in (0.0, 0.0005, 0.004, 0.05):
        seeds = [900 + i for i in range(len(xs))]
        for ne in (0, 2):
            got = mod.fleuret_confirm_batch(
                data, nbins, [(x,) for x in xs], y_col, selected, 12, [gain] * len(xs), 3, [ne] * len(xs), True, seeds, statistic=statistic, jmim_discount_only=bool(flags.get("discount"))
            )
            assert got is not None
            for i, x in enumerate(xs):
                want = _host(data, nbins, x, y_col, selected, 12, gain, 3, ne, True, seeds[i], **flags)
                assert got[i] == tuple(want), (statistic, gain, ne, x, got[i], want)
                seen.add(next(iter(want)))
    assert 0 in seen and max(seen) >= 3, seen


def test_pair_candidates_and_second_order_conditioning_equal_the_host():
    """Candidates of two columns and the conditioning sets of every pair of selected variables (interactions order 2) give the host kernel's counts."""
    if not has_device(cp):
        pytest.skip("no usable device")
    data, nbins = _frame_small()
    y_col = data.shape[1] - 1
    pairs, selected = [(3, 4), (5, 6), (3, 6)], [0, 1, 2]
    seen = set()
    for gain in (0.0, 0.00002, 0.0005, 0.05):
        seeds = [500 + i for i in range(len(pairs))]
        got = mod.fleuret_confirm_batch(data, nbins, pairs, y_col, selected, 12, [gain] * len(pairs), 3, [1] * len(pairs), True, seeds, max_order=2)
        assert got is not None
        for i, xs in enumerate(pairs):
            want = _host(data, nbins, xs, y_col, selected, 12, gain, 3, 1, True, seeds[i], order=2)
            assert got[i] == tuple(want), (gain, xs, got[i], want)
            seen.add(next(iter(want)))
    assert 0 in seen and max(seen) >= 1, seen


def test_worker_segments_equal_the_sum_of_the_parallel_route():
    """With ``worker_loads`` the stop at ``max_failed`` applies per worker and the counts add up as ``get_fleuret_criteria_confidence_parallel`` adds them (each worker its own offset)."""
    if not has_device(cp):
        pytest.skip("no usable device")
    from mlframe.feature_selection.filters._numba_utils import distribute_permutations

    data, nbins = _frame_small()
    y_col = data.shape[1] - 1
    xs, selected, P, workers = [3, 4], [0, 1, 2], 14, 3
    loads = [int(w) for w in distribute_permutations(npermutations=P, n_workers=workers)]
    offs = [0, *np.cumsum(loads)[:-1]]
    for gain in (0.0005, 0.004):
        seeds = [321, 322]
        got = mod.fleuret_confirm_batch(data, nbins, [(x,) for x in xs], y_col, selected, P, [gain] * 2, 2, [1, 1], True, seeds, worker_loads=loads)
        assert got is not None
        for i, x in enumerate(xs):
            total = np.zeros(2, dtype=np.int64)
            for load, off in zip(loads, offs):
                nf, nc = _host(data, nbins, x, y_col, selected, load, gain, 2, 1, True, seeds[i], perm_offset=int(off))
                total += (nf, nc)
            assert got[i] == tuple(int(v) for v in total), (gain, x, got[i], total)
