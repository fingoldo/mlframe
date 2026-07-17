"""Regression tests for the two perf wins:

1. ``_stratified_split_3way`` -- single greedy pass replacing the double
   ``MultilabelStratifiedShuffleSplit`` carve. Must stay a valid, disjoint,
   deterministic, label-balanced multilabel split.
2. ``_maybe_disable_cb_plot`` -- inject ``plot=False`` for CatBoost fits outside
   an interactive notebook (pure-config, no numerics change).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training._split_helpers import _stratified_split, _stratified_split_3way
from mlframe.training._training_loop import _maybe_disable_cb_plot, _in_interactive_notebook


def _make_multilabel(n=4000, k=5, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, k)) < 0.25).astype(int)
    return np.arange(n), y


def test_3way_is_valid_disjoint_partition():
    idx, y = _make_multilabel()
    tr, va, te = _stratified_split_3way(idx, test_size=0.2, val_size=0.2, stratify_y=y, random_state=7)
    allp = np.concatenate([tr, va, te])
    assert len(np.unique(allp)) == len(idx), "must cover every row exactly once"
    assert np.intersect1d(tr, va).size == 0
    assert np.intersect1d(tr, te).size == 0
    assert np.intersect1d(va, te).size == 0


def test_3way_is_deterministic_for_fixed_seed():
    idx, y = _make_multilabel()
    a = _stratified_split_3way(idx, 0.2, 0.2, y, random_state=11)
    b = _stratified_split_3way(idx, 0.2, 0.2, y, random_state=11)
    for x, z in zip(a, b):
        assert np.array_equal(np.sort(x), np.sort(z))


def test_3way_label_balance_comparable_to_two_call_path():
    idx, y = _make_multilabel(n=6000, k=6, seed=3)
    glob = y.mean(0)

    # two-call reference (the old behaviour)
    tr0, te0 = _stratified_split(idx, 0.2, y, random_state=5)
    strat_tr = y[tr0]
    _, va0 = _stratified_split(tr0, 0.25, strat_tr, random_state=5)

    tr, va, te = _stratified_split_3way(idx, 0.2, 0.2, y, random_state=5)

    # New split's per-label deviation must be no worse than a loose bound that
    # the old two-call path also satisfies (both are good stratifiers).
    for fold in (tr, va, te):
        dev = np.abs(y[fold].mean(0) - glob).max()
        assert dev < 0.05, f"label-rate deviation {dev} too large"
    # old path balance for reference (sanity that the bound is meaningful)
    for fold in (te0, va0):
        assert np.abs(y[fold].mean(0) - glob).max() < 0.05


def test_3way_sizes_match_whole_dataset_fractions():
    idx, y = _make_multilabel(n=10000, k=4, seed=1)
    tr, va, te = _stratified_split_3way(idx, 0.2, 0.2, y, random_state=0)
    assert abs(len(te) / len(idx) - 0.2) < 0.02
    assert abs(len(va) / len(idx) - 0.2) < 0.02
    assert abs(len(tr) / len(idx) - 0.6) < 0.02


def test_3way_1d_fallback_is_disjoint_and_stratified():
    rng = np.random.default_rng(2)
    n = 4000
    idx = np.arange(n)
    y = (rng.random(n) < 0.3).astype(int)
    tr, va, te = _stratified_split_3way(idx, 0.2, 0.2, y, random_state=4)
    allp = np.concatenate([tr, va, te])
    assert len(np.unique(allp)) == n
    glob = y.mean()
    for fold in (tr, va, te):
        assert abs(y[fold].mean() - glob) < 0.05


# ---- CatBoost plot=False injection ----


def test_cb_plot_disabled_headless():
    fp: dict = {}
    _maybe_disable_cb_plot("CatBoostClassifier", fp, verbose=False)
    assert fp.get("plot") is False


def test_cb_plot_respects_explicit_user_value():
    fp = {"plot": True}
    _maybe_disable_cb_plot("CatBoostClassifier", fp, verbose=False)
    assert fp["plot"] is True, "must not override an explicit user plot setting"


def test_non_catboost_untouched():
    fp: dict = {}
    _maybe_disable_cb_plot("LGBMClassifier", fp, verbose=False)
    assert "plot" not in fp


def test_interactive_detector_false_under_pytest():
    # pytest runs in a plain interpreter, never a ZMQ Jupyter kernel.
    assert _in_interactive_notebook() is False


@pytest.mark.parametrize("mtype", ["CatBoostClassifier", "CatBoostRegressor"])
def test_cb_smoke_fit_with_plot_disabled(mtype):
    cb = pytest.importorskip("catboost")
    rng = np.random.default_rng(0)
    X = rng.random((300, 6))
    if mtype == "CatBoostClassifier":
        y = (rng.random(300) < 0.5).astype(int)
        model = cb.CatBoostClassifier(iterations=20, verbose=0, allow_writing_files=False)
    else:
        y = rng.random(300)
        model = cb.CatBoostRegressor(iterations=20, verbose=0, allow_writing_files=False)
    fp: dict = {}
    _maybe_disable_cb_plot(mtype, fp, verbose=False)
    model.fit(X, y, **fp)  # must train fine with plot=False
    assert model.predict(X).shape[0] == 300
