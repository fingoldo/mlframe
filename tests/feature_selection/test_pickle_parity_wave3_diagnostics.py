"""Regression PIN: the Wave-1/2/3 default-on MRMR diagnostic artifacts survive
joblib/pickle round-trip and the reloaded accessors return IDENTICAL output.

WHY THIS PIN
------------
mlframe has a recurring failure class -- "runtime caches break pickle" -- where a
fit-state artifact holds an unpicklable live object (closure / numba dispatcher /
lambda / compiled fn), OR a fit-state attr gets excluded by ``__getstate__`` /
``__setstate__`` so a default-on accessor AttributeErrors post-load.

This session shipped a cluster of DEFAULT-ON diagnostics on the fitted MRMR:
  * ``explain_selection()``           (commits 205baa86 / 73be9e3b)
  * ``get_fe_rejection_report()`` + ``fe_rejection_ledger_`` (9adbd36e)
  * ``get_unlabeled_recipe_kinds()``  (ad75e9ff)
  * ``selection_stability_report()`` + ``_stability_replay_state_`` (139f4d7e)
  * ``degenerate_columns_`` (cf005964) / ``fe_provenance_`` / ``_fe_recommended_flags_``

These all round-trip cleanly today. This pin makes a future change that drops one
of them from the pickled state (or stuffs a live cache into one) fail loudly
instead of silently breaking a default-on diagnostic after serialization.

CONTRACTS PINNED
----------------
* P1 (unit): each artifact attribute is present AND each accessor works on the
  RELOADED estimator with no AttributeError / exception.
* P2 (biz_value): explain_selection() is byte-identical pre vs post pickle on the
  canonical fixture (the rejection report + unlabeled kinds too).
* P3: ``_stability_replay_state_`` (the binned-matrix slice W3 stored) round-trips
  byte-identical and keeps the pickle small (size guard so a future bloat regresses).
* P4: sklearn.clone() preserves params; the clone refits and exposes the accessors.
"""
from __future__ import annotations

import os
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _mrmr_fe_on():
    from mlframe.feature_selection.filters.mrmr import MRMR
    return MRMR(
        verbose=0, random_seed=0, dcd_enable=False, cluster_aggregate_enable=False,
        build_friend_graph=False, stability_selection_method="classic",
        retain_artifacts=False, n_jobs=1, fe_hybrid_orth_enable=True, fe_auto=True,
    )


def _canonical_frame(n=800, seed=7):
    """Canonical y = a**2/b + log(c)*sin(d) fixture with engineered survivors + gate
    rejections, plus a constant column and a duplicate column so degenerate_columns_
    is populated -- every diagnostic artifact gets non-empty content."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.uniform(0.5, 2.5, n)
    c = rng.uniform(0.5, 5.0, n)
    d = rng.uniform(0.0, 2 * np.pi, n)
    X = pd.DataFrame({
        "a": a, "b": b, "c": c, "d": d,
        "noise_0": rng.standard_normal(n), "noise_1": rng.standard_normal(n),
        "const_col": np.ones(n),  # degenerate: constant
        "dup_a": a,               # degenerate: duplicate of 'a'
    })
    score = a ** 2 / b + np.log(c) * np.sin(d) + 0.3 * rng.standard_normal(n)
    y = pd.Series((score > np.median(score)).astype(int))
    return X, y


@pytest.fixture(scope="module")
def fitted():
    est = _mrmr_fe_on()
    X, y = _canonical_frame()
    est.fit(X, y)
    return est, X, y


_ARTIFACTS = (
    "fe_rejection_ledger_", "fe_provenance_", "_fe_recommended_flags_",
    "degenerate_columns_", "_stability_replay_state_",
)


def _roundtrip(est):
    fd, path = tempfile.mkstemp(suffix=".joblib")
    os.close(fd)
    try:
        joblib.dump(est, path)
        size = os.path.getsize(path)
        return joblib.load(path), size
    finally:
        os.remove(path)


def test_artifacts_populated_then_survive_dump_load(fitted):
    """P1: every Wave-1/2/3 artifact is present pre-pickle AND survives the reload."""
    est, _, _ = fitted
    # populated (the fixture forces non-trivial content).
    assert est.degenerate_columns_, "degenerate_columns_ should be non-empty on this fixture"
    assert est._stability_replay_state_ is not None

    reloaded, _ = _roundtrip(est)
    for attr in _ARTIFACTS:
        assert hasattr(reloaded, attr), f"{attr} dropped by pickle round-trip"


def test_accessors_work_post_load(fitted):
    """P1: each default-on accessor runs on the RELOADED estimator without raising."""
    est, _, _ = fitted
    reloaded, _ = _roundtrip(est)
    # none of these may AttributeError / raise post-load.
    assert isinstance(reloaded.explain_selection(), str)
    assert isinstance(reloaded.get_fe_rejection_report(), str)
    assert isinstance(reloaded.get_unlabeled_recipe_kinds(), dict)
    assert reloaded.selection_stability_report(n_boot=5) is not None


def test_explain_selection_byte_identical_pre_post(fitted):
    """P2 (biz_value): the human-readable narrative is identical pre vs post pickle."""
    est, _, _ = fitted
    pre_explain = est.explain_selection()
    pre_reject = est.get_fe_rejection_report()
    pre_unlabeled = est.get_unlabeled_recipe_kinds()

    reloaded, _ = _roundtrip(est)
    assert reloaded.explain_selection() == pre_explain
    assert reloaded.get_fe_rejection_report() == pre_reject
    assert reloaded.get_unlabeled_recipe_kinds() == pre_unlabeled


def test_stability_replay_state_roundtrips_and_pickle_stays_small(fitted):
    """P3: the binned-matrix replay state is byte-identical post-load, and the whole
    fitted estimator pickles small -- a future bloat (e.g. storing the full matrix)
    trips this guard."""
    est, _, _ = fitted
    reloaded, size = _roundtrip(est)
    s1, s2 = est._stability_replay_state_, reloaded._stability_replay_state_
    assert s1 is not None and s2 is not None
    assert np.array_equal(s1["cand_codes"], s2["cand_codes"])
    assert s1["cand_names"] == s2["cand_names"]
    assert np.array_equal(s1["y_codes"], s2["y_codes"])
    assert np.array_equal(s1["selected_mask"], s2["selected_mask"])
    # the replay state stores a row-subsampled int32 slice, not the full matrix:
    # the fitted estimator must stay well under a few MB on this small fixture.
    assert size < 3_000_000, f"fitted MRMR pickle bloated to {size} bytes"


def test_clone_preserves_params_and_refits(fitted):
    """P4: sklearn.clone() (params only) yields an estimator that refits and exposes
    the diagnostics -- the constructor-param contract survives the clone path too."""
    from sklearn.base import clone
    est, X, y = fitted
    c = clone(est)
    c.fit(X, y)
    assert isinstance(c.explain_selection(), str)
    for attr in _ARTIFACTS:
        assert hasattr(c, attr)
