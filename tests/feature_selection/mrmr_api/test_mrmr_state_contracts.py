"""Fitted-state contracts of MRMR (mrmr_audit_2026-09-14 CORE-3, CORE-4, CORE-5, CORE-7, CORE-9).

* CORE-3: an explicit ``fit`` never reset the ``partial_fit`` buffer, so ``partial_fit(A); fit(B); partial_fit(C)`` refit on A + C.
* CORE-4: ``_fit_sample_weight_`` (an n-length array read by nothing) rode in every pickle.
* CORE-5: ``_stability_replay_state_`` (a subsample x candidates code matrix, hundreds of MB at production shape) rode in every pickle.
* CORE-7: ``HybridOrthConfig`` had no field for ``fe_hybrid_orth_elasticnet_l1_ratio``, the knob that decides lasso vs ridge.
* CORE-9: with ``partial_fit_window`` below ``partial_fit_min_recompute`` the buffer can never reach the threshold, so the refit came late
  on ``window`` rows while the user believed it used ``min_recompute`` rows of history.
"""

from __future__ import annotations

import logging
import pickle  # nosec B403 - local round-trip of an estimator this test just fitted

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def _data(n=200, m=4, seed=0):
    """A small classification frame with signal on the first two columns."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, m))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(np.int32)
    return pd.DataFrame(X, columns=list("abcd")[:m]), y


def _fast(**kw):
    """A light, deterministic selector."""
    base = dict(full_npermutations=5, baseline_npermutations=3, n_jobs=1, verbose=0, fe_fast_search=False, fe_max_steps=0, interactions_max_order=1, random_seed=4)
    base.update(kw)
    return MRMR(**base)


# ---------------------------------------------------------------------------------------------------------------- CORE-3


def test_explicit_fit_restarts_the_partial_fit_stream():
    """After partial_fit(A); fit(B), the next partial_fit(C) must start a fresh stream: A is forgotten."""
    X, y = _data(n=400, seed=1)
    A, B, C = slice(0, 60), slice(60, 140), slice(140, 190)
    MRMR._FIT_CACHE.clear()
    m = _fast(partial_fit_min_recompute=1)
    m.partial_fit(X.iloc[A], y[A])
    m.fit(X.iloc[B], y[B])
    m.partial_fit(X.iloc[C], y[C])
    buffered = len(m._partial_fit_X_buffer_)
    assert buffered != 60 + 50, "the stale partial_fit buffer from before fit() was resumed (A + C)"
    assert buffered == 50


def test_partial_fit_stream_still_accumulates_across_its_own_refits():
    """Control: consecutive partial_fit calls keep accumulating; only an explicit fit() restarts the stream."""
    X, y = _data(n=400, seed=2)
    MRMR._FIT_CACHE.clear()
    m = _fast(partial_fit_min_recompute=1)
    m.partial_fit(X.iloc[:60], y[:60])
    m.partial_fit(X.iloc[60:140], y[60:140])
    assert len(m._partial_fit_X_buffer_) == 140


# ---------------------------------------------------------------------------------------------------------------- CORE-4


def test_pickled_state_does_not_carry_the_sample_weight_array():
    """The per-row sample weights are consumed during fit and read by nothing afterwards; they must not be pickled."""
    X, y = _data(n=2000, seed=3)
    w = np.random.default_rng(3).uniform(0.5, 1.5, size=len(X))
    MRMR._FIT_CACHE.clear()
    m = _fast().fit(X, y, sample_weight=w)
    state = m.__getstate__()
    assert "_fit_sample_weight_" not in state
    row_arrays = [k for k, v in state.items() if isinstance(v, np.ndarray) and v.shape[:1] == (len(X),)]
    assert not row_arrays, f"row-length arrays in the pickled state: {row_arrays}"
    restored = pickle.loads(pickle.dumps(m))  # nosec B301
    pd.testing.assert_frame_equal(restored.transform(X), m.transform(X))


# ---------------------------------------------------------------------------------------------------------------- CORE-5


def test_oversized_replay_state_is_dropped_on_pickle_and_the_report_says_so(monkeypatch, caplog):
    """Above the byte threshold the replay cache is not pickled, a warning names it, and the report on the reloaded model explains why."""
    X, y = _data(n=600, seed=4)
    MRMR._FIT_CACHE.clear()
    m = _fast().fit(X, y)
    assert getattr(m, "_stability_replay_state_", None), "fixture precondition: the fit must store replay state"
    monkeypatch.setenv("MLFRAME_MRMR_PICKLE_REPLAY_STATE_MAX_MB", "0")
    with caplog.at_level(logging.WARNING):
        blob = pickle.dumps(m)
    assert any(r.levelno >= logging.WARNING and "_stability_replay_state_" in r.getMessage() for r in caplog.records)
    restored = pickle.loads(blob)  # nosec B301
    assert not getattr(restored, "_stability_replay_state_", None)
    report = restored.selection_stability_report(as_text=True)
    assert "not persisted" in report, f"the report on a reloaded model hid why it is empty: {report!r}"


def test_small_replay_state_survives_pickle():
    """Control: under the default threshold a small fit's replay state is kept, so the report still works after reload."""
    X, y = _data(n=600, seed=5)
    MRMR._FIT_CACHE.clear()
    m = _fast().fit(X, y)
    restored = pickle.loads(pickle.dumps(m))  # nosec B301
    assert getattr(restored, "_stability_replay_state_", None)
    assert restored.selection_stability_report(as_text=False)


# ---------------------------------------------------------------------------------------------------------------- CORE-7


def test_every_flat_hybrid_orth_param_has_a_config_field():
    """HybridOrthConfig (with its nested scorers) must be able to set every flat fe_hybrid_orth_* constructor parameter."""
    import inspect

    from mlframe.feature_selection.filters.mrmr._mrmr_config_dataclasses import _HYBRID_ORTH_FIELD_MAP, _HYBRID_ORTH_SCORERS_FIELD_MAP

    flat = {p for p in inspect.signature(MRMR.__init__).parameters if p.startswith("fe_hybrid_orth_")}
    covered = set(_HYBRID_ORTH_FIELD_MAP.values()) | set(_HYBRID_ORTH_SCORERS_FIELD_MAP.values())
    assert not sorted(flat - covered), f"flat params with no HybridOrthConfig field: {sorted(flat - covered)}"


def test_hybrid_orth_config_sets_elasticnet_l1_ratio():
    """The config path must reach the knob that decides lasso vs ridge."""
    from mlframe.feature_selection.filters.mrmr._mrmr_config_dataclasses import HybridOrthConfig

    m = MRMR(hybrid_orth_config=HybridOrthConfig(elasticnet_enable=True, elasticnet_l1_ratio=0.9))
    assert m.fe_hybrid_orth_elasticnet_l1_ratio == 0.9


# ---------------------------------------------------------------------------------------------------------------- CORE-9


def test_window_smaller_than_min_recompute_refits_on_the_window_and_warns(caplog):
    """With window=50 the buffer never exceeds 50 rows, so the effective threshold is 50; say so instead of silently waiting."""
    X, y = _data(n=400, seed=6)
    MRMR._FIT_CACHE.clear()
    m = _fast(partial_fit_window=50, partial_fit_min_recompute=1000)
    m.partial_fit(X.iloc[:80], y[:80])
    with caplog.at_level(logging.WARNING):
        m.partial_fit(X.iloc[80:140], y[80:140])
    assert m._partial_fit_n_since_refit_ == 0, "no refit happened although the window can never hold min_recompute rows"
    assert len(m._partial_fit_X_buffer_) == 50
    assert m.partial_fit_min_recompute == 1000, "the constructor parameter must not be mutated"
    assert any(r.levelno >= logging.WARNING and "partial_fit_window" in r.getMessage() for r in caplog.records)
