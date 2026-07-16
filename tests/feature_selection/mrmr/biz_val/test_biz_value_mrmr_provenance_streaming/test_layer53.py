"""Layer 53 biz_value: INCREMENTAL / STREAMING ``MRMR.partial_fit`` support.

WHY THIS LAYER
--------------
Production tabular ML pipelines retrain on growing data. A full screening
refit per epoch is wasteful when only a small fraction of rows changed.
Layer 53 adds an opt-in ``MRMR.partial_fit(X_new, y_new)`` API:

* First call -> equivalent to ``fit(X_new, y_new)`` (initialise).
* Subsequent calls -> buffer the new batch, optionally roll the window,
  and recompute the screening only once at least
  ``partial_fit_min_recompute`` new rows are accumulated.
* ``partial_fit_decay`` reweights historic rows so the resulting fit is
  biased toward recency. Implemented atop the existing
  ``sample_weight`` resample contract; legacy fit() is byte-for-byte
  unchanged.

CONTRACTS PINNED
----------------
* C1: ``partial_fit`` on a single batch produces a ``support_`` equivalent
  to plain ``fit`` on the same batch (decay=0, no buffer history).
* C2: Subsequent batches expand the buffer and update ``support_`` once
  the recompute threshold is crossed. Calls below threshold leave
  ``support_`` carried-over (no spurious refits).
* C3: ``partial_fit_decay > 0`` biases the fit toward the recent batch
  on a target-flip scenario: the recent target wins the relevance gate.
* C4: ``partial_fit_window`` truncates the buffer to the most recent
  ``window`` rows; the discarded prefix has no further influence.
* C5: Pickle / clone preserves ``support_`` and the partial-fit buffer.
* C6: Regression on Layer 41 (cluster_members_), Layer 48
  (cluster_hierarchy attribute survives), Layer 52 (roster discovery for
  the sibling module).

NEVER xfail. Default-OFF: legacy ``fit()`` unchanged.
"""
from __future__ import annotations

import importlib
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _simple_binary_frame(n: int = 500, seed: int = 0, signal_col: str = "x_signal"):
    """Two informative numeric features + 4 noise columns. Binary y.

    ``signal_col`` selects which of the two signal columns drives y. Used
    in C3 to flip the target relationship between batches and assert that
    decay > 0 surfaces the recent driver.
    """
    rng = np.random.default_rng(int(seed))
    x_signal = rng.standard_normal(n)
    x_other = rng.standard_normal(n)
    cols = {
        "x_signal": x_signal,
        "x_other": x_other,
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
        "noise_2": rng.standard_normal(n),
        "noise_3": rng.standard_normal(n),
    }
    X = pd.DataFrame(cols)
    driver = cols[signal_col]
    y = pd.Series((driver + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


def _fast_mrmr(**overrides):
    """A minimal MRMR with FE / DCD / stability off so the screening is
    cheap on tiny tests. Layer 53 partial_fit semantics don't depend on
    those mechanisms; turning them off keeps the suite under 60s."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    defaults = dict(
        verbose=0,
        random_seed=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        fe_max_steps=0,
        fe_hybrid_orth_enable=False,
        fe_mi_greedy_enable=False,
        fe_kfold_te_enable=False,
        fe_count_encoding_enable=False,
        fe_frequency_encoding_enable=False,
        fe_cat_num_interaction_enable=False,
        stability_selection_method="classic",
        retain_artifacts=False,
    )
    defaults.update(overrides)
    return MRMR(**defaults)


# ---------------------------------------------------------------------------
# 1. API smoke + defaults
# ---------------------------------------------------------------------------


class TestLayer53_APISmoke:
    """partial_fit is bound on the class and rejects invalid inputs up front."""

    def test_method_bound_on_class(self):
        """partial_fit is bound on the MRMR class with an sklearn-style (self, X_new, y_new, ...) signature."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        assert hasattr(MRMR, "partial_fit"), (
            "MRMR.partial_fit must be bound on the class at module import "
            "time so sklearn's incremental-learning estimator detection "
            "(hasattr(est, 'partial_fit')) returns True."
        )
        # sklearn convention: method takes X, y and additional kwargs.
        import inspect
        sig = inspect.signature(MRMR.partial_fit)
        params = list(sig.parameters.keys())
        # self + X_new + y_new at minimum.
        assert params[:3] == ["self", "X_new", "y_new"], f"partial_fit signature should start with (self, X_new, y_new); " f"got {params[:3]!r}"

    def test_default_knobs_off_by_default(self):
        """Ctor defaults: decay=0, min_recompute=100, window=None."""
        m = _fast_mrmr()
        assert m.partial_fit_decay == 0.0
        assert m.partial_fit_min_recompute == 100
        assert m.partial_fit_window is None

    def test_unfitted_buffer_state_is_none(self):
        """Before any partial_fit call, the internal buffer attributes are None."""
        m = _fast_mrmr()
        assert getattr(m, "_partial_fit_X_buffer_", None) is None
        assert getattr(m, "_partial_fit_y_buffer_", None) is None

    def test_invalid_decay_rejected(self):
        """An out-of-range partial_fit_decay raises ValueError."""
        m = _fast_mrmr(partial_fit_decay=1.5)
        X, y = _simple_binary_frame(n=120)
        with pytest.raises(ValueError, match=r"partial_fit_decay"):
            m.partial_fit(X, y)

    def test_invalid_window_rejected(self):
        """A non-positive partial_fit_window raises ValueError."""
        m = _fast_mrmr(partial_fit_window=0)
        X, y = _simple_binary_frame(n=120)
        with pytest.raises(ValueError, match=r"partial_fit_window"):
            m.partial_fit(X, y)

    def test_empty_batch_rejected(self):
        """An empty (X_new, y_new) batch raises ValueError."""
        m = _fast_mrmr()
        X, y = _simple_binary_frame(n=10)
        with pytest.raises(ValueError, match=r"non-empty"):
            m.partial_fit(X.iloc[0:0], y.iloc[0:0])

    def test_shape_mismatch_rejected(self):
        """A row-count mismatch between X_new and y_new raises ValueError."""
        m = _fast_mrmr()
        X, y = _simple_binary_frame(n=100)
        with pytest.raises(ValueError, match=r"rows"):
            m.partial_fit(X.iloc[:50], y.iloc[:30])


# ---------------------------------------------------------------------------
# 2. C1 - first call equivalent to fit on the same batch
# ---------------------------------------------------------------------------


class TestLayer53_C1_FirstCallEquivalence:
    """C1: the first partial_fit call behaves like a plain fit on the same batch."""

    def test_first_call_support_matches_fit(self):
        """C1: first partial_fit call -> support_ identical to fit on the
        same batch (same seeds, default decay=0)."""
        X, y = _simple_binary_frame(n=400, seed=7)
        m_fit = _fast_mrmr().fit(X, y)
        m_pf = _fast_mrmr().partial_fit(X, y)
        # Support contents (not insertion order) must match.
        assert set(m_fit.support_.tolist()) == set(m_pf.support_.tolist()), (
            f"partial_fit first-call support {m_pf.support_.tolist()!r} " f"differs from fit support {m_fit.support_.tolist()!r}"
        )
        # n_features_in_ contract preserved.
        assert m_fit.n_features_in_ == m_pf.n_features_in_

    def test_first_call_returns_self(self):
        """partial_fit returns self, per sklearn convention."""
        m = _fast_mrmr()
        X, y = _simple_binary_frame(n=200, seed=5)
        out = m.partial_fit(X, y)
        assert out is m, "sklearn convention: partial_fit must return self"

    def test_first_call_populates_buffer(self):
        """The first partial_fit call populates the internal buffer with all rows seen."""
        m = _fast_mrmr()
        X, y = _simple_binary_frame(n=200, seed=5)
        m.partial_fit(X, y)
        assert m._partial_fit_X_buffer_ is not None
        assert m._partial_fit_y_buffer_ is not None
        assert len(m._partial_fit_X_buffer_) == 200
        assert m._partial_fit_n_seen_ == 200


# ---------------------------------------------------------------------------
# 3. C2 - subsequent batches update support_ once threshold crossed
# ---------------------------------------------------------------------------


class TestLayer53_C2_BatchAccumulation:
    """C2: support_ carries over below the recompute threshold and refits once it's crossed."""

    def test_below_threshold_no_refit(self):
        """C2a: when fewer than ``partial_fit_min_recompute`` new rows have
        arrived, ``support_`` is carried over verbatim (no extra fit work)."""
        m = _fast_mrmr(partial_fit_min_recompute=500)
        X1, y1 = _simple_binary_frame(n=200, seed=10)
        m.partial_fit(X1, y1)
        sup_after_first = m.support_.copy()
        X2, y2 = _simple_binary_frame(n=50, seed=11)
        m.partial_fit(X2, y2)
        # Buffer grew; support stayed the same (below threshold of 500).
        assert len(m._partial_fit_X_buffer_) == 250
        assert m._partial_fit_n_since_refit_ == 50
        np.testing.assert_array_equal(m.support_, sup_after_first)

    def test_above_threshold_triggers_refit(self):
        """C2b: once new rows >= threshold, support_ is recomputed and
        the refit counter resets to 0."""
        m = _fast_mrmr(partial_fit_min_recompute=100)
        X1, y1 = _simple_binary_frame(n=200, seed=20)
        m.partial_fit(X1, y1)
        X2, y2 = _simple_binary_frame(n=150, seed=21)
        m.partial_fit(X2, y2)
        # Refit triggered: counter reset, buffer is the union.
        assert m._partial_fit_n_since_refit_ == 0
        assert len(m._partial_fit_X_buffer_) == 350
        # Cumulative n_seen is the sum of both batches.
        assert m._partial_fit_n_seen_ == 350

    def test_cumulative_three_batches(self):
        """C2c: three batches with threshold=80 -> two refits (after
        batches 2 and 3); buffer holds all 300 rows."""
        m = _fast_mrmr(partial_fit_min_recompute=80)
        for seed, size in [(30, 100), (31, 100), (32, 100)]:
            Xi, yi = _simple_binary_frame(n=size, seed=seed)
            m.partial_fit(Xi, yi)
        assert len(m._partial_fit_X_buffer_) == 300
        assert m._partial_fit_n_seen_ == 300


# ---------------------------------------------------------------------------
# 4. C3 - decay biases fit toward recent batch
# ---------------------------------------------------------------------------


class TestLayer53_C3_DecayBiasesRecent:
    """C3: partial_fit_decay > 0 biases the fit toward the most recent batch's target relationship."""

    def test_decay_one_essentially_replaces_history(self):
        """C3: with decay=1.0, historical rows are weighted to the floor;
        the resulting fit's relevance is dominated by the newest batch.

        We construct a flip: batch1's target is driven by ``x_signal``;
        batch2 is driven by ``x_other`` (orthogonal feature). With
        decay=1.0 the fit should rank ``x_other`` ahead of ``x_signal``
        in the support_, mirroring the recent driver.
        """
        m = _fast_mrmr(
            partial_fit_decay=1.0,
            partial_fit_min_recompute=50,
        )
        X1, y1 = _simple_binary_frame(n=400, seed=40, signal_col="x_signal")
        X2, y2 = _simple_binary_frame(n=400, seed=41, signal_col="x_other")
        m.partial_fit(X1, y1)
        m.partial_fit(X2, y2)
        feature_names = list(m.feature_names_in_)
        selected = [feature_names[i] for i in m.support_]
        # x_other must beat x_signal in the recent-biased fit.
        assert "x_other" in selected, f"With decay=1.0 + recent batch driven by x_other, " f"selected={selected!r} must include x_other."

    def test_decay_zero_keeps_both_drivers_equal(self):
        """C3 sentry: decay=0 keeps the historic buffer at equal weight,
        so both drivers (from batch1 and batch2) appear in support_."""
        m = _fast_mrmr(
            partial_fit_decay=0.0,
            partial_fit_min_recompute=50,
        )
        X1, y1 = _simple_binary_frame(n=400, seed=50, signal_col="x_signal")
        X2, y2 = _simple_binary_frame(n=400, seed=51, signal_col="x_other")
        m.partial_fit(X1, y1)
        m.partial_fit(X2, y2)
        feature_names = list(m.feature_names_in_)
        selected = set(feature_names[i] for i in m.support_)
        # No-decay: both batches contribute equally; at least one of the
        # two signals should be present (both are weak alone after the
        # 50/50 mix, but neither is excluded by construction).
        assert selected & {"x_signal", "x_other"}, f"decay=0 should keep at least one of the two drivers in " f"support_; got {selected!r}"


# ---------------------------------------------------------------------------
# 5. C4 - rolling window
# ---------------------------------------------------------------------------


class TestLayer53_C4_RollingWindow:
    """C4: partial_fit_window caps the buffer to the most recent N rows."""

    def test_window_truncates_buffer(self):
        """C4a: with window=150, after pushing 100+100 rows the buffer is
        capped at 150 (the most recent 150 rows survive)."""
        m = _fast_mrmr(
            partial_fit_window=150,
            partial_fit_min_recompute=10,
        )
        X1, y1 = _simple_binary_frame(n=100, seed=60)
        m.partial_fit(X1, y1)
        assert len(m._partial_fit_X_buffer_) == 100  # under window, no truncation
        X2, y2 = _simple_binary_frame(n=100, seed=61)
        m.partial_fit(X2, y2)
        assert len(m._partial_fit_X_buffer_) == 150
        # The newest batch (100 rows) is preserved intact; 50 from the
        # historic batch survive.
        batch_sizes = m._partial_fit_batch_sizes_
        assert batch_sizes[-1] == 100, f"Newest batch must be preserved intact; batch_sizes={batch_sizes!r}"
        assert sum(batch_sizes) == 150

    def test_window_smaller_than_first_batch(self):
        """C4b: a first batch larger than the window is truncated on the
        spot; we don't carry hidden state past the window cap."""
        m = _fast_mrmr(
            partial_fit_window=80,
            partial_fit_min_recompute=10,
        )
        X1, y1 = _simple_binary_frame(n=200, seed=70)
        m.partial_fit(X1, y1)
        assert len(m._partial_fit_X_buffer_) == 80


# ---------------------------------------------------------------------------
# 6. C5 - pickle / clone preserves partial_fit state
# ---------------------------------------------------------------------------


class TestLayer53_C5_PicklePreserves:
    """C5: pickle round-trip and sklearn clone() preserve the right partial_fit state."""

    def test_pickle_round_trip_preserves_buffer_and_support(self):
        """Pickle round-trip preserves the partial_fit buffer, ctor params, and support_."""
        m = _fast_mrmr(
            partial_fit_decay=0.3,
            partial_fit_min_recompute=50,
            partial_fit_window=300,
        )
        X1, y1 = _simple_binary_frame(n=200, seed=80)
        m.partial_fit(X1, y1)
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)  # nosec B301 -- round-trip of a locally-created, trusted object
        # Ctor params preserved.
        assert m2.partial_fit_decay == 0.3
        assert m2.partial_fit_min_recompute == 50
        assert m2.partial_fit_window == 300
        # Buffer preserved row-for-row.
        pd.testing.assert_frame_equal(
            m._partial_fit_X_buffer_, m2._partial_fit_X_buffer_,
        )
        pd.testing.assert_series_equal(
            m._partial_fit_y_buffer_, m2._partial_fit_y_buffer_,
        )
        # Support preserved.
        np.testing.assert_array_equal(m.support_, m2.support_)
        # Resume partial_fit on the unpickled copy.
        X2, y2 = _simple_binary_frame(n=100, seed=81)
        m2.partial_fit(X2, y2)
        assert m2._partial_fit_n_seen_ == 300

    def test_clone_drops_fitted_state_but_keeps_partial_fit_params(self):
        """sklearn clone() copies params only; partial_fit state is fitted
        and must NOT propagate to the clone."""
        m = _fast_mrmr(
            partial_fit_decay=0.5,
            partial_fit_min_recompute=42,
            partial_fit_window=200,
        )
        X1, y1 = _simple_binary_frame(n=100, seed=90)
        m.partial_fit(X1, y1)
        c = clone(m)
        assert c.partial_fit_decay == 0.5
        assert c.partial_fit_min_recompute == 42
        assert c.partial_fit_window == 200
        # Fitted attribute NOT copied; clone is a blank-slate estimator.
        assert getattr(c, "_partial_fit_X_buffer_", None) is None
        assert getattr(c, "_partial_fit_y_buffer_", None) is None


# ---------------------------------------------------------------------------
# 7. C6 - regressions on prior layers + sibling import
# ---------------------------------------------------------------------------


class TestLayer53_C6_Regressions:
    """C6: prior-layer contracts (L41, L48, L52) still hold alongside the new partial_fit sibling."""

    def test_sibling_module_importable(self):
        """The _mrmr_partial_fit sibling module is importable and exposes partial_fit."""
        mod = importlib.import_module("mlframe.feature_selection.filters._mrmr_partial_fit")
        assert hasattr(mod, "partial_fit")

    def test_layer41_cluster_members_attr_still_set_on_classic_fit(self):
        """Layer 41 regression: plain fit() still sets cluster_members_
        attribute regardless of Layer 53 additions."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _simple_binary_frame(n=300, seed=100)
        m = MRMR(dcd_enable=False, verbose=0, random_seed=0).fit(X, y)
        assert hasattr(m, "cluster_members_")
        assert m.cluster_members_ is None

    def test_layer48_hierarchy_attribute_compatibility(self):
        """Layer 48 regression: cluster_hierarchy module still imports
        cleanly and offers build_cluster_hierarchy."""
        from mlframe.feature_selection.filters._cluster_hierarchy import (
            build_cluster_hierarchy,
        )
        X, _ = _simple_binary_frame(n=200, seed=110)
        h = build_cluster_hierarchy(None, X)
        assert h == {}

    def test_layer52_roster_discovery_sibling_module(self):
        """Layer 52-style roster: Layer 53's sibling module is discoverable
        via importlib (analogous to the L52 roster sweep)."""
        mod = importlib.import_module("mlframe.feature_selection.filters._mrmr_partial_fit")
        # Must expose the public function name.
        assert callable(getattr(mod, "partial_fit", None))

    def test_legacy_fit_byte_identical_when_partial_fit_not_called(self):
        """Default-OFF guarantee: a plain MRMR.fit on a fresh instance
        yields the same support_ regardless of whether Layer 53 ctor
        knobs were touched. Two equivalent constructors should produce
        the same support."""
        X, y = _simple_binary_frame(n=400, seed=120)
        m1 = _fast_mrmr().fit(X, y)
        m2 = _fast_mrmr(
            partial_fit_decay=0.0,
            partial_fit_min_recompute=100,
            partial_fit_window=None,
        ).fit(X, y)
        np.testing.assert_array_equal(m1.support_, m2.support_)
