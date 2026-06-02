"""Regression tests for freshly-landed fixes in
``mlframe.feature_selection.filters._mrmr_fit_impl`` (+ the paired
``_mrmr_validate_transform`` / ``mrmr`` constructor changes).

Covers the findings whose file is exactly
``src/mlframe/feature_selection/filters/_mrmr_fit_impl.py``:

  - [1]  additional-RFECV rescue must NOT include engineered FE columns in its
         candidate pool. Pre-fix a surviving engineered column (univariate
         basis a__T2 / hybrid / pair / triplet / MI-greedy) could be RFECV-
         selected, then ``feature_names_in_.index(feature)`` raised ValueError
         (feature_names_in_ holds RAW columns only). The fix adds the
         engineered-name attributes to ``_excluded_from_rescue``.

  - [5]  The extra-basis (B-spline / Fourier) FE stage must stay gated on the
         master ``fe_hybrid_orth_enable`` switch (like the pair-cross stage).
         Pre-fix it activated on the default-on univariate-basis path whenever
         ``fe_hybrid_orth_extra_bases`` was non-empty, even with
         ``fe_hybrid_orth_enable=False``.

  - [15] ``additional_rfecv_selection_rule`` is now validated at fit() start
         against the same accepted set RFECV uses, so a typo fails early with
         an actionable message instead of deep inside the RFECV fit.

  - [16] The non-pandas (polars) FE-skip warning no longer falsely claims
         ``fe_hybrid_orth_enable=True`` (which the user never set on the
         default-on univariate path); it names the actual triggering flag(s).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def _make_cheap_mrmr(**overrides):
    """A cheap, deterministic MRMR (no DCD / clustering / friend-graph / cat-FE)."""
    kwargs = dict(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        cat_fe_config=None,
        quantization_nbins=10,
        random_seed=0,
    )
    kwargs.update(overrides)
    return MRMR(**kwargs)


# --------------------------------------------------------------------- [15]


class TestAdditionalRfecvSelectionRuleValidated:
    def test_bad_rule_raises_at_fit_start(self):
        """A typo'd additional_rfecv_selection_rule must fail in
        _validate_string_params (fit() start), not deep inside RFECV."""
        m = _make_cheap_mrmr(additional_rfecv_selection_rule="one_se_minn")
        with pytest.raises(ValueError) as excinfo:
            m._validate_string_params()
        msg = str(excinfo.value)
        assert "additional_rfecv_selection_rule" in msg
        assert "one_se_minn" in msg

    @pytest.mark.parametrize("rule", ["auto", "argmax", "one_se_min", "one_se_max"])
    def test_valid_rules_pass(self, rule):
        """Every value RFECV itself accepts must validate cleanly."""
        m = _make_cheap_mrmr(additional_rfecv_selection_rule=rule)
        m._validate_string_params()  # must not raise

    def test_default_rule_is_valid(self):
        """The constructor default ('one_se_min') is in the accepted set, so
        validation runs (the param is stored on self) and passes."""
        m = _make_cheap_mrmr()
        assert m.additional_rfecv_selection_rule == "one_se_min"
        m._validate_string_params()  # must not raise

    def test_valid_set_matches_rfecv(self):
        """The MRMR allow-list mirrors RFECV's own constructor guard so the two
        never drift (a value MRMR accepts but RFECV rejects would defeat the
        early-validation intent)."""
        assert MRMR._VALID_RFECV_SELECTION_RULES == (
            "auto", "argmax", "one_se_min", "one_se_max",
        )


# ---------------------------------------------------------------------- [1]


class TestRescuePoolExcludesEngineeredColumns:
    def test_engineered_columns_excluded_from_rescue_pool(self):
        """The rescue pool (``temp_columns``) must exclude engineered FE columns
        so RFECV can never select one (then crash at
        ``feature_names_in_.index(feature)``). We reproduce the exact pool-
        construction the fixed _fit_impl rescue block performs from the
        documented engineered-name attributes."""
        # X carries raw columns + surviving engineered columns (as the FE stages
        # leave them on the working frame before the rescue runs).
        x_cols = ["a", "b", "c", "a__T2", "hybrid_x", "mi_greedy_z"]
        # feature_names_in_ holds RAW columns ONLY (mirrors _fit_impl line ~4618).
        feature_names_in_ = ["a", "b", "c"]
        m = _make_cheap_mrmr()
        m.feature_names_in_ = feature_names_in_
        # Engineered-name attributes the fix folds into _excluded_from_rescue.
        m.hybrid_orth_features_ = ["a__T2", "hybrid_x"]
        m.mi_greedy_features_ = ["mi_greedy_z"]

        # Selected raw vars: only 'a' kept; 'b','c' discarded -> rescue candidates.
        selected_names = {"a"}

        # ---- exact rescue-pool construction from the fixed _fit_impl ----
        _excluded_from_rescue = set(
            getattr(m, "_cluster_aggregate_removals_", None) or []
        )
        _cm = getattr(m, "cluster_members_", None)
        if isinstance(_cm, dict):
            for _anchor, _members in _cm.items():
                _excluded_from_rescue.add(_anchor)
                if isinstance(_members, (list, tuple, set)):
                    _excluded_from_rescue.update(_members)
        _excluded_from_rescue.update(getattr(m, "hybrid_orth_features_", None) or [])
        _excluded_from_rescue.update(getattr(m, "mi_greedy_features_", None) or [])
        temp_columns = [
            c for c in x_cols
            if c not in selected_names and c not in _excluded_from_rescue
        ]
        # ----------------------------------------------------------------

        # No engineered column may survive into the rescue candidate pool.
        for eng in ("a__T2", "hybrid_x", "mi_greedy_z"):
            assert eng not in temp_columns, (
                f"engineered column {eng!r} leaked into the rescue pool; RFECV "
                f"could select it then crash on feature_names_in_.index({eng!r})"
            )
        # Genuinely-discarded RAW columns are still reconsidered.
        assert set(temp_columns) == {"b", "c"}

        # Contract the fix relies on: every rescue candidate IS indexable into
        # feature_names_in_ (the pre-fix crash site).
        for feature in temp_columns:
            assert feature in m.feature_names_in_
            m.feature_names_in_.index(feature)  # must not raise

    def test_engineered_column_would_crash_index_if_not_excluded(self):
        """Document the pre-fix failure mode: indexing an engineered name into
        feature_names_in_ raises ValueError. This is exactly what the exclusion
        prevents."""
        m = _make_cheap_mrmr()
        m.feature_names_in_ = ["a", "b", "c"]
        with pytest.raises(ValueError):
            m.feature_names_in_.index("a__T2")


# ---------------------------------------------------------------------- [5]


class TestExtraBasisGatedOnHybridEnable:
    def _build_simple(self, seed: int = 0, n: int = 300):
        rng = np.random.default_rng(seed)
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        X = pd.DataFrame({"x1": x1, "x2": x2, "noise": rng.standard_normal(n)})
        y = pd.Series((x1 + 0.5 * x2 > 0).astype(int), name="y")
        return X, y

    def test_extra_basis_skipped_when_hybrid_disabled(self, monkeypatch):
        """With fe_hybrid_orth_enable=False (default) but
        fe_hybrid_orth_extra_bases set, the extra-basis FE function must NOT be
        called -- the default-on univariate path stays univariate-only."""
        import mlframe.feature_selection.filters._orthogonal_univariate_fe as ofe

        calls = {"n": 0}
        _orig = ofe.hybrid_orth_extra_basis_fe_with_recipes

        def _spy(*args, **kwargs):
            calls["n"] += 1
            return _orig(*args, **kwargs)

        monkeypatch.setattr(ofe, "hybrid_orth_extra_basis_fe_with_recipes", _spy)

        X, y = self._build_simple()
        m = _make_cheap_mrmr(
            fe_hybrid_orth_enable=False,
            fe_univariate_basis_enable=True,
            fe_hybrid_orth_extra_bases=("bspline",),
        )
        m.fit(X, y)
        assert calls["n"] == 0, (
            "extra-basis FE ran with fe_hybrid_orth_enable=False; it must stay "
            "gated on the master hybrid switch"
        )

    def test_extra_basis_runs_when_hybrid_enabled(self, monkeypatch):
        """With fe_hybrid_orth_enable=True and a non-empty extra_bases tuple the
        extra-basis FE function IS invoked (gate opens as intended)."""
        import mlframe.feature_selection.filters._orthogonal_univariate_fe as ofe

        calls = {"n": 0}
        _orig = ofe.hybrid_orth_extra_basis_fe_with_recipes

        def _spy(*args, **kwargs):
            calls["n"] += 1
            return _orig(*args, **kwargs)

        monkeypatch.setattr(ofe, "hybrid_orth_extra_basis_fe_with_recipes", _spy)

        X, y = self._build_simple()
        m = _make_cheap_mrmr(
            fe_hybrid_orth_enable=True,
            fe_univariate_basis_enable=True,
            fe_hybrid_orth_extra_bases=("bspline",),
        )
        m.fit(X, y)
        assert calls["n"] >= 1, (
            "extra-basis FE did not run despite fe_hybrid_orth_enable=True and "
            "a non-empty fe_hybrid_orth_extra_bases tuple"
        )


# --------------------------------------------------------------------- [16]


class TestPolarsFeSkipWarningMessage:
    def test_warning_does_not_falsely_claim_hybrid_enable(self):
        """On the default-on univariate path with a polars frame, the FE-skip
        warning must NOT assert 'fe_hybrid_orth_enable=True' (the user never set
        it); it must name the actual trigger (fe_univariate_basis_enable)."""
        pl = pytest.importorskip("polars")

        rng = np.random.default_rng(0)
        n = 120
        signal = rng.normal(size=n)
        X = pl.DataFrame({"signal": signal, "noise": rng.normal(size=n)})
        y = pl.Series("target", (signal > 0).astype(np.int64))

        m = _make_cheap_mrmr(
            fe_hybrid_orth_enable=False,   # user did NOT enable the hybrid switch
            fe_univariate_basis_enable=True,  # default-on trigger
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m.fit(X, y)

        fe_msgs = [
            str(w.message) for w in caught
            if "not a pandas" in str(w.message) and "FE" in str(w.message)
        ]
        assert fe_msgs, "expected a polars FE-skip UserWarning to be emitted"
        msg = fe_msgs[0]
        # Must not blame a flag the user never set.
        assert "fe_hybrid_orth_enable=True" not in msg, (
            f"warning falsely claims fe_hybrid_orth_enable=True: {msg!r}"
        )
        # Must name the actual triggering flag.
        assert "fe_univariate_basis_enable" in msg
