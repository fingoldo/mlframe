"""Layer 54 biz_value: FE PROVENANCE TRACKING + HUMAN-READABLE REPORT.

WHY THIS LAYER
--------------
Pre-Layer-54 a user asking "which engineered columns landed in support_,
why were they selected, and what's each one's MRMR-gain contribution?"
had to query ~13 scattered fitted attributes (``hybrid_orth_features_``,
``mi_greedy_features_``, ``kfold_te_features_``, etc.), pair them with
the ``_engineered_recipes_`` list, then index into ``mrmr_gains_`` by
selection rank. Layer 54 consolidates that into a single
``fe_provenance_`` DataFrame populated at the end of every ``fit()``
plus a ``get_fe_report()`` renderer for notebooks / logs.

LAYER 54 IMPROVEMENT (pure additive metadata)
---------------------------------------------
1. ``MRMR.fe_provenance_`` -- a pandas DataFrame with columns:
   - ``feature_name``: column name in support_ / engineered output.
   - ``origin``: ``"raw"`` or mechanism label
     (``"hybrid_orth"``/``"mi_greedy"``/``"kfold_te"``/...).
   - ``mechanism_details``: dict-as-string with src_names + mech knobs.
   - ``mrmr_gain``: greedy gain at the moment this feature was added.
   - ``support_rank``: 0-based position in the greedy selection order.

2. ``MRMR.get_fe_report()`` -- single human-readable string with a
   per-origin summary header + the full ``to_string()`` table.

CONTRACTS PINNED
----------------
* C1: ``fe_provenance_`` DataFrame attribute populated after fit() (raw
  data path with no FE switches on).
* C2: One row per support_ feature, raw + engineered.
* C3: Origin labels correctly inferred from the source attr -- raw
  inputs labelled ``"raw"``; engineered features labelled by the
  mechanism that produced them.
* C4: ``mrmr_gain`` values align with ``mrmr_gains_`` at the same
  position when present in the greedy log.
* C5: ``support_rank`` monotonically reflects greedy selection order
  (raw inputs that were ALSO in predictors share the predictor rank;
  engineered names land at their predictor position).
* C6: Pickle and ``sklearn.clone`` semantics: pickle round-trip
  preserves ``fe_provenance_`` content; clone() drops fitted state
  (sklearn convention), the cloned estimator regenerates it on refit.
* C7: ``get_fe_report()`` prints a sensible table with non-empty
  header AND a row per support_ feature.
* C8: Regression on L41 (``cluster_members_``), L48
  (``build_cluster_hierarchy`` import), L52 (roster sweep covers the
  new sibling module), L53 (partial_fit pathway still populates
  ``fe_provenance_`` via the eventual ``self.fit`` call).

NEVER xfail. Pure additive metadata; default-ON (transparency, not
opt-in).
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


def _simple_binary_frame(n: int = 500, seed: int = 0):
    """Six-column binary classification fixture with two informative
    numeric features driving y plus four noise columns. Reused by L53;
    duplicated here to keep the L54 test file self-contained."""
    rng = np.random.default_rng(int(seed))
    x_signal = rng.standard_normal(n)
    x_other = rng.standard_normal(n)
    X = pd.DataFrame({
        "x_signal": x_signal,
        "x_other": x_other,
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
        "noise_2": rng.standard_normal(n),
        "noise_3": rng.standard_normal(n),
    })
    y = pd.Series((x_signal + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


def _multi_signal_frame(n: int = 800, seed: int = 0):
    """Three independent informative columns + three noise columns, so
    greedy MRMR selects at least two predictors and the support_rank
    monotonicity contract has multiple entries to compare."""
    rng = np.random.default_rng(int(seed))
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    X = pd.DataFrame({
        "feat_a": a,
        "feat_b": b,
        "feat_c": c,
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
        "noise_2": rng.standard_normal(n),
    })
    # Sum of three informative columns drives y; each contributes
    # independently so all three earn their predictor slots.
    score = 0.9 * a + 0.9 * b + 0.9 * c + 0.5 * rng.standard_normal(n)
    y = pd.Series((score > 0).astype(int))
    return X, y


def _hybrid_orth_frame(n: int = 1200, seed: int = 99):
    """A frame whose target depends on a PURE polynomial of ``z_main``
    (no linear component) so the hybrid orth FE step at degree 2 has a
    materially better MI vs y than the raw column AND the screening
    surfaces the engineered name in support_. Sized big enough that the
    confirm-predictor permutation gate passes for the engineered He_2
    column on first attempt (n=1200, seed=99 verified empirically)."""
    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal(n)
    X = pd.DataFrame({
        "z_main": z,
        "z_extra": rng.standard_normal(n),
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
    })
    score = z ** 2 - 1.0 + 0.2 * rng.standard_normal(n)
    y = pd.Series((score > 0).astype(int))
    return X, y


def _fast_mrmr(**overrides):
    """A minimal MRMR with FE / DCD / stability OFF so the screening is
    cheap on tiny tests. Matches the L53 style fixture so L54 sits next
    to its predecessor in runtime cost."""
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


def _hybrid_mrmr(**overrides):
    """MRMR with the hybrid-orth FE step enabled so engineered names
    surface in support_. fe_max_steps=1 keeps the run quick."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    defaults = dict(
        verbose=0,
        random_seed=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        fe_max_steps=1,
        fe_hybrid_orth_enable=True,
        fe_hybrid_orth_pair_enable=False,
        fe_hybrid_orth_basis="hermite",
        fe_hybrid_orth_top_k=3,
        fe_hybrid_orth_degrees=(2,),
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


_PROV_COLUMNS = (
    "feature_name", "origin", "mechanism_details",
    "mrmr_gain", "support_rank",
)


# ---------------------------------------------------------------------------
# 1. C1 - fe_provenance_ populated on a vanilla fit
# ---------------------------------------------------------------------------


class TestLayer54_C1_ProvenancePopulated:

    def test_attr_present_after_fit(self):
        X, y = _simple_binary_frame(n=300, seed=1)
        m = _fast_mrmr().fit(X, y)
        assert hasattr(m, "fe_provenance_"), (
            "MRMR.fit() must populate fe_provenance_ on every successful "
            "fit (default-ON Layer 54 contract)."
        )
        assert isinstance(m.fe_provenance_, pd.DataFrame)

    def test_schema_columns_in_order(self):
        X, y = _simple_binary_frame(n=300, seed=2)
        m = _fast_mrmr().fit(X, y)
        assert tuple(m.fe_provenance_.columns) == _PROV_COLUMNS, (
            f"fe_provenance_ column order pinned to {_PROV_COLUMNS!r}; "
            f"got {tuple(m.fe_provenance_.columns)!r}"
        )

    def test_non_empty_for_a_real_fit(self):
        X, y = _simple_binary_frame(n=300, seed=3)
        m = _fast_mrmr().fit(X, y)
        # With informative features in X, screening should pick at least
        # one column. fe_provenance_ has at least one row.
        assert len(m.fe_provenance_) >= 1


# ---------------------------------------------------------------------------
# 2. C2 - one row per support_ feature (raw + engineered)
# ---------------------------------------------------------------------------


class TestLayer54_C2_RowPerSupportFeature:

    def test_row_count_equals_support_plus_engineered(self):
        X, y = _simple_binary_frame(n=400, seed=10)
        m = _fast_mrmr().fit(X, y)
        n_raw = int(np.asarray(m.support_).size)
        n_eng = len(getattr(m, "_engineered_recipes_", []) or [])
        assert len(m.fe_provenance_) == n_raw + n_eng, (
            f"fe_provenance_ rows={len(m.fe_provenance_)}, expected "
            f"n_raw+n_eng={n_raw + n_eng} (support_={n_raw}, "
            f"engineered_recipes={n_eng})"
        )

    def test_feature_names_match_final_output_order(self):
        """Order in fe_provenance_ must mirror transform()'s column order:
        raw support_ names followed by engineered names in
        _engineered_recipes_ order."""
        X, y = _simple_binary_frame(n=400, seed=11)
        m = _fast_mrmr().fit(X, y)
        feature_names_in = list(m.feature_names_in_)
        raw_names = [feature_names_in[i] for i in np.asarray(m.support_).tolist()]
        eng_names = [r.name for r in (m._engineered_recipes_ or [])]
        expected = raw_names + eng_names
        got = list(m.fe_provenance_["feature_name"])
        assert got == expected, (
            f"feature_name order mismatch: got {got!r}, expected {expected!r}"
        )

    def test_engineered_name_with_recipe_included(self):
        """When hybrid-orth FE fires and an engineered name lands in
        support_ with a replayable recipe, it MUST appear in
        fe_provenance_ with a non-``raw`` origin."""
        X, y = _hybrid_orth_frame()
        m = _hybrid_mrmr().fit(X, y)
        assert (m._engineered_recipes_ or []), (
            "Hybrid orth fixture must produce at least one replayable "
            "engineered recipe for this assertion to bite; got 0 recipes. "
            "Strengthen the fixture (n / signal-to-noise) rather than "
            "skipping the test."
        )
        eng_names = {r.name for r in m._engineered_recipes_}
        prov_eng_rows = m.fe_provenance_[
            m.fe_provenance_["feature_name"].isin(eng_names)
        ]
        assert len(prov_eng_rows) == len(eng_names), (
            f"Every engineered name must have a row in fe_provenance_; "
            f"missing: {eng_names - set(prov_eng_rows['feature_name'])!r}"
        )
        # None of the engineered rows should be labelled "raw".
        assert not (prov_eng_rows["origin"] == "raw").any(), (
            f"Engineered rows must not carry origin=='raw'; "
            f"prov_eng_rows=\n{prov_eng_rows!r}"
        )


# ---------------------------------------------------------------------------
# 3. C3 - origin labels correctly inferred
# ---------------------------------------------------------------------------


class TestLayer54_C3_OriginLabels:

    def test_raw_origin_for_input_columns(self):
        X, y = _simple_binary_frame(n=400, seed=20)
        m = _fast_mrmr().fit(X, y)
        feature_names_in = list(m.feature_names_in_)
        raw_names = {feature_names_in[i] for i in np.asarray(m.support_).tolist()}
        prov_raw_rows = m.fe_provenance_[
            m.fe_provenance_["feature_name"].isin(raw_names)
        ]
        assert (prov_raw_rows["origin"] == "raw").all(), (
            f"Raw input columns must carry origin=='raw'; "
            f"got origins={list(prov_raw_rows['origin'])!r}"
        )

    def test_hybrid_orth_origin_for_orth_engineered(self):
        X, y = _hybrid_orth_frame()
        m = _hybrid_mrmr().fit(X, y)
        assert (m._engineered_recipes_ or []), (
            "Hybrid orth fixture must produce at least one recipe; "
            "strengthen the fixture rather than skipping the test."
        )
        # Each replayable orth recipe must end up labelled hybrid_orth.
        # _RECIPE_KIND_TO_ORIGIN maps orth_univariate / orth_pair_cross /
        # orth_spline / orth_fourier / hermite_pair into the hybrid_orth
        # bucket. The fixture's degree-2 Hermite settles on
        # ``orth_univariate``.
        eng_names = [r.name for r in m._engineered_recipes_
                     if str(getattr(r, "kind", "")).startswith("orth_")
                     or str(getattr(r, "kind", "")) == "hermite_pair"]
        assert eng_names, (
            f"No orth-kind / hermite_pair recipes selected on this "
            f"fixture; kinds={[r.kind for r in m._engineered_recipes_]!r}"
        )
        prov_eng = m.fe_provenance_[
            m.fe_provenance_["feature_name"].isin(eng_names)
        ]
        assert (prov_eng["origin"] == "hybrid_orth").all(), (
            f"orth_* recipes must produce origin=='hybrid_orth'; "
            f"got=\n{prov_eng!r}"
        )

    def test_known_origin_set(self):
        """Sanity check: every origin label in fe_provenance_ is drawn
        from the public ``FE_ORIGIN_LABELS`` constant."""
        from mlframe.feature_selection.filters._mrmr_fe_provenance import (
            FE_ORIGIN_LABELS,
        )
        X, y = _simple_binary_frame(n=400, seed=22)
        m = _fast_mrmr().fit(X, y)
        unknown = set(m.fe_provenance_["origin"]) - set(FE_ORIGIN_LABELS)
        assert not unknown, (
            f"fe_provenance_ contains origin labels outside the public "
            f"FE_ORIGIN_LABELS set: {unknown!r}"
        )


# ---------------------------------------------------------------------------
# 4. C4 - mrmr_gain values align with mrmr_gains_
# ---------------------------------------------------------------------------


class TestLayer54_C4_GainAlignsWithMrmrGains:

    def test_gain_values_match_mrmr_gains_at_predictor_rank(self):
        X, y = _simple_binary_frame(n=400, seed=30)
        m = _fast_mrmr().fit(X, y)
        gains_arr = np.asarray(getattr(m, "mrmr_gains_", []))
        if gains_arr.size == 0:
            pytest.skip(
                "mrmr_gains_ empty (legacy fallback path); the gain-"
                "alignment assertion is moot."
            )
        for _, row in m.fe_provenance_.iterrows():
            rank = int(row["support_rank"])
            if 0 <= rank < gains_arr.size:
                assert np.isfinite(row["mrmr_gain"]), (
                    f"Row for {row['feature_name']!r} has rank={rank} "
                    f"but mrmr_gain is NaN; mrmr_gains_={gains_arr.tolist()!r}"
                )
                np.testing.assert_allclose(
                    row["mrmr_gain"], gains_arr[rank], rtol=1e-9,
                    err_msg=(
                        f"Row for {row['feature_name']!r} disagrees with "
                        f"mrmr_gains_[{rank}]={gains_arr[rank]!r}"
                    ),
                )
            else:
                # Rank=-1 ("name not in greedy log") -> mrmr_gain must
                # be NaN by construction.
                assert not np.isfinite(row["mrmr_gain"]), (
                    f"Row for {row['feature_name']!r} has rank={rank} "
                    f"but a finite mrmr_gain={row['mrmr_gain']!r}"
                )

    def test_gain_dtype_is_float(self):
        X, y = _simple_binary_frame(n=400, seed=31)
        m = _fast_mrmr().fit(X, y)
        assert m.fe_provenance_["mrmr_gain"].dtype.kind == "f", (
            f"mrmr_gain dtype must be float; got "
            f"{m.fe_provenance_['mrmr_gain'].dtype!r}"
        )


# ---------------------------------------------------------------------------
# 5. C5 - support_rank monotonic across the greedy log
# ---------------------------------------------------------------------------


class TestLayer54_C5_SupportRankMonotonic:

    def test_predictor_ranks_strictly_increasing(self):
        """Among rows whose name appears in the greedy predictor log
        (rank >= 0), the ranks must be strictly increasing in the order
        they're emitted into fe_provenance_. Uses a multi-signal fixture
        so the predictor log carries at least two entries."""
        X, y = _multi_signal_frame(n=800, seed=40)
        m = _fast_mrmr().fit(X, y)
        ranks = [int(r) for r in m.fe_provenance_["support_rank"] if int(r) >= 0]
        assert len(ranks) >= 2, (
            f"_multi_signal_frame must produce at least 2 predictor "
            f"rows for the monotonicity assertion to bite; got "
            f"ranks={ranks!r}, fe_provenance_=\n{m.fe_provenance_!r}"
        )
        assert all(a < b for a, b in zip(ranks, ranks[1:])), (
            f"support_rank must be strictly increasing among predictor "
            f"rows; got {ranks!r}"
        )

    def test_rank_dtype_int(self):
        X, y = _simple_binary_frame(n=300, seed=41)
        m = _fast_mrmr().fit(X, y)
        assert m.fe_provenance_["support_rank"].dtype.kind in "iu", (
            f"support_rank dtype must be integer; got "
            f"{m.fe_provenance_['support_rank'].dtype!r}"
        )


# ---------------------------------------------------------------------------
# 6. C6 - pickle + clone discipline
# ---------------------------------------------------------------------------


class TestLayer54_C6_PickleCloneDiscipline:

    def test_pickle_round_trip_preserves_dataframe(self):
        X, y = _simple_binary_frame(n=400, seed=50)
        m = _fast_mrmr().fit(X, y)
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)
        assert hasattr(m2, "fe_provenance_")
        assert isinstance(m2.fe_provenance_, pd.DataFrame)
        pd.testing.assert_frame_equal(
            m.fe_provenance_.reset_index(drop=True),
            m2.fe_provenance_.reset_index(drop=True),
        )

    def test_clone_drops_provenance_but_keeps_params(self):
        """sklearn clone() copies ctor params only; fitted attrs (and
        fe_provenance_ counts as one) must NOT propagate."""
        X, y = _simple_binary_frame(n=300, seed=51)
        m = _fast_mrmr().fit(X, y)
        c = clone(m)
        assert getattr(c, "fe_provenance_", None) is None, (
            "clone() must NOT carry fitted fe_provenance_; sklearn "
            "convention is blank-slate clone."
        )
        # Refit on the clone -> provenance regenerated.
        c.fit(X, y)
        assert isinstance(c.fe_provenance_, pd.DataFrame)
        assert len(c.fe_provenance_) >= 1


# ---------------------------------------------------------------------------
# 7. C7 - get_fe_report renders a sensible table
# ---------------------------------------------------------------------------


class TestLayer54_C7_GetFEReport:

    def test_report_includes_header_and_table(self):
        X, y = _simple_binary_frame(n=400, seed=60)
        m = _fast_mrmr().fit(X, y)
        text = m.get_fe_report()
        assert isinstance(text, str) and text
        assert "MRMR FE provenance:" in text, (
            f"Report must include the canonical header marker; got:\n{text}"
        )
        # Every feature_name in the provenance frame must appear in the
        # rendered table.
        for name in m.fe_provenance_["feature_name"]:
            assert str(name) in text, (
                f"feature_name {name!r} missing from rendered report:\n{text}"
            )

    def test_report_on_unfitted_estimator_does_not_raise(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR()
        text = m.get_fe_report()
        assert isinstance(text, str)
        assert "unfitted" in text.lower() or "empty" in text.lower(), (
            f"Unfitted-estimator report must explain the empty state; "
            f"got:\n{text}"
        )

    def test_report_per_origin_counts(self):
        X, y = _simple_binary_frame(n=400, seed=61)
        m = _fast_mrmr().fit(X, y)
        text = m.get_fe_report()
        # The header counts every distinct origin with a non-zero row
        # tally. ``"raw=N"`` MUST appear because raw inputs always land
        # in the provenance frame for this default-OFF fixture.
        assert "raw=" in text, (
            f"Report header must include the raw= count; got:\n{text}"
        )


# ---------------------------------------------------------------------------
# 8. C8 - regressions on prior layers
# ---------------------------------------------------------------------------


class TestLayer54_C8_Regressions:

    def test_layer41_cluster_members_attr_still_set(self):
        """L41 regression: a plain fit still sets cluster_members_ as
        documented; L54's metadata add does not perturb DCD plumbing."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _simple_binary_frame(n=300, seed=70)
        m = MRMR(dcd_enable=False, verbose=0, random_seed=0).fit(X, y)
        assert hasattr(m, "cluster_members_")
        assert m.cluster_members_ is None

    def test_layer48_cluster_hierarchy_module_importable(self):
        """L48 regression: post-hoc hierarchy helper still resolves."""
        from mlframe.feature_selection.filters._cluster_hierarchy import (
            build_cluster_hierarchy,
        )
        X, _ = _simple_binary_frame(n=200, seed=71)
        h = build_cluster_hierarchy(None, X)
        assert h == {}

    def test_layer52_roster_includes_l54_module(self):
        """L52-style roster discovery: the L54 sibling module imports
        cleanly via importlib (i.e. the post-L52 roster expansion is
        still self-consistent)."""
        mod = importlib.import_module(
            "mlframe.feature_selection.filters._mrmr_fe_provenance"
        )
        assert callable(getattr(mod, "compute_fe_provenance", None))
        assert callable(getattr(mod, "get_fe_report", None))
        assert isinstance(getattr(mod, "FE_ORIGIN_LABELS", None), tuple)

    def test_layer53_partial_fit_populates_provenance(self):
        """L53 regression: partial_fit eventually delegates to self.fit,
        so fe_provenance_ MUST be populated after the first batch (just
        like a plain fit)."""
        m = _fast_mrmr()
        X, y = _simple_binary_frame(n=300, seed=72)
        m.partial_fit(X, y)
        assert hasattr(m, "fe_provenance_")
        assert isinstance(m.fe_provenance_, pd.DataFrame)
        # First call equivalent to fit -> non-empty frame.
        assert len(m.fe_provenance_) >= 1

    def test_legacy_pickle_default_fe_provenance_none(self):
        """__setstate__ default-injection contract: an old pickle that
        was created before Layer 54 lands on an unpickled estimator with
        ``fe_provenance_ = None`` (the default seed). The next fit()
        repopulates it."""
        X, y = _simple_binary_frame(n=300, seed=73)
        m = _fast_mrmr().fit(X, y)
        # Simulate a "legacy" pickle by stripping the new attr from the
        # state dict before re-pickling.
        state = m.__getstate__() if hasattr(m, "__getstate__") else m.__dict__.copy()
        state = {k: v for k, v in state.items()
                 if k not in ("fe_provenance_", "_predictors_log_")}
        from mlframe.feature_selection.filters.mrmr import MRMR
        m2 = MRMR()
        m2.__setstate__(state)
        # __setstate__ default should restore fe_provenance_ -> None and
        # _predictors_log_ -> () so callers don't AttributeError.
        assert m2.fe_provenance_ is None
        assert m2._predictors_log_ == ()
