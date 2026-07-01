"""Layer 55 biz_value: COMPREHENSIVE REGRESSION + DIFF VS L52 BASELINE.

WHY THIS LAYER
--------------
Pure VERIFICATION layer (no new prod features). L52 pinned the L1..L51
roster + composite all-on smoke. L53 (partial_fit) and L54 (FE
provenance / get_fe_report) shipped between then and now. Layer 55
re-verifies the cumulative roster has grown, that the composite all-on
kitchen-sink still fits inside a tighter < 60s budget when partial_fit
is wired into the same path, AND that L54's fe_provenance_ DataFrame
covers BOTH raw and engineered features under the composite all-on
configuration (not just under the L54 hybrid_orth-only fixture).

CONTRACTS PINNED
----------------
C1. Cumulative roster size: at least 54 prior layer biz_value test
    modules are discoverable on disk -- one per shipped layer including
    L6..L54 plus the named modules (``extreme``, ``hard_cases``,
    ``ultra``, ``multiway_synergy``, ``quality_metrics``). Catches a
    silent removal of a layer's biz_value file.

C2. All-on smoke: enable every FE switch + DCD + cluster_aggregate on
    the kitchen-sink, then drive the run through ``partial_fit`` (L53)
    instead of a plain ``fit``. The composite must complete inside a
    60s wall-clock budget on this fixture size and shape.

C3. Provenance integration: after the composite all-on
    ``partial_fit`` run, the L54 ``fe_provenance_`` DataFrame must hold
    BOTH ``origin == 'raw'`` rows and at least one non-``raw`` row
    (engineered features). Guards against a regression where the
    composite fit silently bypasses the provenance population step
    (e.g. partial_fit -> fit but provenance only populated on a
    different code path).

NEVER xfail. If the composite fit raises, under-fits, or fails to
populate fe_provenance_ for both raw + engineered, fix prod / the
fixture -- do not relax the contract.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _kitchen_sink(seed: int = 42, n: int = 2000):
    """Smaller kitchen-sink than L52's (n=2000 vs n=3000) so the < 60s
    composite-fit budget under partial_fit is enforceable on CI. Same
    column schema as L52 so the FE switches see the same dtype mix
    (cat_region / cat_user / numeric / threshold)."""
    rng = np.random.default_rng(seed)
    n_users = 50
    user_ids = np.array([f"U_{i:03d}" for i in range(n_users)])
    user_weights = np.linspace(1.0, 50.0, n_users)
    user_weights = user_weights / user_weights.sum()
    cat_user = rng.choice(user_ids, size=n, p=user_weights)
    regions = [f"R{i:02d}" for i in range(20)]
    hot_regions = set(regions[:3])
    cat_region = rng.choice(regions, size=n)
    hot_mask = np.array([(c in hot_regions) for c in cat_region], dtype=float)
    region_means = dict(
        zip(regions, rng.uniform(20.0, 120.0, size=len(regions)))
    )
    price_mean = np.array([region_means[c] for c in cat_region])
    price = price_mean + rng.normal(0.0, 10.0, size=n)
    counts = pd.Series(cat_user).value_counts()
    log_cnt = np.log1p(pd.Series(cat_user).map(counts).to_numpy().astype(float))
    log_cnt_centered = log_cnt - log_cnt.mean()
    x_num1 = rng.standard_normal(n)
    x_num2 = rng.standard_normal(n)
    x_quad = rng.standard_normal(n)
    x_periodic = rng.uniform(-1.0, 1.0, size=n)
    x_threshold = rng.standard_normal(n)
    noise = rng.standard_normal((n, 3))
    box = ((x_threshold > 0.3) & (x_threshold < 1.2)).astype(float)
    logit = (
        0.5 * x_num1
        + 2.0 * (x_quad ** 2 - 1.0)
        + 2.0 * np.sin(2.0 * np.pi * x_periodic)
        + 2.0 * box
        + 2.0 * hot_mask
        + 0.15 * (price - price_mean)
        + 1.0 * log_cnt_centered
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = pd.Series((rng.random(n) < p).astype(int), name="y")
    X = pd.DataFrame({
        "x_num1": x_num1, "x_num2": x_num2, "x_quad": x_quad,
        "x_periodic": x_periodic, "x_threshold": x_threshold,
        "cat_region": cat_region, "cat_user": cat_user, "price": price,
        "n0": noise[:, 0], "n1": noise[:, 1], "n2": noise[:, 2],
    })
    return X, y


def _all_fe_kwargs():
    """Enable every FE switch on MRMR simultaneously. Mirrors L52."""
    return dict(
        fe_hybrid_orth_enable=True,
        fe_hybrid_orth_pair_enable=True,
        fe_hybrid_orth_basis="hermite",
        fe_hybrid_orth_top_k=6,
        fe_hybrid_orth_extra_bases=("spline", "fourier"),
        fe_hybrid_orth_fourier_freqs=(1.0, 2.0),
        fe_hybrid_orth_spline_knots=5,
        fe_mi_greedy_enable=True,
        fe_mi_greedy_top_k=6,
        fe_mi_greedy_include_unary=True,
        fe_mi_greedy_include_binary=True,
        fe_kfold_te_enable=True,
        fe_kfold_te_cols=("cat_region", "cat_user"),
        fe_kfold_te_folds=5,
        fe_kfold_te_smoothing=10.0,
        fe_count_encoding_enable=True,
        fe_count_encoding_cols=("cat_user",),
        fe_frequency_encoding_enable=True,
        fe_frequency_encoding_cols=("cat_user",),
        fe_cat_num_interaction_enable=True,
        fe_cat_num_interaction_cat_cols=("cat_region",),
        fe_cat_num_interaction_num_cols=("price",),
        fe_cat_num_interaction_folds=5,
        fe_cat_num_interaction_smoothing=10.0,
        fe_missingness_indicator_enable=True,
        fe_missingness_count_enable=True,
        fe_missingness_pattern_enable=True,
        fe_pairwise_ratio_enable=True,
        fe_pairwise_log_ratio_enable=True,
        fe_grouped_delta_enable=True,
        fe_lagged_diff_enable=True,
    )


# ---------------------------------------------------------------------------
# C1: cumulative roster size of prior biz_value layer test modules
# ---------------------------------------------------------------------------


class TestLayer55_CumulativeRosterSize:
    """At least 54 prior layer biz_value test modules must be present on
    disk. Catches a silent prune of a shipped layer's test file.
    """

    def test_at_least_54_prior_layer_test_modules_on_disk(self):
        # This module now lives in a themed subpackage; the roster lives one level up in tests/feature_selection/.
        tests_dir = Path(__file__).parent.parent
        # Every biz_value mrmr test file in the directory except this one.
        pattern = "test_biz_value_mrmr_*.py"
        all_files = sorted(p.name for p in tests_dir.glob(pattern))
        # Layers consolidated into themed subpackages (test_biz_value_mrmr_<theme>/) still count:
        # each themed submodule is a relocated prior-layer biz_value test module.
        all_files += sorted(p.name for p in tests_dir.glob("test_biz_value_mrmr_*/test_*.py"))
        prior = [
            n for n in all_files
            if n != "test_layer55.py"
        ]
        assert len(prior) >= 54, (
            f"prior biz_value layer test modules on disk = {len(prior)} "
            f"(< 54). Files seen: {prior!r}"
        )


# ---------------------------------------------------------------------------
# C2 + C3: composite all-on via partial_fit + provenance integration
# ---------------------------------------------------------------------------


class TestLayer55_CompositeAllOnViaPartialFit:
    """The composite kitchen-sink with every FE / DCD / auto switch on
    must complete via L53's partial_fit() inside < 60s AND populate
    L54's fe_provenance_ with both raw and engineered rows."""

    def test_partial_fit_all_on_completes_under_60s_and_populates_provenance(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _kitchen_sink(seed=42, n=2000)

        kwargs = dict(
            verbose=0,
            interactions_max_order=1,
            fe_max_steps=0,
            random_seed=0,
            fe_ntop_features=60,
            quantization_nbins=10,
            dcd_enable=True,
            cluster_aggregate_enable=True,
            cat_fe_config=None,
        )
        kwargs.update(_all_fe_kwargs())
        m = MRMR(**kwargs)

        # Drive the composite through partial_fit() (L53). First call ->
        # initialises and fits on the batch (equivalent to fit() per
        # Layer 53 C1 contract).
        t0 = time.perf_counter()
        m.partial_fit(X, y)
        fit_elapsed = time.perf_counter() - t0

        # C2: 60s wall-clock budget on n=2000 composite all-on.
        assert fit_elapsed <= 60.0, (
            f"composite all-on partial_fit must finish <= 60s; "
            f"got {fit_elapsed:.2f}s"
        )

        # C3: L54 fe_provenance_ DataFrame must be populated and carry
        # BOTH raw and engineered rows under the all-on configuration.
        assert hasattr(m, "fe_provenance_"), (
            "MRMR.partial_fit() must populate fe_provenance_ at the end "
            "of the underlying fit() call (L54 default-ON contract holds "
            "through the L53 partial_fit pathway)."
        )
        prov = m.fe_provenance_
        assert isinstance(prov, pd.DataFrame)
        assert len(prov) >= 1, (
            f"fe_provenance_ must contain at least one row after composite "
            f"all-on fit; got {len(prov)}"
        )
        origins = set(prov["origin"].astype(str).tolist())
        assert "raw" in origins, (
            f"Composite all-on fit produced no raw-origin rows in "
            f"fe_provenance_; origins seen: {origins!r}"
        )
        non_raw = origins - {"raw"}
        assert non_raw, (
            f"Composite all-on fit (every FE switch enabled) produced "
            f"NO engineered-origin rows in fe_provenance_; this means the "
            f"FE pipeline either fired 0 recipes or the provenance "
            f"population step is skipping engineered names. "
            f"origins seen: {origins!r}; "
            f"_engineered_recipes_ count="
            f"{len(getattr(m, '_engineered_recipes_', []) or [])}"
        )


class TestLayer55_ProvenanceAuditTrail:
    """Regression sensor for the produced-recipes audit ledger.

    ``fe_provenance_`` is an AUDIT TRAIL, not a survivor list: it must
    surface every engineered column the FE stages PRODUCED this fit, even
    the ones the greedy CMI screen / accuracy gate / cross-stage dedup
    dropped before support finalisation. Pre-fix, ``fe_provenance_`` drained
    only the post-reconciliation survivor rosters, so on a kitchen-sink
    frame where the screen keeps the strongest ~5 of ~18 produced columns
    the ledger reported only 3-5 mechanisms -- the audit / pickle-replay
    path could no longer recover which mechanism produced each engineered
    column. The fix snapshots the full produced set into
    ``_produced_recipes_`` before the screen runs and emits one ledger row
    per produced column (survivors keep their greedy rank; screened-out
    columns get support_rank == -1).
    """

    def _build(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _kitchen_sink(seed=42, n=2000)
        kwargs = dict(
            verbose=0,
            interactions_max_order=1,
            fe_max_steps=0,
            random_seed=0,
            fe_ntop_features=60,
            quantization_nbins=10,
            dcd_enable=True,
            cluster_aggregate_enable=True,
            cat_fe_config=None,
        )
        kwargs.update(_all_fe_kwargs())
        m = MRMR(**kwargs).fit(X, y)
        return m, X

    def test_provenance_covers_every_produced_recipe(self):
        m, _ = self._build()
        produced_names = {
            str(r.name) for r in (getattr(m, "_produced_recipes_", []) or [])
            if getattr(r, "name", None) is not None
        }
        assert produced_names, (
            "kitchen-sink all-on fit produced 0 engineered recipes; the FE "
            "pipeline regressed (no stage emitted a recipe)."
        )
        prov_names = set(m.fe_provenance_["feature_name"].astype(str).tolist())
        missing = produced_names - prov_names
        assert not missing, (
            f"fe_provenance_ is missing rows for produced engineered "
            f"columns: {sorted(missing)!r}. The audit ledger must surface "
            f"every column the FE stages produced, survivor or not."
        )

    def test_screened_out_columns_carry_audit_rank_minus_one(self):
        m, _ = self._build()
        survivors = {str(c) for c in (getattr(m, "_engineered_features_", []) or [])}
        produced = {
            str(r.name) for r in (getattr(m, "_produced_recipes_", []) or [])
            if getattr(r, "name", None) is not None
        }
        dropped = produced - survivors
        # The kitchen-sink frame is built so the screen keeps only the
        # strongest handful of the produced columns -> at least one drop.
        assert dropped, (
            "expected at least one produced engineered column to be "
            "screened out on the kitchen-sink frame; if every produced "
            "column survived, strengthen the fixture rather than weakening "
            "this contract."
        )
        prov = m.fe_provenance_
        for name in dropped:
            row = prov[prov["feature_name"].astype(str) == name]
            assert len(row) == 1, (
                f"screened-out column {name!r} must have exactly one ledger "
                f"row; got {len(row)}"
            )
            assert int(row["support_rank"].iloc[0]) == -1, (
                f"screened-out column {name!r} must carry support_rank == -1 "
                f"(it never entered the greedy selection); got "
                f"{int(row['support_rank'].iloc[0])}"
            )
            assert str(row["origin"].iloc[0]) not in ("raw", "engineered_unknown"), (
                f"screened-out column {name!r} must keep its mechanism "
                f"origin in the ledger; got {row['origin'].iloc[0]!r}"
            )

    def test_survivor_rosters_stay_subset_of_feature_names_out(self):
        # The audit-trail completion must NOT leak screened-out names into
        # the user-facing survivor rosters (the layer28 subset contract).
        m, _ = self._build()
        names_out = set(map(str, m.get_feature_names_out()))
        for attr in (
            "hybrid_orth_features_", "kfold_te_features_",
            "count_encoding_features_", "frequency_encoding_features_",
            "cat_num_interaction_features_", "pairwise_ratio_features_",
        ):
            roster = getattr(m, attr, None) or []
            leaked = [str(c) for c in roster if str(c) not in names_out]
            assert not leaked, (
                f"{attr} leaked screened-out names not in "
                f"get_feature_names_out(): {leaked!r}. The produced-recipes "
                f"audit ledger must live in fe_provenance_ only, never in "
                f"the survivor rosters."
            )
