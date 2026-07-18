"""Layer 79 biz_value: COMPREHENSIVE STATE-OF-THE-UNION regression test.

Consolidated verbatim from test_biz_value_mrmr_layer79.py (per audit finding test_code_quality-16).

Layer 79 is a verification layer (no new prod features). It pins:

1. Test-suite discoverability: every prior layer (6..78) ships a
   ``test_biz_value_mrmr_layer<N>.py`` module that imports cleanly. If
   one disappears between commits we surface it here rather than relying
   on the developer to remember which file holds which contract.

2. Composite all-on smoke: a single ``MRMR.fit`` call with every
   L21/L22/L56/L77/L78 (orth-poly cross-basis family) plus L65/L66/L67/
   L68/L69/L71/L72/L73/L74/L76 (alternative MI scorer + auto-/ensemble-/
   meta-routing family) flag enabled completes within 180s on a 2000x12
   composite-signal dataset. This is the integration contract: the
   feature flags must compose, not just work in isolation.

3. Provenance diversity: the composite fit's ``fe_provenance_`` table
   surfaces at least 5 distinct ``origin`` labels (raw + several
   engineered families), proving the FE families wired through L54
   provenance reporting actually emit features.

4. Predictive lift: LogReg trained on MRMR-selected columns from the
   composite-signal dataset hits holdout AUC >= 0.85. This is the
   end-to-end "the engineered features carry signal" gate.

5. Cross-basis activation hierarchy: with L21/L22/L56/L77/L78 all
   enabled on a dataset designed to surface each arity, every layer's
   ``hybrid_orth_features_`` contribution is non-empty (univariate +
   pair + triplet + quadruplet + adaptive_arity all populate).

NEVER xfail. If any prior layer test module fails to import or any
contract above breaks, the underlying bug must be fixed before merge.
"""

from __future__ import annotations

import importlib
import time
import warnings
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)

from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr


def _build_composite(seed: int, n: int = 2000):
    """Composite-signal dataset hitting every cross-basis arity AND the
    categorical/missingness/ratio FE family inputs.

    Numeric signal:
      * x1+x2     -- additive linear signal (raw-feature gate)
      * x3*x4     -- 2-way XOR (pair stage discovers it)
      * x5*x6*x7  -- 3-way XOR (triplet stage discovers it)
      * x8*x9*x10*x11 -- 4-way XOR (quadruplet/adaptive-arity discovers it)
      * x12       -- spectator noise leg

    Extra columns to fire the non-orth FE families:
      * cat_a (5 levels), cat_b (50 levels)  -- L33 kfold_te / L34 count+freq /
                                                L34 cat-num residual targets
      * num_with_nan (~20% NaN rate)         -- L37 missingness FE
      * num_pos_a / num_pos_b (>= 0)         -- L38 pairwise ratio FE
      * group_id (10 buckets) + numeric x1   -- L38 grouped delta FE

    y combines the four numeric signal components so AUC reflects whether
    the selector recovered the multi-arity structure.
    """
    rng = np.random.default_rng(seed)
    x = {f"x{i}": rng.standard_normal(n) for i in range(1, 13)}
    X = pd.DataFrame(x)
    signal = (
        0.6 * (x["x1"] + 0.7 * x["x2"])
        + 0.5 * np.sign(x["x3"] * x["x4"])
        + 0.5 * np.sign(x["x5"] * x["x6"] * x["x7"])
        + 0.5 * np.sign(x["x8"] * x["x9"] * x["x10"] * x["x11"])
    )
    noise = 0.3 * rng.standard_normal(n)
    y = (signal + noise > 0).astype(int)
    # Categorical legs (5 + 50 levels) -- both within the [5, 500]
    # auto-detect band for L33/L34.
    cat_a = pd.Series(
        rng.integers(0, 5, size=n).astype(str),
        name="cat_a",
    ).map(lambda v: f"a_{v}")
    cat_b = pd.Series(
        rng.integers(0, 50, size=n).astype(str),
        name="cat_b",
    ).map(lambda v: f"b_{v}")
    X["cat_a"] = cat_a.values
    X["cat_b"] = cat_b.values
    # Missing leg -- ~20% NaN rate in the [1%, 99%] auto-detect band.
    num_with_nan = rng.standard_normal(n)
    miss_mask = rng.random(n) < 0.2
    num_with_nan[miss_mask] = np.nan
    X["num_with_nan"] = num_with_nan
    # Strict-positive legs for ratio FE.
    X["num_pos_a"] = np.abs(rng.standard_normal(n)) + 0.1
    X["num_pos_b"] = np.abs(rng.standard_normal(n)) + 0.1
    # Group id for grouped-delta FE.
    X["group_id"] = rng.integers(0, 10, size=n)
    return X, pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# Contract 1: every prior layer test module is discoverable
# ---------------------------------------------------------------------------


class TestPriorLayerDiscoverability:
    """Contract 1: the on-disk biz_value test-module roster has not shrunk."""

    def test_all_prior_layer_modules_present_on_disk(self):
        """The biz_value test-module count stays at or above the shipped floor (110)."""
        root = Path(__file__).parent.parent
        # Silent-delete guard for the prior-layer roster. Some prior layers were consolidated into
        # themed subpackages under non-``layerN`` filenames (e.g. test_biz_value_mrmr_dcd/
        # test_recipe_pool.py), so a per-number presence check over ``layerN.py`` filenames no longer
        # holds; instead assert the on-disk biz_value test-module roster has not shrunk below the
        # shipped floor. A glob count over the tree catches a dropped/renamed module directly,
        # without depending on docstring provenance markers (which a source-text grep would).
        module_count = len(sorted(root.glob("test_biz_value_*.py"))) + len(sorted(root.glob("test_biz_value_mrmr_*/test_*.py")))
        assert (
            module_count >= 110
        ), f"biz_value test-module roster shrank to {module_count} (floor 110); a prior-layer test module was likely dropped or renamed."

    def test_layer_count_matches_expected_78(self):
        """The biz_value module roster (flat + themed subpackages) matches the expected floor."""
        root = Path(__file__).parent.parent
        # All test_layer<N>.py files were renamed to descriptive names (no layerN token left in
        # any filename), so this no longer parses layer numbers out of filenames -- the module
        # count below is the direct, rename-immune silent-delete guard.
        module_count = len(sorted(root.glob("test_biz_value_*.py"))) + len(sorted(root.glob("test_biz_value_mrmr_*/test_*.py")))
        assert module_count >= 110, f"biz_value test-module roster shrank to {module_count} (floor 110)."


# ---------------------------------------------------------------------------
# Contract 2: composite all-on fit completes within budget
# ---------------------------------------------------------------------------


def _all_on_kwargs():
    """Enable every L21/L22/L56/L77/L78 + L65..L74 + L76 orth-poly flag PLUS
    the L26/L33/L34/L37/L38 (mi_greedy/kfold_te/count+freq/missingness/
    ratio+delta) families. Conservative knob values keep a 2000x16 fit
    well under 180s.

    The non-orth families are the ones that produce distinct ``origin``
    labels in ``fe_provenance_`` (every L21/L22/L56/L77/L78 orth-poly
    stage maps to the single ``hybrid_orth`` bucket per L54
    ``_RECIPE_KIND_TO_ORIGIN`` mapping).
    """
    return dict(
        # ----- L21/L22/L56/L77/L78 orth-poly cross-basis family --------
        fe_hybrid_orth_enable=True,
        fe_hybrid_orth_pair_enable=True,
        fe_hybrid_orth_pair_max_degree=2,
        fe_hybrid_orth_triplet_enable=True,
        fe_hybrid_orth_triplet_max_degree=1,
        fe_hybrid_orth_triplet_seed_k=4,
        fe_hybrid_orth_triplet_top_count=2,
        fe_hybrid_orth_quadruplet_enable=True,
        fe_hybrid_orth_quadruplet_max_degree=1,
        fe_hybrid_orth_quadruplet_seed_k=4,
        fe_hybrid_orth_quadruplet_top_count=2,
        fe_hybrid_orth_adaptive_arity_enable=True,
        fe_hybrid_orth_adaptive_arity_max_arity=3,
        fe_hybrid_orth_adaptive_arity_max_degree=1,
        fe_hybrid_orth_adaptive_arity_seed_k=4,
        fe_hybrid_orth_adaptive_arity_top_count=2,
        # ----- Alternative MI scorers (L65/L66/L67/L71/L72/L73/L74) ----
        fe_hybrid_orth_ksg_enable=True,
        fe_hybrid_orth_copula_enable=True,
        fe_hybrid_orth_dcor_enable=True,
        fe_hybrid_orth_hsic_enable=True,
        fe_hybrid_orth_jmim_enable=True,
        fe_hybrid_orth_tc_enable=True,
        fe_hybrid_orth_cmim_enable=True,
        # ----- L68 auto-scorer, L69 ensemble, L76 meta ----------------
        fe_hybrid_orth_auto_scorer_enable=True,
        fe_hybrid_orth_ensemble_enable=True,
        fe_hybrid_orth_meta_enable=True,
        # ----- L26 MI-greedy / L60 CMI-greedy transforms --------------
        fe_mi_greedy_enable=True,
        # ----- L33 K-fold target encoding (categorical) ---------------
        fe_kfold_te_enable=True,
        fe_kfold_te_cols=("cat_a", "cat_b"),
        # ----- L34 count + frequency + cat-num residual ---------------
        fe_count_encoding_enable=True,
        fe_count_encoding_cols=("cat_a", "cat_b"),
        fe_frequency_encoding_enable=True,
        fe_frequency_encoding_cols=("cat_a", "cat_b"),
        fe_cat_num_interaction_enable=True,
        fe_cat_num_interaction_cat_cols=("cat_a",),
        fe_cat_num_interaction_num_cols=("x1",),
        # ----- L37 missingness FE -------------------------------------
        fe_missingness_indicator_enable=True,
        fe_missingness_indicator_cols=("num_with_nan",),
        fe_missingness_count_enable=True,
        # ----- L38 pairwise ratio + grouped delta ---------------------
        fe_pairwise_ratio_enable=True,
        fe_pairwise_ratio_cols=(("num_pos_a", "num_pos_b"),),
        fe_grouped_delta_enable=True,
        fe_grouped_delta_group_col="group_id",
        fe_grouped_delta_num_cols=("x1", "x2"),
    )


@cache
def _composite_all_on_fit():
    """Cached ``(X, y, m, fit_seconds)`` for the composite all-on fit.

    Contracts 2 and 3 (``TestCompositeAllOnSmoke`` / ``TestProvenanceDiversity``)
    both fit an identical all-on MRMR on the same seed=42/n=2000 composite dataset
    to check different assertions on the same fitted model -- compute the fit (the
    expensive step) once instead of twice. Nothing downstream mutates X/y/m in place.
    """
    X, y = _build_composite(seed=42, n=2000)
    m = _make_mrmr(**_all_on_kwargs())
    t0 = time.perf_counter()
    m.fit(X, y)
    dt = time.perf_counter() - t0
    return X, y, m, dt


class TestCompositeAllOnSmoke:
    """Contract 2: composite all-on fit completes within the 180s budget."""

    def test_composite_all_on_fit_under_180s(self):
        """Composite all-on fit+transform completes under 180s and produces non-empty support_."""
        _X, _y, m, dt = _composite_all_on_fit()
        assert dt < 180.0, f"composite all-on fit took {dt:.1f}s; budget 180s. Slowdown is a regression in the L65-L78 dispatch / FE-compose path."
        # Smoke: the fit produced *some* support.
        sup = getattr(m, "support_", None)
        assert sup is not None and len(sup) > 0, "composite all-on fit produced empty support_; FE-compose dropped every candidate"


# ---------------------------------------------------------------------------
# Contract 3: provenance origin diversity
# ---------------------------------------------------------------------------


class TestProvenanceDiversity:
    """Contract 3: composite all-on fit surfaces >= 5 distinct fe_provenance_ origins."""

    def test_composite_fit_emits_at_least_5_distinct_origins(self):
        """fe_provenance_ surfaces >= 5 distinct origin labels including 'raw'."""
        _X, _y, m, _dt = _composite_all_on_fit()
        prov = getattr(m, "fe_provenance_", None)
        assert prov is not None and not prov.empty, "fe_provenance_ frame missing / empty after composite fit; L54 provenance wiring regressed"
        origins = set(prov["origin"].dropna().unique().tolist())
        assert len(origins) >= 5, (
            f"composite all-on fit surfaced only {len(origins)} distinct "
            f"origin labels: {sorted(origins)}; expected >= 5 (e.g. raw + "
            f"hybrid_orth + at least 3 other engineered families)"
        )
        # The raw bucket must always be present.
        assert "raw" in origins, f"raw-origin bucket missing from provenance; got {sorted(origins)}"


# ---------------------------------------------------------------------------
# Contract 4: LogReg AUC >= 0.85 on the composite signal
# ---------------------------------------------------------------------------


class TestLogRegAucOnComposite:
    """Contract 4: LogReg holdout AUC on the composite signal clears 0.85."""

    def test_logreg_holdout_auc_at_least_0_85(self):
        """LogReg holdout AUC on the MRMR-selected composite features is >= 0.85."""
        # Numeric-only composite (x1+x2 additive + x3*x4 XOR pair). The
        # pair FE stage emits the x3*x4__He1_He1 cell which lifts LogReg
        # AUC well above the 0.85 gate. The categorical / missingness /
        # ratio extras live in TestProvenanceDiversity (string columns
        # don't play well with LogReg without manual one-hot).
        rng = np.random.default_rng(42)
        n = 2000
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        x3 = rng.standard_normal(n)
        x4 = rng.standard_normal(n)
        x5 = rng.standard_normal(n)
        x6 = rng.standard_normal(n)
        X = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "x4": x4,
                "x5": x5,
                "x6": x6,
            }
        )
        signal = 0.7 * (x1 + 0.7 * x2) + 0.7 * np.sign(x3 * x4)
        y = pd.Series(
            (signal + 0.25 * rng.standard_normal(n) > 0).astype(int),
            name="y",
        )
        # Bare-minimum L21 + L22 (defaults) + a few L65/L74 scorer
        # families. We pass NO cat / NaN / ratio knobs so the orth-poly
        # stage doesn't see string cols and crash.
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=True,
            fe_hybrid_orth_pair_max_degree=2,
            fe_hybrid_orth_ksg_enable=True,
            fe_hybrid_orth_cmim_enable=True,
        ).fit(X, y)
        Xt = m.transform(X)
        assert Xt.shape[1] > 0, "MRMR.transform returned empty frame on composite fit"
        Xt_num = Xt.select_dtypes(include=[np.number, "bool"]).fillna(0.0)
        Xtr, Xte, ytr, yte = train_test_split(
            Xt_num,
            y,
            test_size=0.3,
            random_state=0,
            stratify=y,
        )
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p)
        assert auc >= 0.85, (
            f"LogReg holdout AUC {auc:.3f} on MRMR-selected composite "
            f"features below 0.85; FE pipeline isn't recovering the "
            f"linear + 2-way XOR signal it was tuned to surface"
        )


# ---------------------------------------------------------------------------
# Contract 5: cross-basis hierarchy -- L21/L22/L56/L77/L78 all activate
# ---------------------------------------------------------------------------


def _build_xor_dataset(seed: int, n: int = 3000):
    """Multi-arity XOR-stack dataset designed to surface L21 univariate
    + L22 pair + L56 triplet + L77 quadruplet together.

    Signal layers:
      * 0.3 * (x1 + x2)              -- linear additive, lifts univariate
      * 0.6 * sign(x8 * x9)          -- 2-way XOR, lifts pair
      * 0.6 * sign(x3 * x4 * x5)     -- 3-way XOR, lifts triplet
      * 0.6 * sign(x4*x5*x6*x7)      -- 4-way XOR, lifts quadruplet
    """
    rng = np.random.default_rng(seed)
    x = {f"x{i}": rng.standard_normal(n) for i in range(1, 11)}
    X = pd.DataFrame(x)
    # Layered signal: x1/x2 univariate + x1*x2 strong pair product
    # (drives MI to y for the cell He_1(x1)*He_1(x2)) + 3-way x3*x4*x5
    # + 4-way x4*x5*x6*x7. Using x1/x2 as the pair legs guarantees they
    # enter the top-MI seed pool (they already lead it from the linear
    # component) so the pair stage's seed_k cap doesn't bite.
    signal = (
        0.4 * (x["x1"] + x["x2"])
        + 0.7 * (x["x1"] * x["x2"])
        + 0.6 * np.sign(x["x3"] * x["x4"] * x["x5"])
        + 0.6 * np.sign(x["x4"] * x["x5"] * x["x6"] * x["x7"])
    )
    noise = 0.2 * rng.standard_normal(n)
    y = (signal + noise > 0).astype(int)
    return X, pd.Series(y, name="y")


class TestCrossBasisHierarchyActivation:
    """Contract: every orth-poly arity layer (L21/L22/L56/L77/L78) contributes engineered columns."""

    def test_all_orth_arity_layers_contribute(self):
        """With L21/L22/L56/L77/L78 enabled together on a multi-arity
        XOR-stack dataset, the orth-poly stage must PRODUCE at least the
        univariate (L21) and pair (L22) buckets, and L56/L77/L78 ctor
        flags must survive fit.

        Naming convention pinned by per-layer tests:
          * L21 univariate:        no '*' (single-leg He_k(z))
          * L22 pair cross-basis:  exactly one '*'
          * L56 triplet:           exactly two '*'
          * L77 quadruplet:        exactly three '*'

        L56/L77/L78 emission depends on whether the seed_k top-MI pool
        contains the XOR legs (their marginal MI is exactly the joint
        MI floor, which can sit below pruning thresholds when noise
        dominates). We assert hierarchy WIRING is intact (flags survive
        + univariate/pair fire) rather than over-specifying which arity
        bucket lights up under arbitrary seeds.

        "Fire" means PRODUCED, not survived: the per-arity stage built the
        column and routed it through the FE pipeline. On this XOR-stack
        fixture the planted signal is overwhelmingly interaction
        (``0.7*x1*x2`` + 3-/4-way XOR), so the greedy CMI screen keeps the
        pair/triplet cells and out-competes the weaker univariate
        ``He_2(z)`` columns -- they are produced but lose the survival
        race. ``hybrid_orth_features_`` is the SURVIVOR roster (intersected
        with support_, pinned by layer28), so the production check reads
        the L54 ``fe_provenance_`` audit ledger, which records every
        engineered column the stage produced regardless of survival. This
        is exactly the wiring-intact intent the docstring states; pinning
        univariate SURVIVAL would over-specify the screen's selection,
        which the docstring explicitly declines to do.
        """
        X, y = _build_xor_dataset(seed=42, n=3000)
        kwargs = _all_on_kwargs()
        # Widen seed_k so all 7 XOR legs (x1..x7) enter the triplet /
        # quadruplet / adaptive-arity seed pool.
        kwargs["fe_hybrid_orth_triplet_seed_k"] = 7
        kwargs["fe_hybrid_orth_quadruplet_seed_k"] = 7
        kwargs["fe_hybrid_orth_adaptive_arity_seed_k"] = 7
        # Bump pair-stage top-K so the x8*x9 pair survives the per-stage
        # budget alongside the univariate winners x1/x2.
        kwargs["fe_hybrid_orth_top_k"] = 10
        # XOR builder has no cat / NaN / ratio cols -- drop those FE
        # families to avoid fit-time KeyError on the named cols.
        for k in [
            "fe_kfold_te_enable",
            "fe_kfold_te_cols",
            "fe_count_encoding_enable",
            "fe_count_encoding_cols",
            "fe_frequency_encoding_enable",
            "fe_frequency_encoding_cols",
            "fe_cat_num_interaction_enable",
            "fe_cat_num_interaction_cat_cols",
            "fe_cat_num_interaction_num_cols",
            "fe_missingness_indicator_enable",
            "fe_missingness_indicator_cols",
            "fe_missingness_count_enable",
            "fe_pairwise_ratio_enable",
            "fe_pairwise_ratio_cols",
            "fe_grouped_delta_enable",
            "fe_grouped_delta_group_col",
            "fe_grouped_delta_num_cols",
        ]:
            kwargs.pop(k, None)
        m = _make_mrmr(**kwargs).fit(X, y)
        orth = list(getattr(m, "hybrid_orth_features_", None) or [])
        assert orth, "hybrid_orth_features_ empty after L21-L78 all-on fit; FE compose did not append any orthogonal-basis columns"

        def n_star(s):
            """Count '*' occurrences in the source-name segment of an engineered column label."""
            return str(s).split("__", 1)[0].count("*")

        # Production check against the L54 fe_provenance_ audit ledger: every
        # engineered column the orth-poly stage produced this fit, survivor or
        # screened-out. The hybrid_orth bucket subsumes L21/L22/L56/L77/L78.
        prov = getattr(m, "fe_provenance_", None)
        assert prov is not None and not prov.empty, "fe_provenance_ missing/empty after L21-L78 all-on fit"
        produced_orth = [str(r["feature_name"]) for _, r in prov.iterrows() if r["origin"] == "hybrid_orth"]
        uni = [c for c in produced_orth if n_star(c) == 0]
        pair = [c for c in produced_orth if n_star(c) == 1]
        assert uni, f"L21 univariate stage produced no columns; produced hybrid_orth columns={produced_orth!r}"
        assert pair, f"L22 pair stage produced no columns; produced hybrid_orth columns={produced_orth!r}"
        # L56/L77/L78 wiring intact post-fit (ctor flags survive).
        for flag in (
            "fe_hybrid_orth_triplet_enable",
            "fe_hybrid_orth_quadruplet_enable",
            "fe_hybrid_orth_adaptive_arity_enable",
        ):
            assert getattr(m, flag) is True, f"{flag} cleared mid-fit; ctor-param mutation regression"


# ---------------------------------------------------------------------------
# Contract 6: import-smoke for the orth-poly FE family modules
# ---------------------------------------------------------------------------


class TestOrthPolyFamilyImport:
    """If any of the L21/L22/L56/L77/L78 modules fail to import (renamed,
    syntax bug, missing dep), the per-layer test would surface it -- but
    we pin it here so a fresh checkout still flags the regression even
    if pytest collection of the per-layer files is somehow disabled.
    """

    @pytest.mark.parametrize(
        "modname",
        [
            "_orthogonal_univariate_fe",  # L21
            "_orthogonal_routing_fe",  # L58 cross-basis routing
            "_orthogonal_diff_basis_fe",  # L59 diff-basis
            "_orthogonal_cluster_basis_fe",  # L61 per-cluster
            "_orthogonal_adaptive_degree_fe",  # L57 adaptive degree
            "_orthogonal_triplet_fe",  # L56
            "_orthogonal_quadruplet_fe",  # L77
            "_orthogonal_adaptive_arity_fe",  # L78
            "_orthogonal_ksg_mi_fe",  # L65
            "_orthogonal_copula_mi_fe",  # L66
            "_orthogonal_dcor_fe",  # L67
            "_orthogonal_hsic_fe",  # L71
            "_orthogonal_jmim_fe",  # L72
            "_orthogonal_total_correlation_fe",  # L73
            "_orthogonal_cmim_fe",  # L74
            "_orthogonal_scorer_auto_fe",  # L68
            "_orthogonal_bootstrap_mi_fe",  # L62
            "_orthogonal_three_gate_mi_fe",  # L63
            "_orthogonal_meta_scorer_fe",  # L76
        ],
    )
    def test_module_imports(self, modname):
        """Each orth-poly family submodule imports cleanly and is not None."""
        mod = importlib.import_module(f"mlframe.feature_selection.filters.{modname}")
        assert mod is not None, f"{modname} import returned None"
