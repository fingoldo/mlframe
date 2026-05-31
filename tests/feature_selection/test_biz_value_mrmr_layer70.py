"""Layer 70 biz_value: COMPREHENSIVE 69-LAYER REGRESSION + COMPOSITE ALL-ON.

Pure VERIFICATION layer (no new prod surface). Layer 64 already pinned a
kitchen-sink "every L21-L63 FE knob on" composite; Layer 70 extends the
roster to L65-L69 (KSG / copula / dCor / auto / ensemble MI scorers) and
tightens the gates so a regression in any of the five new MI ranking
paths fails this layer rather than silently degrading downstream AUC on
unrelated tests.

Why a separate layer (not an extension of L64)?

* The L65-L69 scorers are alternative MI-ranking paths over the SAME
  engineered values L21 emits; they fire as four independent passes
  inside the hybrid_orth stage when their master switches are flipped
  on. Wiring them all into the L64 ctor and re-running L64 would muddy
  the L64 contract (which pins the L21-L63 stack on the SAME budget).
* The L70 spec doubles the L64 fit budget (120s vs 90s) because four
  extra MI passes per source column add measurable wall time and the
  L64 budget is too tight once all five new scorers are on at once.
* The provenance / collision / roster floors are pinned IDENTICALLY to
  L64 to catch the regression class where adding the L65-L69 paths
  silently drops a pre-existing mechanism out of the ledger.

What this layer pins
--------------------

* ``TestAllOnCompositeFitsInBudget``: the all-on composite fit on the
  kitchen-sink frame completes inside 120s.
* ``TestProvenanceDiversityFloor``: ``fe_provenance_`` carries >= 4
  distinct engineered-origin labels (excluding ``raw`` /
  ``engineered_unknown``).
* ``TestLogRegAucFloor``: downstream LogReg holdout AUC >= 0.85 on the
  kitchen-sink frame.
* ``TestRosterSizeAtLeast69``: the biz_value layer module roster on disk
  covers at least 69 layers (L6..L70 contiguous = 65 layerN.py modules +
  5 catch-all = 70 total; floor at 69 absorbs one missing-file slack
  for future layer rename audits).
* ``TestNoEngineeredNameCollisions``: every engineered column emitted
  by the all-on fit is unique within and across specific-bucket
  rosters.

NEVER xfail. NEVER mask bugs via runtime workarounds.
"""
from __future__ import annotations

import glob
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data fixture: reuse the L64 kitchen-sink generator verbatim so a frame
# regression fails BOTH layers in lock-step.
# ---------------------------------------------------------------------------


def _build_kitchen_sink_frame(seed: int = 0, n: int = 1500):
    """Multi-signal frame -- identical to the Layer 64 builder. Reused
    here rather than imported so a refactor of the L64 test file does
    not silently shift the L70 contract data."""
    rng = np.random.default_rng(int(seed))

    x_gauss = rng.standard_normal(n)
    x_gauss_corr = (
        x_gauss * 0.95
        + rng.standard_normal(n) * np.sqrt(1 - 0.95 ** 2)
    )
    x_uni = rng.uniform(-1, 1, n)
    x_uni_corr = (
        x_uni * 0.95
        + rng.uniform(-1, 1, n) * np.sqrt(1 - 0.95 ** 2)
    )

    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)

    cat_lo = pd.Series(rng.choice(list("ABCDE"), size=n))
    cat_med = pd.Series(rng.choice([f"v{k}" for k in range(20)], size=n))
    region = pd.Series(rng.choice(list("NSWE"), size=n))

    t = pd.Series(np.arange(n, dtype="int64"))
    temperature = pd.Series(
        20.0 + 5.0 * np.sin(np.arange(n) / 50.0) + rng.standard_normal(n)
    )

    maybe_a = rng.standard_normal(n).astype(float)
    maybe_b = rng.standard_normal(n).astype(float)
    mask_a = rng.random(n) < 0.10
    mask_b = rng.random(n) < 0.10
    maybe_a[mask_a] = np.nan
    maybe_b[mask_b] = np.nan

    X = pd.DataFrame({
        "x_gauss": x_gauss,
        "x_gauss_corr": x_gauss_corr,
        "x_uni": x_uni,
        "x_uni_corr": x_uni_corr,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "cat_lo": cat_lo,
        "cat_med": cat_med,
        "region": region,
        "t": t,
        "temperature": temperature,
        "maybe_missing_a": maybe_a,
        "maybe_missing_b": maybe_b,
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
        "noise_2": rng.standard_normal(n),
    })

    cat_lo_effect = cat_lo.map(
        {"A": 1.0, "B": 0.5, "C": 0.0, "D": -0.5, "E": -1.0}
    ).astype(float).to_numpy()
    score = (
        1.2 * x_gauss
        + 0.8 * x_uni
        + 0.6 * (x1 ** 2 - 1.0)
        + 0.6 * (x1 * x2 * x3)
        + 0.5 * cat_lo_effect
        + 0.3 * np.where(mask_a, 1.0, 0.0)
        + 0.5 * rng.standard_normal(n)
    )
    y = pd.Series((score > np.median(score)).astype(int), name="y")
    return X, y


# ---------------------------------------------------------------------------
# All-on composite ctor: L21..L63 (mirrored from L64) PLUS L65..L69
# ---------------------------------------------------------------------------


def _build_all_on_mrmr():
    """MRMR with every public FE master switch from L21-L69 flipped ON.

    Includes the L64 kitchen-sink roster AND the four L65-L69 MI scorer
    paths (KSG, copula, dCor, auto, ensemble). DCD + cluster_aggregate
    + friend_graph stay off so the wall-clock budget is governed by FE
    cost only, mirroring L64.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR
    return MRMR(
        verbose=0,
        random_seed=0,
        interactions_max_order=1,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        fe_max_steps=1,
        # L21 hybrid orth (master + pair).
        fe_hybrid_orth_enable=True,
        fe_hybrid_orth_pair_enable=True,
        fe_hybrid_orth_degrees=(2, 3),
        fe_hybrid_orth_basis="auto",
        fe_hybrid_orth_top_k=3,
        # L32 extra non-poly bases (spline + Fourier).
        fe_hybrid_orth_extra_bases=("spline", "fourier"),
        fe_hybrid_orth_fourier_freqs=(1.0, 2.0),
        fe_hybrid_orth_spline_knots=5,
        # L56 triplet.
        fe_hybrid_orth_triplet_enable=True,
        fe_hybrid_orth_triplet_max_degree=1,
        fe_hybrid_orth_triplet_seed_k=4,
        fe_hybrid_orth_triplet_top_count=2,
        # L57 adaptive degree.
        fe_hybrid_orth_adaptive_degree_enable=True,
        # L58 conditional routing.
        fe_hybrid_orth_conditional_routing_enable=True,
        fe_hybrid_orth_conditional_routing_top_k=3,
        # L59 diff-basis.
        fe_hybrid_orth_diff_basis_enable=True,
        fe_hybrid_orth_diff_basis_corr_threshold=0.7,
        fe_hybrid_orth_diff_basis_top_k=2,
        # L61 cluster-basis.
        fe_hybrid_orth_cluster_basis_enable=True,
        fe_hybrid_orth_cluster_basis_top_k=2,
        # L62 bootstrap MI.
        fe_hybrid_orth_bootstrap_enable=True,
        fe_hybrid_orth_bootstrap_n_boot=5,
        # L63 three-gate OOF MI.
        fe_hybrid_orth_three_gate_enable=True,
        fe_hybrid_orth_three_gate_n_folds=3,
        # L65 KSG MI ranking.
        fe_hybrid_orth_ksg_enable=True,
        fe_hybrid_orth_ksg_n_neighbors=3,
        # L66 copula MI ranking.
        fe_hybrid_orth_copula_enable=True,
        fe_hybrid_orth_copula_n_bins=20,
        # L67 distance-correlation ranking.
        fe_hybrid_orth_dcor_enable=True,
        fe_hybrid_orth_dcor_n_sample=300,
        # L68 per-column scorer auto-selection.
        fe_hybrid_orth_auto_scorer_enable=True,
        fe_hybrid_orth_auto_scorer_n_boot=3,
        # L69 ensemble rank-fusion.
        fe_hybrid_orth_ensemble_enable=True,
        fe_hybrid_orth_ensemble_aggregator="mean_rank",
        # L26 MI-greedy.
        fe_mi_greedy_enable=True,
        fe_mi_greedy_top_k=3,
        # L60 CMI-greedy.
        fe_mi_greedy_cmi_enable=True,
        fe_mi_greedy_cmi_top_k=3,
        # L33 K-fold target encoding.
        fe_kfold_te_enable=True,
        fe_kfold_te_cols=("cat_lo", "cat_med"),
        fe_kfold_te_folds=3,
        # L34 count + freq + cat-num residual.
        fe_count_encoding_enable=True,
        fe_count_encoding_cols=("cat_med",),
        fe_frequency_encoding_enable=True,
        fe_frequency_encoding_cols=("cat_med",),
        fe_cat_num_interaction_enable=True,
        fe_cat_num_interaction_cat_cols=("cat_lo",),
        fe_cat_num_interaction_num_cols=("x_gauss",),
        fe_cat_num_interaction_folds=3,
        # L37 missingness.
        fe_missingness_indicator_enable=True,
        fe_missingness_indicator_cols=("maybe_missing_a", "maybe_missing_b"),
        fe_missingness_count_enable=True,
        fe_missingness_pattern_enable=True,
        # L38 pairwise / grouped delta / lagged diff.
        fe_pairwise_ratio_enable=True,
        fe_pairwise_ratio_cols=("x_uni", "x_uni_corr"),
        fe_pairwise_log_ratio_enable=True,
        fe_pairwise_log_ratio_cols=("x_uni", "x_uni_corr"),
        fe_grouped_delta_enable=True,
        fe_grouped_delta_group_col="region",
        fe_grouped_delta_num_cols=("temperature",),
        fe_lagged_diff_enable=True,
        fe_lagged_diff_time_col="t",
        fe_lagged_diff_value_cols=("temperature",),
        fe_lagged_diff_periods=(1, 2),
        retain_artifacts=False,
    )


# Engineered roster attribute names mirrored from L64 so the
# collision check covers identical surface.
_ENGINEERED_ROSTER_ATTRS: tuple[str, ...] = (
    "hybrid_orth_features_",
    "mi_greedy_features_",
    "kfold_te_features_",
    "count_encoding_features_",
    "frequency_encoding_features_",
    "cat_num_interaction_features_",
    "missingness_indicator_features_",
    "missingness_count_features_",
    "missingness_pattern_features_",
    "pairwise_ratio_features_",
    "pairwise_log_ratio_features_",
    "grouped_delta_features_",
    "lagged_diff_features_",
)


# ---------------------------------------------------------------------------
# Module-scoped fitted estimator (amortise the all-on fit cost)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def all_on_fitted():
    """Module-scoped fixture: one all-on fit shared by every L70 test."""
    X, y = _build_kitchen_sink_frame(seed=0)
    m = _build_all_on_mrmr()
    t0 = time.perf_counter()
    m.fit(X, y)
    elapsed = time.perf_counter() - t0
    return m, X, y, elapsed


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


class TestAllOnCompositeFitsInBudget:

    def test_fit_inside_120s_budget(self, all_on_fitted):
        """All-on (L21-L69) composite fit on the kitchen-sink frame must
        complete inside 120s. Budget is doubled vs L64's 90s because the
        L65-L69 paths add 4 extra per-column MI ranking passes."""
        _, _, _, elapsed = all_on_fitted
        assert elapsed < 120.0, (
            f"L70 all-on composite fit took {elapsed:.1f}s; budget is "
            f"120s. A regression in one of the L65-L69 MI scorer paths "
            f"most likely added an O(n_boot * n_cols^2) blow-up; profile "
            f"the new layer rather than relaxing the gate."
        )


class TestProvenanceDiversityFloor:

    def test_provenance_has_at_least_4_engineered_origins(self, all_on_fitted):
        """``fe_provenance_`` must carry >= 4 DISTINCT engineered-origin
        labels (excluding ``raw`` and ``engineered_unknown``) on the
        kitchen-sink frame. L64 floor was 3; L70 raises to 4 because the
        kitchen-sink fixture exercises >= 10 mechanisms and the four
        extra L65-L69 paths feed the same hybrid_orth bucket -- the new
        floor catches the regression class where only the orth bucket
        emits and every L33/L34/L37/L38 mechanism early-returns silently.
        """
        m = all_on_fitted[0]
        assert hasattr(m, "fe_provenance_"), (
            "MRMR must populate fe_provenance_ on every successful fit "
            "(L54 contract)."
        )
        prov = m.fe_provenance_
        assert isinstance(prov, pd.DataFrame), (
            f"fe_provenance_ must be a DataFrame; got {type(prov).__name__}"
        )
        engineered_origins = {
            o for o in prov["origin"].tolist()
            if o not in ("raw", "engineered_unknown")
        }
        assert len(engineered_origins) >= 4, (
            f"fe_provenance_ engineered origins = "
            f"{sorted(engineered_origins)!r}; L70 floor is 4 distinct "
            f"labels. All-on FE on the composite frame should populate "
            f"at minimum hybrid_orth + one categorical encoding + one "
            f"missingness bucket + one pairwise/lagged bucket."
        )


class TestLogRegAucFloor:

    def test_logreg_holdout_auc_at_least_0_85(self, all_on_fitted):
        """End-to-end biz_value: LogReg on the selected + engineered view
        achieves holdout AUC >= 0.85 on the kitchen-sink frame. Same
        floor as L64 -- the L65-L69 scorers should not regress this
        gate; if anything the auto / ensemble paths should keep it
        stable across re-runs."""
        m, X, y, _ = all_on_fitted
        Xt = m.transform(X)
        Xt_num = Xt.select_dtypes(include=[np.number]).copy()
        if Xt_num.shape[1] == 0:
            pytest.fail(
                "MRMR.transform returned no numeric columns; downstream "
                "model cannot consume the selected view. Hard regression "
                "in the FE/selection stack."
            )
        Xt_num = Xt_num.fillna(Xt_num.median(numeric_only=True))
        n = len(y)
        rng = np.random.default_rng(0)
        idx = rng.permutation(n)
        cut = int(0.7 * n)
        train_idx, test_idx = idx[:cut], idx[cut:]
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(
            Xt_num.iloc[train_idx].to_numpy(),
            y.iloc[train_idx].to_numpy(),
        )
        proba = clf.predict_proba(Xt_num.iloc[test_idx].to_numpy())[:, 1]
        auc = float(roc_auc_score(y.iloc[test_idx].to_numpy(), proba))
        assert auc >= 0.85, (
            f"L70 all-on composite AUC={auc:.3f}; floor is 0.85. The "
            f"signal in the fixture (linear + quadratic + 3-way + "
            f"categorical + missingness) should be trivially separable "
            f"once the FE stack lands the right augmented support; if "
            f"AUC drops here, one of the L65-L69 scorer paths is "
            f"degrading the selection."
        )


class TestRosterSizeAtLeast69:

    def test_layer_module_roster_at_least_69(self):
        """The biz_value layer test module roster on disk must cover
        at least 69 layers (L6..L70 contiguous = 65 layerN.py modules +
        5 catch-all = 70 total). Floor at 69 absorbs one missing-file
        slack for future layer rename / consolidation audits.

        Catches the silent-delete / silent-rename regression class.
        """
        this_dir = os.path.dirname(os.path.abspath(__file__))
        matched = sorted(glob.glob(
            os.path.join(this_dir, "test_biz_value_mrmr_layer*.py")
        ))
        rx = re.compile(r"test_biz_value_mrmr_layer(\d+)\.py$")
        layer_numbers: set[int] = set()
        for path in matched:
            mobj = rx.search(os.path.basename(path))
            if mobj is not None:
                layer_numbers.add(int(mobj.group(1)))
        catchall_required = (
            "test_biz_value_mrmr_extreme.py",
            "test_biz_value_mrmr_hard_cases.py",
            "test_biz_value_mrmr_multiway_synergy.py",
            "test_biz_value_mrmr_quality_metrics.py",
            "test_biz_value_mrmr_ultra.py",
        )
        catchall_on_disk = [
            n for n in catchall_required
            if os.path.isfile(os.path.join(this_dir, n))
        ]
        total = len(layer_numbers) + len(catchall_on_disk)
        assert total >= 69, (
            f"Combined biz_value module roster size = {total}; floor is "
            f"69 (Layer 70 spec). Discovered layer numbers: "
            f"{sorted(layer_numbers)!r}; catch-alls on disk: "
            f"{catchall_on_disk!r}."
        )
        # L70 itself must be on disk -- the floor above could pass even
        # if L70 was somehow missing as long as other layers compensate.
        # Pin L70 explicitly so a future rename does not slip past.
        assert 70 in layer_numbers, (
            f"L70 layer module not discovered on disk; layer numbers "
            f"present: {sorted(layer_numbers)!r}."
        )


class TestNoEngineeredNameCollisions:

    def test_no_duplicate_engineered_names_post_fit(self, all_on_fitted):
        """Every engineered column name produced by the all-on fit must
        be unique within each specific-bucket roster AND across them
        (excluding the cumulative hybrid_orth / mi_greedy trackers
        which legitimately re-collect specific-bucket names).

        Pins the bug class where two FE stages converge on the same
        canonical name -- the resulting ``X[name]`` returns a 2-column
        DataFrame instead of a Series and downstream rank/correlation
        paths explode.
        """
        m = all_on_fitted[0]
        # Within-roster: every specific bucket must have unique entries.
        within_specific = tuple(
            a for a in _ENGINEERED_ROSTER_ATTRS
            if a != "hybrid_orth_features_"
        )
        per_roster_dupes: dict[str, list[str]] = {}
        for attr in within_specific:
            roster = getattr(m, attr, None)
            if roster is None:
                continue
            try:
                names = list(roster)
            except Exception:
                continue
            seen: set[str] = set()
            dupes: list[str] = []
            for nm in names:
                if nm in seen:
                    dupes.append(str(nm))
                else:
                    seen.add(nm)
            if dupes:
                per_roster_dupes[attr] = dupes
        assert not per_roster_dupes, (
            f"Engineered roster(s) contain duplicate column names: "
            f"{per_roster_dupes!r}. A regression in one FE stage is "
            f"emitting the same name twice; downstream X[name] selects "
            f"a 2-column DataFrame and rank/correlation paths break."
        )
        # Cross-roster: specific buckets must not share a name.
        specific_rosters = tuple(
            a for a in _ENGINEERED_ROSTER_ATTRS
            if a not in ("hybrid_orth_features_", "mi_greedy_features_")
        )
        owner: dict[str, str] = {}
        cross_dupes: dict[str, list[str]] = {}
        for attr in specific_rosters:
            roster = getattr(m, attr, None)
            if roster is None:
                continue
            for nm in roster:
                if nm in owner and owner[nm] != attr:
                    cross_dupes.setdefault(str(nm), []).append(attr)
                    if owner[nm] not in cross_dupes[str(nm)]:
                        cross_dupes[str(nm)].append(owner[nm])
                else:
                    owner[nm] = attr
        assert not cross_dupes, (
            f"Engineered column name(s) emitted by MULTIPLE specific FE "
            f"buckets: {cross_dupes!r}. Two distinct mechanisms are "
            f"producing identically-named columns; the provenance ledger "
            f"can no longer attribute the column to its origin "
            f"unambiguously."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--no-cov"])
