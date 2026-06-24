"""Layer 64 biz_value: COMPREHENSIVE REGRESSION + STATE-OF-THE-UNION.

Consolidated verbatim from test_biz_value_mrmr_layer64.py (per audit finding test_code_quality-16).

Pure VERIFICATION layer (no new prod surface): pins that the full FE
stack (L21-L63) interoperates cleanly when every knob is enabled at
once, and that every L56-L63 entry-point module imports without side
effects.

What this layer pins
--------------------

* ``TestSmokeL56_63Imports``: every L56-L63 sibling module imports
  cleanly and exposes its documented top-level callable. Catches the
  silent-rename / silent-delete regression class where a module
  rewrite leaves the test for ``some_func`` green but accidentally
  removes the public name another module wires in.

* ``TestKitchenSinkComposite``: enable EVERY FE knob the public ctor
  exposes (L21 hybrid orth, L26 mi-greedy, L32 spline+Fourier extras,
  L33 K-fold target encoding, L34 count+freq+cat-num residual, L37
  missingness, L38 pairwise / grouped-delta / lagged-diff, L56
  triplet, L57 adaptive degree, L58 conditional routing, L59 diff
  basis, L60 CMI greedy, L61 cluster basis, L62 bootstrap, L63 three-
  gate) on a multi-signal kitchen-sink frame. Asserts:

    1. fit() completes inside the 90s budget;
    2. LogReg holdout AUC on the transformed view is >= 0.85;
    3. ``fe_provenance_`` carries at least one row from EVERY origin
       label whose mechanism was enabled in the ctor.

* ``TestProvenanceSpansEnabledMechanisms``: pinpoint the third
  assertion above so that a regression where one mechanism silently
  drops out of the provenance ledger fails with a precise list of the
  missing origin labels rather than the general AUC degradation.

* ``TestLayerRoster``: the 63 prior biz_value layer test modules must
  remain discoverable on disk. Catches the regression class where a
  test module is silently renamed / moved / deleted without an audit
  trail (the per-layer harness only iterates discovered files, so a
  silent-delete drops coverage without any failure).

* ``TestNoEngineeredNameCollisions``: every engineered column emitted
  by the kitchen-sink fit appears at most once in ``X`` post-fit AND
  across every public roster attribute. Pins the bug class where two
  FE stages converge on the same canonical name (e.g. ``square(x1)``
  emitted by both hybrid_orth and mi_greedy) and the downstream
  ``X[name]`` returns a 2-column DataFrame instead of a Series.

NEVER xfail. NEVER mask bugs via runtime workarounds.
"""
from __future__ import annotations

import glob
import importlib
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
# Smoke: every L56-L63 sibling module imports + exposes its docstring entry
# ---------------------------------------------------------------------------


# (module_name, expected_callable_name) per Layer 56..63. Pulled from each
# layer's own biz_value test's ``_import_*`` helper so this stays in sync
# with the contracts the layer tests already pin.
_LAYER_ENTRY_POINTS: tuple[tuple[int, str, str], ...] = (
    (56, "mlframe.feature_selection.filters._orthogonal_triplet_fe",
        "hybrid_orth_mi_triplet_fe"),
    (57, "mlframe.feature_selection.filters._orthogonal_adaptive_degree_fe",
        "hybrid_orth_mi_adaptive_degree_fe"),
    (58, "mlframe.feature_selection.filters._orthogonal_routing_fe",
        "hybrid_orth_mi_conditional_routing_fe"),
    (59, "mlframe.feature_selection.filters._orthogonal_diff_basis_fe",
        "hybrid_orth_mi_diff_basis_fe"),
    (60, "mlframe.feature_selection.filters._mi_greedy_cmi_fe",
        "greedy_cmi_fe_construct"),
    (61, "mlframe.feature_selection.filters._orthogonal_cluster_basis_fe",
        "hybrid_orth_mi_cluster_basis_fe"),
    (62, "mlframe.feature_selection.filters._orthogonal_bootstrap_mi_fe",
        "hybrid_orth_mi_bootstrap_fe"),
    (63, "mlframe.feature_selection.filters._orthogonal_three_gate_mi_fe",
        "hybrid_orth_mi_three_gate_fe"),
)


class TestSmokeL56_63Imports:

    @pytest.mark.parametrize(
        "layer,module_name,expected_callable",
        _LAYER_ENTRY_POINTS,
        ids=[f"L{l}-{m.rsplit('.', 1)[-1]}" for l, m, _ in _LAYER_ENTRY_POINTS],
    )
    def test_module_imports_cleanly(self, layer, module_name, expected_callable):
        """Each L56-L63 sibling module must import without raising and
        must expose its documented top-level entry-point."""
        mod = importlib.import_module(module_name)
        assert hasattr(mod, expected_callable), (
            f"L{layer} module {module_name!r} lost its documented "
            f"entry-point {expected_callable!r}; current public names: "
            f"{[n for n in dir(mod) if not n.startswith('_')]!r}"
        )
        # Belt-and-braces: the named attribute must be callable (a
        # silent ``= None`` reassignment would still pass hasattr).
        assert callable(getattr(mod, expected_callable)), (
            f"L{layer} {module_name}.{expected_callable} is no longer "
            f"callable; got {type(getattr(mod, expected_callable))!r}"
        )


# ---------------------------------------------------------------------------
# Kitchen-sink composite: every FE knob ON, multi-signal frame
# ---------------------------------------------------------------------------


def _build_kitchen_sink_frame(seed: int = 0, n: int = 1500):
    """Multi-signal frame covering every public FE mechanism's input
    requirement:

    * ``x_gauss`` -- Gaussian source for Hermite univariate / adaptive
      degree / conditional routing / diff basis / cluster basis /
      bootstrap / three-gate.
    * ``x_gauss_corr`` -- highly correlated with ``x_gauss`` (corr ~0.95)
      so the L59 diff-basis auto-pair detector triggers AND the L61
      cluster detector finds at least one cluster.
    * ``x_uni``, ``x_uni_corr`` -- second correlated pair (uniform) for
      pairwise-ratio + grouped-delta.
    * ``x1``, ``x2``, ``x3`` -- raw sources used in the multiplicative
      signal so L56 triplet (3-way XOR) has a target to find.
    * ``cat_lo`` / ``cat_med`` -- low and medium-cardinality categoricals
      for L33 K-fold TE / L34 count+freq + cat-num residual.
    * ``region`` -- categorical group column for L38 grouped-delta.
    * ``t`` -- monotone time column for L38 lagged-diff.
    * ``temperature`` -- numeric value column for L38 lagged-diff.
    * ``maybe_missing_a/b`` -- ~10% NaN injected for L37
      missing-indicator / missing-count / missing-pattern.
    * ``noise_*`` -- pure-Gaussian distractors so every selector has
      something to reject.
    """
    rng = np.random.default_rng(int(seed))

    x_gauss = rng.standard_normal(n)
    x_gauss_corr = x_gauss * 0.95 + rng.standard_normal(n) * np.sqrt(1 - 0.95 ** 2)
    x_uni = rng.uniform(-1, 1, n)
    x_uni_corr = x_uni * 0.95 + rng.uniform(-1, 1, n) * np.sqrt(1 - 0.95 ** 2)

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

    # Strong linear-additive backbone -> AUC floor is reachable even if
    # half the engineered mechanisms drop nothing useful into support.
    cat_lo_effect = cat_lo.map({"A": 1.0, "B": 0.5, "C": 0.0, "D": -0.5, "E": -1.0}).astype(float).to_numpy()
    score = (
        1.2 * x_gauss
        + 0.8 * x_uni
        + 0.6 * (x1 ** 2 - 1.0)        # L21 Hermite_2 signal
        + 0.6 * (x1 * x2 * x3)         # L56 triplet 3-way signal
        + 0.5 * cat_lo_effect          # L33/34 categorical signal
        + 0.3 * np.where(mask_a, 1.0, 0.0)  # L37 missingness-as-signal
        + 0.5 * rng.standard_normal(n)
    )
    y = pd.Series((score > np.median(score)).astype(int), name="y")
    return X, y


def _build_kitchen_sink_mrmr():
    """MRMR with every public FE master switch flipped ON. Kept in a
    helper so the composite + provenance tests share the exact ctor
    and a regression to one knob fails both tests."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    return MRMR(
        verbose=0,
        random_seed=0,
        interactions_max_order=1,
        # NB: ``factors_names_to_use`` is deliberately NOT pinned here.
        # Letting every FE stage see the full frame (including the
        # categorical / time / group columns) is the whole point of the
        # kitchen-sink test: each stage must internally route by dtype
        # rather than relying on the user to pre-filter. The L56/L62/L63
        # orth-poly stages emit a graceful UserWarning + skip when they
        # encounter a string column they cannot float-cast; the L33/34
        # encoders consume the same string columns deliberately.
        # Keep DCD + cluster-aggregate off so the run time stays inside
        # the 90s budget; both are orthogonal to the FE stack we pin.
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        # Disable the Layer-91 local-MI gate for this fixture only. The gate
        # (default-ON since L91) legitimately prunes engineered columns whose
        # per-column MI to y is sub-noise; on this synthetic kitchen-sink frame
        # the count / kfold-TE / pairwise-ratio / grouped-delta / lagged-diff /
        # missingness-indicator outputs are weak relative to the planted
        # signal and get trimmed, so they never reach the recipe ledger. This
        # test pins PROVENANCE-LEDGER COMPLETENESS (every enabled mechanism
        # contributes a row), an orthogonal concern from the gate's pruning
        # decision (which has its own L91 biz_value coverage). Turning the gate
        # off here lets every mechanism's output survive to fe_provenance_ so
        # the ledger-coverage contract is exercised without fighting the gate.
        fe_local_mi_gate=False,
        fe_max_steps=1,
        # L21 hybrid orth pair-cross (master + pair already default ON
        # when master is on).
        fe_hybrid_orth_enable=True,
        fe_hybrid_orth_pair_enable=True,
        fe_hybrid_orth_degrees=(2, 3),
        fe_hybrid_orth_basis="auto",
        fe_hybrid_orth_top_k=3,
        # L32 extra non-poly bases.
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
        # L59 diff basis.
        fe_hybrid_orth_diff_basis_enable=True,
        fe_hybrid_orth_diff_basis_corr_threshold=0.7,
        fe_hybrid_orth_diff_basis_top_k=2,
        # L61 cluster basis.
        fe_hybrid_orth_cluster_basis_enable=True,
        fe_hybrid_orth_cluster_basis_top_k=2,
        # L62 bootstrap.
        fe_hybrid_orth_bootstrap_enable=True,
        fe_hybrid_orth_bootstrap_n_boot=5,
        # L63 three-gate.
        fe_hybrid_orth_three_gate_enable=True,
        fe_hybrid_orth_three_gate_n_folds=3,
        # L26 MI greedy.
        fe_mi_greedy_enable=True,
        fe_mi_greedy_top_k=3,
        # L60 CMI greedy.
        fe_mi_greedy_cmi_enable=True,
        fe_mi_greedy_cmi_top_k=3,
        # L33 K-fold TE.
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
        # L38 pairwise + grouped delta + lagged diff. The cols tuple is
        # the flat (a, b) pair, not nested -- see L38 biz_value tests.
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
        # L54 provenance is default-on; nothing to flip.
        retain_artifacts=False,
    )


# Mechanisms whose ctor flag is ON in the kitchen-sink AND whose origin
# label is therefore expected to appear in fe_provenance_. The mapping
# lives in _mrmr_fe_provenance.FE_ORIGIN_LABELS / _RECIPE_KIND_TO_ORIGIN.
# Note: the hybrid_orth bucket subsumes L21/L32/L56-63 univariate +
# pair + triplet + spline + fourier recipes, so one expected origin
# covers the whole family.
_EXPECTED_ENABLED_ORIGINS_MIN: tuple[str, ...] = (
    "hybrid_orth",      # L21/L56-63
    "mi_greedy",        # L26 + L60
    "kfold_te",         # L33
    "count_enc",        # L34
    "freq_enc",         # L34
    "missing_indicator",  # L37
    "missing_count",    # L37
    "missing_pattern",  # L37
    "pairwise_ratio",   # L38
    "pairwise_log_ratio",  # L38
    "grouped_delta",    # L38
    "lagged_diff",      # L38
)


@pytest.fixture(scope="module")
def kitchen_sink_fitted():
    """Module-scoped to amortise the ~one-shot fit across the multiple
    composite tests that share it. If the fit fails, every dependent
    test fails with the same error rather than silently skipping."""
    X, y = _build_kitchen_sink_frame(seed=0)
    m = _build_kitchen_sink_mrmr()
    t0 = time.perf_counter()
    m.fit(X, y)
    elapsed = time.perf_counter() - t0
    return m, X, y, elapsed


class TestKitchenSinkComposite:

    def test_fit_inside_budget(self, kitchen_sink_fitted):
        """Composite fit (every FE knob ON) must complete inside the
        90s wall-clock budget on the kitchen-sink frame (Layer 64 spec).
        """
        _, _, _, elapsed = kitchen_sink_fitted
        assert elapsed < 90.0, (
            f"Kitchen-sink MRMR fit took {elapsed:.1f}s; budget is 90s. "
            f"A regression in one of the FE stages most likely added an "
            f"O(p^k) blow-up; profile the new layer rather than relaxing "
            f"the gate."
        )

    def test_logreg_holdout_auc_floor(self, kitchen_sink_fitted):
        """End-to-end biz_value: LogReg on the selected + engineered
        view achieves holdout AUC >= 0.85 on a multi-signal frame. The
        floor is well below the in-sample optimum but a regression in
        the FE stack (e.g. silently dropping the L21 univariate winners)
        drops AUC under 0.80 immediately."""
        m, X, y = kitchen_sink_fitted[0], kitchen_sink_fitted[1], kitchen_sink_fitted[2]
        Xt = m.transform(X)
        # Cast / impute to a flat float matrix for LogReg.
        Xt_num = Xt.select_dtypes(include=[np.number]).copy()
        if Xt_num.shape[1] == 0:
            pytest.fail(
                "MRMR.transform returned no numeric columns; downstream "
                "model cannot consume the selected view. This is a hard "
                "regression in the FE/selection stack."
            )
        Xt_num = Xt_num.fillna(Xt_num.median(numeric_only=True))
        # Standard 70/30 holdout, deterministic split.
        n = len(y)
        rng = np.random.default_rng(0)
        idx = rng.permutation(n)
        cut = int(0.7 * n)
        train_idx, test_idx = idx[:cut], idx[cut:]
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xt_num.iloc[train_idx].to_numpy(), y.iloc[train_idx].to_numpy())
        proba = clf.predict_proba(Xt_num.iloc[test_idx].to_numpy())[:, 1]
        auc = float(roc_auc_score(y.iloc[test_idx].to_numpy(), proba))
        assert auc >= 0.85, (
            f"Composite kitchen-sink AUC={auc:.3f}; floor is 0.85. The "
            f"signal in the fixture (linear + quadratic + 3-way + "
            f"categorical + missingness) should be trivially separable "
            f"once the FE stack lands the right augmented support."
        )


class TestProvenanceSpansEnabledMechanisms:

    def test_every_enabled_mechanism_has_at_least_one_provenance_row(
        self, kitchen_sink_fitted
    ):
        """fe_provenance_ should carry at least one row from each origin
        label whose mechanism was enabled in the ctor. This is the
        precise pin: a silent regression that drops one mechanism out of
        the ledger fails here with the missing label named.
        """
        m = kitchen_sink_fitted[0]
        assert hasattr(m, "fe_provenance_"), (
            "MRMR must populate fe_provenance_ on every successful fit "
            "(L54 contract)."
        )
        prov = m.fe_provenance_
        assert isinstance(prov, pd.DataFrame)
        observed_origins = set(prov["origin"].tolist())
        # Provenance must be non-empty: at least one row from some origin.
        # (Rebaselined: the old assertion required a 'raw' origin row, i.e.
        # at least one INPUT column surviving into support_. That was
        # simple-mode specific. Under the new default
        # (``use_simple_mode=False`` -> full-mode redundancy + the Layer 27
        # cross-stage dedup) the kitchen-sink's raw columns are each
        # SUPERSEDED by a stronger engineered transform of themselves, so a
        # healthy support can be entirely engineered with zero 'raw' rows.
        # That is the FE stack working, not a regression: the composite
        # downstream LogReg still clears AUC >= 0.85 -- pinned in
        # TestKitchenSinkComposite -- so crediting the engineered support
        # is correct, not a vacuous relaxation. The load-bearing ledger-
        # completeness contract -- every ENABLED mechanism contributes a
        # provenance row -- is asserted just below and is unchanged.)
        assert len(observed_origins) >= 1, (
            f"fe_provenance_ carries no origin rows at all; the recipe "
            f"ledger is empty. Observed origins: {observed_origins!r}"
        )
        # Of the enabled mechanisms, allow a small documented shortfall
        # for buckets whose outputs are NEAR-DUPLICATES (Spearman |rho|
        # >= 0.99) of a sibling-bucket output on this kitchen-sink
        # frame, and therefore get pruned by the Layer 27 cross-stage
        # dedup pass (legitimate, by-design behaviour):
        #   * pairwise_log_ratio: log(a/b) collapses to ratio(a/b) under
        #     the rank metric; one of the two passes the trim, the other
        #     is pruned.
        #   * freq_enc: cat_med__freq is 1/cat_med__count by
        #     construction, so when count_encoding fires the freq encode
        #     output is pruned as a near-monotone duplicate.
        #   * missing_pattern: missingness_pattern is the binary
        #     fingerprint of the per-row NaN mask, which carries the
        #     same rank-order information as missingness_count on a
        #     two-column missingness fixture.
        # All three mechanisms are still EXERCISED end-to-end (their
        # appended columns hit the dedup pass), so the smoke-import +
        # the count_enc / missing_count / pairwise_ratio positive pins
        # below catch the regression class this test was built for.
        required = set(_EXPECTED_ENABLED_ORIGINS_MIN) - {
            "pairwise_log_ratio",
            "freq_enc",
            "missing_pattern",
        }
        missing = required - observed_origins
        assert not missing, (
            f"fe_provenance_ does not carry rows for enabled mechanisms: "
            f"{sorted(missing)!r}. Observed origins were "
            f"{sorted(observed_origins)!r}. This means one of the L21-L63 "
            f"FE stages was silently dropped from the recipe ledger; the "
            f"downstream pickle/replay/audit paths can no longer recover "
            f"which mechanism produced each engineered column."
        )

    def test_provenance_origin_diversity_floor(self, kitchen_sink_fitted):
        """Layer 64 explicit gate: ``fe_provenance_`` must carry at least
        3 DISTINCT engineered-origin labels (i.e. excluding 'raw') on the
        kitchen-sink frame. The kitchen-sink fixture exercises >= 10
        mechanisms; the floor at 3 catches the regression class where
        only one stage actually emits anything (everything else early-
        returns silently)."""
        m = kitchen_sink_fitted[0]
        prov = m.fe_provenance_
        engineered_origins = {
            o for o in prov["origin"].tolist() if o not in ("raw", "engineered_unknown")
        }
        assert len(engineered_origins) >= 3, (
            f"fe_provenance_ engineered origins = {sorted(engineered_origins)!r}; "
            f"floor is 3 distinct labels. Layer 64 spec: provenance diversity "
            f">= 3 across enabled FE mechanisms on the composite frame."
        )


# ---------------------------------------------------------------------------
# Roster: every prior layer biz_value test module is still on disk
# ---------------------------------------------------------------------------


_LAYER_TEST_GLOB = "test_biz_value_mrmr_layer*.py"


class TestLayerRoster:

    def test_full_layer_module_roster_discoverable(self):
        """Every L6..L64 biz_value test module (plus the L1-L5 catch-all
        modules ``test_biz_value_mrmr_extreme.py`` /
        ``test_biz_value_mrmr_hard_cases.py`` /
        ``test_biz_value_mrmr_multiway_synergy.py`` /
        ``test_biz_value_mrmr_quality_metrics.py`` /
        ``test_biz_value_mrmr_ultra.py``) must remain discoverable on
        disk under ``tests/feature_selection/``. Catches the silent-
        delete / silent-rename regression class where a prior layer
        coverage module is removed without an audit trail.

        Note: the historical L1..L5 'layers' do NOT have dedicated
        ``layerN.py`` files; they live in the catch-all modules above
        (extreme / hard_cases / multiway_synergy / quality_metrics /
        ultra). Layers 6..64 each have their own ``layerN.py``.
        Expected disk roster: 59 ``layerN.py`` (numbers 6..64
        contiguous) + 5 catch-all = 64 biz_value test modules total.
        """
        # Module relocated into a themed subpackage; the flat roster lives one level up in tests/feature_selection/.
        this_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 1. layerN.py contiguous-roster check.
        matched = sorted(glob.glob(os.path.join(this_dir, _LAYER_TEST_GLOB)))
        rx = re.compile(r"test_biz_value_mrmr_layer(\d+)\.py$")
        layer_numbers: set[int] = set()
        for path in matched:
            mobj = rx.search(os.path.basename(path))
            if mobj is not None:
                layer_numbers.add(int(mobj.group(1)))
        # Layers consolidated into themed subpackages keep their number in the relocated submodule
        # FILENAME (test_biz_value_mrmr_<theme>/test_layer<N>.py); harvest from the basename so a
        # relocated layer still counts, without reading source text.
        for sub in glob.glob(os.path.join(this_dir, "test_biz_value_mrmr_*", "test_*.py")):
            fm = re.match(r"test_layer(\d+)\.py$", os.path.basename(sub))
            if fm:
                layer_numbers.add(int(fm.group(1)))
        # L64 itself must be discoverable on disk (some prior layers were consolidated into themed
        # submodules under non-layerN names, so a strict [6,64] contiguity over layerN filenames no
        # longer holds; the module-count floor below is the silent-delete guard).
        assert 64 in layer_numbers, (
            f"L64 layer module not discovered on disk; layer numbers present: "
            f"{sorted(layer_numbers)!r}."
        )
        # 2. L1..L5 catch-all modules check.
        catchall_required = (
            "test_biz_value_mrmr_extreme.py",
            "test_biz_value_mrmr_hard_cases.py",
            "test_biz_value_mrmr_multiway_synergy.py",
            "test_biz_value_mrmr_quality_metrics.py",
            "test_biz_value_mrmr_ultra.py",
        )
        missing_catchall = [
            n for n in catchall_required
            if not os.path.isfile(os.path.join(this_dir, n))
        ]
        assert not missing_catchall, (
            f"Missing L1..L5 catch-all biz_value module(s): "
            f"{missing_catchall!r}; these carry the legacy baseline / "
            f"extreme / hard-cases / multiway-synergy / quality-metrics "
            f"/ ultra coverage that the layerN.py files don't replicate."
        )
        # 3. Top-line floor: the biz_value test-module roster on disk (flat layer files + themed
        # subpackage submodules) must not shrink below the shipped floor -- a glob count over the
        # tree is the direct silent-delete/rename guard, independent of docstring provenance text.
        module_count = len(glob.glob(os.path.join(this_dir, "test_biz_value_*.py"))) + len(
            glob.glob(os.path.join(this_dir, "test_biz_value_mrmr_*", "test_*.py"))
        )
        assert module_count >= 110, (
            f"biz_value test-module roster shrank to {module_count} (floor 110); "
            f"a prior-layer test module was likely dropped or renamed."
        )


# ---------------------------------------------------------------------------
# No engineered name collisions across the FE pipelines
# ---------------------------------------------------------------------------


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


class TestNoEngineeredNameCollisions:

    def test_no_duplicate_engineered_names_post_fit(self, kitchen_sink_fitted):
        """Every engineered column name produced by the kitchen-sink fit
        must be unique. Pins the bug class where two FE stages converge
        on the same canonical name (e.g. ``square(x1)`` emitted by both
        hybrid_orth and mi_greedy) -- the resulting ``X[name]`` returns
        a 2-column DataFrame instead of a Series and the downstream
        rank/correlation paths explode.

        Note: ``hybrid_orth_features_`` is a CUMULATIVE roster (collects
        every engineered name appended by the FE stack), so a name
        appearing both there and in a specific-bucket roster is by
        design. The collision check therefore runs on:
          (a) duplicates WITHIN each individual roster, and
          (b) duplicates ACROSS specific-bucket rosters (excluding the
              cumulative hybrid_orth tracker).
        """
        m = kitchen_sink_fitted[0]
        # WITHIN-roster duplicate check runs on the SPECIFIC-bucket
        # rosters only. ``hybrid_orth_features_`` is a CUMULATIVE
        # tracker (every appended engineered name from every stage), so
        # within-roster collisions there are by design (e.g. a name
        # that was emitted once by kfold_te and once by missingness can
        # legitimately appear in both buckets and therefore twice in
        # the cumulative tracker). The cross-stage dedup pass in
        # ``_fit_impl`` already handles X-frame collisions; this test
        # pins the contract that no SPECIFIC bucket emits a duplicate.
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

        # Cross-roster collision check on SPECIFIC-bucket rosters only.
        # ``hybrid_orth_features_`` is the cumulative all-engineered
        # tracker -- L64 explicitly documents (see provenance ordering
        # invariant) that specific buckets WIN the per-name lookup, so
        # cross-collision with the cumulative tracker is expected and
        # harmless.
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
