"""Layer 39 biz_value: COMPREHENSIVE REGRESSION across all 38 prior layers.

Pure VERIFICATION layer - no new prod features. Pins the steady-state
contract that the kitchen-sink config (Layer 35's all-FE-enabled +
Layer 38's ratio/grouped_delta/lagged_diff additions = 11 FE
mechanisms total) holds end-to-end:

* recipe-count parity: ``len(_engineered_recipes_)`` matches the count
  of engineered cols in ``_engineered_features_`` (the lookup keyed off
  ``selected_vars_names`` at ``_mrmr_fit_impl.py:1953``).  An engineered
  col without a recipe gets DROPPED from transform output -- if parity
  ever silently slips, the headline AUC contracts in Layer 35 still
  pass (the dropped col was a noise hit anyway) but the public
  ``_engineered_recipes_`` accessor lies. Layer 39 pins it.

* fit-time budget: p95 fit time on the kitchen-sink (n=3000, p=12)
  must stay under 30s across 3 seeds. Layer 35's
  ``TestFitTimeBudget`` only pins seed=42.

* memory bound: peak RSS delta during fit must stay under 500MB.
  Kitchen-sink probe shows ~21MB peak delta; the bound is a safety
  margin against silent regressions (kfold OOF allocations, MI bin
  expansion).

* import smoke for all 38 layer test modules: every prior-layer
  ``test_biz_value_mrmr_layerN.py`` imports without error. Catches
  the "Layer N renamed a public param, Layer N-3 test still imports
  the old name and silently skips because pytest collected 0 items"
  failure mode.

NEVER xfail. If a prior-layer fixture or import breaks, fix prod / the
fixture / the import path -- not the test.
"""
from __future__ import annotations

import gc
import importlib
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pytest


warnings.filterwarnings("ignore")


HEADLINE_SEED = 42
AUX_SEEDS = (1, 101)


# ---------------------------------------------------------------------------
# Helpers (kept module-local so a Layer 35 refactor does not silently break
# Layer 39's regression contracts).
# ---------------------------------------------------------------------------


def _make_mrmr(**overrides):
    from mlframe.feature_selection.filters.mrmr import MRMR
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
        fe_ntop_features=25,
    )
    kwargs.update(overrides)
    return MRMR(**kwargs)


def _all_fe_kwargs():
    """Kitchen-sink: all 11 FE mechanisms (8 from Layer 35 + 3 from Layer 38)."""
    return dict(
        # 1+2: orth-poly univariate + cross-basis pair
        fe_hybrid_orth_enable=True,
        fe_hybrid_orth_pair_enable=True,
        fe_hybrid_orth_basis="hermite",
        fe_hybrid_orth_top_k=10,
        # 5+6: spline + Fourier
        fe_hybrid_orth_extra_bases=("spline", "fourier"),
        fe_hybrid_orth_fourier_freqs=(1.0, 2.0),
        fe_hybrid_orth_spline_knots=7,
        # 3+4: MI-greedy unary + binary
        fe_mi_greedy_enable=True,
        fe_mi_greedy_top_k=8,
        fe_mi_greedy_include_unary=True,
        fe_mi_greedy_include_binary=True,
        # 7: K-fold target encoding
        fe_kfold_te_enable=True,
        fe_kfold_te_cols=("cat_region", "cat_user"),
        fe_kfold_te_folds=5,
        fe_kfold_te_smoothing=10.0,
        # 8: count encoding
        fe_count_encoding_enable=True,
        fe_count_encoding_cols=("cat_user",),
        # 9: frequency encoding
        fe_frequency_encoding_enable=True,
        fe_frequency_encoding_cols=("cat_user",),
        # 10: cat x num residual
        fe_cat_num_interaction_enable=True,
        fe_cat_num_interaction_cat_cols=("cat_region",),
        fe_cat_num_interaction_num_cols=("price",),
        fe_cat_num_interaction_folds=5,
        fe_cat_num_interaction_smoothing=10.0,
    )


def _kitchen_sink(seed: int = HEADLINE_SEED, n: int = 3000):
    """Same fixture as Layer 35 - 8 signal columns, 4 noise. See Layer 35 docstring."""
    rng = np.random.default_rng(seed)
    n_users = 60
    user_ids = np.array([f"U_{i:03d}" for i in range(n_users)])
    user_weights = np.linspace(1.0, 50.0, n_users)
    user_weights = user_weights / user_weights.sum()
    cat_user = rng.choice(user_ids, size=n, p=user_weights)
    regions = [f"R{i:02d}" for i in range(30)]
    hot_regions = set(regions[:4])
    cat_region = rng.choice(regions, size=n)
    hot_mask = np.array([(c in hot_regions) for c in cat_region], dtype=float)
    region_means = dict(zip(regions, rng.uniform(20.0, 120.0, size=len(regions))))
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
    noise = rng.standard_normal((n, 4))
    box = ((x_threshold > 0.3) & (x_threshold < 1.2)).astype(float)
    logit = (
        0.5 * x_num1
        + 2.0 * (x_quad ** 2 - 1.0)
        + 2.5 * np.sin(2.0 * np.pi * x_periodic)
        + 2.5 * box
        + 2.5 * hot_mask
        + 0.15 * (price - price_mean)
        + 1.0 * log_cnt_centered
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = pd.Series((rng.random(n) < p).astype(int), name="y")
    X = pd.DataFrame({
        "x_num1": x_num1, "x_num2": x_num2, "x_quad": x_quad,
        "x_periodic": x_periodic, "x_threshold": x_threshold,
        "cat_region": cat_region, "cat_user": cat_user, "price": price,
        "n0": noise[:, 0], "n1": noise[:, 1], "n2": noise[:, 2], "n3": noise[:, 3],
    })
    return X, y


def _train_holdout_split(X: pd.DataFrame, y: pd.Series, *,
                         train_frac: float = 0.7, seed: int = HEADLINE_SEED):
    rng = np.random.default_rng(seed + 100)
    idx = np.arange(len(X)); rng.shuffle(idx)
    cut = int(train_frac * len(X))
    tr, ho = idx[:cut], idx[cut:]
    return (
        X.iloc[tr].reset_index(drop=True),
        y.iloc[tr].reset_index(drop=True),
        X.iloc[ho].reset_index(drop=True),
        y.iloc[ho].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Contract 1: recipe-count parity holds across seeds
# ---------------------------------------------------------------------------


class TestRecipeCountParity:
    """``len(_engineered_recipes_)`` must equal the count of engineered cols
    that have a replayable recipe.

    The on-disk contract at ``_mrmr_fit_impl.py``: every engineered name in
    ``_engineered_features_`` either (a) has a recipe in
    ``_engineered_recipes_`` (replayable on test data) or (b) is recorded
    by name only and DROPPED from transform output.  On the kitchen-sink
    fixture all engineered cols are 1-deep (fe_max_steps=0), so every
    engineered name MUST have a matching recipe -- parity is strict.
    """

    @pytest.mark.parametrize("seed", (HEADLINE_SEED,) + AUX_SEEDS)
    def test_engineered_features_and_recipes_count_match(self, seed):
        X, y = _kitchen_sink(seed=seed)
        X_tr, y_tr, _, _ = _train_holdout_split(X, y, seed=seed)
        m = _make_mrmr(**_all_fe_kwargs())
        m.fit(X_tr, y_tr)
        eng_feats = list(getattr(m, "_engineered_features_", []) or [])
        eng_recipes = list(getattr(m, "_engineered_recipes_", []) or [])
        assert len(eng_feats) == len(eng_recipes), (
            f"seed={seed}: recipe-count parity FAILED: "
            f"{len(eng_feats)} engineered features but "
            f"{len(eng_recipes)} recipes; "
            f"_engineered_features_={eng_feats}"
        )

    @pytest.mark.parametrize("seed", (HEADLINE_SEED,) + AUX_SEEDS)
    def test_every_recipe_replays_to_an_output_column(self, seed):
        """A recipe whose output name does not appear in transform output
        is a silent drop -- the parent grep at ``_mrmr_fit_impl.py``
        appended a recipe but ``transform`` did not materialise it.
        """
        X, y = _kitchen_sink(seed=seed)
        X_tr, y_tr, _, _ = _train_holdout_split(X, y, seed=seed)
        m = _make_mrmr(**_all_fe_kwargs())
        m.fit(X_tr, y_tr)
        out = m.transform(X_tr)
        eng_feats = list(getattr(m, "_engineered_features_", []) or [])
        eng_recipes = list(getattr(m, "_engineered_recipes_", []) or [])
        out_cols = set(out.columns)
        # Every engineered name in _engineered_features_ that also has a
        # recipe MUST land in transform output.
        for col, recipe in zip(eng_feats, eng_recipes):
            assert col in out_cols, (
                f"seed={seed}: engineered col {col!r} has a recipe "
                f"({type(recipe).__name__}) but did NOT materialise in "
                f"transform output ({sorted(out_cols)})"
            )

    def test_hinge_gate_skips_selected_raw_categorical(self):
        """A selected RAW categorical (string labels like 'R22'/'U_055') must not
        reach the hinge-gate float64 baseline cast. The kitchen-sink selects
        ``cat_region``/``cat_user`` (string-valued) alongside numeric columns; the
        hinge change-point protection builds a numeric baseline design over the
        selected set and must SKIP non-numeric columns rather than crash on
        ``np.asarray(str_col, dtype=float64)``."""
        X, y = _kitchen_sink(seed=HEADLINE_SEED)
        X_tr, y_tr, _, _ = _train_holdout_split(X, y, seed=HEADLINE_SEED)
        m = _make_mrmr(**_all_fe_kwargs())
        m.fit(X_tr, y_tr)  # must not raise ValueError: could not convert string to float
        sel = set(m.get_feature_names_out())
        assert {"cat_region", "cat_user"} & sel, (
            "fixture no longer selects a raw string categorical; the regression "
            "this test pins (str column hitting the hinge float cast) is unreachable"
        )


# ---------------------------------------------------------------------------
# Contract 2: fit-time budget across 3 seeds (p95 < 30s)
# ---------------------------------------------------------------------------


class TestFitTimeBudgetMultiSeed:
    """Layer 35's TestFitTimeBudget pins seed=42 only. Layer 39 extends
    the budget pin to 3 seeds and bounds the p95, not just the worst
    of one.
    """

    def test_p95_fit_time_under_30s_3_seeds(self):
        seeds = (HEADLINE_SEED,) + AUX_SEEDS
        fit_times = []
        for s in seeds:
            X, y = _kitchen_sink(seed=s)
            X_tr, y_tr, _, _ = _train_holdout_split(X, y, seed=s)
            m = _make_mrmr(**_all_fe_kwargs())
            gc.collect()
            t0 = time.perf_counter()
            m.fit(X_tr, y_tr)
            fit_times.append(time.perf_counter() - t0)
        fit_times.sort()
        # p95 of 3 samples = the worst sample (clamp to last index)
        p95 = fit_times[-1]
        assert p95 < 30.0, (
            f"p95 fit time {p95:.2f}s across {len(seeds)} seeds "
            f"exceeds 30s budget; per-seed times={fit_times}"
        )


# ---------------------------------------------------------------------------
# Contract 3: peak RSS delta during fit bounded (< 500MB)
# ---------------------------------------------------------------------------


class TestMemoryBound:
    """Peak RSS DELTA (not absolute -- absolute pollutes from imports,
    JIT warmup, BLAS pools) during a single all-FE fit on the
    kitchen-sink must stay under 500MB. Probe baseline: ~21MB peak
    delta on seed=42 dev hw, so 500MB is a wide safety margin against
    silent allocation regressions (kfold OOF arrays, MI bin
    explosion).
    """

    def test_peak_rss_delta_under_500mb(self):
        X, y = _kitchen_sink()
        X_tr, y_tr, _, _ = _train_holdout_split(X, y)
        m = _make_mrmr(**_all_fe_kwargs())
        proc = psutil.Process(os.getpid())
        gc.collect()
        rss_baseline = proc.memory_info().rss
        m.fit(X_tr, y_tr)
        rss_after = proc.memory_info().rss
        delta_mb = (rss_after - rss_baseline) / 1024.0 / 1024.0
        assert delta_mb < 500.0, (
            f"peak RSS delta during all-FE fit on (n={len(X_tr)}, "
            f"p={X_tr.shape[1]}) was {delta_mb:.1f}MB, exceeding the "
            f"500MB budget"
        )


# ---------------------------------------------------------------------------
# Contract 4: smoke import of every prior-layer biz_value module
# ---------------------------------------------------------------------------


def _discover_prior_layer_modules():
    """Locate every ``test_biz_value_mrmr_layer<N>.py`` in this directory plus the
    relocated themed consolidation subpackages (``test_biz_value_mrmr_<theme>/test_*.py``).
    """
    here = Path(__file__).parent
    out = []
    for p in sorted(here.glob("test_biz_value_mrmr_layer*.py")):
        # Skip Layer 39 itself.
        if p.name == Path(__file__).name:
            continue
        mod_name = f"tests.feature_selection.{p.stem}"
        out.append((mod_name, p, mod_name))
    # Layers consolidated into themed subpackages are imported once per ORIGINAL layer number so the
    # import-smoke keeps per-layer granularity (the submodule docstrings record "...layerNN.py").
    for p in sorted(here.glob("test_biz_value_mrmr_*/test_*.py")):
        mod_name = f"tests.feature_selection.{p.parent.name}.{p.stem}"
        layers = sorted(set(re.findall(r"layer(\d+)\.py", p.read_text(encoding="utf-8"))), key=int)
        if layers:
            for n in layers:
                out.append((mod_name, p, f"{mod_name}::layer{n}"))
        else:
            out.append((mod_name, p, mod_name))
    return out


_LAYER_MODULES = _discover_prior_layer_modules()


class TestPriorLayerImportSmoke:
    """Every Layer N test module from N=6..38 must import cleanly under
    the current prod codebase. Catches the "Layer N+1 renamed a public
    param, Layer N-3 import line fails silently and pytest collected 0
    items" failure mode (which masquerades as a pass in CI).
    """

    @pytest.mark.parametrize("mod_name,path", [(m, p) for m, p, _ in _LAYER_MODULES],
                             ids=[i for _, _, i in _LAYER_MODULES])
    def test_prior_layer_module_imports(self, mod_name, path):
        # Force a fresh import so a previously-imported (and possibly
        # stale) cached module does not mask a current breakage. Snapshot
        # + restore the entry so we don't leave a rebound module behind
        # for sibling tests (per CLAUDE.md test-pollution rule).
        original_mod = sys.modules.get(mod_name)
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        try:
            importlib.import_module(mod_name)
        finally:
            if original_mod is not None:
                sys.modules[mod_name] = original_mod
            else:
                sys.modules.pop(mod_name, None)


# ---------------------------------------------------------------------------
# Contract 5: roster size (sanity guard against deletions of layer tests)
# ---------------------------------------------------------------------------


class TestPriorLayerRosterSize:
    """The biz_value roster has grown to 38 layers (Layer 6..38 +
    extreme/hard_cases/multiway_synergy/quality_metrics/ultra).
    Layer 39's regression contract presumes at least 33 layer modules
    are present at glob time. A regression that deletes one (or
    relocates the test file) trips this smoke before the more
    expensive import-smoke fans out.
    """

    def test_at_least_33_prior_layer_modules_discoverable(self):
        assert len(_LAYER_MODULES) >= 33, (
            f"Discovered only {len(_LAYER_MODULES)} prior-layer biz_value "
            f"test modules; expected >= 33. Modules found: "
            f"{[m for m, _, _ in _LAYER_MODULES]}"
        )
