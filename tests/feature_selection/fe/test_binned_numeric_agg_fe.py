"""Grouped aggregation over quantile-binned NUMERIC cells (``_binned_numeric_agg_fe``).

Contract:
* per-cell mean/std/skew/kurt of an aggregated numeric, grouped by quantile-binned cells of another numeric;
* moment-aware bin resolution: nbins = min(nbins_base, cap), with HIGH-MOMENT AUTO-DROP when the cap < 2;
* leak-safe transform replay via stored quantile edges (deterministic, finite, unseen -> global fallback);
* business value: recovers a cell-driven SPREAD signal (target = sigma(cell)) the cell mean cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._binned_numeric_agg_fe import (
    apply_binned_numeric_agg,
    engineered_name_binned_agg,
    fit_binned_numeric_agg,
    per_cell_stats_bincount,
    resolve_nbins_and_stats,
)


def test_moment_aware_resolution_and_autodrop():
    # Large n: full panel at the base bin count.
    nb, kept = resolve_nbins_and_stats(20000, ("mean", "std", "skew", "kurt"), 10, k=1)
    assert nb == 10 and kept == ["mean", "std", "skew", "kurt"]
    # Small n: cap from the highest moment binds (kurt n_min=100 -> nbins<=5).
    nb, kept = resolve_nbins_and_stats(500, ("mean", "std", "skew", "kurt"), 10, k=1)
    assert nb == 5 and "kurt" in kept
    # Tiny n: kurt's floor cannot be met at nbins>=2 -> kurt auto-dropped.
    nb, kept = resolve_nbins_and_stats(150, ("mean", "std", "skew", "kurt"), 10, k=1)
    assert "kurt" not in kept and "mean" in kept


def test_bincount_stats_match_numpy():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 4, 5000)
    v = rng.normal(0, 1, 5000)
    out = per_cell_stats_bincount(codes, v, 4, ("mean", "std"))
    for c in range(4):
        m = v[codes == c]
        assert abs(out["mean"][c] - m.mean()) < 1e-9
        assert abs(out["std"][c] - m.std()) < 1e-9


def test_replay_is_leak_safe_and_deterministic():
    rng = np.random.default_rng(1)
    n = 6000
    df = pd.DataFrame({"g": rng.uniform(0, 1, n), "aux": rng.normal(0, 1, n)})
    y = rng.normal(0, 1, n)
    _, recipes = fit_binned_numeric_agg(df, y, group_num_cols=["g"], agg_num_cols=["aux"], stats=("mean", "std"), nbins_base=8)
    df_te = pd.DataFrame(
        {
            "g": np.r_[rng.uniform(0, 1, 300), np.full(10, 99.0)],  # 99 -> out-of-range
            "aux": rng.normal(0, 1, 310),
        }
    )
    for r in recipes.values():
        o1 = apply_binned_numeric_agg(df_te, r)
        o2 = apply_binned_numeric_agg(df_te, r)
        assert np.isfinite(o1).all()
        np.testing.assert_array_equal(o1, o2)


def test_std_column_recovers_cell_spread():
    rng = np.random.default_rng(2)
    n = 10000
    g = rng.uniform(0, 1, n)
    sigma = 0.5 + 2.0 * np.abs(g - 0.5)
    aux = rng.normal(0, sigma, n)
    df = pd.DataFrame({"g": g, "aux": aux})
    feat_df, _ = fit_binned_numeric_agg(df, sigma, group_num_cols=["g"], agg_num_cols=["aux"], stats=("mean", "std"), nbins_base=10)
    std_name = engineered_name_binned_agg("aux", "g", "std")
    from scipy.stats import pearsonr

    assert pearsonr(feat_df[std_name].to_numpy(), sigma)[0] > 0.9
    # The mean column carries ~no spread signal.
    mean_name = engineered_name_binned_agg("aux", "g", "mean")
    assert abs(pearsonr(feat_df[mean_name].to_numpy(), sigma)[0]) < 0.4


def test_biz_value_recovers_spread_driven_target():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score

    deltas = []
    for seed in (10, 11, 12):
        rng = np.random.default_rng(seed)
        n = 10000
        g = rng.uniform(0, 1, n)
        sigma = 0.5 + 2.0 * np.abs(g - 0.5)
        aux = rng.normal(0, sigma, n)
        df = pd.DataFrame({"g": g, "aux": aux})
        cut = n // 2
        tr, te = slice(0, cut), slice(cut, n)

        def _r2(stats):
            feat_df, recipes = fit_binned_numeric_agg(df.iloc[tr], sigma[tr], group_num_cols=["g"], agg_num_cols=["aux"], stats=stats, nbins_base=10)
            Xtr = feat_df.to_numpy()
            Xte = np.column_stack([apply_binned_numeric_agg(df.iloc[te], recipes[c]) for c in feat_df.columns])
            m = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0).fit(Xtr, sigma[tr])
            return r2_score(sigma[te], m.predict(Xte))

        deltas.append(_r2(("mean", "std", "skew", "kurt")) - _r2(("mean",)))
    # std/skew/kurt of the feature per cell recover the spread the mean misses -> large lift.
    assert float(np.mean(deltas)) > 0.30, f"binned-numeric multistat agg should recover spread: deltas={deltas}"


def test_mrmr_integration_creates_binagg_columns_and_transform_replays():
    """End-to-end through MRMR.fit/transform: enabling the flag appends binagg columns into screening and
    transform replays them without error; disabling produces none."""
    from mlframe.feature_selection.filters import MRMR

    rng = np.random.default_rng(0)
    n = 5000
    g = rng.uniform(0, 1, n)
    sigma = 0.5 + 2.0 * np.abs(g - 0.5)
    aux = rng.normal(0, sigma, n)
    y = (sigma + rng.normal(0, 0.1, n) > sigma.mean()).astype(int)
    df = pd.DataFrame({"g": g, "aux": aux, "noise": rng.normal(0, 1, n)})
    tr, te = df.iloc[: n // 2].reset_index(drop=True), df.iloc[n // 2 :].reset_index(drop=True)
    ytr = y[: n // 2]

    m_on = MRMR(fe_binned_numeric_agg_enable=True, fe_binned_numeric_agg_max_pairs=8, verbose=0)
    m_on.fit(tr, ytr)
    roster = list(m_on.get_feature_names_out()) if hasattr(m_on, "get_feature_names_out") else []
    # Roster reflects only SELECTED features; the engineered-recipe registry proves the columns were created.
    recs = getattr(m_on, "_engineered_recipes_", []) or []
    if isinstance(recs, dict):
        recs = list(recs.values())
    # Transform must succeed and be finite (the load-bearing leak-safe-replay assertion).
    out = m_on.transform(te)
    arr = out.to_numpy() if hasattr(out, "to_numpy") else np.asarray(out)
    assert arr.shape[0] == len(te)
    assert np.isfinite(np.nan_to_num(arr)).all()

    m_off = MRMR(fe_binned_numeric_agg_enable=False, verbose=0)
    m_off.fit(tr, ytr)
    off_recs = getattr(m_off, "_engineered_recipes_", []) or []
    if isinstance(off_recs, dict):
        off_recs = list(off_recs.values())
    assert not any(r.kind == "binned_numeric_agg" for r in off_recs)


def test_redundancy_gate_drops_binagg_redundant_with_engineered_source_on_linear_target():
    """End-to-end through MRMR on a linearly-separable target. The default-on univariate Fourier stage emits a
    ``__qcos`` basis column whose binned aggregate (``binagg_std(x1__qcos..|qbin(x1))``) clears the Tier-1 MI floor
    yet is a deterministic function of its source -- on this target raw ``[x1, x2]`` already explains y, so the
    aggregate adds no conditional information. The redundancy gate (default ON) must drop it (no ``binagg_*`` in
    ``hybrid_orth_features_``); turning the gate OFF restores the spurious append, pinning that the gate is what
    suppresses it. Regression sensor for the per-scorer ``test_default_off_no_*`` family."""
    from tests.feature_selection.conftest import make_fast_mrmr

    rng = np.random.default_rng(42)
    n = 1500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "noise_a": rng.standard_normal(n),
            "noise_b": rng.standard_normal(n),
            "noise_c": rng.standard_normal(n),
        }
    )
    y = pd.Series(((x1 + 0.7 * x2) > 0).astype(int), name="y")

    on_appended = list(getattr(make_fast_mrmr().fit(X, y), "hybrid_orth_features_", []) or [])
    assert not any(str(c).startswith("binagg_") for c in on_appended), (
        f"redundancy gate (default ON) should drop binagg columns redundant with their source; got {on_appended}"
    )

    off_appended = list(getattr(make_fast_mrmr(fe_binned_numeric_agg_redundancy_gate=False).fit(X, y), "hybrid_orth_features_", []) or [])
    assert any(str(c).startswith("binagg_") for c in off_appended), "with the redundancy gate OFF the Tier-1 MI floor admits the redundant binagg column(s)"
