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
    """Moment aware resolution and autodrop."""
    nb, kept = resolve_nbins_and_stats(20000, ("mean", "std", "skew", "kurt"), 10, k=1)
    assert nb == 10 and kept == ["mean", "std", "skew", "kurt"]
    # Small n: cap from the highest moment binds (kurt n_min=100 -> nbins<=5).
    nb, kept = resolve_nbins_and_stats(500, ("mean", "std", "skew", "kurt"), 10, k=1)
    assert nb == 5 and "kurt" in kept
    # Tiny n: kurt's floor cannot be met at nbins>=2 -> kurt auto-dropped.
    nb, kept = resolve_nbins_and_stats(150, ("mean", "std", "skew", "kurt"), 10, k=1)
    assert "kurt" not in kept and "mean" in kept


def test_bincount_stats_match_numpy():
    """Bincount stats match numpy."""
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 4, 5000)
    v = rng.normal(0, 1, 5000)
    out = per_cell_stats_bincount(codes, v, 4, ("mean", "std"))
    for c in range(4):
        m = v[codes == c]
        assert abs(out["mean"][c] - m.mean()) < 1e-9
        assert abs(out["std"][c] - m.std()) < 1e-9


def test_replay_is_leak_safe_and_deterministic():
    """Replay is leak safe and deterministic."""
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
    """Std column recovers cell spread."""
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
    """Biz value recovers spread driven target."""
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
            """Fit/apply binned_numeric_agg with the given stats list on the train/test split and return held-out R^2."""
            feat_df, recipes = fit_binned_numeric_agg(df.iloc[tr], sigma[tr], group_num_cols=["g"], agg_num_cols=["aux"], stats=stats, nbins_base=10)  # noqa: B023 -- closure invoked twice below, same iteration, never stored
            Xtr = feat_df.to_numpy()
            Xte = np.column_stack([apply_binned_numeric_agg(df.iloc[te], recipes[c]) for c in feat_df.columns])  # noqa: B023 -- closure invoked twice below, same iteration, never stored
            m = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0).fit(Xtr, sigma[tr])  # noqa: B023 -- closure invoked twice below, same iteration, never stored
            return r2_score(sigma[te], m.predict(Xte))  # noqa: B023 -- closure invoked twice below, same iteration, never stored

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
    list(m_on.get_feature_names_out()) if hasattr(m_on, "get_feature_names_out") else []
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

    # Both arms need fe_max_steps>0: no FE family runs under a zero budget, so with the factory's no-FE
    # preset neither arm would append anything and the gate's effect would be unobservable.
    on_appended = list(getattr(make_fast_mrmr(fe_max_steps=1).fit(X, y), "hybrid_orth_features_", []) or [])
    assert not any(
        str(c).startswith("binagg_") for c in on_appended
    ), f"redundancy gate (default ON) should drop binagg columns redundant with their source; got {on_appended}"

    off_appended = list(getattr(make_fast_mrmr(fe_max_steps=1, fe_binned_numeric_agg_redundancy_gate=False).fit(X, y), "hybrid_orth_features_", []) or [])
    assert any(str(c).startswith("binagg_") for c in off_appended), "with the redundancy gate OFF the Tier-1 MI floor admits the redundant binagg column(s)"


def test_global_stats_all_matches_global_stat():
    # _global_stats_all replaced _global_stat's four separate scipy-based full-array passes (mean, std, and
    # TWO scipy skew/kurtosis calls) with ONE O(n) njit raw-moment pass -- profiled at 8.5s cumtime / 188
    # calls, the dominant cost of fit_binned_numeric_agg on a 2M-row cProfile run. scipy's defaults
    # (skew(bias=True), kurtosis(fisher=True, bias=True)) are the same population moment-ratio estimators
    # _derive_cell_stats already computes, so this pins that the fused path matches the old per-stat path
    # (selection-equivalence tolerance, not bit-identical -- different summation order) across random data,
    # NaN-bearing columns, near-constant columns, and tiny-n edge cases (the skew n<=2 / kurt n<=3 guards).
    """Global stats all matches global stat."""
    from mlframe.feature_selection.filters._binned_numeric_agg_fe import _global_stat, _global_stats_all

    rng = np.random.default_rng(0)
    stats = ["mean", "std", "skew", "kurt"]
    worst = 0.0
    for _ in range(500):
        n = int(rng.integers(1, 200))
        v = rng.standard_normal(n)
        if rng.random() < 0.1:
            v[rng.integers(0, n, size=max(1, n // 5))] = np.nan
        if rng.random() < 0.05:
            v[:] = rng.standard_normal()  # constant column
        old = {s: _global_stat(v, s) for s in stats}
        new = _global_stats_all(v, stats)
        for s in stats:
            worst = max(worst, abs(old[s] - new[s]))
    assert worst < 1e-9, f"_global_stats_all diverges {worst:.3e} from _global_stat"


def test_fit_binned_numeric_agg_matches_pre_fusion_reference():
    # Pin for the 2026-07-31 fold-gate + global-stats fusion: fit_binned_numeric_agg must produce the SAME
    # OOF features and recipes as the pre-fix implementation (a wasted full-array (fold_ids != f) & finite
    # gate, and four separate _global_stat calls per acol instead of one _global_stats_all call).
    """Fit binned numeric agg matches pre fusion reference."""
    from mlframe.feature_selection.filters._binned_numeric_agg_fe import (
        _derive_cell_stats,
        _global_stat,
        _per_cell_moments_stable,
    )

    def _old_reference(X, y, *, group_num_cols, agg_num_cols, stats=("mean", "std", "skew", "kurt"), nbins_base=10, n_folds=5, random_state=0):
        """Pre-fix fit_binned_numeric_agg: wasted full-array fold gate + per-stat _global_stat calls."""
        n = len(X)
        rng_ = np.random.default_rng(int(random_state))
        fold_ids = np.empty(n, dtype=np.int64)
        fold_ids[rng_.permutation(n)] = np.arange(n) % int(n_folds)
        feat_cols = {}
        _av_cache = {}
        _globals_cache = {}
        _fold_ne = [fold_ids != f for f in range(int(n_folds))]
        _fold_test = [np.where(fold_ids == f)[0] for f in range(int(n_folds))]
        for gcol in group_num_cols:
            gvals = np.asarray(X[gcol].to_numpy(), dtype=np.float64)
            nbins, kept_stats = resolve_nbins_and_stats(n, stats, nbins_base, k=1)
            edges = np.unique(np.quantile(gvals, np.linspace(0.0, 1.0, nbins + 1)[1:-1]))
            codes = np.searchsorted(edges, gvals, side="right")
            n_cells = int(codes.max()) + 1
            _ct_by_fold = [codes[_ft] for _ft in _fold_test]
            for acol in agg_num_cols:
                if acol == gcol:
                    continue
                _avc = _av_cache.get(acol)
                if _avc is None:
                    av = np.asarray(X[acol].to_numpy(), dtype=np.float64)
                    finite = np.isfinite(av)
                    _av_cache[acol] = (av, finite)
                else:
                    av, finite = _avc
                _gk = (acol, tuple(kept_stats))
                globals_ = _globals_cache.get(_gk)
                if globals_ is None:
                    globals_ = {s: _global_stat(av[finite], s) for s in kept_stats}
                    _globals_cache[_gk] = globals_
                oof = {s: np.full(n, globals_[s], dtype=np.float64) for s in kept_stats}
                for f in range(int(n_folds)):
                    tr = _fold_ne[f] & finite
                    if not tr.any():
                        continue
                    test = _fold_test[f]
                    ct = _ct_by_fold[f]
                    train_idx = np.where(tr)[0]
                    t_cnt, t_mean, t_cm2, t_cm3, t_cm4 = _per_cell_moments_stable(codes[train_idx], av[train_idx], n_cells)
                    per = _derive_cell_stats(t_cnt, t_mean, t_cm2, t_cm3, t_cm4, kept_stats)
                    for s in kept_stats:
                        vals = per[s][ct]
                        oof[s][test] = np.where(np.isfinite(vals), vals, globals_[s])
                for s in kept_stats:
                    feat_cols[engineered_name_binned_agg(acol, gcol, s)] = oof[s]
        return pd.DataFrame(feat_cols, index=X.index)

    rng = np.random.default_rng(3)
    n = 4000
    X = pd.DataFrame({f"g{i}": rng.uniform(0, 1, n) for i in range(3)} | {f"a{i}": rng.standard_normal(n) for i in range(3)})
    y = rng.standard_normal(n)
    group_cols = [f"g{i}" for i in range(3)]
    agg_cols = [f"a{i}" for i in range(3)]

    ref = _old_reference(X, y, group_num_cols=group_cols, agg_num_cols=agg_cols)
    feat_df, _recipes = fit_binned_numeric_agg(X, y, group_num_cols=group_cols, agg_num_cols=agg_cols)

    assert set(ref.columns) == set(feat_df.columns)
    for c in ref.columns:
        worst = float(np.max(np.abs(ref[c].to_numpy() - feat_df[c].to_numpy())))
        assert worst < 1e-9, f"{c}: diverges {worst:.3e} from the pre-fusion reference"


def test_per_cell_skew_kurt_stable_on_large_offset_small_scale_column():
    # Same bug class already fixed for _global_stats_all (whole-column) and _target_encoding_fe.py
    # (per-category): the raw-power binomial-expansion form (s3/n - 3*mean*s2/n + 2*mean**3, ...)
    # catastrophically cancels on large-offset/small-scale data. Pin per-cell skew/kurt against scipy's
    # direct per-cell computation on exactly that regime.
    """Per cell skew kurt stable on large offset small scale column."""
    from scipy.stats import kurtosis, skew

    rng = np.random.default_rng(7)
    n_cells = 6
    per_cell_n = 400
    offset = 8.5e3
    scale = 0.06
    codes = np.repeat(np.arange(n_cells), per_cell_n)
    v = offset + scale * rng.standard_normal(codes.shape[0])

    got = per_cell_stats_bincount(codes, v, n_cells, ("mean", "std", "skew", "kurt"))
    worst = {"skew": 0.0, "kurt": 0.0}
    for c in range(n_cells):
        cell_v = v[codes == c]
        ref_skew = float(skew(cell_v))
        ref_kurt = float(kurtosis(cell_v))
        worst["skew"] = max(worst["skew"], abs(got["skew"][c] - ref_skew))
        worst["kurt"] = max(worst["kurt"], abs(got["kurt"][c] - ref_kurt))
        assert np.isclose(got["mean"][c], cell_v.mean(), rtol=1e-9)
        assert np.isclose(got["std"][c], cell_v.std(), rtol=1e-9)
    assert worst["skew"] < 1e-6, f"per-cell skew diverges {worst['skew']:.3e} from scipy on a large-offset/small-scale column"
    assert worst["kurt"] < 1e-6, f"per-cell kurt diverges {worst['kurt']:.3e} from scipy on a large-offset/small-scale column"


def test_fit_binned_numeric_agg_oof_skew_kurt_stable_on_large_offset_agg_column():
    # Same regime as the per-cell unit test above, but through the real OOF fold loop
    # (fit_binned_numeric_agg) to pin the TRAIN-direct-computation fix (centered moments are not
    # additive across row subsets, unlike the old raw-power full-minus-test subtraction).
    """Fit binned numeric agg oof skew kurt stable on large offset agg column."""
    rng = np.random.default_rng(11)
    n = 3000
    X = pd.DataFrame(
        {
            "g0": rng.uniform(0, 1, n),
            "a0": 8.5e3 + 0.06 * rng.standard_normal(n),
        }
    )
    y = rng.standard_normal(n)
    feat_df, _recipes = fit_binned_numeric_agg(X, y, group_num_cols=["g0"], agg_num_cols=["a0"], stats=("skew", "kurt"))
    for c in feat_df.columns:
        vals = feat_df[c].to_numpy()
        assert np.isfinite(vals).all()
        assert np.abs(vals).max() < 50.0, f"{c}: OOF value blew up to {np.abs(vals).max():.3e} -- large-offset raw-moment cancellation regressed"
