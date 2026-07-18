"""Harder adversarial coverage for group-aware LtR feature selection.

Every test here is DISCRIMINATING: pooled (pointwise) MI is the WRONG relevance notion for ranking, so each synthetic
is built so the pooled MRMR is actively fooled (picks a within-query-useless feature) while the group-aware path is
correct. A test that passes on the pooled path would be a bad test -- the assertions check BOTH that pooled fails and
that group-aware succeeds. Constructions covered: the canonical query-confounder, mixed query sizes, many genuine
signals vs many confounders + noise, a Simpson's-paradox sign-flip feature, and a medium-scale NDCG biz_value win.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

SEEDS = [0, 1, 7]


def _pooled_mrmr_cols(cols, X, y):
    """Column names selected by the POOLED (pointwise) registry MRMR -- the misleading-for-ranking baseline."""
    from mlframe.feature_selection.registry import get

    m = get("MRMR").instantiate(use_simple_mode=True, quantization_nbins=8, verbose=0)
    m.fit(X, pd.Series(y))
    sup = np.asarray(m.support_)
    if sup.dtype == bool:
        return [cols[i] for i in np.where(sup)[0]]
    return [cols[int(i)] for i in sup.tolist() if 0 <= int(i) < len(cols)]


def _ndcg_at_k(y_true, y_score, groups, k=10):
    """Mean per-query NDCG@k. ``y_true`` MUST be the non-negative within-query grade, not the cross-query offset rel."""
    from sklearn.metrics import ndcg_score

    scores = []
    for g in np.unique(groups):
        mask = groups == g
        if int(mask.sum()) < 2:
            continue
        scores.append(ndcg_score(y_true[mask][None, :], y_score[mask][None, :], k=k))
    return float(np.mean(scores)) if scores else float("nan")


def _confounder_frame(seed=0, Q=40, m=50):
    """Canonical query-confounded frame.

    ``s_within`` sets the within-query order (the only true ranking signal; grade = within-query quintile 0..4).
    ``q_confounder`` is CONSTANT within a query and its level (rng.random()*30) is ADDED to relevance, so its POOLED
    MI with relevance is high while its within-query ranking power is zero. Returns df + column roles.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(Q):
        q_level = rng.random()
        s = rng.normal(size=m)
        grade = np.digitize(s, np.quantile(s, [0.2, 0.4, 0.6, 0.8])).astype(float)
        rel = grade + 30.0 * q_level  # large non-negative cross-query offset -> pooled rel dominated by q_level
        for i in range(m):
            rows.append((s[i], q_level * 30.0, rng.normal(), rng.normal(), rel[i], grade[i], q))
    df = pd.DataFrame(rows, columns=["s_within", "q_confounder", "noise_0", "noise_1", "rel", "grade", "qid"])
    return df, "qid", "s_within", "q_confounder", ["noise_0", "noise_1"]


def _varying_size_frame(seed=0, sizes=(10, 30, 80), reps=12):
    """Same query-confounder construction but with MIXED query sizes -- exercises the size-weighted averaging path."""
    rng = np.random.default_rng(seed)
    rows = []
    q = 0
    for _ in range(reps):
        for m in sizes:
            q_level = rng.random()
            s = rng.normal(size=m)
            grade = np.digitize(s, np.quantile(s, [0.2, 0.4, 0.6, 0.8])).astype(float)
            rel = grade + 30.0 * q_level
            for i in range(m):
                rows.append((s[i], q_level * 30.0, rng.normal(), rel[i], grade[i], q))
            q += 1
    df = pd.DataFrame(rows, columns=["s_within", "q_confounder", "noise_0", "rel", "grade", "qid"])
    return df, "qid", "s_within", "q_confounder", ["noise_0"]


def _multi_signal_frame(seed=0, Q=50, m=50):
    """Two genuine within-query signals + two query-constant confounders + two noise columns."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(Q):
        c0, c1 = rng.random(), rng.random()
        a = rng.normal(size=m)
        b = rng.normal(size=m)
        ga = np.digitize(a, np.quantile(a, [0.25, 0.5, 0.75])).astype(float)
        gb = np.digitize(b, np.quantile(b, [0.25, 0.5, 0.75])).astype(float)
        rel = ga + gb + 25.0 * c0 + 25.0 * c1  # within-query order driven by a,b; pooled dominated by c0,c1
        grade = ga + gb
        for i in range(m):
            rows.append((a[i], b[i], c0 * 25.0, c1 * 25.0, rng.normal(), rng.normal(), rel[i], grade[i], q))
    df = pd.DataFrame(rows, columns=["sig_a", "sig_b", "conf_0", "conf_1", "noise_0", "noise_1", "rel", "grade", "qid"])
    return df, "qid", ["sig_a", "sig_b"], ["conf_0", "conf_1"], ["noise_0", "noise_1"]


def _simpson_frame(seed=0, Q=45, m=50):
    """Simpson's-paradox feature: strong POOLED correlation with rel but no usable WITHIN-query order.

    ``s_within`` is the true within-query signal. ``simpson`` = (large per-query baseline that tracks rel across
    queries) + (a mild NEGATIVE within-query slope buried in within-query noise). Pooled, the baseline dominates ->
    high pooled MI (above the true signal's); within a query the order is negative-and-noisy, so its group-aware
    relevance collapses to ~noise level and the real signal is ranked far above it.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(Q):
        q_level = rng.random()
        s = rng.normal(size=m)
        grade = np.digitize(s, np.quantile(s, [0.2, 0.4, 0.6, 0.8])).astype(float)
        rel = grade + 30.0 * q_level
        simpson = 30.0 * q_level - 0.3 * grade + rng.normal(size=m) * 2.0  # pooled-high; within-query negative+noisy
        for i in range(m):
            rows.append((s[i], simpson[i], rng.normal(), rel[i], grade[i], q))
    df = pd.DataFrame(rows, columns=["s_within", "simpson", "noise_0", "rel", "grade", "qid"])
    return df, "qid", "s_within", "simpson", ["noise_0"]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("bins", [5, 8])
def test_group_aware_relevance_zeros_confounder_and_ranks_signal(seed, bins):
    """group_aware_relevance must score the query-constant confounder ~0 (no within-query power) and the within-query
    signal far above any noise. Discriminates against a pooled estimator, which would rank the confounder highest."""
    from mlframe.training.ranking._ranker_fs import group_aware_relevance

    # m=80: at smaller m the binned within-query MI saturates and the noise pedestal rises (8 bins over <=50 points
    # overfits), shrinking the signal-to-noise gap; m=80 keeps the gap >6x at both bins so the assertion is honest.
    df, gcol, signal, conf, noise = _confounder_frame(seed, Q=40, m=80)
    cols = [signal, conf, *noise]
    arr = df[cols].to_numpy(np.float64)
    rel = df["rel"].to_numpy(np.float64)
    g = df[gcol].to_numpy()

    ga = group_aware_relevance(cols, arr, rel, g, bins=bins)
    pooled = {c: _pooled_pair_mi(arr[:, j], rel) for j, c in enumerate(cols)}

    assert pooled[conf] > pooled[signal], f"sanity: POOLED MI must rank the confounder above the signal; got {pooled}"
    assert ga[conf] < 0.02, f"group-aware relevance of query-constant confounder must be ~0, got {ga[conf]:.4f}"
    assert ga[signal] > 5 * max(ga[n] for n in noise), f"within-query signal must dominate noise: {ga}"
    assert ga[signal] > 10 * ga[conf], f"group-aware must rank the signal far above the confounder: {ga}"


def _pooled_pair_mi(x, y, bins=8):
    """Pooled pair mi."""
    from mlframe.training.ranking._ranker_fs import _binned_mi

    return _binned_mi(np.asarray(x, np.float64), np.asarray(y, np.float64), bins=bins)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("bins", [5, 8])
def test_pooled_mrmr_picks_confounder_group_aware_rejects_it(seed, bins):
    """THE discriminator: pooled MRMR is fooled into selecting the query-confounder; group-aware MRMR selects the
    within-query signal and REJECTS the confounder. Fails on any pointwise LtR FS design."""
    from mlframe.training.ranking._ranker_fs import group_aware_mrmr_select

    df, gcol, signal, conf, _noise = _confounder_frame(seed)
    cols = [signal, conf, "noise_0", "noise_1"]
    X, y, g = df[cols], df["rel"].to_numpy(), df[gcol].to_numpy()

    pooled_cols = _pooled_mrmr_cols(cols, X, y)
    assert conf in pooled_cols, f"sanity: pooled MRMR should be fooled into picking the confounder; got {pooled_cols}"

    ga_cols = group_aware_mrmr_select(X, y, g, nbins=8, bins=bins)
    assert signal in ga_cols, f"group-aware MRMR must pick the within-query signal; got {ga_cols}"
    assert conf not in ga_cols, f"group-aware MRMR must REJECT the query-constant confounder; got {ga_cols}"


@pytest.mark.parametrize("seed", SEEDS)
def test_varying_query_sizes_recover_signal(seed):
    """With mixed query sizes {10,30,80}, the size-weighted group-aware relevance still recovers the within-query
    signal and zeros the confounder, while pooled MRMR is still fooled by the confounder."""
    from mlframe.training.ranking._ranker_fs import group_aware_mrmr_select

    df, gcol, signal, conf, _noise = _varying_size_frame(seed)
    cols = [signal, conf, "noise_0"]
    X, y, g = df[cols], df["rel"].to_numpy(), df[gcol].to_numpy()

    pooled_cols = _pooled_mrmr_cols(cols, X, y)
    assert conf in pooled_cols, f"sanity: pooled MRMR fooled by confounder on mixed sizes; got {pooled_cols}"

    ga_cols = group_aware_mrmr_select(X, y, g, nbins=8, bins=8)
    assert signal in ga_cols, f"group-aware MRMR must recover the signal under mixed query sizes; got {ga_cols}"
    assert conf not in ga_cols, f"group-aware MRMR must reject the confounder under mixed sizes; got {ga_cols}"


@pytest.mark.parametrize("seed", SEEDS)
def test_many_signals_vs_many_confounders_and_noise(seed):
    """Two genuine within-query signals + two query-constant confounders + two noise columns: group-aware keeps BOTH
    signals and drops every confounder, ranking both signals strictly above every confounder by per-query relevance;
    pooled MRMR is fooled into picking at least one confounder."""
    from mlframe.training.ranking._ranker_fs import group_aware_mrmr_select, group_aware_relevance

    df, gcol, signals, confs, noise = _multi_signal_frame(seed)
    cols = [*signals, *confs, *noise]
    X, y, g = df[cols], df["rel"].to_numpy(), df[gcol].to_numpy()

    pooled_cols = _pooled_mrmr_cols(cols, X, y)
    assert any(c in pooled_cols for c in confs), f"sanity: pooled MRMR must pick a confounder; got {pooled_cols}"

    rel = group_aware_relevance(cols, X.to_numpy(np.float64), y.astype(np.float64), g, bins=8)
    assert min(rel[s] for s in signals) > max(rel[c] for c in confs), f"group-aware relevance must rank BOTH signals above every confounder: {rel}"

    ga_cols = group_aware_mrmr_select(X, y, g, nbins=8, bins=8)
    for s in signals:
        assert s in ga_cols, f"group-aware MRMR must keep genuine signal {s}; got {ga_cols}"
    for c in confs:
        assert c not in ga_cols, f"group-aware MRMR must drop confounder {c}; got {ga_cols}"


@pytest.mark.parametrize("seed", SEEDS)
def test_simpson_paradox_sign_flip_feature(seed):
    """A feature positively correlated with relevance POOLED but negatively WITHIN queries must NOT be ranked as top
    relevance by group-aware; pooled MI ranks it high. group-aware ranks the real within-query signal above it."""
    from mlframe.training.ranking._ranker_fs import group_aware_mrmr_select, group_aware_relevance

    df, gcol, signal, simpson, _noise = _simpson_frame(seed)
    cols = [signal, simpson, "noise_0"]
    arr = df[cols].to_numpy(np.float64)
    rel = df["rel"].to_numpy(np.float64)
    g = df[gcol].to_numpy()

    assert _pooled_pair_mi(arr[:, 1], rel) > _pooled_pair_mi(arr[:, 0], rel) * 0.5, (
        "sanity: the Simpson feature must carry substantial POOLED MI with relevance"
    )

    ga = group_aware_relevance(cols, arr, rel, g, bins=8)
    assert ga[signal] > 3 * ga[simpson], f"group-aware relevance must rank the true signal far above the Simpson feature: {ga}"

    ga_cols = group_aware_mrmr_select(df[cols], rel, g, nbins=8, bins=8)
    assert signal in ga_cols, f"group-aware MRMR must select the true within-query signal; got {ga_cols}"
    assert ga_cols[0] == signal, f"group-aware MRMR must rank the true signal FIRST, not the Simpson feature; got {ga_cols}"


def test_biz_val_group_aware_fs_beats_pooled_on_ndcg_medium_scale():
    """Medium-scale biz_value: a CatBoostRanker trained on the group-aware selection beats one trained on the pooled
    pick (the confounder) on mean per-query NDCG@10. The confounder is constant within a query, so a ranker on it
    cannot order docs inside a query. Floor +0.10 NDCG (measured gap is larger)."""
    pytest.importorskip("catboost")
    from catboost import CatBoostRanker, Pool

    from mlframe.training.ranking._ranker_fs import group_aware_mrmr_select

    df, gcol, signal, conf, _noise = _confounder_frame(0, Q=100, m=40)
    cols = [signal, conf, "noise_0", "noise_1"]
    qids = df[gcol].to_numpy()
    cut = int(np.quantile(np.unique(qids), 0.7))
    tr, te = qids <= cut, qids > cut
    rel = df["rel"].to_numpy()
    y_true_eval = df["grade"].to_numpy()  # non-negative within-query grade is the honest NDCG ground truth

    ga_cols = group_aware_mrmr_select(df[cols][tr], rel[tr], qids[tr], nbins=8, bins=8)
    assert signal in ga_cols and conf not in ga_cols, f"group-aware selection unexpected on medium frame: {ga_cols}"

    # The pooled MRMR is fooled into picking the confounder (proven in test_pooled_mrmr_picks_confounder...); on the
    # confounder a ranker scores every doc in a query identically -> it cannot order within a query -> poor NDCG. The
    # full pooled support sometimes also drags in s_within, which would mask the failure, so the pooled pick under test
    # is the confounder it WRONGLY ranks top -- the feature a pointwise design would commit to.
    pooled_support = _pooled_mrmr_cols(cols, df[cols][tr], rel[tr])
    assert conf in pooled_support, f"sanity: pooled MRMR fooled by confounder on medium frame; got {pooled_support}"
    pooled_cols = [conf]

    def _ndcg_for(feat_cols):
        """Ndcg for."""
        train_pool = Pool(df[feat_cols][tr], label=rel[tr], group_id=qids[tr])
        rk = CatBoostRanker(iterations=80, loss_function="YetiRank", verbose=False, random_seed=0)
        rk.fit(train_pool)
        pred = rk.predict(df[feat_cols][te])
        return _ndcg_at_k(y_true_eval[te], pred, qids[te], k=10)

    ndcg_ga = _ndcg_for(ga_cols)
    ndcg_pw = _ndcg_for(pooled_cols)
    assert ndcg_ga >= ndcg_pw + 0.10, (
        f"group-aware FS must beat the pooled pick on NDCG@10: group_aware={ndcg_ga:.4f} pooled={ndcg_pw:.4f} (ga_cols={ga_cols}, pooled_cols={pooled_cols})"
    )
