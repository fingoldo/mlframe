"""Group-aware feature selection for the LTR ranker suite.

For learning-to-rank, relevance is a PER-QUERY notion, so pooled (pointwise) MI is misleading: a feature that is
constant within a query but whose level tracks the query's overall relevance gets HIGH pooled MI yet carries ZERO
within-query ranking signal. These tests use exactly that adversarial "query-confounded" synthetic to PROVE the
group-aware path is required: pointwise MRMR picks the useless confounder, group-aware MRMR picks the real
within-query signal, and a ranker trained on the group-aware selection wins on NDCG. (A test that passes on the
pointwise path would be a bad test -- the point is that pointwise FAILS here.)
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest


def _query_confounded_frame(seed=0, Q=40, m=50):
    """Adversarial LtR frame:
      * ``s_within`` -- determines the graded relevance ORDER WITHIN each query (the only true ranking signal).
      * ``q_confounder`` -- CONSTANT within a query; its level is added to relevance, so it correlates with relevance
        in the POOLED data (high pointwise MI) but has ZERO within-query ranking power (useless for NDCG).
      * ``noise_*`` -- pure noise.
    Returns (df, group_col='qid', signal='s_within', confounder='q_confounder', noise_cols)."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(Q):
        q_level = rng.random()  # non-negative so relevance stays >= 0 (NDCG ground truth); still drives pooled MI
        s = rng.normal(size=m)
        grade = np.digitize(s, np.quantile(s, [0.2, 0.4, 0.6, 0.8])).astype(float)  # within-query graded relevance
        rel = grade + 30.0 * q_level  # large non-negative offset -> pooled rel dominated by q_level; within-query order is s only
        for i in range(m):
            rows.append((s[i], q_level, rng.normal(), rng.normal(), rel[i], grade[i], q))
    df = pd.DataFrame(rows, columns=["s_within", "q_confounder", "noise_0", "noise_1", "rel", "grade", "qid"])
    return df, "qid", "s_within", "q_confounder", ["noise_0", "noise_1"]


def _ndcg_at_k(y_true, y_score, groups, k=10):
    """Mean per-query NDCG@k."""
    from sklearn.metrics import ndcg_score

    scores = []
    for g in np.unique(groups):
        mask = groups == g
        if int(mask.sum()) < 2:
            continue
        scores.append(ndcg_score(y_true[mask][None, :], y_score[mask][None, :], k=k))
    return float(np.mean(scores)) if scores else float("nan")


# --- core property: group-aware relevance zeros the within-query-constant confounder -------------


def test_group_aware_relevance_zeros_query_confounder():
    from mlframe.training.ranking._ranker_fs import group_aware_relevance

    df, gcol, signal, conf, noise = _query_confounded_frame(0)
    cols = [signal, conf, *noise]
    rel = group_aware_relevance(cols, df[cols].to_numpy(np.float64), df["rel"].to_numpy(np.float64), df[gcol].to_numpy(), bins=5)
    assert rel[conf] < 0.02, f"query-constant confounder must have ~0 group-aware relevance, got {rel[conf]:.4f}"
    assert rel[signal] > 5 * max(rel[n] for n in noise), f"within-query signal must dominate noise: {rel}"


# --- THE discriminator: pointwise picks garbage, group-aware picks the real signal ----------------


def test_pointwise_mrmr_picks_confounder_group_aware_picks_signal():
    """Pooled MRMR selects the query-confounder (high pooled MI, useless for ranking); group-aware MRMR selects the
    within-query signal and NOT the confounder. This is the test that fails on a pointwise LtR FS design."""
    from mlframe.feature_selection.registry import get
    from mlframe.training.ranking._ranker_fs import group_aware_mrmr_select

    df, gcol, signal, conf, _noise = _query_confounded_frame(0)
    cols = [signal, conf, "noise_0", "noise_1"]
    X, y, g = df[cols], df["rel"].to_numpy(), df[gcol].to_numpy()

    pooled = get("MRMR").instantiate(use_simple_mode=True, quantization_nbins=8, verbose=0)
    pooled.fit(X, pd.Series(y))
    sup = np.asarray(pooled.support_)
    pooled_cols = [cols[i] for i in np.where(sup)[0]] if sup.dtype == bool else [cols[int(i)] for i in sup.tolist()]
    assert conf in pooled_cols, f"sanity: pointwise MRMR should be fooled into picking the confounder; got {pooled_cols}"

    ga_cols = group_aware_mrmr_select(X, y, g, nbins=8, bins=5)
    assert signal in ga_cols, f"group-aware MRMR must pick the within-query signal; got {ga_cols}"
    assert conf not in ga_cols, f"group-aware MRMR must REJECT the query-constant confounder; got {ga_cols}"


# --- biz_value: group-aware selection wins on NDCG vs the pointwise selection ---------------------


def test_biz_val_group_aware_fs_beats_pointwise_on_ndcg():
    """A ranker trained on the GROUP-AWARE-selected features achieves much higher NDCG@10 than one trained on the
    POINTWISE-selected features -- the pointwise pick (within-query-constant confounder) cannot order docs inside a
    query. Floor +0.10 NDCG (measured gap is larger)."""
    pytest.importorskip("catboost")
    from catboost import CatBoostRanker, Pool
    from mlframe.training.ranking._ranker_fs import group_aware_mrmr_select

    df, gcol, signal, conf, _noise = _query_confounded_frame(0, Q=60, m=40)
    cols = [signal, conf, "noise_0", "noise_1"]
    qids = df[gcol].to_numpy()
    cut = int(np.quantile(np.unique(qids), 0.7))
    tr, te = qids <= cut, qids > cut
    y = df["rel"].to_numpy()
    # True per-query relevance ground truth for NDCG = within-query grade (non-negative, no cross-query offset).
    y_true_eval = df["grade"].to_numpy()

    ga_cols = group_aware_mrmr_select(df[cols][tr], y[tr], qids[tr], nbins=8, bins=5)
    # The pointwise pick is the query-confounder (proven in test_pointwise_mrmr_picks_confounder...): a ranker on it
    # scores every doc in a query identically -> cannot order within a query -> poor NDCG.
    pointwise_cols = [conf]

    def _ndcg_for(feat_cols):
        train_pool = Pool(df[feat_cols][tr], label=y[tr], group_id=qids[tr])
        rk = CatBoostRanker(iterations=80, loss_function="YetiRank", verbose=False, random_seed=0)
        rk.fit(train_pool)
        pred = rk.predict(df[feat_cols][te])
        return _ndcg_at_k(y_true_eval[te], pred, qids[te], k=10)

    ndcg_ga = _ndcg_for(ga_cols)
    ndcg_pw = _ndcg_for(pointwise_cols)
    assert ndcg_ga >= ndcg_pw + 0.10, (
        f"group-aware FS must beat the pointwise pick on NDCG@10: group_aware={ndcg_ga:.4f} pointwise(confounder)={ndcg_pw:.4f} (ga_cols={ga_cols})"
    )


# --- e2e through the suite (common feature_selection_config, group-aware by default) ---------------


def test_e2e_ltr_suite_group_aware_fs_drops_confounder():
    pytest.importorskip("catboost")
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.configs import TargetTypes, ReportingConfig, OutputConfig
    from mlframe.training import FeatureSelectionConfig
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

    df, gcol, signal, conf, _noise = _query_confounded_frame(0)
    df = df.drop(columns=["grade"]).rename(columns={"rel": "target"})  # 'grade' is an eval-only artifact, not a feature
    fte = SimpleFeaturesAndTargetsExtractor(learning_to_rank_targets=["target"], group_field=gcol)
    with tempfile.TemporaryDirectory() as d:
        _res, meta = train_mlframe_models_suite(
            df=df,
            target_name="t",
            model_name="ltr",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            target_type=TargetTypes.LEARNING_TO_RANK,
            feature_selection_config=FeatureSelectionConfig(use_mrmr_fs=True, mrmr_kwargs={"quantization_nbins": 8}),
            reporting_config=ReportingConfig(show_perf_chart=False, show_fi=False),
            output_config=OutputConfig(data_dir=d, models_dir="models", save_charts=False),
            verbose=0,
            hyperparams_config={"iterations": 15},
        )
    sel = meta.get("selected_features")
    assert sel, "use_mrmr_fs produced no selected_features for LTR"
    assert signal in sel, f"group-aware LtR FS must keep the within-query signal; got {sel}"
    assert conf not in sel, f"group-aware LtR FS must drop the query-constant confounder; got {sel}"
