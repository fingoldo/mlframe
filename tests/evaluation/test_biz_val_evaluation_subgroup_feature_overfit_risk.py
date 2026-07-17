"""biz_value test for ``evaluation.flag_subgroup_only_feature_overfit_risk``.

The win: reproduces the exact source scenario (train ~90/10 cash/revolving, test ~99/1) and confirms a
revolving-loan-only feature's CV gain gets flagged as overfit-risk, while an equivalent feature scoped to a
subgroup with STABLE train/test prevalence is correctly NOT flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mlframe.evaluation.subgroup_feature_overfit_risk import (
    flag_subgroup_only_feature_overfit_risk,
    rank_subgroup_feature_overfit_risk,
)


def test_biz_val_flags_shifted_subgroup_feature_as_overfit_risk():
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({"loan_type": rng.choice(["cash", "revolving"], size=5000, p=[0.90, 0.10])})
    test_df = pd.DataFrame({"loan_type": rng.choice(["cash", "revolving"], size=5000, p=[0.99, 0.01])})

    result = flag_subgroup_only_feature_overfit_risk(train_df, test_df, subgroup_col="loan_type", feature_subgroup_value="revolving", cv_delta=0.006)

    assert result["subgroup_shifted"] is True
    assert result["overfit_risk_flag"] is True
    assert result["prevalence_ratio"] > 5.0


def test_stable_subgroup_feature_is_not_flagged():
    rng = np.random.default_rng(1)
    train_df = pd.DataFrame({"region": rng.choice(["north", "south"], size=5000, p=[0.5, 0.5])})
    test_df = pd.DataFrame({"region": rng.choice(["north", "south"], size=5000, p=[0.52, 0.48])})

    result = flag_subgroup_only_feature_overfit_risk(train_df, test_df, subgroup_col="region", feature_subgroup_value="south", cv_delta=0.004)

    assert result["subgroup_shifted"] is False
    assert result["overfit_risk_flag"] is False


def test_zero_cv_delta_never_flagged_even_if_shifted():
    rng = np.random.default_rng(2)
    train_df = pd.DataFrame({"loan_type": rng.choice(["cash", "revolving"], size=3000, p=[0.9, 0.1])})
    test_df = pd.DataFrame({"loan_type": rng.choice(["cash", "revolving"], size=3000, p=[0.99, 0.01])})

    result = flag_subgroup_only_feature_overfit_risk(train_df, test_df, subgroup_col="loan_type", feature_subgroup_value="revolving", cv_delta=0.0)
    assert result["subgroup_shifted"] is True
    assert result["overfit_risk_flag"] is False


def test_biz_val_rank_subgroup_feature_overfit_risk_orders_by_known_severity():
    """Five candidate subgroup-only features, each scoped to its own column with a DIFFERENT, engineered
    train/test prevalence shift (stable -> mild -> moderate -> severe -> extreme) but the SAME cv_delta, so
    the true overfit-risk ordering is known by construction (driven purely by shift magnitude). Proves the
    ranking mode recovers that ordering exactly (Spearman rho == 1.0), which lets a caller with several
    subgroup-only features prioritize which to drop/re-validate first instead of only getting a per-feature
    boolean.
    """
    rng = np.random.default_rng(3)
    n = 8000
    # (column, train_p_minority, test_p_minority) -- increasing gap => increasing true severity.
    specs = [
        ("col_stable", 0.30, 0.30),
        ("col_mild", 0.30, 0.20),
        ("col_moderate", 0.30, 0.10),
        ("col_severe", 0.30, 0.04),
        ("col_extreme", 0.30, 0.005),
    ]
    train_data = {}
    test_data = {}
    for col, p_train, _ in specs:
        train_data[col] = rng.choice(["majority", "minority"], size=n, p=[1 - p_train, p_train])
    for col, _, p_test in specs:
        test_data[col] = rng.choice(["majority", "minority"], size=n, p=[1 - p_test, p_test])
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    candidates = [{"feature_name": col, "subgroup_col": col, "feature_subgroup_value": "minority", "cv_delta": 0.005} for col, _, _ in specs]
    ranking = rank_subgroup_feature_overfit_risk(train_df, test_df, candidates)

    true_severity_order = [col for col, _, _ in specs]  # col_stable (least severe) -> col_extreme (most severe)
    true_rank = {col: rank for rank, col in enumerate(true_severity_order)}
    predicted_severity = [ranking.loc[ranking["feature_name"] == col, "risk_score"].iloc[0] for col in true_severity_order]
    rho, _ = spearmanr(list(range(len(true_severity_order))), predicted_severity)
    assert rho >= 0.99

    # the most severe candidate must be ranked first, the stable one last -- the actual "what to drop first" answer.
    assert ranking.iloc[0]["feature_name"] == "col_extreme"
    assert ranking.iloc[-1]["feature_name"] == "col_stable"
    assert ranking.iloc[0]["risk_score"] > ranking.iloc[-1]["risk_score"]


def test_biz_val_rank_subgroup_feature_overfit_risk_missing_value_ranks_last():
    train_df = pd.DataFrame({"loan_type": ["cash"] * 10})
    test_df = pd.DataFrame({"loan_type": ["cash"] * 10})
    candidates = [
        {"feature_name": "present", "subgroup_col": "loan_type", "feature_subgroup_value": "cash", "cv_delta": 0.01},
        {"feature_name": "absent", "subgroup_col": "loan_type", "feature_subgroup_value": "nonexistent", "cv_delta": 0.01},
    ]
    ranking = rank_subgroup_feature_overfit_risk(train_df, test_df, candidates)
    absent_row = ranking[ranking["feature_name"] == "absent"].iloc[0]
    assert absent_row["risk_score"] == 0.0
    assert bool(absent_row["overfit_risk_flag"]) is False


def test_missing_subgroup_value_returns_no_report():
    train_df = pd.DataFrame({"loan_type": ["cash"] * 10})
    test_df = pd.DataFrame({"loan_type": ["cash"] * 10})
    result = flag_subgroup_only_feature_overfit_risk(train_df, test_df, subgroup_col="loan_type", feature_subgroup_value="nonexistent", cv_delta=0.01)
    assert result["subgroup_report_row"] is None
    assert result["overfit_risk_flag"] is False
