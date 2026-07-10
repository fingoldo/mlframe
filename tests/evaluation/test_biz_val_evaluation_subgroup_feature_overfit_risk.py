"""biz_value test for ``evaluation.flag_subgroup_only_feature_overfit_risk``.

The win: reproduces the exact source scenario (train ~90/10 cash/revolving, test ~99/1) and confirms a
revolving-loan-only feature's CV gain gets flagged as overfit-risk, while an equivalent feature scoped to a
subgroup with STABLE train/test prevalence is correctly NOT flagged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.evaluation.subgroup_feature_overfit_risk import flag_subgroup_only_feature_overfit_risk


def test_biz_val_flags_shifted_subgroup_feature_as_overfit_risk():
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({"loan_type": rng.choice(["cash", "revolving"], size=5000, p=[0.90, 0.10])})
    test_df = pd.DataFrame({"loan_type": rng.choice(["cash", "revolving"], size=5000, p=[0.99, 0.01])})

    result = flag_subgroup_only_feature_overfit_risk(
        train_df, test_df, subgroup_col="loan_type", feature_subgroup_value="revolving", cv_delta=0.006
    )

    assert result["subgroup_shifted"] is True
    assert result["overfit_risk_flag"] is True
    assert result["prevalence_ratio"] > 5.0


def test_stable_subgroup_feature_is_not_flagged():
    rng = np.random.default_rng(1)
    train_df = pd.DataFrame({"region": rng.choice(["north", "south"], size=5000, p=[0.5, 0.5])})
    test_df = pd.DataFrame({"region": rng.choice(["north", "south"], size=5000, p=[0.52, 0.48])})

    result = flag_subgroup_only_feature_overfit_risk(
        train_df, test_df, subgroup_col="region", feature_subgroup_value="south", cv_delta=0.004
    )

    assert result["subgroup_shifted"] is False
    assert result["overfit_risk_flag"] is False


def test_zero_cv_delta_never_flagged_even_if_shifted():
    rng = np.random.default_rng(2)
    train_df = pd.DataFrame({"loan_type": rng.choice(["cash", "revolving"], size=3000, p=[0.9, 0.1])})
    test_df = pd.DataFrame({"loan_type": rng.choice(["cash", "revolving"], size=3000, p=[0.99, 0.01])})

    result = flag_subgroup_only_feature_overfit_risk(
        train_df, test_df, subgroup_col="loan_type", feature_subgroup_value="revolving", cv_delta=0.0
    )
    assert result["subgroup_shifted"] is True
    assert result["overfit_risk_flag"] is False


def test_missing_subgroup_value_returns_no_report():
    train_df = pd.DataFrame({"loan_type": ["cash"] * 10})
    test_df = pd.DataFrame({"loan_type": ["cash"] * 10})
    result = flag_subgroup_only_feature_overfit_risk(
        train_df, test_df, subgroup_col="loan_type", feature_subgroup_value="nonexistent", cv_delta=0.01
    )
    assert result["subgroup_report_row"] is None
    assert result["overfit_risk_flag"] is False
