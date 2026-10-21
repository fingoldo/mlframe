"""End-to-end survival of the opt-in encoding, missingness, ratio and grouped FE families of MRMR.

Same contract as ``test_biz_value_mrmr_fe_family_survival``: every default-True ``fe_*_enable`` flag is off, only the family under test is on (with
the column arguments that family documents), and the scenario is one only that family can express. These families are default-off but documented
as supported, so a gate that silently kills one of them is as much a defect as for the default-on families.

=====================  ==================================================  ===========================  ============================
family                 flags                                               recipe kind                  scenario / downstream model
=====================  ==================================================  ===========================  ============================
count_encoding         fe_count_encoding_enable (+ _cols)                  count_encoded                P(y) sigmoid of log row count of a user id, logistic
frequency_encoding     fe_frequency_encoding_enable (+ _cols)              frequency_encoded            same data, logistic
cat_num_interaction    fe_cat_num_interaction_enable (+ cat/num cols)      cat_num_residual             y = price above its category mean, logistic
missingness_indicator  fe_missingness_indicator_enable                     missing_indicator            y depends on a column being missing, logistic
missingness_count      fe_missingness_count_enable                         missingness_count            y depends on how many of 6 fields are missing, logistic
missingness_pattern    fe_missingness_pattern_enable                       missingness_pattern          y depends on a joint missing pattern, boosted tree
pairwise_ratio         fe_pairwise_ratio_enable (+ _cols)                  div                          y = 0.5 < revenue / cost < 2, logistic
pairwise_log_ratio     fe_pairwise_log_ratio_enable (+ _cols)              log_div                      same data, logistic
grouped_delta          fe_grouped_delta_enable (+ group / num col)         grouped_delta                y = age above its region mean, logistic
lagged_diff            fe_lagged_diff_enable (+ time / value cols)         lagged_diff                  y = sign of the first difference, logistic
=====================  ==================================================  ===========================  ============================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.feature_selection.mrmr.biz_val._fe_family_survival_common import (
    checked,
    Family,
    check_created_and_survived,
    check_false_positive_rate,
    check_lift,
    check_noise_control,
    check_power,
    check_replay,
)

N = 6000


def build_count_signal(seed: int):
    """``P(y=1)`` is a sigmoid of the log of how often a user id occurs; 8 heavy users and a tail of singletons."""
    rng = np.random.default_rng(seed)
    heavy_w = np.array([700, 600, 500, 450, 400, 350, 300, 300])
    heavy = np.concatenate([np.repeat(f"U_HEAVY_{i}", int(w)) for i, w in enumerate(heavy_w)])
    rare = np.array([f"U_RARE_{i:05d}" for i in range(N - len(heavy))])
    cat_user = np.concatenate([heavy, rare])
    rng.shuffle(cat_user)
    counts = pd.Series(cat_user).value_counts()
    log_cnt = np.log1p(pd.Series(cat_user).map(counts).to_numpy().astype(np.float64))
    p = 1.0 / (1.0 + np.exp(-2.0 * (log_cnt - np.median(log_cnt))))
    y = (rng.random(N) < p).astype(int)
    return pd.DataFrame({"cat_user": cat_user, "noise_a": rng.standard_normal(N), "noise_b": rng.standard_normal(N)}), y


def build_cat_num_residual(seed: int):
    """20 categories with their own price mean; ``y = 1`` when the price is above its category mean."""
    rng = np.random.default_rng(seed)
    levels = [f"CAT_{i:02d}" for i in range(20)]
    means = rng.uniform(10.0, 110.0, size=20)
    cat = rng.choice(levels, size=N)
    cat_mean = pd.Series(cat).map(dict(zip(levels, means))).to_numpy()
    price = cat_mean + rng.normal(0.0, 10.0, size=N)
    y = (rng.random(N) < 1.0 / (1.0 + np.exp(-0.2 * (price - cat_mean)))).astype(int)
    return pd.DataFrame({"cat": cat, "price": price, "noise": rng.standard_normal(N)}), y


def build_missing_indicator(seed: int):
    """``credit_history`` is missing for 30% of rows and ``y`` is almost determined by that, not by the observed values."""
    rng = np.random.default_rng(seed)
    miss = rng.random(N) < 0.30
    ch = rng.standard_normal(N)
    ch[miss] = np.nan
    p = 1.0 / (1.0 + np.exp(-(-2.5 + 5.0 * miss + 0.1 * rng.standard_normal(N))))
    X = pd.DataFrame({"credit_history": ch, **{f"noise_{k}": rng.standard_normal(N) for k in "abcd"}})
    return X, (rng.random(N) < p).astype(int)


def build_missing_count(seed: int):
    """Six fields each missing at 25%; ``y`` is likely when four or more are missing."""
    rng = np.random.default_rng(seed)
    masks = rng.random((N, 6)) < 0.25
    p = 1.0 / (1.0 + np.exp(-(masks.sum(1) - 3.5) * 1.5))
    data = {}
    for j in range(6):
        col = rng.standard_normal(N)
        col[masks[:, j]] = np.nan
        data[f"field_{j}"] = col
    for k in range(3):
        data[f"noise_{k}"] = rng.standard_normal(N)
    return pd.DataFrame(data), (rng.random(N) < p).astype(int)


def build_missing_pattern(seed: int):
    """Three latent clusters differ only in which of ``field_a, field_b, field_c`` are missing; cluster C (a and b missing) is positive."""
    rng = np.random.default_rng(seed)
    cl = rng.choice([0, 1, 2], size=N, p=[0.4, 0.4, 0.2])
    a, b, c = rng.standard_normal(N), rng.standard_normal(N), rng.standard_normal(N)
    c[cl == 0] = np.nan
    a[cl >= 1] = np.nan
    b[cl == 2] = np.nan
    y = (rng.random(N) < np.where(cl == 2, 0.9, 0.1)).astype(int)
    X = pd.DataFrame({"field_a": a, "field_b": b, "field_c": c, "noise_x": rng.standard_normal(N), "noise_y": rng.standard_normal(N)})
    return X, y


def build_ratio_band(seed: int):
    """``y = 0.5 < revenue / cost < 2`` with both columns log-uniform over three orders of magnitude."""
    rng = np.random.default_rng(seed)
    rev = np.exp(rng.uniform(0, np.log(1000.0), N))
    cost = np.exp(rng.uniform(0, np.log(1000.0), N))
    band = ((rev / cost > 0.5) & (rev / cost < 2.0)).astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-(-3.0 + 6.0 * band + 0.1 * rng.standard_normal(N))))
    X = pd.DataFrame({"revenue": rev, "cost": cost, **{f"noise_{k}": rng.standard_normal(N) for k in "abcd"}})
    return X, (rng.random(N) < p).astype(int)


def build_grouped_delta(seed: int):
    """Each of 6 regions has its own mean age; ``y`` depends on the age deviation inside the region."""
    rng = np.random.default_rng(seed)
    regions = [f"R_{i}" for i in range(6)]
    means = dict(zip(regions, rng.uniform(25.0, 65.0, size=6)))
    region = rng.choice(regions, size=N)
    rmean = np.array([means[r] for r in region])
    age = rmean + rng.normal(0.0, 6.0, size=N)
    p = 1.0 / (1.0 + np.exp(-(-0.2 + 1.5 * (age > rmean))))
    X = pd.DataFrame({"region": region, "age": age, **{f"noise_{k}": rng.standard_normal(N) for k in "abc"}})
    return X, (rng.random(N) < p).astype(int)


def build_lagged_diff(seed: int):
    """A random-walk temperature observed in shuffled row order next to its time stamp; ``y`` follows the sign of the first difference."""
    rng = np.random.default_rng(seed)
    temp = 20.0 + np.cumsum(rng.normal(0.0, 1.0, N))
    diff = np.concatenate([[0.0], np.diff(temp)])
    perm = rng.permutation(N)
    t, temp, diff = np.arange(N)[perm], temp[perm], diff[perm]
    p = 1.0 / (1.0 + np.exp(-(-0.1 + 2.5 * (diff > 0))))
    X = pd.DataFrame({"t": t, "temperature": temp, **{f"noise_{k}": rng.standard_normal(N) for k in "abc"}})
    return X, (rng.random(N) < p).astype(int)


_C = "classification"
FAMILIES = [
    Family(
        "count_encoding", {"fe_count_encoding_enable": True}, ("count_encoded",), build_count_signal, _C, "linear", {"fe_count_encoding_cols": ("cat_user",)}
    ),
    Family(
        "frequency_encoding",
        {"fe_frequency_encoding_enable": True},
        ("frequency_encoded",),
        build_count_signal,
        _C,
        "linear",
        {"fe_frequency_encoding_cols": ("cat_user",)},
    ),
    Family(
        "cat_num_interaction",
        {"fe_cat_num_interaction_enable": True},
        ("cat_num_residual",),
        build_cat_num_residual,
        _C,
        "linear",
        {"fe_cat_num_interaction_cat_cols": ("cat",), "fe_cat_num_interaction_num_cols": ("price",)},
    ),
    Family("missingness_indicator", {"fe_missingness_indicator_enable": True}, ("missing_indicator",), build_missing_indicator, _C, "linear"),
    Family(
        "missingness_count",
        {"fe_missingness_count_enable": True},
        ("missingness_count",),
        build_missing_count,
        _C,
        "linear",
        {"fe_missingness_indicator_cols": tuple(f"field_{j}" for j in range(6))},
    ),
    Family(
        "missingness_pattern",
        {"fe_missingness_pattern_enable": True},
        ("missingness_pattern",),
        build_missing_pattern,
        _C,
        "linear",
        {"fe_missingness_indicator_cols": ("field_a", "field_b", "field_c"), "fe_missingness_pattern_top_k": 4},
    ),
    Family("pairwise_ratio", {"fe_pairwise_ratio_enable": True}, ("pairwise_ratio",), build_ratio_band, _C, "linear", {"fe_pairwise_ratio_cols": ("revenue", "cost")}, pattern=r"^ratio__"),
    Family("pairwise_log_ratio", {"fe_pairwise_log_ratio_enable": True}, ("pairwise_ratio",), build_ratio_band, _C, "linear", {"fe_pairwise_log_ratio_cols": ("revenue", "cost")}, pattern=r"^log_ratio__"),
    Family(
        "grouped_delta",
        {"fe_grouped_delta_enable": True},
        ("grouped_delta",),
        build_grouped_delta,
        _C,
        "linear",
        {"fe_grouped_delta_group_col": "region", "fe_grouped_delta_num_cols": ("age",)},
    ),
    Family(
        "lagged_diff",
        {"fe_lagged_diff_enable": True},
        ("lagged_diff",),
        build_lagged_diff,
        _C,
        "linear",
        {"fe_lagged_diff_time_col": "t", "fe_lagged_diff_value_cols": ("temperature",), "fe_lagged_diff_periods": (1, 2)},
    ),
]
_IDS = [f.name for f in FAMILIES]


@pytest.mark.parametrize("family", checked(FAMILIES, "created"))
def test_family_column_is_created_and_survives_selection(family):
    """A recipe of the family's kind exists and at least one of its columns is in the selected output."""
    check_created_and_survived(family)


@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_family_replay_on_a_held_out_frame_is_finite(family):
    """``transform`` of the held-out rows has the selected shape and only finite engineered values."""
    check_replay(family)


@pytest.mark.parametrize("family", checked(FAMILIES, "lift"))
def test_family_lifts_a_held_out_model(family):
    """The selected output beats the raw columns on held-out rows for every seed, by more than twice the seed-to-seed spread of the lift."""
    check_lift(family)


@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_family_adds_nothing_on_a_permuted_target(family):
    """With the target independent of every column no column of the family is selected."""
    check_noise_control(family)


@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("family", checked(FAMILIES, "power"))
def test_family_detection_power_over_the_seed_panel(family):
    """The family's column is created and selected on enough seeds that the Clopper-Pearson lower bound of the detection rate reaches the policy floor."""
    check_power(family)


@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_family_false_positive_rate_over_the_seed_panel(family):
    """On permuted targets the Clopper-Pearson upper bound of the rate of runs selecting a column of the family stays below the policy ceiling."""
    check_false_positive_rate(family)
