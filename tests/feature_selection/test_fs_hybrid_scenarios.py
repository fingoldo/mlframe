"""Structural verification of the adversarial Phase-0 feature-selection scenarios.

Three families of assertion:

* **Determinism** -- every generator is a pure function of ``(name, seed)``; two calls with the same seed are
  bit-identical (``assert_array_equal``, never ``allclose``), and stream naming uses ``blake2b`` so it cannot
  drift with ``PYTHONHASHSEED``.
* **Varsortability** -- every emitted column carries unit variance, so the ``VarianceSortArm`` control cannot
  read the generative order off the marginal variances (Reisach, Seiler & Drton, NeurIPS 2021).
* **Planted structure** -- each bed is checked for the property it exists to test: zero marginal MI on XOR
  operands with non-zero joint MI, individually-weak members on ``group_additive``, a compensating pair,
  independence on the null controls, and information genuinely destroyed by cluster aggregation.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mlframe.feature_selection._benchmarks.fs_hybrid.adversarial_scenarios import (
    ADVERSARIAL_SCENARIOS,
    GATE_SCENARIOS,
    compensable_pair,
    expected_to_break_index,
    fdr_under_budget,
    group_additive,
    latent_replicates_private_delta,
    linear_gaussian_lowdim_n200,
    null_p10,
    probe_flood_p1000,
    stable_name_hash,
    stream_for,
    xor2,
    xor3,
    xor3_plus_marginal_decoy,
)

@cache
def _cached(name: str, seed: int = 0):
    """Generate a bed once per (name, seed) so the p=1000 beds are not rebuilt by every read-only test."""
    return ADVERSARIAL_SCENARIOS[name](seed)


ALL_NAMES = tuple(ADVERSARIAL_SCENARIOS)
#: Beds cheap enough to regenerate several times inside a single test.
SMALL_NAMES = tuple(name for name in ALL_NAMES if "p1000" not in name)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _discretize(values: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Return equal-frequency bin labels for ``values`` (ties collapsed into the lower bin)."""
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
    return np.searchsorted(edges, values, side="right")


def _mutual_information(labels: np.ndarray, y: np.ndarray) -> float:
    """Plug-in mutual information in nats between an integer-labelled variable and a binary target."""
    joint = pd.crosstab(labels, y).to_numpy(dtype=float)
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = joint * np.log(joint / (px * py))
    return float(np.nansum(terms))


def _marginal_mi(column: np.ndarray, y: np.ndarray, n_bins: int = 8) -> float:
    """Mutual information between one continuous column and the target, after equal-frequency binning."""
    return _mutual_information(_discretize(column, n_bins), y)


def _joint_mi(frame: pd.DataFrame, names: list, y: np.ndarray) -> float:
    """Mutual information between the JOINT cell of several columns (binarised at zero) and the target."""
    codes = np.zeros(len(frame), dtype=np.int64)
    for name in names:
        codes = codes * 2 + (frame[name].to_numpy() > 0.0).astype(np.int64)
    return _mutual_information(codes, y)


def _holdout_auc(frame: pd.DataFrame, y: pd.Series, names: list) -> float:
    """Honest holdout AUC of a logistic model fitted on ``names`` (first half train, second half test)."""
    half = len(frame) // 2
    x_train = frame.loc[:, names].to_numpy()[:half]
    x_test = frame.loc[:, names].to_numpy()[half:]
    model = LogisticRegression(max_iter=2000).fit(x_train, y.to_numpy()[:half])
    return float(roc_auc_score(y.to_numpy()[half:], model.predict_proba(x_test)[:, 1]))


def _point_biserial(frame: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Absolute correlation of every column with the binary target."""
    target = y.to_numpy(dtype=float)
    return frame.apply(lambda col: abs(float(np.corrcoef(col.to_numpy(), target)[0, 1])))


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_stable_name_hash_is_not_pythonhashseed_dependent() -> None:
    """The name hash is a fixed blake2b digest, pinned here so a switch to builtin ``hash()`` fails loudly."""
    assert stable_name_hash("xor3") == int.from_bytes(__import__("hashlib").blake2b(b"xor3", digest_size=8).digest(), "big")
    assert stable_name_hash("xor3") != stable_name_hash("xor2")


def test_streams_are_addressed_by_name_not_position() -> None:
    """Adding a stream name cannot shift another stream's draws: each path is hashed independently."""
    first = stream_for(7, "scenario", "features").standard_normal(5)
    inserted = stream_for(7, "scenario", "brand_new_stream").standard_normal(5)
    again = stream_for(7, "scenario", "features").standard_normal(5)
    assert_array_equal(first, again)
    assert not np.array_equal(first, inserted)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_scenario_is_bit_identical_across_calls(name: str) -> None:
    """Two calls with the same seed produce bit-identical X, y and column order."""
    builder = ADVERSARIAL_SCENARIOS[name]
    x_a, y_a, truth_a = builder(3)
    x_b, y_b, truth_b = builder(3)
    assert list(x_a.columns) == list(x_b.columns)
    assert_array_equal(x_a.to_numpy(), x_b.to_numpy())
    assert_array_equal(y_a.to_numpy(), y_b.to_numpy())
    assert truth_a["base"] == truth_b["base"]
    assert truth_a["expected_to_break"] == truth_b["expected_to_break"]


@pytest.mark.parametrize("name", SMALL_NAMES)
def test_different_seeds_give_different_draws(name: str) -> None:
    """Distinct seeds must actually move the data, otherwise replication across seeds is a fiction."""
    x_a, _, _ = ADVERSARIAL_SCENARIOS[name](0)
    x_b, _, _ = ADVERSARIAL_SCENARIOS[name](1)
    assert not np.array_equal(x_a.to_numpy(), x_b.to_numpy())


# ---------------------------------------------------------------------------
# contract: truth shape, expected_to_break, varsortability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_NAMES)
def test_truth_dict_keeps_the_existing_shape(name: str) -> None:
    """Every bed emits the legacy ``truth`` keys plus the pre-registration fields."""
    frame, target, truth = _cached(name)
    for key in ("base", "relevant", "noise", "interaction_operands", "quadratic_operands"):
        assert isinstance(truth[key], list)
    assert isinstance(truth["expected_to_break"], tuple) and truth["expected_to_break"]
    assert set(truth["base"]) <= set(frame.columns)
    assert set(truth["noise"]) <= set(frame.columns)
    assert set(truth["base"]).isdisjoint(truth["noise"])
    assert set(truth["pre_std_scale"]) == set(frame.columns)
    assert target.isin((0, 1)).all()


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_column_is_standardised_so_variance_sorting_is_at_chance(name: str) -> None:
    """Unit variance everywhere: a VarianceSortArm sees an exact tie and cannot exploit the generative order."""
    frame, _, truth = _cached(name)
    stds = frame.std(ddof=0).to_numpy()
    assert np.allclose(stds, 1.0, atol=1e-9)
    scales = np.array(list(truth["pre_std_scale"].values()))
    assert np.all(scales > 0.0)
    assert scales.std() > 0.0 or len(scales) == 1  # the recorded pre-standardisation scales are real


@pytest.mark.parametrize("name", ("group_additive", "latent_replicates_private_delta", "fdr_under_budget"))
def test_variance_sorting_recovers_no_more_than_chance(name: str) -> None:
    """Ranking features by marginal variance recovers the informative set at the base rate, across seeds."""
    recalls = []
    for seed in range(4):
        frame, _, truth = _cached(name, seed)
        informative = set(truth["base"])
        ranked = frame.var(ddof=0).sort_values(ascending=False).index[: len(informative)]
        recalls.append(len(informative.intersection(ranked)) / len(informative))
    chance = len(informative) / frame.shape[1]
    assert float(np.mean(recalls)) <= chance + 0.25


def test_every_arm_named_is_expected_to_break_at_least_two_beds() -> None:
    """Pre-registration coverage: no arm is listed as broken by a single bed only."""
    index = expected_to_break_index()
    assert index, "no expected_to_break entries at all"
    singletons = {arm: beds for arm, beds in index.items() if len(beds) < 2}
    assert not singletons, f"arms broken by fewer than two beds: {singletons}"
    assert "mrmr" in index and len(index["mrmr"]) >= 6


def test_null_beds_are_declared_as_the_entry_gate() -> None:
    """The null controls run first and are flagged as gating; that flag is what disqualifies noisy arms."""
    assert GATE_SCENARIOS
    for name in GATE_SCENARIOS:
        _, _, truth = _cached(name)
        assert truth["gate"] is True
        assert truth["metric"] == "n_selected"
        assert truth["base"] == []


# ---------------------------------------------------------------------------
# planted structure, one test per scenario
# ---------------------------------------------------------------------------


def test_xor2_operands_have_zero_marginal_mi_but_a_large_joint_mi() -> None:
    """Both XOR operands are marginally uninformative; the pair carries most of a bit."""
    frame, target, truth = xor2(0)
    y = target.to_numpy()
    for name in truth["base"]:
        assert _marginal_mi(frame[name].to_numpy(), y) < 4e-3
    assert _joint_mi(frame, list(truth["base"]), y) > 0.30


def test_xor3_is_blind_marginally_and_pairwise() -> None:
    """3-way parity: singles AND pairs carry ~zero MI; only the full triple is informative."""
    frame, target, truth = xor3(0)
    y = target.to_numpy()
    operands = list(truth["base"])
    for name in operands:
        assert _marginal_mi(frame[name].to_numpy(), y) < 4e-3
    for i in range(len(operands)):
        for j in range(i + 1, len(operands)):
            assert _joint_mi(frame, [operands[i], operands[j]], y) < 4e-3
    assert _joint_mi(frame, operands, y) > 0.30


def test_xor3_plus_marginal_decoy_plants_both_failure_modes() -> None:
    """The decoy dominates every marginal ranking while the operands sit at zero, as designed."""
    frame, target, truth = xor3_plus_marginal_decoy(0)
    y = target.to_numpy()
    decoy_mi = _marginal_mi(frame["marginal_decoy"].to_numpy(), y)
    assert decoy_mi > 0.05
    for name in truth["base"]:
        assert _marginal_mi(frame[name].to_numpy(), y) < 4e-3
    # The decoy is a lossy reflection of the parity, so the operands strictly dominate it jointly.
    assert _joint_mi(frame, list(truth["base"]), y) > decoy_mi


def test_group_additive_members_are_individually_weak_and_jointly_strong() -> None:
    """No member of the group is separable on its own; the sum of the ten is highly predictive."""
    frame, target, truth = group_additive(0)
    members = list(truth["base"])
    singles = [_holdout_auc(frame, target, [name]) for name in members]
    assert max(singles) < 0.60
    assert _holdout_auc(frame, target, members) > 0.68
    noise_mi = np.median([_marginal_mi(frame[name].to_numpy(), target.to_numpy()) for name in truth["noise"]])
    member_mi = np.median([_marginal_mi(frame[name].to_numpy(), target.to_numpy()) for name in members])
    assert member_mi > noise_mi  # weak, but not literally invisible: the difficulty is the ranking, not detection


def test_compensable_pair_is_collinear_and_only_the_difference_predicts() -> None:
    """The pair correlates above 0.9 yet only their difference separates the classes."""
    frame, target, truth = compensable_pair(0)
    a, b = frame["comp_a"].to_numpy(), frame["comp_b"].to_numpy()
    assert float(np.corrcoef(a, b)[0, 1]) > 0.9
    assert _holdout_auc(frame, target, ["comp_a"]) < 0.60
    assert _holdout_auc(frame, target, ["comp_b"]) < 0.60
    assert _holdout_auc(frame, target, ["comp_a", "comp_b"]) > 0.80
    assert _holdout_auc(frame, target, ["cluster_decoy"]) < 0.58  # the third cluster member carries nothing
    assert "cluster_decoy" in truth["noise"]


def test_fdr_under_budget_grades_relevance_and_declares_fdr_as_the_metric() -> None:
    """Coefficients decay geometrically, so recall can only be bought with false discoveries."""
    frame, target, truth = fdr_under_budget(0)
    assert truth["metric"] == "fdr"
    assert 0.0 < truth["nominal_fdr"] < 1.0
    coefficients = truth["coefficients"]
    values = [coefficients[name] for name in truth["base"]]
    assert values == sorted(values, reverse=True)
    correlations = _point_biserial(frame, target)
    strong = [name for name in truth["base"] if coefficients[name] > 0.3]
    weak = [name for name in truth["base"] if coefficients[name] < 0.12]
    assert correlations[strong].mean() > correlations[weak].mean()
    assert correlations[weak].mean() < correlations[truth["noise"]].max()  # weak members are genuinely buried


def test_probe_flood_has_a_thousand_columns_of_matched_probes() -> None:
    """Eight informative features among ~1000 probes whose marginals are indistinguishable after scaling."""
    frame, target, truth = probe_flood_p1000(0)
    assert frame.shape[1] == 1000
    assert len(truth["base"]) == 8
    correlations = _point_biserial(frame, target)
    assert correlations[truth["base"]].min() > correlations[truth["noise"]].quantile(0.99)
    # Matched probes: after standardisation nothing distinguishes a probe from a signal column distributionally.
    assert abs(frame[truth["noise"]].kurt().median() - frame[truth["base"]].kurt().median()) < 1.5


@pytest.mark.parametrize("name", GATE_SCENARIOS)
def test_null_beds_are_independent_of_the_target(name: str) -> None:
    """``y`` is drawn from its own stream, so no column may exceed the sampling null by a wide margin."""
    frame, target, truth = _cached(name)
    assert truth["relevant"] == []
    correlations = _point_biserial(frame, target)
    threshold = 5.0 / np.sqrt(len(frame))  # 5 sampling standard errors
    assert correlations.max() < threshold
    assert len(frame) * min(float(target.mean()), 1.0 - float(target.mean())) >= 4000


def test_null_p10_row_count_is_sized_from_the_minority_class() -> None:
    """Row counts come from the minority count, not the total: at least 5000 rows in the smaller class."""
    frame, target, _ = null_p10(0)
    minority = min(int(target.sum()), len(target) - int(target.sum()))
    assert minority >= 4700  # 5000 nominal, binomial slack
    assert len(frame) >= 10000


def test_latent_replicates_lose_information_when_the_cluster_is_collapsed() -> None:
    """Keeping the cluster WHOLE beats both a medoid and a mean aggregate: the private deltas drive ``y``."""
    frame, target, truth = latent_replicates_private_delta(0)
    group = list(truth["jointly_necessary_group"])
    assert truth["must_keep_whole"] is True
    whole = _holdout_auc(frame, target, [*group, "indep"])
    single = _holdout_auc(frame, target, [group[0], "indep"])
    aggregated = frame.assign(_agg=frame[group].mean(axis=1))
    mean_only = _holdout_auc(aggregated, target, ["_agg", "indep"])
    assert whole > single + 0.03
    assert whole > mean_only + 0.03
    # The cluster really is a cluster: the replicates are strongly mutually correlated.
    corr = frame[group].corr().to_numpy()
    assert np.median(corr[np.triu_indices(len(group), k=1)]) > 0.35


def test_linear_gaussian_lowdim_is_small_n_and_parametrically_recoverable() -> None:
    """n=200 with 5 informative Gaussian columns: the t-statistic separates them from the 15 probes."""
    frame, target, truth = linear_gaussian_lowdim_n200(0)
    assert frame.shape == (200, 20)
    assert len(truth["base"]) == 5 and len(truth["noise"]) == 15
    correlations = _point_biserial(frame, target)
    assert correlations[truth["base"]].mean() > correlations[truth["noise"]].mean() * 1.5
    assert _holdout_auc(frame, target, list(truth["base"])) > 0.70
