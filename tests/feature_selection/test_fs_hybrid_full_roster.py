"""The full roster, and the pre-registration rules about it that were stated and never enforced.

`BENCHMARK_PREREGISTRATION.md` section 8 says a meta-test requires every arm to appear in at least two beds'
`expected_to_break`. There was no such test. Measured when this file was written, six of the then
twenty-one arms were named fewer than twice -- four of them genuine methods -- and nothing noticed, because a
rule without an executor is a sentence.

The same file pins the wrapper-estimator guard to names the roster actually builds. That guard stops a
wrapper being scored on its own objective, and it had been looking up `rfecv_lgbm` while the arm is named
`rfecv`, so it returned "not a wrapper" for every real arm.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from mlframe.data.datasets.scenarios import SCENARIOS
from mlframe.feature_selection._benchmarks.fs_hybrid._arms import build_arm_roster
from mlframe.feature_selection._benchmarks.fs_hybrid._arms_wrappers import WRAPPER_MAX_WIDTH, ProbabilityAsPrediction, _ordinal_from_removal
from mlframe.feature_selection._benchmarks.fs_hybrid._roster import WRAPPER_INTERNAL_ESTIMATOR, is_control_arm

#: The roster as it stood before this change. Cells are keyed by arm name, so every one of these must survive.
ORIGINAL_ARMS = (
    "all-features", "variance-sort", "univariate-mi", "skb-f", "skb-mi", "select-fdr", "sfm-lgbm", "lars-order",
    "boruta", "ace", "knockoffs", "mrmr", "rfecv", "boruta-shap", "shap-proxied", "rank-vote", "byproduct-ensemble",
)

#: Arms registered only up to `WRAPPER_MAX_WIDTH` columns.
WIDTH_GATED = ("forward-select", "greedy-backward", "zero-importance")


def _beds() -> List:
    """Every registered bed."""
    return list(SCENARIOS.values()) if isinstance(SCENARIOS, dict) else list(SCENARIOS)


def _bed_width(bed) -> int:
    """How many feature columns a bed carries at its declared defaults."""
    return len(bed.builder(**dict(bed.defaults)).features)


def _roster_names(n_features: int = 30, relevant: bool = True) -> List[str]:
    """The arm names the roster builds at a width, with or without a declared answer key."""
    return sorted(build_arm_roster(n_features, relevant=["x0"] if relevant else None))


def test_every_method_is_predicted_to_break_on_at_least_two_beds() -> None:
    """The pre-registration's section 8 rule, enforced: a method no bed expects to beat is a method nobody tested."""
    named = Counter(arm for bed in _beds() for arm in bed.expected_to_break)
    unpredicted = [arm for arm in _roster_names() if not is_control_arm(arm) and named[arm] < 2]
    assert unpredicted == [], f"these methods are expected to break on fewer than two beds: {unpredicted}"


def test_every_predicted_arm_is_a_real_arm() -> None:
    """A prediction naming an arm that does not exist is a prediction that can never be scored."""
    real = set(_roster_names())
    ghosts = sorted({arm for bed in _beds() for arm in bed.expected_to_break} - real)
    assert ghosts == [], f"beds predict breaks for arms the roster never builds: {ghosts}"


def test_a_width_gated_arm_is_only_predicted_where_it_runs() -> None:
    """A prediction on a bed wider than the gate is unscorable: the arm is never built there."""
    unscorable = [
        (arm, bed.name, _bed_width(bed)) for bed in _beds() for arm in bed.expected_to_break if arm in WIDTH_GATED and _bed_width(bed) > WRAPPER_MAX_WIDTH
    ]
    assert unscorable == [], f"width-gated arms predicted on beds they never run on: {unscorable}"


def test_every_wrapper_estimator_key_names_an_arm_the_roster_builds() -> None:
    """The guard's lookup table must be keyed by real names, or it silently returns 'not a wrapper'."""
    real = set(_roster_names())
    stale = sorted(set(WRAPPER_INTERNAL_ESTIMATOR) - real)
    assert stale == [], f"wrapper-estimator keys that name no arm: {stale}"


def test_the_original_arm_names_survive() -> None:
    """Existing results are keyed by these names; renaming one would orphan every cell it ever produced."""
    missing = [name for name in ORIGINAL_ARMS if name not in _roster_names()]
    assert missing == [], f"original arm names dropped from the roster: {missing}"


def test_the_roster_more_than_doubled_and_covers_every_paradigm() -> None:
    """The plan's paradigms each have at least one member, so no family is judged by a proxy of another."""
    names = set(_roster_names())
    assert len(names) >= 45
    paradigm_members = {
        "information criteria": {"it-mim", "it-cmim", "it-jmim", "it-relax"},
        "MRMR implementation variants": {"mrmr-pld", "mrmr-relax", "mrmr-tree-rescued", "mrmr-grouped", "mrmr-grouped-expand", "mrmr-stability"},
        "wrappers": {"forward-select", "greedy-backward", "zero-importance", "noise-floor", "unanimous-permutation"},
        "stability and bandits": {"bandit", "bandit-ensemble", "cascade-stable"},
        "cascades": {"cascade"},
        "registry-built": {"rfecv-registry", "boruta-shap-registry"},
        "univariate filters": {"ksg-mi", "relevance-table", "near-noise-auc", "permutation-topk", "unsupervised-prescreen"},
        "references": {"oracle-informative", "all-except-informative"},
    }
    absent = {paradigm: sorted(members - names) for paradigm, members in paradigm_members.items() if members - names}
    assert absent == {}, f"paradigms with missing members: {absent}"


def test_quadratic_wrappers_exist_only_up_to_the_width_cap() -> None:
    """Past the cap they are absent rather than present and timing out, which would read as unreliability."""
    narrow = set(_roster_names(WRAPPER_MAX_WIDTH))
    wide = set(_roster_names(WRAPPER_MAX_WIDTH + 1))
    assert set(WIDTH_GATED) <= narrow
    assert set(WIDTH_GATED).isdisjoint(wide)


def test_the_oracle_pair_exists_only_when_the_bed_declares_an_answer_key() -> None:
    """On a real bed with no truth the two references would be meaningless, so they must not be built."""
    assert {"oracle-informative", "all-except-informative"} <= set(_roster_names(relevant=True))
    assert {"oracle-informative", "all-except-informative"}.isdisjoint(_roster_names(relevant=False))


def test_the_oracle_pair_selects_exactly_the_key_and_exactly_its_complement() -> None:
    """The two references partition the columns: together they are every column, and they share none."""
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(60, 5)), columns=[f"x{i}" for i in range(5)])
    y = (X["x0"] > 0).astype(int).to_numpy()
    roster = build_arm_roster(5, relevant=["x0", "x3"])
    oracle = roster["oracle-informative"]().run(X, y).support
    complement = roster["all-except-informative"]().run(X, y).support
    assert list(X.columns[oracle]) == ["x0", "x3"]
    assert not np.any(oracle & complement)
    assert np.all(oracle | complement)


def test_removal_order_ranks_survivors_jointly_above_every_removed_column() -> None:
    """Survivors tie at the top -- they are a set -- and a later removal outranks an earlier one."""
    score = _ordinal_from_removal(["a", "b", "c", "d"], kept=["b", "d"], removed_in_order=["a", "c"])
    assert score[1] == score[3], "survivors must tie: the selector never ordered them"
    assert score[1] > score[2] > score[0], "later removal must outrank earlier removal"


def test_probability_as_prediction_turns_predict_into_a_probability() -> None:
    """A wrapper scoring through `predict` must see a probability, or its AUC collapses to one threshold."""
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + 0.3 * rng.normal(size=200) > 0).astype(int)
    model = clone(ProbabilityAsPrediction(LogisticRegression())).fit(X, y)
    predicted = model.predict(X)
    assert predicted.dtype.kind == "f"
    assert np.unique(predicted).size > 2, "a hard label would take only two values"
    np.testing.assert_allclose(predicted, model.predict_proba(X)[:, 1])


def test_cmim_by_the_chain_rule_equals_the_direct_conditional_mutual_information() -> None:
    """CMIM is derived from the JMIM kernel as I(X;Y|Z) = I(X,Z;Y) - I(Z;Y); check it against a direct table."""
    from mlframe.feature_selection.filters._jmim_scorer import _joint_mi_3d_njit

    rng = np.random.default_rng(2)
    n = 5000
    x = rng.integers(0, 4, n).astype(np.int64)
    z = rng.integers(0, 3, n).astype(np.int64)
    y = ((x + z + rng.integers(0, 2, n)) % 3).astype(np.int64)
    zeros = np.zeros(n, dtype=np.int64)
    via_chain_rule = float(_joint_mi_3d_njit(x, z, y, 4, 3, 3)) - float(_joint_mi_3d_njit(z, zeros, y, 3, 1, 3))

    direct = 0.0
    for zv in range(3):
        mask = z == zv
        pz = mask.mean()
        joint = np.zeros((4, 3))
        np.add.at(joint, (x[mask], y[mask]), 1.0)
        joint /= joint.sum()
        outer = joint.sum(axis=1, keepdims=True) * joint.sum(axis=0, keepdims=True)
        nz = joint > 0
        direct += pz * float((joint[nz] * np.log(joint[nz] / outer[nz])).sum())
    assert via_chain_rule == pytest.approx(direct, abs=1e-9)


def test_the_information_scorers_disagree_on_a_redundant_bed() -> None:
    """On a bed with an exact copy, MIM takes the copy and CMIM does not -- the family is not one method.

    MIM ranks by relevance alone, so the copy of the strongest column is its second pick. CMIM conditions on
    what it already has, and a copy carries nothing given its original.
    """
    from mlframe.feature_selection._benchmarks.fs_hybrid._arms_it_family import InformationGreedyArm

    rng = np.random.default_rng(3)
    n = 3000
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    X["x0_copy"] = X["x0"]
    y = (1.6 * X["x0"] + 0.8 * X["x1"] + 0.4 * rng.normal(size=n) > 0).astype(int).to_numpy()
    mim = InformationGreedyArm("mim", k=2).run(X, y)
    cmim = InformationGreedyArm("cmim", k=2).run(X, y)
    mim_top: Dict[str, bool] = {c: True for c in X.columns[mim.support]}
    cmim_top: Dict[str, bool] = {c: True for c in X.columns[cmim.support]}
    assert "x0_copy" in mim_top and "x0" in mim_top, "MIM should spend its budget on the copy"
    assert "x1" in cmim_top and not ("x0" in cmim_top and "x0_copy" in cmim_top), "CMIM should reject the copy"


def test_the_forecast_list_in_code_is_the_table_in_the_preregistration() -> None:
    """The tier runs `PREREGISTERED_2E_ARMS`; the document says which arms were forecast. They must agree."""
    import re
    from pathlib import Path

    from mlframe.feature_selection._benchmarks.fs_hybrid._roster import PREREGISTERED_2E_ARMS

    doc = (Path(__file__).resolve().parents[2] / "docs" / "BENCHMARK_PREREGISTRATION.md").read_text(encoding="utf-8")
    section = doc[doc.index("## 2e.") : doc.index("\n## ", doc.index("## 2e.") + 5)]
    in_table = set(re.findall(r"^\| `([^`]+)` \| `[^`]+` \|", section, flags=re.M))
    assert in_table == set(PREREGISTERED_2E_ARMS)


def test_the_predictions_tier_runs_every_forecast_arm_on_every_bed_that_names_it() -> None:
    """Derived from the registry, so a prediction added to a bed can never be left out of the run."""
    from mlframe.feature_selection._benchmarks.fs_hybrid._roster import PREREGISTERED_2E_ARMS
    from mlframe.feature_selection._benchmarks.fs_hybrid._tiers import TIERS, TIER_NAMES, get_tier

    tier = get_tier("predictions")
    forecast = set(PREREGISTERED_2E_ARMS)
    expected_beds = {bed.name for bed in _beds() if forecast & set(bed.expected_to_break)}
    assert set(tier.scenarios) == expected_beds
    assert set(tier.arms) == forecast
    assert "predictions" in TIER_NAMES and "predictions" not in TIERS, "built on request, not at import"


def _wide_bed(seed: int):
    """A sixty-column bed: past the width cap, so the quadratic wrappers are not built on it."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(240, 60)), columns=[f"c{i}" for i in range(60)])
    y = (X["c0"] + 0.5 * rng.normal(size=240) > 0).astype(int).to_numpy()
    return X, y, {"relevant": ["c0"]}


def test_a_width_gated_arm_is_skipped_on_a_wide_bed_and_a_typo_still_stops_the_run(tmp_path) -> None:
    """The two absences differ: a wrapper past its cap is skipped, a name no roster builds is an error."""
    import orjson

    from mlframe.feature_selection._benchmarks.fs_hybrid.run_experiment import run_grid

    results = tmp_path / "cells.jsonl"
    run_grid(scenarios=[("wide", _wide_bed)], dataset_seeds=[0], cv_seeds=[0], results_path=str(results), arms=["greedy-backward"])
    arms_run = {orjson.loads(line)["arm"] for line in results.read_text(encoding="utf-8").splitlines() if line.strip()}
    assert "greedy-backward" not in arms_run
    assert "all-features" in arms_run, "the null hypothesis still runs, so the bed is not silently empty"

    with pytest.raises(ValueError, match="no such name"):
        run_grid(scenarios=[("wide", _wide_bed)], dataset_seeds=[0], cv_seeds=[0], results_path=str(tmp_path / "b.jsonl"), arms=["greedy-backwards"])
