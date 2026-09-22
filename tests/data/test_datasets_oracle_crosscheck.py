"""The oracle's exact mutual information, checked against a library this repository did not write.

The oracle's closed form and every test of it were written here, against one understanding of one formula.
A shared misunderstanding survives that arrangement intact, which is why the comparison below is against
`dit` -- a third-party package with its own distribution object and its own `I(X;Y)`.

Both sides are given the same exact joint law, so a disagreement cannot be a binning or sampling artefact.
One of the tests below is the teeth check: with a deliberately wrong value substituted for the oracle's,
the comparison must fail, or the agreements prove nothing.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

dit = pytest.importorskip("dit", reason="the oracle cross-check needs the third-party dit package")

from mlframe.data.datasets._oracle_crosscheck import (
    assert_oracle_agrees_with_dit,
    crosscheck_exact_mi,
    crosscheck_summary,
    exact_joint,
    mutual_information_via_dit,
)


def _discrete_bed(n_levels: int, n_samples: int = 4000, seed: int = 1) -> Dict[str, Any]:
    """Return integer level codes and the exact `P(y=1 | row)` they drive, as the generator would."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, n_levels, size=n_samples)
    centred = (codes - (n_levels - 1) / 2.0) / max(n_levels - 1, 1)
    logit = 3.6 * centred + rng.normal(0.0, 0.3, n_samples)
    return {"codes": codes, "true_prob": 1.0 / (1.0 + np.exp(-logit))}


@pytest.mark.parametrize("n_levels", [2, 5, 17, 64])
def test_the_oracles_exact_mi_matches_dit_to_floating_point(n_levels: int) -> None:
    """At every level count the oracle's grid separates, the two implementations agree to ~1e-16 nats."""
    bed = _discrete_bed(n_levels)
    checks = crosscheck_exact_mi({"x": bed["codes"]}, bed["true_prob"])

    assert len(checks) == 1
    check = checks[0]
    assert check.n_levels == n_levels
    assert check.difference < 1e-12
    assert check.agrees


def test_the_exact_joint_is_a_distribution_over_the_realised_column() -> None:
    """The joint sums to one, its column marginal is the level shares, and its target marginal is the mean law."""
    bed = _discrete_bed(9)
    joint, levels = exact_joint(bed["codes"], bed["true_prob"])

    assert levels.size == 9
    assert float(joint.sum()) == pytest.approx(1.0)
    shares = np.bincount(bed["codes"], minlength=9) / bed["codes"].size
    np.testing.assert_allclose(joint.sum(axis=1), shares)
    assert float(joint[:, 1].sum()) == pytest.approx(float(np.mean(bed["true_prob"])))


def test_an_independent_column_carries_no_information_on_either_side() -> None:
    """With the law constant across levels both sides return zero rather than a small positive number."""
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 8, size=4000)
    flat = np.full(codes.size, 0.3)
    joint, _ = exact_joint(codes, flat)

    assert abs(mutual_information_via_dit(joint)) < 1e-12
    checks = crosscheck_exact_mi({"flat": codes}, flat)
    assert abs(checks[0].ours) < 1e-12
    assert checks[0].agrees


def test_a_column_the_grid_cannot_separate_is_skipped_rather_than_compared() -> None:
    """Above the oracle's own bin count two levels share a bin, so the comparison declines instead of failing."""
    bed = _discrete_bed(200, n_samples=6000)
    assert crosscheck_exact_mi({"x": bed["codes"]}, bed["true_prob"]) == []


def test_the_summary_is_none_when_nothing_was_checkable() -> None:
    """An empty run reports `None`, never a record saying zero columns were verified."""
    assert crosscheck_summary([]) is None

    bed = _discrete_bed(5)
    summary = crosscheck_summary(assert_oracle_agrees_with_dit({"x": bed["codes"]}, bed["true_prob"]))
    assert summary is not None
    assert summary["columns_checked"] == 1
    assert summary["all_agree"] is True
    assert summary["estimator"] == "dit.shannon.mutual_information"


def test_a_wrong_exact_value_is_caught_rather_than_tolerated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Substituting a value off by a thousandth of a nat must raise -- otherwise the agreement proves nothing.

    A thousandth is far smaller than any effect the benchmark measures, so this also pins that the check's
    tolerance is tight enough to notice a formula error rather than only a catastrophic one.
    """
    bed = _discrete_bed(5)
    honest = crosscheck_exact_mi({"x": bed["codes"]}, bed["true_prob"])[0]

    def _slightly_wrong(column: np.ndarray, true_prob: np.ndarray, n_bins: int = 64) -> float:
        """Return the oracle's own answer, moved by a thousandth of a nat."""
        return honest.ours + 1e-3

    monkeypatch.setattr("mlframe.data.datasets._oracle._exact_mi_independent", _slightly_wrong)
    with pytest.raises(ValueError, match="disagrees with dit"):
        assert_oracle_agrees_with_dit({"x": bed["codes"]}, bed["true_prob"])


def test_mismatched_lengths_raise_rather_than_broadcasting_into_a_number() -> None:
    """A column and a law describing different row counts is a caller error, not something to align silently."""
    with pytest.raises(ValueError, match="same rows"):
        exact_joint(np.array([0, 1, 2]), np.array([0.5, 0.5]))


def test_a_law_outside_zero_one_raises() -> None:
    """A probability above one means the link escaped its squashing function, which must not reach a joint."""
    with pytest.raises(ValueError, match="probability in"):
        exact_joint(np.array([0, 1]), np.array([0.5, 1.4]))


def test_the_run_manifest_carries_the_crosscheck_verdict() -> None:
    """A run's manifest must say whether an outside implementation confirmed the oracle, not leave it open."""
    from mlframe.feature_selection._benchmarks.fs_hybrid._manifest import build_manifest

    manifest = build_manifest(
        results_path="cells.jsonl",
        scenarios=["linear_ceiling_sweep"],
        arms=["all-features"],
        dataset_seeds=[0],
        cv_seeds=[0],
        protocol_version="test",
    )
    record = manifest["oracle_crosscheck"]

    assert record["ran"] is True
    assert record["all_agree"] is True
    assert record["columns_checked"] >= 2
    assert record["worst_difference_nats"] < 1e-12
