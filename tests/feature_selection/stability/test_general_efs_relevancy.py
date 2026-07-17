"""Unit and biz_value tests for mlframe.feature_selection.general.

Public surface: ``estimate_features_relevancy``, ``run_efs``, ``benchmark_mi_algos``.

biz_value: on synthetic data with 2 known-informative features mixed with 5 noise
columns, ``estimate_features_relevancy`` must drop all 5 noise columns and keep both
informative ones. This locks in the actual relevancy contract (permutation +
baseline-MI passing).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mlframe.feature_selection.general import (
    benchmark_mi_algos,
    estimate_features_relevancy,
)
from mlframe.feature_selection.mi import (
    chatgpt_compute_mutual_information,
    grok_compute_mutual_information,
)


def _build_bins_with_known_signal(n: int = 500, n_informative: int = 2, n_noise: int = 5, seed: int = 0) -> pl.DataFrame:
    """Build a polars frame of int8 binned columns where the first column is the
    target, then ``n_informative`` features that share entropy with target, then
    ``n_noise`` independent features. All values in [0, 15)."""
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 15, size=n, dtype=np.int8)
    cols = [target]
    names = ["target"]
    for i in range(n_informative):
        # Informative: noisy linear combo of target — shares enough entropy that
        # MI is comfortably above the permuted baseline.
        signal = (target + rng.integers(0, 3, size=n, dtype=np.int8)) % 15
        cols.append(signal.astype(np.int8))
        names.append(f"inf_{i}")
    for i in range(n_noise):
        cols.append(rng.integers(0, 15, size=n, dtype=np.int8))
        names.append(f"noise_{i}")
    arr = np.column_stack(cols).astype(np.int8)
    return pl.DataFrame(arr, schema=names)


# ----------------------------------------------------------------------------
# estimate_features_relevancy
# ----------------------------------------------------------------------------


def test_relevancy_returns_4_tuple():
    bins = _build_bins_with_known_signal(n=300)
    result = estimate_features_relevancy(
        bins=bins,
        target_columns=["target"],
        mi_algorithms_ranking=[grok_compute_mutual_information],
        benchmark_mi_algorithms=False,
        min_randomized_permutations=3,
        min_permuted_mi_evaluations=5,
        verbose=0,
    )
    assert len(result) == 4, f"estimate_features_relevancy must return 4-tuple; got {len(result)}"


def test_relevancy_returns_4_tuple_shape():
    """Tuple shape: (cols_to_drop, mi_matrix, all_permuted_mis, mi_algorithms_ranking)."""
    bins = _build_bins_with_known_signal(n=300)
    cols_to_drop, mi, perm_mi, ranking = estimate_features_relevancy(
        bins=bins,
        target_columns=["target"],
        mi_algorithms_ranking=[grok_compute_mutual_information],
        benchmark_mi_algorithms=False,
        min_randomized_permutations=3,
        min_permuted_mi_evaluations=5,
        verbose=0,
    )
    assert isinstance(cols_to_drop, list)
    assert mi.shape == (1, bins.shape[1]), f"MI matrix shape mismatch — got {mi.shape}, expected (1, {bins.shape[1]})"
    assert isinstance(perm_mi, dict)
    assert "target" in perm_mi, "permuted MI dict must key by target name"
    assert isinstance(ranking, list)


def test_relevancy_biz_value_drops_noise_keeps_signal():
    """biz_value: noise features MUST be in drop list; informative features MUST NOT."""
    bins = _build_bins_with_known_signal(n=500, n_informative=2, n_noise=5)
    cols_to_drop, _, _, _ = estimate_features_relevancy(
        bins=bins,
        target_columns=["target"],
        mi_algorithms_ranking=[grok_compute_mutual_information],
        benchmark_mi_algorithms=False,
        min_randomized_permutations=5,
        min_permuted_mi_evaluations=10,
        verbose=0,
    )
    drops = set(cols_to_drop)
    # All 5 noise cols must be dropped
    for i in range(5):
        assert f"noise_{i}" in drops, f"noise_{i} must be in drop list; got {drops}"
    # Neither informative col should be dropped
    for i in range(2):
        assert f"inf_{i}" not in drops, f"inf_{i} must NOT be dropped (informative); got {drops}"
    # Target itself must never be in drop list
    assert "target" not in drops


def test_relevancy_rejects_zero_permutations():
    """Wave 31 guard: min_randomized_permutations < 1 must raise ValueError (was assert pre-wave-31)."""
    bins = _build_bins_with_known_signal(n=200)
    with pytest.raises(ValueError, match="min_randomized_permutations"):
        estimate_features_relevancy(
            bins=bins,
            target_columns=["target"],
            mi_algorithms_ranking=[grok_compute_mutual_information],
            benchmark_mi_algorithms=False,
            min_randomized_permutations=0,
            min_permuted_mi_evaluations=5,
            verbose=0,
        )


def test_relevancy_handles_multiple_targets():
    """Multi-target call: MI matrix gets one row per target."""
    bins = _build_bins_with_known_signal(n=300, n_informative=2, n_noise=3)
    # Rename second informative col as a second target
    _cols_to_drop, mi, perm_mi, _ = estimate_features_relevancy(
        bins=bins,
        target_columns=["target", "inf_1"],
        mi_algorithms_ranking=[grok_compute_mutual_information],
        benchmark_mi_algorithms=False,
        min_randomized_permutations=3,
        min_permuted_mi_evaluations=5,
        verbose=0,
    )
    assert mi.shape[0] == 2, f"MI matrix must have 1 row per target; got {mi.shape}"
    assert "target" in perm_mi
    assert "inf_1" in perm_mi


def test_relevancy_deterministic_under_fixed_seed():
    """Same seed + same data must produce same drop list. Lock in reproducibility."""
    bins = _build_bins_with_known_signal(n=400, seed=123)
    np.random.seed(7)
    out1 = estimate_features_relevancy(
        bins=bins,
        target_columns=["target"],
        mi_algorithms_ranking=[grok_compute_mutual_information],
        benchmark_mi_algorithms=False,
        min_randomized_permutations=5,
        min_permuted_mi_evaluations=10,
        verbose=0,
    )
    np.random.seed(7)
    out2 = estimate_features_relevancy(
        bins=bins,
        target_columns=["target"],
        mi_algorithms_ranking=[grok_compute_mutual_information],
        benchmark_mi_algorithms=False,
        min_randomized_permutations=5,
        min_permuted_mi_evaluations=10,
        verbose=0,
    )
    assert out1[0] == out2[0], f"deterministic under fixed seed must match; got {out1[0]!r} vs {out2[0]!r}"
    # MI matrix is also computed deterministically
    np.testing.assert_allclose(out1[1], out2[1], rtol=1e-12)


def test_relevancy_with_explicit_mi_ranking_skips_benchmark():
    """If caller provides ``mi_algorithms_ranking`` AND ``benchmark_mi_algorithms=False``,
    no time should be spent benchmarking; the supplied first entry must be used."""
    bins = _build_bins_with_known_signal(n=200)
    # Use a deliberately distinct MI estimator and verify it is consulted.
    _cols_to_drop, _, _, ranking = estimate_features_relevancy(
        bins=bins,
        target_columns=["target"],
        mi_algorithms_ranking=[chatgpt_compute_mutual_information],
        benchmark_mi_algorithms=False,
        min_randomized_permutations=3,
        min_permuted_mi_evaluations=5,
        verbose=0,
    )
    assert ranking[0] is chatgpt_compute_mutual_information, "explicit mi_algorithms_ranking[0] must propagate through to the returned ranking"


# ----------------------------------------------------------------------------
# Statistical calibration: Miller-Madow bias correction + Benjamini-Hochberg FDR
# ----------------------------------------------------------------------------


def _build_pure_null_bins(n: int = 2000, n_features: int = 60, seed: int = 0) -> pl.DataFrame:
    """Target and every feature drawn INDEPENDENTLY -> no feature is genuinely relevant.

    The raw plug-in MI is positively biased even here, so the pre-fix ``>=`` exceedance over the RAW
    MI over-selects a chunk of these pure-noise features. The bias-corrected + BH-controlled test must
    keep the false-positive count small.
    """
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, 15, size=n, dtype=np.int8)]
    names = ["target"]
    for i in range(n_features):
        cols.append(rng.integers(0, 15, size=n, dtype=np.int8))
        names.append(f"noise_{i}")
    arr = np.column_stack(cols).astype(np.int8)
    return pl.DataFrame(arr, schema=names)


def test_relevancy_fdr_controls_false_positives_on_pure_null():
    """On a pure-null dataset (many independent features), the calibrated test must NOT retain more than
    a tiny handful of features. FDR control at alpha=0.05 bounds expected false discoveries; we allow a
    small slack for the discrete asymptotics."""
    bins = _build_pure_null_bins(n=2000, n_features=60, seed=0)
    cols_to_drop, _, _, _ = estimate_features_relevancy(
        bins=bins,
        target_columns=["target"],
        mi_algorithms_ranking=[grok_compute_mutual_information],
        benchmark_mi_algorithms=False,
        min_randomized_permutations=5,
        min_permuted_mi_evaluations=10,
        fdr_alpha=0.05,
        verbose=0,
    )
    n_features = bins.shape[1] - 1  # minus target
    kept = n_features - len([c for c in cols_to_drop if c != "target"])
    # Expect near-zero survivors on pure noise; allow <=3 as discrete-asymptotic slack.
    assert kept <= 3, f"pure-null over-selection: {kept}/{n_features} noise features retained; FDR failed to control false positives"


def test_relevancy_keeps_genuine_signal_under_fdr():
    """The FDR gate must NOT be so aggressive that it discards genuinely-relevant features. With a couple
    of informative features embedded among noise, those informative ones survive (are not dropped)."""
    bins = _build_bins_with_known_signal(n=2000, n_informative=2, n_noise=40, seed=1)
    cols_to_drop, _, _, _ = estimate_features_relevancy(
        bins=bins,
        target_columns=["target"],
        mi_algorithms_ranking=[grok_compute_mutual_information],
        benchmark_mi_algorithms=False,
        min_randomized_permutations=5,
        min_permuted_mi_evaluations=10,
        fdr_alpha=0.05,
        verbose=0,
    )
    drops = set(cols_to_drop)
    for i in range(2):
        assert f"inf_{i}" not in drops, f"inf_{i} is genuinely relevant but was dropped under FDR; got {drops}"


# ----------------------------------------------------------------------------
# benchmark_mi_algos — slow path, only one smoke test
# ----------------------------------------------------------------------------


@pytest.mark.slow
def test_benchmark_mi_algos_returns_sorted_list():
    """Smoke test: ``benchmark_mi_algos`` ranks the input estimators by runtime and
    returns them in order. n=1e6 is hardcoded inside the function so this is slow."""
    algos = [grok_compute_mutual_information, chatgpt_compute_mutual_information]
    out = benchmark_mi_algos(algos, verbose=0)
    assert isinstance(out, list)
    assert len(out) == len(algos), f"benchmark_mi_algos must preserve all input algos; got {len(out)}"
    for a in out:
        assert callable(a), f"every returned entry must be callable; got {a!r}"
