"""Edge-case coverage for the feature_selection public surface.

Targets the boundary / degenerate paths of five modules that the broader suite exercises only on
happy-path inputs:

* ``mi`` -- the three parallel MI kernels (grok / chatgpt / deepseek): n_bins boundaries (1, 127, 128),
  empty data, out-of-range bin codes, self-information == entropy, cross-implementation agreement.
* ``pre_screen`` -- ``compute_unsupervised_drops`` (None / empty / single-row / constant / all-null /
  string dtype / null-fraction boundary / variance-threshold / protected) and ``apply_drops``
  (idempotence / missing columns / cross-backend), with explicit pandas-vs-polars parity.
* ``general.estimate_features_relevancy`` -- 0-permutation guard, single-target selection, the
  mutate-and-restore contract on the input frame, and the quantile baseline path.
* ``structure_discovery.discover_structure`` -- gcd recovery, single-class / degenerate y, 2D y skip,
  empty X, and the 0-false-discovery guarantee on a smooth noise frame.
* ``compare_selectors`` -- empty / None guards, empty feature set, Jaccard + consensus values,
  mixed fitted/unfitted handling, name de-duplication, unreadable-support skip.

Behavioural assertions only (values / raises / invariants) -- never bare ``is not None``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.mi import (
    grok_compute_mutual_information,
    chatgpt_compute_mutual_information,
    deepseek_compute_mutual_information,
)
from mlframe.feature_selection.pre_screen import compute_unsupervised_drops, apply_drops
from mlframe.feature_selection.general import estimate_features_relevancy
from mlframe.feature_selection import discover_structure, compare_selectors

pl = pytest.importorskip("polars")


MI_KERNELS = [
    pytest.param(grok_compute_mutual_information, id="grok"),
    pytest.param(chatgpt_compute_mutual_information, id="chatgpt"),
    pytest.param(deepseek_compute_mutual_information, id="deepseek"),
]


def _observed_entropy(codes: np.ndarray) -> float:
    """Shannon entropy (nats) of an integer-coded 1-D array, over OCCUPIED bins only."""
    counts = np.bincount(np.asarray(codes).ravel())
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


# ============================================================================================
# mi.py -- the three MI kernels
# ============================================================================================


@pytest.mark.parametrize("kernel", MI_KERNELS)
@pytest.mark.parametrize("n_bins", [1, 5, 15, 127, 128])
def test_mi_self_information_equals_entropy(kernel, n_bins):
    """MI(x, x) must equal H(x) at every n_bins boundary (1 == single-bin collapse -> 0; 127/128 ==
    the int8 kernel's upper edge). This pins both the histogram sizing and the log arithmetic."""
    rng = np.random.default_rng(7)
    n = 4000
    hi = max(1, n_bins)  # for n_bins=1 codes collapse to a single value -> H=0
    x = rng.integers(0, hi, size=n).astype(np.int8)
    data = np.column_stack([x, x]).astype(np.int8)

    mi = float(kernel(data, [0], n_bins=n_bins)[0, 1])
    expected = _observed_entropy(x)  # 0.0 exactly when n_bins == 1
    assert abs(mi - expected) < 1e-6, f"MI(x,x)={mi} should equal H(x)={expected} for n_bins={n_bins}"


@pytest.mark.parametrize("kernel", MI_KERNELS)
def test_mi_independent_variables_near_zero(kernel):
    """Two independent uniforms have true MI 0; the plug-in estimate stays tiny (biased-positive only)."""
    rng = np.random.default_rng(11)
    n = 4000
    a = rng.integers(0, 5, n).astype(np.int8)
    b = rng.integers(0, 5, n).astype(np.int8)
    data = np.column_stack([a, b]).astype(np.int8)
    mi = float(kernel(data, [0], n_bins=5)[0, 1])
    assert 0.0 <= mi < 0.02, f"independent-pair MI should be ~0, got {mi}"


def test_mi_kernels_agree_cross_implementation():
    """The three kernels exist to cross-validate MI under different summation orders (see mi.py
    header): on identical binned input they MUST agree to near machine precision."""
    rng = np.random.default_rng(3)
    n = 3000
    # y depends on a but not b -> a moderate, non-trivial MI value to compare across kernels.
    a = rng.integers(0, 6, n)
    y = (a + rng.integers(0, 2, n)) % 6
    b = rng.integers(0, 6, n)
    data = np.column_stack([y, a, b]).astype(np.int8)

    g = grok_compute_mutual_information(data, [0], n_bins=6)
    c = chatgpt_compute_mutual_information(data, [0], n_bins=6)
    d = deepseek_compute_mutual_information(data, [0], n_bins=6)
    assert g.shape == c.shape == d.shape == (1, 3)
    np.testing.assert_allclose(g, c, atol=1e-9, rtol=0)
    np.testing.assert_allclose(g, d, atol=1e-9, rtol=0)
    # sanity: the informative column carries clearly more MI than the noise column
    assert g[0, 1] > g[0, 2] + 0.1


@pytest.mark.parametrize("kernel", MI_KERNELS)
def test_mi_empty_data_returns_zeros(kernel):
    """n_samples == 0 must not divide by zero; every kernel returns a zero (n_targets, n_cols) matrix."""
    empty = np.zeros((0, 3), dtype=np.int8)
    out = kernel(empty, [0, 1], n_bins=5)
    assert out.shape == (2, 3)
    assert np.all(out == 0.0)


@pytest.mark.parametrize("kernel", MI_KERNELS)
@pytest.mark.parametrize("bad_value", [200, -1])
def test_mi_out_of_range_codes_raise(kernel, bad_value):
    """Bin codes outside [0, 127] would silently corrupt the int8-indexed histogram; the shared
    validator must raise ValueError BEFORE the cast rather than write out of bounds."""
    data = np.array([[0, bad_value], [1, 3], [2, 4]], dtype=np.int16)
    with pytest.raises(ValueError, match=r"\[0, 127\]"):
        kernel(data, [0], n_bins=5)


# ============================================================================================
# pre_screen.py -- compute_unsupervised_drops / apply_drops
# ============================================================================================


def _mk(backend: str, data: dict):
    """Build a polars or pandas DataFrame from data depending on the requested backend."""
    return pl.DataFrame(data) if backend == "polars" else pd.DataFrame(data)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_drops_none_and_empty_return_empty(backend):
    """None input and a 0-row frame short-circuit to [] (nothing to fit on)."""
    assert compute_unsupervised_drops(None) == []
    empty = _mk(backend, {"a": []})
    assert compute_unsupervised_drops(empty) == []


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_drops_single_row_drops_numeric(backend):
    """A single-row numeric frame has undefined (NaN / null) variance for every column, so all
    numeric columns are dropped as effectively constant."""
    df = _mk(backend, {"a": [1.0], "b": [2.0]})
    assert compute_unsupervised_drops(df) == ["a", "b"]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_drops_constant_allnull_keep_signal_and_string(backend):
    """Constant + all-null columns drop; a varying numeric column and a non-constant string column
    are both kept (the variance rule is numeric-only, string handled downstream)."""
    n = 100
    rng = np.random.default_rng(0)
    data = {
        "const": [1.0] * n,
        "vary": rng.standard_normal(n).tolist(),
        "allnull": [None] * n,
        "s": ["x"] * 50 + ["y"] * 50,
    }
    drops = compute_unsupervised_drops(_mk(backend, data))
    assert drops == ["allnull", "const"]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_drops_null_fraction_boundary(backend):
    """Drop rule is null_count > 0.99 * n (strict). At n=1000 the cutoff is 990: 991 nulls drops,
    985 nulls (with real variance in the non-null values) is kept."""
    n = 1000
    over = [float(i % 9 + 1) for i in range(9)] + [None] * (n - 9)  # 991 nulls -> drop
    under = [float(i % 15 + 1) for i in range(15)] + [None] * (n - 15)  # 985 nulls -> keep
    drops = compute_unsupervised_drops(_mk(backend, {"over": over, "under": under}))
    assert drops == ["over"]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_drops_variance_threshold_custom(backend):
    """A caller-supplied variance_threshold drops low-variance columns above the 0.0 default."""
    rng = np.random.default_rng(1)
    n = 1000
    lowvar = (rng.standard_normal(n) * 0.1).tolist()  # var ~ 0.01
    hivar = (rng.standard_normal(n) * 5).tolist()  # var ~ 25
    drops = compute_unsupervised_drops(_mk(backend, {"lowvar": lowvar, "hivar": hivar}), variance_threshold=1.0)
    assert drops == ["lowvar"]


def test_drops_protected_columns_never_dropped():
    """A column named in protected_columns is never dropped even when it is constant."""
    n = 100
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"const": [1.0] * n, "vary": rng.standard_normal(n)})
    assert compute_unsupervised_drops(df, protected_columns=["const"]) == []
    assert compute_unsupervised_drops(df) == ["const"]


def test_drops_pandas_polars_parity():
    """The polars and pandas branches must return the identical drop set for the same data."""
    n = 200
    rng = np.random.default_rng(4)
    data = {
        "const": [3.14] * n,
        "vary": rng.standard_normal(n).tolist(),
        "allnull": [None] * n,
    }
    assert compute_unsupervised_drops(pd.DataFrame(data)) == compute_unsupervised_drops(pl.DataFrame(data))


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_apply_drops_idempotent_and_missing(backend):
    """apply_drops is idempotent (re-applying a done drop is a no-op), tolerates missing column
    names, and returns the frame unchanged for an empty drop list -- on both backends."""
    df = _mk(backend, {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    once = apply_drops(df, ["b"])
    twice = apply_drops(once, ["b"])
    assert list(once.columns) == ["a", "c"]
    assert list(twice.columns) == ["a", "c"]
    assert list(apply_drops(df, ["does_not_exist"]).columns) == ["a", "b", "c"]
    assert list(apply_drops(df, []).columns) == ["a", "b", "c"]


# ============================================================================================
# general.py -- estimate_features_relevancy
# ============================================================================================


def _relevancy_bins(seed: int = 0, n: int = 600) -> pl.DataFrame:
    """A tiny binned frame: target in [0,5), 'copy' == target (perfect signal), 'const' (dead),
    'noise' (independent). Codes stay in [0,14] for the kernels' default n_bins=15."""
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 5, n).astype(np.int64)
    return pl.DataFrame(
        {
            "target": t,
            "copy": t.copy(),
            "const": np.zeros(n, dtype=np.int64),
            "noise": rng.integers(0, 5, n).astype(np.int64),
        }
    )


def test_relevancy_zero_permutations_raises():
    """min_randomized_permutations < 1 is rejected up front (no baseline could be built)."""
    bins = _relevancy_bins()
    with pytest.raises(ValueError, match="min_randomized_permutations must be >= 1"):
        estimate_features_relevancy(
            bins=bins,
            target_columns=["target"],
            benchmark_mi_algorithms=False,
            min_randomized_permutations=0,
            verbose=0,
        )


def test_relevancy_single_target_keeps_signal_drops_dead():
    """Single-target run: the perfect-copy feature is never dropped, the constant column always is,
    and the MI matrix has one row per target across all columns."""
    bins = _relevancy_bins()
    result = estimate_features_relevancy(
        bins=bins,
        target_columns=["target"],
        benchmark_mi_algorithms=False,
        min_randomized_permutations=20,
        min_permuted_mi_evaluations=50,
        verbose=0,
    )
    assert len(result) == 4
    drop, orig_mi, _all_perm, _ranking = result
    assert orig_mi.shape == (1, bins.shape[1])
    assert "copy" in bins.columns and "copy" not in drop, "perfect-signal feature must survive"
    assert "const" in drop, "dead constant column must be dropped"
    assert "target" not in drop, "the target column is never itself dropped"


def test_relevancy_does_not_mutate_input_bins():
    """The permutation loop shuffles target columns of a to_numpy() copy and restores them; the
    caller's polars frame must be byte-for-byte unchanged after the call."""
    bins = _relevancy_bins(seed=5)
    before = bins.clone()
    estimate_features_relevancy(
        bins=bins,
        target_columns=["target"],
        benchmark_mi_algorithms=False,
        min_randomized_permutations=15,
        min_permuted_mi_evaluations=45,
        verbose=0,
    )
    assert bins.equals(before), "estimate_features_relevancy must not scramble the caller's frame"


def test_relevancy_quantile_baseline_path():
    """The permuted_max_mi_quantile branch (nanquantile baseline instead of nanmax) still runs and
    still drops the dead constant column."""
    bins = _relevancy_bins(seed=9)
    drop, _orig_mi, _all_perm, _rank = estimate_features_relevancy(
        bins=bins,
        target_columns=["target"],
        benchmark_mi_algorithms=False,
        min_randomized_permutations=20,
        min_permuted_mi_evaluations=50,
        permuted_max_mi_quantile=0.95,
        verbose=0,
    )
    assert isinstance(drop, list)
    assert "const" in drop
    assert "copy" not in drop


# ============================================================================================
# structure_discovery.py -- discover_structure
# ============================================================================================


def test_discover_gcd_relation_recovered():
    """y = gcd(a, b) is a number-theoretic relationship a correlation matrix cannot express; the
    integer-lattice detector must surface it as a 'gcd' relation over the two source columns, with a
    finite (strongly-significant) permutation p-value."""
    rng = np.random.default_rng(12345)
    a = rng.integers(1, 40, 2000)
    b = rng.integers(1, 40, 2000)
    X = pd.DataFrame({"price": a, "quantity": b, "noise": rng.normal(size=2000)})
    y = np.gcd(a, b)
    report = discover_structure(X, y, significance_n_perm=20)
    gcd_rels = [r for r in report.relations if r.kind == "gcd"]
    assert gcd_rels, f"gcd(price, quantity) not discovered; got kinds {[r.kind for r in report.relations]}"
    g = gcd_rels[0]
    assert set(g.columns) == {"price", "quantity"}
    assert g.mi > 0.0
    assert np.isfinite(g.p_value) and g.p_value <= 0.1, "strong gcd signal should be highly significant"


def test_discover_single_class_y_yields_empty_report():
    """A constant (single-class) y carries no MI, so no detector responds -- empty report, no crash,
    and it is NOT flagged as 'skipped' (a 1-D y is a valid input, just uninformative)."""
    rng = np.random.default_rng(0)
    a = rng.integers(1, 40, 1500)
    b = rng.integers(1, 40, 1500)
    X = pd.DataFrame({"price": a, "quantity": b})
    report = discover_structure(X, np.zeros(1500, dtype=int), significance_n_perm=0)
    assert len(report.relations) == 0
    assert report.skipped is None
    assert not report  # __bool__ is False on an empty report


def test_discover_2d_y_is_skipped_with_warning():
    """A 2-D (multilabel / multi-target) y is out of scope for the discrete detectors: the report is
    marked skipped and empty, and a UserWarning is emitted."""
    rng = np.random.default_rng(1)
    a = rng.integers(1, 40, 1200)
    b = rng.integers(1, 40, 1200)
    X = pd.DataFrame({"price": a, "quantity": b})
    y2d = np.column_stack([np.gcd(a, b), np.gcd(a, b)])
    with pytest.warns(UserWarning, match="2D / multi-target y"):
        report = discover_structure(X, y2d, significance_n_perm=0)
    assert report.skipped == "2D / multi-target y"
    assert len(report.relations) == 0


def test_discover_empty_x_yields_empty_report():
    """X with zero feature columns has nothing to scan; the report carries n_columns == 0 and no
    relations rather than raising."""
    y = np.random.default_rng(2).integers(0, 3, 100)
    report = discover_structure(pd.DataFrame(index=range(100)), y, significance_n_perm=0)
    assert report.n_columns == 0
    assert len(report.relations) == 0


def test_discover_noise_frame_no_false_discovery():
    """The headline anti-false-discovery guarantee: a smooth continuous-noise frame with a random
    target yields an EMPTY report (each detector's permutation-null gate rejects everything)."""
    rng = np.random.default_rng(20)
    n = 2000
    X = pd.DataFrame({f"g{i}": rng.standard_normal(n) for i in range(4)})
    y = rng.standard_normal(n)
    report = discover_structure(X, y, significance_n_perm=0)
    assert len(report.relations) == 0, f"noise frame should discover nothing; got {[r.kind for r in report.relations]}"


# ============================================================================================
# compare_selectors.py
# ============================================================================================


class _FakeSelector:
    """Minimal pre-fitted selector: a boolean support_ mask + feature_names_in_ (the support_ path
    of _extract_selected). Enough for compare_selectors without any heavy real selector."""

    def __init__(self, name: str, support, names_in):
        self._compare_name = name
        self.support_ = np.asarray(support, dtype=bool)
        self.feature_names_in_ = np.asarray(names_in, dtype=object)


@pytest.fixture
def _cmp_frame():
    """Cmp frame."""
    names = ["f0", "f1", "f2", "f3"]
    X = pd.DataFrame(np.random.default_rng(0).standard_normal((20, 4)), columns=names)
    return X, names


def test_compare_none_and_empty_selectors_raise(_cmp_frame):
    """A missing or empty selector collection is a usage error, not a silent empty report."""
    X, _ = _cmp_frame
    with pytest.raises(ValueError):
        compare_selectors(X, selectors=None)
    with pytest.raises(ValueError):
        compare_selectors(X, selectors=[])


def test_compare_empty_feature_set_raises():
    """A frame with zero columns would make every selector trivially 'agree' on nothing; reject it."""
    empty_cols = pd.DataFrame(index=range(5))
    sel = _FakeSelector("A", [], [])
    with pytest.raises(ValueError, match=">= 1 column"):
        compare_selectors(empty_cols, selectors=[sel])


def test_compare_jaccard_and_consensus_values(_cmp_frame):
    """Two pre-fitted selectors picking {f0,f1} and {f1,f2}: exact Jaccard (1/3) and per-feature
    consensus counts (f1 by both, f0/f2 by one, f3 by none)."""
    X, names = _cmp_frame
    A = _FakeSelector("A", [True, True, False, False], names)
    B = _FakeSelector("B", [False, True, True, False], names)
    cmp = compare_selectors(X, selectors=[A, B], fit=False)
    assert cmp.n_selectors == 2
    assert cmp.jaccard.loc["A", "B"] == pytest.approx(1.0 / 3.0)
    assert cmp.jaccard.loc["A", "A"] == pytest.approx(1.0)
    assert cmp.consensus.to_dict() == {"f0": 1, "f1": 2, "f2": 1, "f3": 0}
    # report() renders without error and names both selectors
    text = cmp.report()
    assert "2 selector(s)" in text and "PAIRWISE JACCARD" in text


def test_compare_mixed_fitted_skips_unfitted(_cmp_frame):
    """With fit=False an un-fitted selector is skipped with a recorded reason while the fitted one
    still contributes -- the report degrades gracefully instead of aborting."""
    from types import SimpleNamespace

    X, names = _cmp_frame
    A = _FakeSelector("A", [True, False, True, False], names)
    unfitted = SimpleNamespace()  # no support accessor -> _is_fitted False
    cmp = compare_selectors(X, selectors={"A": A, "U": unfitted}, fit=False)
    assert cmp.n_selectors == 1
    assert "U" in cmp.skipped and "not fitted" in cmp.skipped["U"]
    assert set(cmp.agreement.columns) == {"A"}


def test_compare_deduplicates_identical_names(_cmp_frame):
    """Two selectors sharing a display name are disambiguated (A, A#2), not silently merged."""
    X, names = _cmp_frame
    A1 = _FakeSelector("A", [True, False, False, False], names)
    A2 = _FakeSelector("A", [False, True, False, False], names)
    cmp = compare_selectors(X, selectors=[A1, A2], fit=False)
    assert set(cmp.jaccard.columns) == {"A", "A#2"}
    # distinct picks -> disjoint -> Jaccard 0
    assert cmp.jaccard.loc["A", "A#2"] == pytest.approx(0.0)


def test_compare_unreadable_support_is_skipped(_cmp_frame):
    """A selector that LOOKS fitted (has feature_names_in_) but exposes no support accessor is
    skipped with a 'no readable support' note, not crashed on."""
    from types import SimpleNamespace

    X, names = _cmp_frame
    good = _FakeSelector("good", [True, True, False, False], names)
    bad = SimpleNamespace(feature_names_in_=np.asarray(names, dtype=object), _compare_name="bad")
    cmp = compare_selectors(X, selectors=[good, bad], fit=False)
    assert cmp.n_selectors == 1
    assert "bad" in cmp.skipped and "no readable support" in cmp.skipped["bad"]
