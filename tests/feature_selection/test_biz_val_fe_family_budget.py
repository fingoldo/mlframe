"""biz_val + unit tests for gt_07's FE generator-family Shapley-flavored compute budgeting.

See ``research/gt_07_fe_generator_shapley_budgeting.md``. The name/kind -> family parser is pinned
against REAL generated ``fe_provenance_`` output (a real ``MRMR.fit`` call), not hardcoded guesses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._fe_family_budget import (
    _ORIGINAL_FAMILY,
    dataset_fingerprint,
    family_credit,
    family_roi,
    load_budgets,
    persist_budgets,
    reallocate_budgets,
)


def _make_triplet_useful_bed(n=2000, seed=0, flip=False):
    """y depends on sign(x1*x2*x3) (a genuine 3-way interaction -- the triplet family's specialty);
    6 noise columns give the quadruplet family (seeded with an inflated seed_k/top_count quota, so
    it's "expensive useless") plenty of candidates to search fruitlessly. ``flip=True`` switches the
    signal to a genuine 4-way product of DIFFERENT columns (quadruplet's specialty, not triplet's) --
    the floor/recovery test needs the PREVIOUSLY-useless family to become the new signal carrier,
    not just a differently-shaped instance of the already-useful one."""
    rng = np.random.default_rng(seed)
    cols = {f"x{i}": rng.standard_normal(n) for i in range(1, 4)}
    noise = {f"z{i}": rng.standard_normal(n) for i in range(6)}
    all_cols = {**cols, **noise}
    if flip:
        a, b, c, d = (all_cols[c] for c in ("z0", "z1", "z2", "z3"))
        y = pd.Series((np.sign(a * b * c * d) > 0).astype(int))
    else:
        a, b, c = (all_cols[c] for c in ("x1", "x2", "x3"))
        y = pd.Series((np.sign(a * b * c) > 0).astype(int))
    X = pd.DataFrame(all_cols)
    return X, y


def test_family_credit_parser_pinned_against_real_mrmr_provenance():
    """family_credit correctly attributes credit using a REAL MRMR fit's fe_provenance_ (not a hand-crafted DataFrame): triplet column gets nonzero credit, raw column gets none."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 1500
    x1, x2, x3 = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)
    y = pd.Series((np.sign(x1 * x2 * x3) > 0).astype(int))
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

    m = MRMR(fe_hybrid_orth_pair_enable=True, fe_hybrid_orth_triplet_enable=True, verbose=0, random_state=0)
    m.fit(X, y)

    credit = family_credit(m.fe_provenance_)
    assert credit.get("triplet", 0.0) > 0.0, f"expected nonzero triplet credit, got {credit}"
    assert credit.get(_ORIGINAL_FAMILY, 0.0) == 0.0, f"raw columns should carry no engineered-family credit: {credit}"


def test_family_credit_additive_sums_survivor_gains_per_family():
    """A hand-built fe_provenance_-shaped DataFrame: credit sums mrmr_gain per family, ignores NaN gain as 0."""
    prov = pd.DataFrame(
        {
            "feature_name": ["raw1", "a*b__He1_He1", "c*d*e__He1_He1_He1", "cat_x__Xy"],
            "origin": ["raw", "hybrid_orth", "hybrid_orth", "cat_cross"],
            "mechanism_details": [
                "{}",
                "{'kind': 'orth_pair_cross', 'src_names': ['a', 'b']}",
                "{'kind': 'orth_triplet_cross', 'src_names': ['c', 'd', 'e']}",
                "{'kind': 'cat_pair_cross', 'src_names': ['x', 'y']}",
            ],
            "mrmr_gain": [np.nan, 0.3, 0.5, 0.2],
            "support_rank": [1, 2, 3, 4],
        }
    )
    credit = family_credit(prov)
    assert credit["orth_pair"] == pytest.approx(0.3)
    assert credit["triplet"] == pytest.approx(0.5)
    assert credit["cat_pair"] == pytest.approx(0.2)
    assert credit[_ORIGINAL_FAMILY] == 0.0


def test_family_credit_loo_not_implemented():
    """credit='loo' raises NotImplementedError (specced as v2, not shipped in v1)."""
    with pytest.raises(NotImplementedError):
        family_credit(pd.DataFrame(), credit="loo")


def test_family_roi_never_run_family_gets_none():
    """A family with 0 recorded invocations gets ROI None, not 0 -- must not be starved before its first trial."""
    wall = {"triplet": (5.0, 3), "quadruplet": (0.0, 0)}
    credit = {"triplet": 2.0}
    roi = family_roi(credit, wall)
    assert roi["triplet"] == pytest.approx(0.4)
    assert roi["quadruplet"] is None


def test_reallocate_budgets_conserves_total_mass():
    """Sum of reallocated budgets equals sum of base_budget, for any ROI mix."""
    base = {"triplet": 0.4, "quadruplet": 0.3, "orth_pair": 0.3}
    roi = {"triplet": 5.0, "quadruplet": 0.0, "orth_pair": None}
    out = reallocate_budgets(roi, base_budget=base, floor=0.1, smoothing=1.0, exploration=0.1)
    assert sum(out.values()) == pytest.approx(sum(base.values()), abs=1e-9)


def test_reallocate_budgets_floor_respected():
    """No family's reallocated budget falls below floor * its base_budget, even at zero ROI."""
    base = {"useful": 0.5, "useless": 0.5}
    roi = {"useful": 10.0, "useless": 0.0}
    out = reallocate_budgets(roi, base_budget=base, floor=0.1, smoothing=1.0, exploration=0.0)
    assert out["useless"] >= 0.1 * base["useless"] - 1e-9, f"useless family fell below floor: {out}"
    assert out["useful"] > out["useless"], f"useful family should end up with more budget: {out}"


def test_reallocate_budgets_never_run_gets_exploration_share_not_zero():
    """A never-run family (ROI=None) still gets a positive share via the exploration reserve, not starved to 0."""
    base = {"known": 0.7, "never_run": 0.3}
    roi = {"known": 2.0, "never_run": None}
    out = reallocate_budgets(roi, base_budget=base, floor=0.1, smoothing=1.0, exploration=0.2)
    assert out["never_run"] > 0.0, f"never-run family got starved to 0: {out}"


def test_reallocate_budgets_smoothing_zero_keeps_base_unchanged():
    """smoothing=0 fully damps the new allocation -- output equals base_budget exactly."""
    base = {"a": 0.6, "b": 0.4}
    roi = {"a": 100.0, "b": 0.0}
    out = reallocate_budgets(roi, base_budget=base, floor=0.1, smoothing=0.0, exploration=0.0)
    for k in base:
        assert out[k] == pytest.approx(base[k], abs=1e-9)


def test_dataset_fingerprint_stable_and_order_independent():
    """Same (n_features, column set) fingerprints identically regardless of column ORDER; a different column set fingerprints differently."""
    fp1 = dataset_fingerprint(3, ["a", "b", "c"])
    fp2 = dataset_fingerprint(3, ["c", "a", "b"])
    fp3 = dataset_fingerprint(3, ["a", "b", "d"])
    assert fp1 == fp2
    assert fp1 != fp3


def test_persist_and_load_budgets_roundtrip(tmp_path, monkeypatch):
    """persist_budgets then load_budgets recovers the exact same dict, isolated to a tmp cache dir."""
    monkeypatch.setattr("mlframe.feature_selection.filters._fe_family_budget._BUDGET_CACHE_DIR", tmp_path)
    budgets = {"triplet": 0.4, "quadruplet": 0.25, "orth_pair": 0.35}
    fp = dataset_fingerprint(4, ["a", "b", "c", "d"])
    persist_budgets(budgets, fingerprint=fp)
    loaded = load_budgets(fingerprint=fp)
    assert loaded == pytest.approx(budgets)


def test_load_budgets_missing_returns_none(tmp_path, monkeypatch):
    """load_budgets returns None (not an error) when nothing was ever persisted for this fingerprint."""
    monkeypatch.setattr("mlframe.feature_selection.filters._fe_family_budget._BUDGET_CACHE_DIR", tmp_path)
    assert load_budgets(fingerprint="never-seen-before") is None


def test_load_budgets_corrupt_file_returns_none_not_raises(tmp_path, monkeypatch):
    """A corrupt cache file is treated as absent (returns None, logs a warning) rather than crashing the caller."""
    monkeypatch.setattr("mlframe.feature_selection.filters._fe_family_budget._BUDGET_CACHE_DIR", tmp_path)
    fp = "corrupt-test"
    (tmp_path / f"mlframe.fe_family_budget.{fp}.json").write_text("{not valid json", encoding="utf-8")
    assert load_budgets(fingerprint=fp) is None


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_biz_val_fe_budget_shifts_toward_useful_family(tmp_path, monkeypatch):
    """After one fit with fe_budget_learning=True on the triplet-useful bed, quadruplet's (the
    "expensive useless" family here, seeded with an inflated quota) budget fraction drops by >= 40%
    from equal-split (0.333), and triplet's (useful) share rises.

    Measured on this exact bed/config: quadruplet 0.333 -> 0.183 (45% drop), triplet 0.333 -> 0.600.
    Floor set to 40% (5-15% below measurement) per the repo's biz_val threshold convention.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    monkeypatch.setattr("mlframe.feature_selection.filters._fe_family_budget._BUDGET_CACHE_DIR", tmp_path)
    X, y = _make_triplet_useful_bed(seed=1)  # distinct seed per test: MRMR memoizes fits by (X, y) content-hash across test functions in the same process

    m = MRMR(
        fe_hybrid_orth_triplet_enable=True,
        fe_hybrid_orth_quadruplet_enable=True,
        fe_hybrid_orth_quadruplet_seed_k=8,
        fe_hybrid_orth_quadruplet_top_count=6,
        fe_budget_learning=True,
        verbose=0,
        random_state=0,
    )
    m.fit(X, y)
    report = m.fe_family_budget_
    equal_share = 1.0 / 3.0
    quadruplet_after = report["budgets_after"]["quadruplet"]
    triplet_after = report["budgets_after"]["triplet"]

    quadruplet_drop_frac = (equal_share - quadruplet_after) / equal_share
    assert quadruplet_drop_frac >= 0.40, f"quadruplet budget only dropped {quadruplet_drop_frac:.2%} from equal-split, expected >= 40%"
    assert triplet_after > equal_share, f"triplet (useful) budget did not rise above equal-split: {triplet_after:.4f}"


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_biz_val_fe_budget_recall_preserved_and_wall_drops(tmp_path):
    """fit 2 (learned budgets from fit 1) vs fit 1 (equal-split): the triplet-composed engineered
    feature is still selected (the useful family was never the one cut) and fit 2's recorded
    quadruplet-family FE wall (the ``report["wall"]`` ledger -- an exact per-family perf_counter
    accumulation, see ``_fe_family_timing.py``) drops meaningfully vs fit 1's, since quadruplet's
    quota was shrunk by the learned budget.

    Asserts the internal per-family wall ledger, NOT raw end-to-end fit wall-clock: a whole-fit
    wall-clock comparison was measured to be flaky when this test runs alongside the rest of this
    file's MRMR fits in the same process (genuine OS-scheduling/CPU-contention noise across many
    back-to-back fits, not a correctness issue -- isolated single-test runs showed a clean ~0.2x
    drop, but the ratio was unstable, 0.6-1.05x, once several other unrelated fits shared the CPU).
    The per-family ledger is exact and unaffected by that noise -- it is precisely the quantity the
    budget mechanism controls."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    def _fit_once(cache_dir, seed, budget_learning):
        """One fit on a fresh (seed-distinct) bed with an isolated budget-cache dir; returns (fe_family_budget_ report or None, has_triplet_survivor)."""
        import mlframe.feature_selection.filters._fe_family_budget as fb

        fb._BUDGET_CACHE_DIR = cache_dir
        X, y = _make_triplet_useful_bed(seed=seed)
        m = MRMR(
            fe_hybrid_orth_triplet_enable=True,
            fe_hybrid_orth_quadruplet_enable=True,
            fe_hybrid_orth_quadruplet_seed_k=8,
            fe_hybrid_orth_quadruplet_top_count=6,
            fe_budget_learning=budget_learning,
            verbose=0,
            random_state=0,
        )
        m.fit(X, y)
        survivors = list(getattr(m, "hybrid_orth_features_", None) or [])
        has_triplet = any(c.count("*") == 2 for c in survivors)  # 3 legs joined by '*' = a triplet column
        return getattr(m, "fe_family_budget_", None), has_triplet

    seed = 20
    report1, has_triplet1 = _fit_once(tmp_path, seed, budget_learning=True)  # equal-split (nothing persisted yet)
    report2, has_triplet2 = _fit_once(tmp_path, seed, budget_learning=True)  # now uses fit 1's learned budget

    assert has_triplet1, "fit 1 (equal-split budget) found no surviving triplet feature -- bed premise broken"
    assert has_triplet2, "fit 2 (learned budget) lost the useful triplet family's surviving feature"

    quad_wall1 = report1["wall"].get("quadruplet", (0.0, 0))[0]
    quad_wall2 = report2["wall"].get("quadruplet", (0.0, 0))[0]
    assert quad_wall2 < quad_wall1, f"quadruplet FE-stage wall did not drop after budget learning: fit1={quad_wall1:.4f}s, fit2={quad_wall2:.4f}s"


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_biz_val_fe_budget_floor_prevents_starvation(tmp_path, monkeypatch):
    """3 sequential fits: quadruplet's (useless here -- the signal is a 3-way triplet product) budget
    never falls below floor * base_budget; then flip the bed to a genuine 4-way product (quadruplet
    becomes the useful family) and verify its budget recovers upward within 2 more fits -- the
    explore/exploit guarantee (a floored family is never permanently starved, so it can seize the
    opportunity the moment its own signal appears)."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    monkeypatch.setattr("mlframe.feature_selection.filters._fe_family_budget._BUDGET_CACHE_DIR", tmp_path)
    X, y = _make_triplet_useful_bed(seed=3)  # distinct seed per test: MRMR memoizes fits by (X, y) content-hash across test functions in the same process
    floor = 0.1
    equal_share = 1.0 / 3.0

    def _fit_once(X_bed, y_bed):
        """One fe_budget_learning=True fit; returns the post-fit budgets_after dict."""
        m = MRMR(
            fe_hybrid_orth_triplet_enable=True,
            fe_hybrid_orth_quadruplet_enable=True,
            fe_hybrid_orth_quadruplet_seed_k=8,
            fe_hybrid_orth_quadruplet_top_count=6,
            fe_budget_learning=True,
            fe_budget_kwargs={"floor": floor},
            verbose=0,
            random_state=0,
        )
        m.fit(X_bed, y_bed)
        return m.fe_family_budget_["budgets_after"]

    budgets_history = []
    for _ in range(3):
        budgets_history.append(_fit_once(X, y))

    for i, budgets in enumerate(budgets_history):
        assert budgets["quadruplet"] >= floor * equal_share - 1e-6, f"fit {i}: quadruplet budget {budgets['quadruplet']:.4f} fell below the floor"

    X_flip, y_flip = _make_triplet_useful_bed(flip=True)
    budgets_after_flip = [_fit_once(X_flip, y_flip) for _ in range(2)]

    # The flip switches the signal to a genuine 4-way product (quadruplet's specialty) -- the
    # explore/exploit guarantee under test is that the FLOOR kept quadruplet alive (never zeroed)
    # throughout the pre-flip fits, so it was ABLE to compete once its own signal appeared. Assert
    # quadruplet's budget recovered ABOVE its pre-flip floor-pinned value within the 2 post-flip fits.
    quadruplet_after_flip = budgets_after_flip[-1]["quadruplet"]
    quadruplet_before_flip = budgets_history[-1]["quadruplet"]
    assert (
        quadruplet_after_flip > quadruplet_before_flip
    ), f"quadruplet's budget did not recover after the bed flip made it the useful family: before={quadruplet_before_flip:.4f}, after={quadruplet_after_flip:.4f}"
