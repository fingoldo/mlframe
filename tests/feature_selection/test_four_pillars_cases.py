"""The extended four-pillar cases generate what they claim, deterministically, and their contracts tell a right selection from a wrong one.

No selector is fitted here: the contracts are exercised on stub selectors with a known output, and the generators on their own.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection._benchmarks.four_pillars_cases import CASES, f2_monotone_decoys, g_parity_k
from mlframe.feature_selection._benchmarks.four_pillars_contracts import cover_verdict, cp_lower, cp_upper, f2_verdict, null_verdict, panel, refs


class _Stub:
    """Selector stand-in: fixed output names and a transform that evaluates a callable on the frame."""

    def __init__(self, names, fn=None):
        """Remember the output names and the transform."""
        self._names = list(names)
        self._fn = fn

    def get_feature_names_out(self):
        """The fixed output names."""
        return np.array(self._names)

    def transform(self, X):
        """The callable applied to the frame, one column per name."""
        return self._fn(X)


def _frame_of(case, seed):
    """Generate one case at a small size, returning the frame and the target regardless of the case kind."""
    kwargs = dict(case.gen_kwargs)
    if case.kind != "f2" and "n" in case.gen.__code__.co_varnames and "n" not in kwargs:
        kwargs["n"] = 600
    out = case.gen(seed, **kwargs)
    return out[0], out[1]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_every_case_generates_a_deterministic_aligned_frame(case):
    """Two draws with the same seed are identical, a different seed differs, and X and y have the same length."""
    X1, y1 = _frame_of(case, 3)
    X2, y2 = _frame_of(case, 3)
    X3, _ = _frame_of(case, 4)
    assert len(X1) == len(np.asarray(y1))
    pd.testing.assert_frame_equal(X1, X2)
    assert np.array_equal(np.asarray(y1), np.asarray(y2), equal_nan=True) if np.asarray(y1).dtype.kind == "f" else np.array_equal(np.asarray(y1), np.asarray(y2))
    assert not X1.equals(X3)


def test_case_names_are_unique():
    """The registry keys cases by name, so a duplicate would silently hide one."""
    names = [c.name for c in CASES]
    assert len(names) == len(set(names))


def test_a_parity_target_has_no_marginal_signal_in_any_operand():
    """3-way parity: each operand alone is independent of y, the three together determine it."""
    X, y = g_parity_k(0, k=3, n=4000, p_noise=5)
    y = np.asarray(y)
    for c in ("x1", "x2", "x3"):
        assert abs(np.corrcoef(X[c], y)[0, 1]) < 0.06
    assert np.array_equal(X[["x1", "x2", "x3"]].to_numpy().sum(1).astype(int) % 2, y)


def test_refs_matches_whole_identifiers_only():
    """``a`` is found in ``mul(a,b)`` but not in ``a_dup``, ``sqrt`` or ``quad``."""
    assert refs("mul(a,b)", ["a", "b", "c"]) == {"a", "b"}
    assert refs("a_dup", ["a"]) == set()
    assert refs("sqrt(c)", ["a", "c"]) == {"c"}


def test_clopper_pearson_bounds_match_the_documented_values():
    """0 of 59 bounds a rate below 5 percent, 0 of 30 below 9.5 percent, and 20 of 20 bounds the rate above 86 percent."""
    assert cp_upper(0, 59) == pytest.approx(0.049, abs=0.001)
    assert cp_upper(0, 30) == pytest.approx(0.095, abs=0.001)
    assert cp_lower(20, 20) == pytest.approx(0.861, abs=0.001)


def test_panel_needs_the_bound_not_the_rate():
    """9 of 10 successes is a rate of 0.9 but its lower bound is under the gate, so it is INCONCLUSIVE rather than PASS."""
    rows = [{"ok": True, "n_forbidden": 0}] * 9 + [{"ok": False, "n_forbidden": 0}]
    assert panel(rows, power_gate=0.9, fp_gate=0.3)["verdict"] == "INCONCLUSIVE"
    assert panel([{"ok": True, "n_forbidden": 0}] * 59, power_gate=0.9, fp_gate=0.05)["verdict"] == "PASS"


def test_null_verdict_fails_on_any_selected_column():
    """A pure-noise fit passes only when nothing is selected."""
    assert null_verdict(_Stub([]))["ok"]
    assert not null_verdict(_Stub(["noise_3"]))["ok"]


def test_cover_verdict_accepts_the_composite_and_rejects_noise_or_missing_operands():
    """``mul(x1,mul(x2,x3))`` covers the triple; a noise token, a missing operand or too many columns each fail."""
    ok = cover_verdict(_Stub(["mul(x1,mul(x2,x3))"]), [["x1", "x2", "x3"]], max_cols=1)
    assert ok["ok"]
    assert not cover_verdict(_Stub(["mul(x1,x2)"]), [["x1", "x2", "x3"]])["ok"]
    assert not cover_verdict(_Stub(["mul(x1,mul(x2,x3))", "noise_4"]), [["x1", "x2", "x3"]])["ok"]
    assert not cover_verdict(_Stub(["x1*x2*x3", "x1"]), [["x1", "x2", "x3"]], max_cols=1)["ok"]


def test_cover_verdict_counts_a_monotone_twin_as_its_source_and_a_partial_and_as_forbidden():
    """``x1_exp`` normalises to ``x1``; a selected ``AB`` partial decoy is forbidden by name."""
    twins = cover_verdict(_Stub(["mul(x1_exp,mul(x2,x3_cube))"]), [["x1", "x2", "x3"]], normalize=("_exp", "_cube"))
    assert twins["ok"]
    partial = cover_verdict(_Stub(["mul(A,mul(B,C))", "AB"]), [["A", "B", "C"]], forbidden_prefixes=(), forbidden_exact=("AB", "AC", "BC"))
    assert not partial["ok"] and partial["forbidden"] == ["AB"]


def test_f2_verdict_rejects_a_wrong_form_column_that_mentions_the_right_operands():
    """The name-token test passes any column that mentions a, b, c, d; the held-out rank check does not."""
    gen = f2_monotone_decoys
    right = _Stub(["f2"], lambda X: (X["a"] ** 2 / X["b"] + np.log(np.abs(X["c"]) + 1e-9) * np.sin(X["d"])).to_numpy())
    wrong = _Stub(["mul(a,b,c,d)"], lambda X: (X["a"] * X["b"] * X["c"] * X["d"]).to_numpy())
    assert f2_verdict(right, gen, 0, n=1500)["ok"]
    assert not f2_verdict(wrong, gen, 0, n=1500)["ok"]
    noisy = _Stub(["f2", "e"], lambda X: np.c_[right._fn(X), X["e"]])
    assert f2_verdict(noisy, gen, 0, n=1500)["noise"] == ["e"]
