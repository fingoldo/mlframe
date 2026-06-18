"""biz_value + unit tests for ``discover_structure`` -- the standalone discrete-structure EDA tool.

biz_value: on synthetics with KNOWN structure (gcd / modular / regime-switch / argmax), ``discover_structure`` surfaces the right relation
with the right columns + parameter, clearing a measured MI / lift floor. specificity: a smooth / linear / noise frame returns an EMPTY
report (the anti-false-discovery guarantee). robustness: 2D y skips cleanly with a warning; all-noise + tiny frames don't crash.
human-readable: ``str(report)`` carries the column names + kind.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection import discover_structure, StructureReport, DiscoveredRelation


N = 2000


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_biz_val_discover_gcd_classification_top_relation():
    rng = _rng(0)
    a = rng.integers(1, 40, N)
    b = rng.integers(1, 40, N)
    X = pd.DataFrame({"price": a, "quantity": b, "noise": rng.normal(size=N), "c3": rng.integers(0, 5, N)})
    y = np.gcd(a, b)
    report = discover_structure(X, y)
    assert isinstance(report, StructureReport)
    assert report.relations, "gcd target must produce at least one discovered relation"
    top = report.relations[0]
    assert top.kind == "gcd"
    assert set(top.columns) == {"price", "quantity"}
    assert top.mi >= 0.8, f"gcd MI floor (measured ~0.97); got {top.mi}"
    assert top.lift >= 3.0, f"gcd lift floor (measured ~6.4x); got {top.lift}"


def test_biz_val_discover_gcd_regression_variant():
    rng = _rng(2)
    a = rng.integers(1, 40, N)
    b = rng.integers(1, 40, N)
    X = pd.DataFrame({"p": a, "q": b, "noise": rng.normal(size=N)})
    y = np.gcd(a, b).astype(float) + rng.normal(0, 0.1, N)  # continuous y -> internal qcut binning
    report = discover_structure(X, y)
    assert report.relations
    top = report.relations[0]
    assert top.kind == "gcd"
    assert set(top.columns) == {"p", "q"}


def test_biz_val_discover_modular_modulus():
    rng = _rng(1)
    a = rng.integers(0, 50, N)
    b = rng.integers(0, 50, N)
    X = pd.DataFrame({"a": a, "b": b, "z": rng.normal(size=N)})
    y = (a + b) % 7
    report = discover_structure(X, y)
    mod = [r for r in report.relations if r.kind in ("modular", "parity")]
    assert mod, "a (a+b) mod 7 target must surface a modular relation"
    top = mod[0]
    assert set(top.columns) == {"a", "b"}
    # The escalate stage may pin a MULTIPLE of the true modulus (7 | 14 | 21 ... all carry the residue signal).
    assert top.parameter is not None and int(round(top.parameter)) % 7 == 0, f"modulus must be a multiple of 7; got {top.parameter}"
    assert top.mi >= 0.5


def test_biz_val_discover_gate_regime_switch():
    rng = _rng(3)
    base = rng.normal(size=N)
    discount = rng.normal(size=N)
    tenure = rng.normal(size=N)
    X = pd.DataFrame({"base": base, "discount": discount, "tenure": tenure, "n1": rng.normal(size=N)})
    y = np.where(tenure > 0.4, discount, base)
    yb = pd.qcut(y, 10, labels=False, duplicates="drop")
    report = discover_structure(X, yb)
    gate = [r for r in report.relations if r.kind.startswith("gate")]
    assert gate, "a regime-switch target must surface a gate relation"
    top = gate[0]
    assert top.kind == "gate_select"
    # gate select cols = (a, b, c) for ``c>tau ? a : b``; here a=discount, b=base, c=tenure.
    assert top.columns[-1] == "tenure", f"gating column must be tenure; got {top.columns}"
    assert set(top.columns) == {"base", "discount", "tenure"}
    assert top.parameter is not None
    assert top.mi >= 1.0


def test_biz_val_discover_argmax():
    rng = _rng(4)
    c0, c1, c2 = (rng.normal(size=N) for _ in range(3))
    X = pd.DataFrame({"x0": c0, "x1": c1, "x2": c2, "n": rng.normal(size=N)})
    y = np.argmax(np.stack([c0, c1, c2], axis=1), axis=1)
    report = discover_structure(X, y)
    am = [r for r in report.relations if r.kind == "argmax"]
    assert am, "an argmax target must surface an argmax relation"
    top = am[0]
    assert set(top.columns) == {"x0", "x1", "x2"}
    assert top.mi >= 0.7


def test_specificity_linear_frame_empty():
    rng = _rng(5)
    X = pd.DataFrame({f"f{i}": rng.normal(size=N) for i in range(8)})
    y = 2 * X["f0"] + X["f1"] - 0.5 * X["f2"] + rng.normal(0, 0.1, N)
    report = discover_structure(X, y)
    assert len(report.relations) == 0, f"linear frame must yield 0 discoveries; got {[r.kind for r in report.relations]}"


def test_specificity_noise_frame_empty():
    rng = _rng(6)
    X = pd.DataFrame({f"g{i}": rng.normal(size=N) for i in range(8)})
    y = rng.normal(size=N)
    report = discover_structure(X, y)
    assert len(report.relations) == 0
    assert "no discrete structural relationships detected" in str(report)


def test_robustness_2d_y_skips_with_warning():
    rng = _rng(7)
    a = rng.integers(1, 40, 500)
    b = rng.integers(1, 40, 500)
    X = pd.DataFrame({"a": a, "b": b})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        report = discover_structure(X, np.stack([a, b], axis=1))
    assert len(report.relations) == 0
    assert report.skipped is not None
    assert any("2D" in str(wi.message) or "multi-target" in str(wi.message) for wi in w)


def test_robustness_tiny_frame_no_crash():
    rng = _rng(8)
    X = pd.DataFrame({"a": rng.integers(0, 5, 30), "b": rng.integers(0, 5, 30)})
    y = rng.integers(0, 2, 30)
    report = discover_structure(X, y)  # must not raise
    assert isinstance(report, StructureReport)


def test_numpy_X_fallback():
    rng = _rng(9)
    a = rng.integers(1, 40, N)
    b = rng.integers(1, 40, N)
    Xnp = np.column_stack([a, b, rng.normal(size=N)])
    y = np.gcd(a, b)
    report = discover_structure(Xnp, y)
    assert report.relations
    assert report.relations[0].kind == "gcd"
    assert report.relations[0].columns == ("f0", "f1")  # positional names


def test_human_readable_str_contains_columns_and_kind():
    rng = _rng(0)
    a = rng.integers(1, 40, N)
    b = rng.integers(1, 40, N)
    X = pd.DataFrame({"price": a, "quantity": b, "noise": rng.normal(size=N)})
    y = np.gcd(a, b)
    report = discover_structure(X, y)
    s = str(report)
    assert "gcd" in s
    assert "price" in s and "quantity" in s
    assert "StructureReport" in s


def test_biz_val_max_int_cols_budget_guard_skips_above_cap():
    """``max_int_cols`` is the budget guard on the modular + integer-lattice sweeps: when the number of integer
    columns EXCEEDS the cap, those sweeps are skipped entirely, so a real gcd structure goes undiscovered; raising
    the cap to admit all integer columns recovers it. Measured (seed=0, N=2000, 10 integer columns): cap=5 -> 0
    relations (gcd skipped); cap=10 -> gcd surfaced. The DELTA between the two caps is the guard's whole behaviour."""
    rng = _rng(0)
    a = rng.integers(1, 40, N)
    b = rng.integers(1, 40, N)
    cols = {"price": a, "quantity": b}
    for j in range(8):  # 2 signal + 8 filler integer columns = 10 integer columns total
        cols[f"i{j}"] = rng.integers(0, 30, N)
    X = pd.DataFrame(cols)
    y = np.gcd(a, b)

    capped = discover_structure(X, y, significance_n_perm=0, max_int_cols=5)
    assert not any(r.kind == "gcd" for r in capped.relations), (
        f"10 integer columns > cap 5 must skip the lattice sweep -> no gcd; got {[r.kind for r in capped.relations]}"
    )

    uncapped = discover_structure(X, y, significance_n_perm=0, max_int_cols=10)
    gcd = [r for r in uncapped.relations if r.kind == "gcd"]
    assert gcd, "raising max_int_cols to admit all 10 integer columns must recover the gcd structure"
    assert set(gcd[0].columns) == {"price", "quantity"}


def test_biz_val_nbins_resolution_recovers_multivalued_structure():
    """``nbins`` sets the resolution of the internal y-binning that drives the MI estimate. On a CONTINUOUS gcd
    target (binned via qcut(nbins) inside discover_structure) too-coarse binning crushes the multi-valued discrete
    structure: 2 bins collapse gcd's many distinct values into a near-binary signal, while finer bins resolve it.
    Measured (seed=2, N=2000): gcd MI rises monotonically 0.203 (nbins=2) -> 1.050 (nbins=20), ~5x. Floors pinned
    10-15% below the measured endpoints; assert the coarse-vs-fine MI DELTA, not just discovery."""
    rng = _rng(2)
    a = rng.integers(1, 40, N)
    b = rng.integers(1, 40, N)
    X = pd.DataFrame({"p": a, "q": b, "noise": rng.normal(size=N)})
    yc = np.gcd(a, b).astype(float) + rng.normal(0, 0.1, N)

    coarse = discover_structure(X, yc, significance_n_perm=0, nbins=2)
    fine = discover_structure(X, yc, significance_n_perm=0, nbins=20)
    gcd_coarse = [r for r in coarse.relations if r.kind == "gcd"]
    gcd_fine = [r for r in fine.relations if r.kind == "gcd"]
    assert gcd_coarse and gcd_fine, "gcd structure must be discoverable at both binning resolutions"
    mi_coarse, mi_fine = gcd_coarse[0].mi, gcd_fine[0].mi
    assert mi_coarse <= 0.30, f"nbins=2 must crush the multi-valued gcd MI (measured ~0.20); got {mi_coarse}"
    assert mi_fine >= 0.90, f"nbins=20 must resolve the gcd structure (measured ~1.05); got {mi_fine}"
    assert mi_fine >= 3.0 * mi_coarse, (
        f"finer binning must lift the recovered gcd MI >=3x over coarse (measured ~5x); "
        f"coarse={mi_coarse:.3f} fine={mi_fine:.3f}"
    )


def test_top_k_cap():
    rng = _rng(0)
    a = rng.integers(1, 40, N)
    b = rng.integers(1, 40, N)
    X = pd.DataFrame({"price": a, "quantity": b, "c3": rng.integers(0, 8, N), "c4": rng.integers(0, 8, N)})
    y = np.gcd(a, b)
    report = discover_structure(X, y, top_k=2)
    assert len(report.relations) <= 2


def test_include_filter():
    rng = _rng(0)
    a = rng.integers(1, 40, N)
    b = rng.integers(1, 40, N)
    X = pd.DataFrame({"price": a, "quantity": b, "noise": rng.normal(size=N)})
    y = np.gcd(a, b)
    report = discover_structure(X, y, include=("lattice",))
    assert all(r.kind in ("gcd", "lcm", "bitwise_and") for r in report.relations)
    assert any(r.kind == "gcd" for r in report.relations)


def test_mrmr_discovered_structure_accessor():
    """The fitted-MRMR ``discovered_structure_`` accessor reads the frozen FE recipes into a StructureReport."""
    from mlframe.feature_selection.structure_discovery import structure_report_from_recipes
    from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe

    recipes = [
        EngineeredRecipe(name="il_gcd__p__q", kind="pairwise_integer_lattice", src_names=("p", "q"), extra={"op": "gcd"}),
        EngineeredRecipe(name="pmod_sum__a__b__m7", kind="pairwise_modular", src_names=("a", "b"), extra={"op": "sum", "modulus": 7}),
        EngineeredRecipe(name="gate_select__d__base__ten__t0.4", kind="conditional_gate", src_names=("d", "base", "ten"),
                         extra={"mode": "select", "tau": 0.4}),
        EngineeredRecipe(name="argmax__x0__x1__x2", kind="row_argmax", src_names=("x0", "x1", "x2")),
        EngineeredRecipe(name="orth_he2__c", kind="orth_univariate", src_names=("c",), extra={"basis": "hermite", "degree": 2}),
    ]
    report = structure_report_from_recipes(recipes, n_columns=10)
    kinds = {r.kind for r in report.relations}
    assert kinds == {"gcd", "modular", "gate_select", "argmax"}  # orth_univariate is not a structural FE kind -> excluded
    gcd = next(r for r in report.relations if r.kind == "gcd")
    assert gcd.columns == ("p", "q")
    mod = next(r for r in report.relations if r.kind == "modular")
    assert mod.parameter == 7.0


def test_biz_val_significance_pvalue_strong_on_gcd():
    """A genuine gcd structure must get a strong (small) permutation p-value, reported in the description."""
    rng = _rng(0)
    a = rng.integers(1, 40, N)
    b = rng.integers(1, 40, N)
    X = pd.DataFrame({"price": a, "quantity": b, "noise": rng.normal(size=N)})
    y = (np.gcd(a, b) >= 4).astype(int)
    report = discover_structure(X, y, significance_n_perm=200)
    assert report.relations, "gcd target must produce a discovered relation"
    top = report.relations[0]
    assert np.isfinite(top.p_value) and top.p_value < 0.05, f"gcd structure must be significant; p={top.p_value}"
    assert "p<" in top.description or "p=" in top.description, f"description must carry the p-value: {top.description}"


def test_biz_val_significance_off_omits_pvalue():
    """significance_n_perm=0 disables the test: p_value is nan and the p is omitted from the description."""
    rng = _rng(0)
    a = rng.integers(1, 40, N)
    b = rng.integers(1, 40, N)
    X = pd.DataFrame({"price": a, "quantity": b, "noise": rng.normal(size=N)})
    y = (np.gcd(a, b) >= 4).astype(int)
    report = discover_structure(X, y, significance_n_perm=0)
    assert report.relations
    top = report.relations[0]
    assert np.isnan(top.p_value), "p_value must be nan when significance testing is off"
    assert "p<" not in top.description and "p=" not in top.description, f"no p-value when off: {top.description}"


def test_biz_val_significance_does_not_create_false_discovery_on_noise():
    """Significance testing must not manufacture discoveries: a pure-noise frame stays empty even with the deeper null on."""
    rng = _rng(3)
    X = pd.DataFrame({f"f{i}": rng.normal(size=N) for i in range(5)})
    y = (rng.normal(size=N) > 0).astype(int)
    report = discover_structure(X, y, significance_n_perm=200)
    assert len(report) == 0, f"noise frame must yield 0 discovered relations; got {[r.description for r in report]}"
