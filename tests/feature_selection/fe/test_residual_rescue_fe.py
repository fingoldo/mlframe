"""Residual rescue: a term that is weak in rank space but large in variance is admitted on the residual of the selected design, noise is not."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._residual_rescue_fe import residual_rescue
from mlframe.feature_selection.filters.mrmr import MRMR


def _two_term(n=20000, seed=0, outlier_share=0.03):
    """y = a**2/b + 3*log(c)*sin(d) + 0.3*e: the ratio holds most of the variance through a few outlier rows, the (c, d) term dominates every rank statistic."""
    r = np.random.default_rng(seed)
    a, e, d = (r.uniform(-3, 3, n) for _ in range(3))
    b = r.uniform(0.25, 15, n)
    c = r.uniform(0.001, 15, n)
    out = r.random(n) < outlier_share
    a = np.where(out, a * 2.5, a)
    b = np.where(out & (r.random(n) < 0.3), 0.25, b)
    y = a**2 / b + 3 * np.log(c) * np.sin(d) + 0.3 * e
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}), y


def test_weak_in_rank_space_term_is_admitted_first():
    """The ratio, 83% of the variance but a rank-MI of 0.07, is the first column admitted on top of the (c, d) design."""
    X, y = _two_term()
    design = {"d": X["d"].to_numpy(), "cd": np.log(X["c"].to_numpy()) * np.sin(X["d"].to_numpy())}
    pool = {"a": X["a"].to_numpy(), "b": X["b"].to_numpy(), "a2/b": (X["a"] ** 2 / X["b"]).to_numpy()}
    got = residual_rescue(design, pool, y)
    assert got and got[0][0] == "a2/b"


def test_pure_noise_pool_is_family_wise_controlled():
    """Thirty pure-noise columns are admitted in at most 3 of 20 calls (family-wise 5% per call)."""
    X, y = _two_term(n=6000)
    design = {"d": X["d"].to_numpy(), "cd": np.log(X["c"].to_numpy()) * np.sin(X["d"].to_numpy()), "a2/b": (X["a"] ** 2 / X["b"]).to_numpy(), "e": X["e"].to_numpy()}
    hits = 0
    for seed in range(20):
        pool = {f"z{j}": np.random.default_rng(1000 * seed + j).standard_normal(len(y)) for j in range(30)}
        hits += bool(residual_rescue(design, pool, y))
    assert hits <= 3  # family-wise 5% per call: 20 calls, a binomial tail far below this bar


def test_copy_of_a_design_column_scores_nothing():
    """An affine copy and a monotone transform of a design column leave nothing in the residual."""
    X, y = _two_term()
    cd = np.log(X["c"].to_numpy()) * np.sin(X["d"].to_numpy())
    design = {"cd": cd, "a2/b": (X["a"] ** 2 / X["b"]).to_numpy(), "e": X["e"].to_numpy(), "d": X["d"].to_numpy()}
    assert residual_rescue(design, {"cd_copy": cd * 2.0 + 1.0, "cd_monotone": np.arctan(cd)}, y) == []


def test_practical_effect_floor_is_a_knob():
    """min_relative_gain and max_admit gate the admission."""
    X, y = _two_term()
    design = {"d": X["d"].to_numpy(), "cd": np.log(X["c"].to_numpy()) * np.sin(X["d"].to_numpy())}
    pool = {"a2/b": (X["a"] ** 2 / X["b"]).to_numpy()}
    assert residual_rescue(design, pool, y, min_relative_gain=0.0)
    assert residual_rescue(design, pool, y, min_relative_gain=0.99) == []
    assert residual_rescue(design, pool, y, max_admit=0) == []


def _operands(names):
    """Single-letter source columns named by the selected features (wavelet / hinge suffixes after ``__`` ignored)."""
    out = set()
    for n in names:
        word = ""
        for ch in str(n).split("__")[0] + " ":
            if ch.isalnum() or ch == "_":
                word += ch
            else:
                if len(word) == 1:
                    out.add(word)
                word = ""
    return out


@pytest.mark.timeout(900)
def test_estimator_recovers_the_outlier_masked_ratio_only_with_the_stage():
    """two_pairs_strong under 3% outliers: ``a**2/b`` holds 83% of the variance in a few rows, its rank MI is 0.07 against 1.85 for the (c, d) term, so the greedy drops both operands."""
    from tests.feature_selection import _synthetic_distributions as sd

    domains = {"a": sd.DOMAIN_ANY, "b": sd.DOMAIN_DIVISOR, "c": sd.DOMAIN_POSITIVE, "d": sd.DOMAIN_ANY, "e": sd.DOMAIN_ANY}
    data = sd.sample_operands(seed=42, n=20000, domains=domains, profile="with_outliers")
    y = data["a"] ** 2 / data["b"] + 3.0 * np.log(data["c"]) * np.sin(data["d"]) + 0.3 * data["e"]
    X = pd.DataFrame({k: data[k] for k in domains})
    on = MRMR(verbose=0, random_state=42).fit(X, pd.Series(y, name="y"))
    off = MRMR(verbose=0, random_state=42, fe_residual_rescue_enable=False).fit(X, pd.Series(y, name="y"))
    assert "a" in _operands(on.get_feature_names_out())
    assert "a" not in _operands(off.get_feature_names_out())


def test_constructor_default_and_legacy_state():
    """Constructor default is on; a pre-feature pickle reloads with it off."""
    assert MRMR().fe_residual_rescue_enable is True
    assert MRMR(fe_residual_rescue_enable=False).fe_residual_rescue_enable is False
    from mlframe.feature_selection.filters.mrmr._mrmr_setstate_defaults import _SETSTATE_LEGACY_DEFAULTS as D

    assert D["fe_residual_rescue_enable"] is False


def test_candidates_built_from_the_design_sources_are_not_eligible():
    """An operand view of one design column is a refinement and not eligible; a joint feature of operands held only in separate columns is."""
    from mlframe.feature_selection.filters._mrmr_fit_impl._residual_rescue_stage import _is_refinement, _raw_sources

    raw = frozenset({"a", "b", "c", "d", "e"})
    compound = _raw_sources("add(mul(log(c),sin(d)),div(sqr(a),abs(b)))", raw)
    assert compound == {"a", "b", "c", "d"}
    assert _raw_sources("a__haar_j1k0", raw) == {"a"}
    assert _raw_sources("binagg_mean(e|qbin(d))", raw) == {"d", "e"}
    assert _is_refinement(_raw_sources("oofwarp(a)", raw), [compound])
    assert not _is_refinement(_raw_sources("binagg_mean(e|qbin(d))", raw), [compound])
    separate = [frozenset({"a"}), frozenset({"b"})]
    assert not _is_refinement(_raw_sources("mul(log(a),sin(b))", raw), separate)
    assert _is_refinement(_raw_sources("sqr(a)", raw), separate)


class _Stub:
    """Minimal estimator stand-in carrying the constructor knobs the stage reads."""

    def __init__(self, policy="drop", cell=True):
        self.redundancy_policy = policy
        self.fe_oof_cell2d_enable = cell
        self.fe_residual_rescue_operand_min_linear_gain = 0.001


def test_prune_drops_the_noise_columns_and_keeps_the_real_ones_only_under_the_drop_policy():
    """Under ``redundancy_policy='drop'`` a raw or engineered column without held-out gain is pruned and a real term is kept; the default policy prunes nothing."""
    from mlframe.feature_selection.filters._mrmr_fit_impl._residual_rescue_stage import _unsupported_columns

    X, y = _two_term(n=20000, seed=3)
    raw = frozenset(X.columns)
    cd = np.log(X["c"].to_numpy()) * np.sin(X["d"].to_numpy())
    rng = np.random.default_rng(5)
    design = {"a2/b": (X["a"] ** 2 / X["b"]).to_numpy(), "cd": cd, "e": X["e"].to_numpy(), "noise_engineered": rng.standard_normal(len(y)) * cd.std()}
    design["a"] = X["a"].to_numpy()  # an operand of the ratio that the ratio already holds
    pruned = _unsupported_columns(_Stub(), design, y, raw, residual_rescue)
    assert "noise_engineered" in pruned and "e" not in pruned and "cd" not in pruned
    assert _unsupported_columns(_Stub(policy="emit_both"), design, y, raw, residual_rescue) == []


def test_cell_tables_of_the_residual_find_a_pair_interaction_the_vocabulary_may_miss():
    """A weak interaction of two columns under a dominant term is tabulated on the residual of the design; an additive target yields no table."""
    from mlframe.feature_selection.filters._mrmr_fit_impl._residual_rescue_stage import _cell_table_candidates

    r = np.random.default_rng(0)
    n = 20000
    X = pd.DataFrame({k: r.uniform(0, 1, n) for k in "abcde"})
    big = 6.0 * np.sin(6.0 * X["a"].to_numpy()) * X["b"].to_numpy()
    y = big + np.log(X["c"].to_numpy() + 1.0) * np.sin(2 * X["d"].to_numpy())
    design = {"big": big}
    pool: dict = {}
    got = _cell_table_candidates(_Stub(), X, y, design, frozenset(X.columns), [frozenset({"a", "b"})], pool)
    assert "oofcell(c,d)" in got and "oofcell(c,d)" in pool
    additive = X["a"].to_numpy() + X["c"].to_numpy() ** 2
    assert _cell_table_candidates(_Stub(), X, additive, {"a": X["a"].to_numpy()}, frozenset(X.columns), [frozenset({"a"})], {}) == {}
    assert _cell_table_candidates(_Stub(cell=False), X, y, design, frozenset(X.columns), [], {}) == {}


def test_design_residual_shrinks_with_the_true_design():
    """The residual of the right design is far smaller than the residual of an empty one, on the standardised scale."""
    from mlframe.feature_selection.filters._residual_rescue_fe import design_residual

    X, y = _two_term(n=6000, seed=4)
    full = {"a2/b": (X["a"] ** 2 / X["b"]).to_numpy(), "cd": np.log(X["c"].to_numpy()) * np.sin(X["d"].to_numpy())}
    assert np.var(design_residual(full, y)) < 0.2 * np.var(design_residual({}, y))


def test_linear_oos_gain_sees_a_private_linear_term_that_a_proxy_compound_hides():
    """A raw beside ``add(exp(a), ...)`` has a private linear term only a linear fit sees; a raw the compound holds exactly has none."""
    from mlframe.feature_selection.filters._residual_rescue_fe import linear_oos_gain

    r = np.random.default_rng(0)
    n = 20000
    a, d = r.uniform(0, 1, n), r.uniform(0, 1, n)
    compound = np.exp(a) + np.sin(3 * d)
    y_private = compound + 4.0 * a + 0.1 * r.standard_normal(n)
    y_exact = compound + 0.1 * r.standard_normal(n)
    t_priv, rel_priv = linear_oos_gain({"compound": compound}, a, y_private)
    _t_exact, rel_exact = linear_oos_gain({"compound": compound}, a, y_exact)
    assert t_priv > 10 and rel_priv > 0.01
    assert rel_exact < 0.001


def test_operand_pruning_follows_the_linear_gain_and_the_existing_knob():
    """A raw operand with a private linear term is kept, an operand the compound holds is pruned, and ``fe_drop_redundant_raw_operands=False`` keeps both."""
    from mlframe.feature_selection.filters._mrmr_fit_impl._residual_rescue_stage import _unsupported_columns

    r = np.random.default_rng(1)
    n = 20000
    a, d, b = r.uniform(0, 1, n), r.uniform(0, 1, n), r.uniform(0.2, 1, n)
    compound = np.exp(a) + np.sin(3 * d)
    ratio = a**2 / b
    y = compound + ratio + 4.0 * a
    design = {"compound": compound, "ratio(a,b)": ratio, "a": a, "b": b}
    raw = frozenset({"a", "b", "d"})
    stub = _Stub()
    stub.fe_drop_redundant_raw_operands = True
    # the names must spell the operands for the source scan: the stub design uses real FE-style names
    design = {"add(exp(a),sin(d))": compound, "div(sqr(a),b)": ratio, "a": a, "b": b}
    pruned = _unsupported_columns(stub, design, y, raw, residual_rescue)
    assert "a" not in pruned and "b" in pruned
    stub.fe_drop_redundant_raw_operands = False
    assert "b" not in _unsupported_columns(stub, design, y, raw, residual_rescue)


def test_nothing_is_pruned_when_there_are_too_few_rows_for_evidence():
    """With fewer than MIN_ROWS rows the held-out test cannot run: an empty verdict must not be read as 'no gain' (it once pruned every column of a 442-row diabetes fit)."""
    from mlframe.feature_selection.filters._mrmr_fit_impl._residual_rescue_stage import _unsupported_columns

    r = np.random.default_rng(2)
    n = 400
    X = {k: r.standard_normal(n) for k in "abc"}
    y = X["a"] + 0.5 * X["b"] + 0.1 * r.standard_normal(n)
    design = {"a": X["a"], "b": X["b"], "mul(a,b)": X["a"] * X["b"]}
    assert _unsupported_columns(_Stub(), design, y, frozenset("abc"), residual_rescue) == []


def test_classification_targets_are_skipped_by_the_stage():
    """The stage is for continuous targets: a 2-class label vector must leave the selection as it was."""
    from mlframe.feature_selection.filters._mrmr_fit_impl._residual_rescue_stage import run_residual_rescue

    class _Fit(_Stub):
        """Stub with the rescue on and a two-class label vector as the continuous target."""

        fe_residual_rescue_enable = True
        _fe_prewarp_y_continuous_ = (np.arange(3000) % 2).astype(np.float64)

    X = pd.DataFrame({"a": np.arange(3000, dtype=np.float64), "b": np.arange(3000, dtype=np.float64)[::-1]})
    assert run_residual_rescue(_Fit(), X, [0], ["a", "b"], {}, set(), 0) is False
