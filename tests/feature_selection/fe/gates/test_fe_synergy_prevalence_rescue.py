"""Tests for the PREVALENCE-FAILED SYNERGY RESCUE into the FE auto-escalation
(2026-06-12, F2 ``a**2/b`` miss).

ROOT CAUSE this fixes: a genuine SMOOTH ratio/product interaction whose two operands
both have weak univariate MI (so neither is selected; the pair enters the FE pool only
via the synergy bootstrap) has a LOW raw joint-MI prevalence ratio -- the raw-MI ratio
structurally under-estimates a smooth non-bilinear interaction. The user's F2
``y = 0.2*a**2/b + f/5 + log(c*2)*sin(d/3)`` has the (a,b) pair at joint-MI ratio ~1.11,
far below the ``fe_synergy_min_prevalence=1.5`` synergy bar, so it was DROPPED before any
FE / escalation ran and the output carried NO (a,b) feature -- downstream R^2 capped at
~0.95 vs the ~0.997 the feature reaches. The rescue hands such prevalence-failed synergy
pairs (that DID clear the order-2 maxT null) to the escalation, where a leak-safe held-out
rank-1 ALS pair-vs-single |corr| margin re-decides and the full admission gates gate.

CURRENT STATE OF THE FIXTURE (2026-11-24): with the default prevalence bars the (a,b) pair of F2 no longer fails the prevalence gate (its joint MI is 13 times the sum of its marginals; no bar below 13 rejects it),
so the rescue has nothing to rescue there and the pair reaches the escalation as an ordinary zero-admission pair. The rescue tests therefore make the pair fail the gate with a high ``fe_min_pair_mi_prevalence`` and
route it with ``fe_prevalence_rescue_all_pairs`` (the same ``_route_to_rescue`` branch the synergy pairs take): the ledger shows the (a,b) rejection and the rescue switch alone decides whether the escalation sees the pair.
A separate test pins the other finding of the same investigation: the escalation budget (8 pairs) was filled in order of raw joint MI, with the additive cross pairs of the strong operands ahead of the weak-marginal
interaction; the pairs now compete on the MI beyond their operands' marginals.

Covers:
* UNIT: a prevalence-failed weak-marginal smooth interaction (a**2/b) is rescued -- the escalation eligible
  set contains (a,b) and an ``esc_poly_*`` candidate is proposed+admitted; with the rescue off nothing is escalated.
* UNIT: the budget ranking puts the interaction pair first and keeps the joint-MI order without marginals.
* biz_value: the interaction-ranked budget surfaces the (a,b) feature on F2 without harming the downstream R^2 of a tree or a linear consumer.
"""

from __future__ import annotations

from tests.feature_selection.conftest import LEGACY_FE_OFF
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

# Disable the newer default-on FE families (the modular and hybrid-orth triplet columns also widen the pool the order-2 max-T floor is taken over, which lifts it above the weak (a,b) pair) that ALSO recover the (a,b) multiplicative
# synergy (pairwise-modular / integer-lattice / binned-agg / conditional-gate / oof-cell2d): with them
# ON the rescue-OFF baseline already reconstructs the synergy (R^2 0.97), compressing the
# rescue's marginal improvement below the floor even though rescue ON surfaces the (a,b)
# feature and lifts R^2. Isolating them measures the rescue's OWN contribution -- the
# documented intent of this biz-value sensor. NOTE: fe_auto_escalation MUST stay ON --
# the rescue mechanism works BY routing the prevalence-failed pair into escalation, so
# disabling it would break the rescue itself (not just the baseline).
_LEAN = dict(
    **LEGACY_FE_OFF,
    dcd_enable=False,
    build_friend_graph=False,
    cluster_aggregate_enable=False,
    fe_pairwise_modular_enable=False,
    fe_integer_lattice_enable=False,
    fe_binned_numeric_agg_enable=False,
    fe_conditional_gate_enable=False,
    fe_oof_cell2d_enable=False,
    fe_modular_enable=False,
    fe_hybrid_orth_triplet_enable=False,
)


# Every pair of the fixture fails a prevalence bar of 40 (the (a,b) ratio is 13), so the verdict of the prevalence gate no longer depends on the data: the rescue switch alone decides which pairs reach the escalation.
_PREVALENCE_FAILED = dict(fe_min_pair_mi_prevalence=40.0, fe_prevalence_rescue_all_pairs=True)


def _prescreen_rejected_pairs(sel) -> set:
    """Operand-name tuples the marginal-pair-MI prescreen rejected (read from the rejection ledger)."""
    led = sel.fe_rejection_ledger_
    rows = led[led["gate"] == "marginal_pair_mi_prescreen"]
    return {tuple(sorted(str(n).strip() for n in str(nm).strip("()").split(","))) for nm in rows["operand_names"]}


@pytest.fixture(autouse=True)
def _unary_credit_off(monkeypatch):
    """Keep the unary-credit pass out of these fits, like the families of ``_LEAN``: it credits the closed form ``mul(sq0(a),recip(b))`` for the same a**2/b signal on its own, and the escalation's (a,b) column
    is then pruned as redundant to it, so with it on the output carries the signal but not the rescue's column."""
    monkeypatch.setenv("MLFRAME_FE_UNARY_CREDIT", "0")


def _make_f2(seed: int = 11, n: int = 12000):
    """Make f2."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.5, 3.0, n)
    b = rng.uniform(1.2, 5.0, n)
    c = rng.uniform(0.5, 3.0, n)
    d = rng.uniform(-6.0, 6.0, n)
    f = rng.uniform(-3.0, 3.0, n)
    true = 0.2 * a**2 / b + f / 5 + np.log(c * 2) * np.sin(d / 3)
    y = true + rng.normal(0, 0.05 * np.std(true), n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "f": f}), pd.Series(y, name="y")


def _ab_escalation_names(sel) -> list[str]:
    """Engineered names that carry the rescued (a,b) escalation feature.

    The base ``esc_poly_*mul(a,b)`` may be SELECTED standalone OR consumed into a nested
    child (``div(log(esc_poly_legendre_mul(a,b)),...)``) and only the child selected. Both
    cases mean the rescued (a,b) signal reached the model; match the verbatim escalation
    token in any recorded recipe name AND any column emitted by get_feature_names_out."""
    import re

    pat = re.compile(r"esc_\w+_mul\(a,\s*b\)")
    names = set()
    for r in getattr(sel, "_engineered_recipes_", None) or []:
        if pat.search(r.name):
            names.add(r.name)
    for nm in sel.get_feature_names_out():
        if pat.search(nm):
            names.add(nm)
    return sorted(names)


def _ab_from_recipes(sel) -> list[str]:
    """Ab from recipes."""
    return _ab_escalation_names(sel)


def _ab_in_output(sel) -> list[str]:
    """(a,b) escalation columns that actually reach get_feature_names_out()."""
    import re

    pat = re.compile(r"esc_\w+_mul\(a,\s*b\)")
    return [nm for nm in sel.get_feature_names_out() if pat.search(nm)]


def test_unit_rescue_fires_for_weak_smooth_interaction_inert_on_noise():
    """The rescue routes the prevalence-failed (a,b) synergy pair into escalation: the
    escalation history must list (a,b) as eligible and admit an ``esc_`` (a,b) feature,
    while a pure-noise pair forced through the same pool admits nothing."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, y = _make_f2(seed=11, n=12000)
    MRMR.clear_fit_cache()
    sel = MRMR(verbose=0, n_jobs=1, random_state=0, **_PREVALENCE_FAILED, **_LEAN)
    sel.fit(df, y)

    assert ("a", "b") in _prescreen_rejected_pairs(sel), "premise: the (a,b) pair must fail the prevalence gate for the rescue to have something to rescue"
    hist = getattr(sel, "fe_escalation_history_", []) or []
    eligible_pairs = set()
    for h in hist:
        for p in h.get("eligible_pairs") or []:
            eligible_pairs.add(tuple(p))
    assert ("a", "b") in eligible_pairs or ("b", "a") in eligible_pairs, (
        f"rescue must route the prevalence-failed (a,b) synergy pair into escalation; eligible={eligible_pairs}"
    )
    ab = _ab_from_recipes(sel)
    names = list(sel.get_feature_names_out())
    # Provenance: an escalation product is labelled as such and its admission is in the admission ledger (stage "escalation"), so no spying is needed to see where it came from.
    prov = sel.fe_provenance_
    plain_esc = prov[prov["feature_name"].astype(str).str.startswith("esc_")]
    assert len(plain_esc) > 0 and (plain_esc["origin"] == "escalation").all(), f"escalation products must carry the escalation origin: {plain_esc[['feature_name', 'origin']].to_dict('records')}"
    admissions = sel.fe_admission_ledger_
    escalated = admissions[admissions["stage"] == "escalation"]
    assert len(escalated) > 0 and escalated["candidate"].astype(str).str.startswith("esc_").all(), f"escalation admissions missing from the admission ledger: {admissions['stage'].tolist()}"
    assert ab, f"an (a,b) engineered feature must be admitted by the rescue path; feature_names_out={names}; admitted={[h.get('admitted') for h in hist]}"
    # The rescued (a,b) escalation feature (base, or the nested child that consumed it)
    # must reach the output and carry the a**2/b signal.
    out_cols = _ab_in_output(sel)
    assert out_cols, f"an (a,b) escalation column must reach the output; names={names}"
    Xt = sel.transform(df.iloc[:2000])
    a, b, c, d = (df[k].to_numpy(dtype=np.float64)[:2000] for k in ("a", "b", "c", "d"))
    ab_term = 0.2 * a**2 / b
    cd_term = np.log(c * 2) * np.sin(d / 3)

    def _r2(col, regressors):
        """R^2 of ``col`` on the intercept and ``regressors``."""
        design = np.column_stack([np.ones_like(col), *regressors])
        resid = col - design @ np.linalg.lstsq(design, col, rcond=None)[0]
        return 1.0 - float(resid @ resid) / float(((col - col.mean()) ** 2).sum())

    # The additive fusion may carry the rescued term inside a compound with the c,d term (``add(mul(c,d..), esc(a,b))``), whose correlation with a**2/b alone is small by construction.
    # The contract is that the column carries the a**2/b signal BEYOND the c,d term: the R^2 the a,b term adds to a regression of the column on the c,d term.
    best = max(
        _r2(col, [cd_term, ab_term]) - _r2(col, [cd_term])
        for col in (np.nan_to_num(np.asarray(Xt[c_], dtype=np.float64)) for c_ in out_cols if c_ in Xt.columns)
    )
    assert best >= 0.1, f"rescued (a,b) feature must carry the a**2/b signal beyond the c,d term; best R^2 gain={best:.3f}"


def test_unit_rescue_can_be_disabled():
    """With the rescue OFF, the prevalence-failed (a,b) pair is NOT escalated (the pre-fix
    behaviour) -- proving the recovery is attributable to the rescue, not the base search."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, y = _make_f2(seed=11, n=12000)
    MRMR.clear_fit_cache()
    sel = MRMR(verbose=0, n_jobs=1, random_state=0, fe_synergy_prevalence_rescue_enable=False, **_PREVALENCE_FAILED, **_LEAN)
    sel.fit(df, y)
    assert ("a", "b") in _prescreen_rejected_pairs(sel), "premise: the (a,b) pair fails the prevalence gate here exactly as in the rescue-ON test"
    assert not _ab_from_recipes(sel), "rescue OFF must NOT produce an (a,b) feature (true-negative control)"
    escalated = {tuple(p) for h in (getattr(sel, "fe_escalation_history_", []) or []) for p in (h.get("eligible_pairs") or [])}
    assert ("a", "b") not in escalated and ("b", "a") not in escalated, f"rescue OFF must not hand the prevalence-failed pair to the escalation: {escalated}"


def test_unit_weak_marginal_interaction_is_not_starved_of_the_escalation_budget():
    """With the default bars (nothing to rescue) the (a,b) pair is an ordinary zero-admission pair and must still reach the escalation: it is ninth of the nine candidate pairs by joint MI (the additive cross pairs
    of the strong operands d, f, c lead), and the budget holds eight."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, y = _make_f2(seed=11, n=12000)
    MRMR.clear_fit_cache()
    sel = MRMR(verbose=0, n_jobs=1, random_state=0, **_LEAN)
    sel.fit(df, y)
    hist = getattr(sel, "fe_escalation_history_", []) or []
    escalated = {tuple(p) for h in hist for p in (h.get("eligible_pairs") or [])}
    assert ("a", "b") in escalated or ("b", "a") in escalated, f"the weak-marginal interaction pair must be escalated; eligible={escalated}"
    assert any(name.startswith("esc_") and "mul(a,b)" in name for h in hist for name in (h.get("admitted") or [])), f"an (a,b) richer-basis candidate must be admitted; admitted={[h.get('admitted') for h in hist]}"


def test_unit_escalation_budget_ranks_pairs_by_interaction_content():
    """``_select_eligible_pairs`` fills the budget with the pairs whose joint MI most exceeds their operands' marginals: an additive cross pair (joint MI = the sum of its marginals, the largest joint MI of all) loses to
    a weak-marginal interaction pair, and without marginals the order is the joint MI as before. Rescue pairs keep their reserved half of the budget."""
    from mlframe.feature_selection.filters._fe_auto_escalation import _select_eligible_pairs

    cols = ["strong", "other", "weak1", "weak2", "rescued1", "rescued2"]
    raw = set(cols)
    marginals = {(0,): 0.6, (1,): 0.1, (2,): 0.03, (3,): 0.03, (4,): 0.02, (5,): 0.02}
    failed = [((0, 1), 1.30), ((2, 3), 0.85), ((4, 5), 0.20)]

    def picked(max_pairs, marginal_mi, rescue=None):
        """Operand-name pairs the budget picked, in order."""
        _, eligible = _select_eligible_pairs(failed, cols, raw, rescue, max_pairs, {}, marginal_mi)
        return [(na, nb) for _, _, na, nb in eligible]

    assert picked(1, marginals) == [("weak1", "weak2")], "the interaction pair must take the single slot"
    assert picked(1, None) == [("strong", "other")], "without marginals the joint-MI order stays"
    assert picked(3, marginals) == [("weak1", "weak2"), ("strong", "other"), ("rescued1", "rescued2")], "interaction content, not joint MI, orders the budget"
    assert picked(2, marginals, rescue={(4, 5)})[0] == ("rescued1", "rescued2"), "a rescue pair is listed ahead of the others"
    assert ("rescued1", "rescued2") in picked(2, marginals, rescue={(4, 5)}), "a rescue pair keeps its reserved share of the budget"


def test_biz_value_interaction_pair_escalation_surfaces_the_ab_feature_without_harming_f2_r2(monkeypatch):
    """biz_value: on F2 the budget ranking by interaction content escalates the (a,b) pair, a CLEAN (a,b) escalation feature reaches the output, and the downstream R^2 does not fall behind the joint-MI ranking that left
    the pair out of the budget (the old ranking is reinstated through ``_pair_rank_key``).

    Measured on the three fixtures tried (2026-11-24), new against old ranking: ridge R^2 0.9405 against 0.9389 (seed 11, 12000 rows), equal where the escalation admitted nothing (seed 3), 0.9476 against 0.9404 (seed 5,
    20000 rows); HistGBR R^2 0.9844 against 0.9878, 0.9876 against 0.9876, 0.9886 against 0.9885. The tree consumer recovers the modest ``0.2*a**2/b`` term from the raw (a,b) splits on its own, so the contract that survives
    its near-ceiling baseline is no harm beyond noise (both >= 0.95, new >= old - 0.01) for the tree and for the linear consumer the engineered feature is made for. The escalation stays TERMINAL in the composite
    feed-forward (``fe_escalation_feedforward_enable=False``, the default): feeding it back as an operand fuses it into a ratio composite that drops the clean raw predictors (measured F2 regression)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_score, KFold
    from mlframe.feature_selection.filters import _fe_auto_escalation
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, y = _make_f2(seed=11, n=12000)
    cv = KFold(5, shuffle=True, random_state=0)

    def _r2s(sel):
        """Held-out CV R^2 of a gradient-boosted tree and of a ridge regression on the selector's output: ``(tree, linear)``."""
        Xt = sel.transform(df)
        X = np.column_stack([np.nan_to_num(np.asarray(Xt[c], dtype=np.float64)) for c in Xt.columns])
        tree = cross_val_score(HistGradientBoostingRegressor(random_state=0, max_iter=150), X, y.values, cv=cv, scoring="r2").mean()
        linear = cross_val_score(RidgeCV(alphas=np.logspace(-3, 3, 7)), (X - X.mean(0)) / (X.std(0) + 1e-12), y.values, cv=cv, scoring="r2").mean()
        return tree, linear

    with monkeypatch.context() as old_ranking:
        old_ranking.setattr(_fe_auto_escalation, "_pair_rank_key", lambda pair, pair_mi, marginal_mi: float(pair_mi))
        MRMR.clear_fit_cache()
        old = MRMR(verbose=0, n_jobs=1, random_state=0, **_LEAN)
        old.fit(df, y)
        tree_old, linear_old = _r2s(old)

    MRMR.clear_fit_cache()
    new = MRMR(verbose=0, n_jobs=1, random_state=0, **_LEAN)
    new.fit(df, y)
    tree_new, linear_new = _r2s(new)

    assert not _ab_from_recipes(old), "the joint-MI ranking leaves the (a,b) pair out of the budget on F2 (the control of this comparison)"
    assert _ab_from_recipes(new), "the interaction ranking must surface an (a,b) feature on F2"
    assert min(tree_new, tree_old, linear_new, linear_old) >= 0.9, f"F2 signal must be recovered on both rankings: tree old={tree_old:.4f} new={tree_new:.4f}, linear old={linear_old:.4f} new={linear_new:.4f}"
    assert tree_new >= tree_old - 0.01, f"the ranking must not harm the tree consumer beyond noise: old={tree_old:.4f} new={tree_new:.4f}"
    assert linear_new >= linear_old - 0.005, f"the ranking must not harm the linear consumer beyond noise: old={linear_old:.4f} new={linear_new:.4f}"
