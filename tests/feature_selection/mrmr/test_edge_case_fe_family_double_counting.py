"""Edge case: FE family double-counting between fe_pairwise_modular and fe_conditional_gate.

Both families are reachable when ``fe_discrete_structural_operators_enable=True``. On a target that is the OR of a modular-arithmetic
event and a threshold-gated interaction over the SAME raw columns, both families' detectors fire and each family proposes several
variants. The two families capture DIFFERENT components of y: on this fixture CMI(gate; y | modular) ~0.28 exceeds MI(gate; y) ~0.16,
so selecting one column of each is the correct, non-redundant answer. The double-counting risk is WITHIN a family: the gate variants
that swap one operand (``gate_select__a__b__c`` vs ``gate_select__extra__b__c``) are near-duplicates (CMI ~0.007 given the other), and
MRMR's redundancy gate must not select more than one of them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters.mrmr import MRMR


def _mixed_modular_and_gate_fixture(seed: int = 1, n: int = 6000, m: int = 7):
    """y is the OR of a pure modular-arithmetic event ((a+b) mod m >= m//2, pairwise_modular's own
    detection target) and a threshold-gated interaction (y = a if c>tau else b, conditional_gate's
    own detection target) -- both over the SAME raw columns a/b, so both families' own detectors
    have genuine, partially-overlapping signal to catch."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 100, n)
    b = rng.integers(0, 100, n)
    mod_sig = ((a + b) % m >= (m // 2)).astype(int)
    c = rng.uniform(0.0, 1.0, n)
    tau = 0.5
    gate_val = np.where(c > tau, a, b)
    gate_sig = (gate_val > np.median(gate_val)).astype(int)
    y = ((mod_sig + gate_sig) >= 1).astype(int)
    X = pd.DataFrame(
        {
            "a": a.astype(float),
            "b": b.astype(float),
            "c": c,
            "extra": (a % 2).astype(float),
            "n0": rng.integers(0, 50, n).astype(float),
            "n1": rng.integers(0, 50, n).astype(float),
        }
    )
    return X, y


def _kw(**overrides):
    """Fast-fitting default MRMR constructor kwargs, overridable per test."""
    base = dict(
        random_state=42,
        verbose=0,
        n_jobs=1,
        full_npermutations=2,
        baseline_npermutations=2,
        fe_max_steps=1,
        skip_retraining_on_same_content=False,
        fe_discrete_structural_operators_enable=True,
        fe_pairwise_modular_enable=True,
        fe_conditional_gate_enable=True,
    )
    base.update(overrides)
    return base


def _proposed_by_family(m, origin: str) -> list:
    """Engineered columns a family proposed this fit, from ``fe_provenance_`` (the per-family rosters keep only the output columns)."""
    prov = m.fe_provenance_
    return prov.loc[prov["origin"] == origin, "feature_name"].astype(str).tolist()


def test_both_families_propose_candidates_on_the_mixed_fixture():
    """Sanity gate: on this fixture BOTH families' own MI-based detectors must actually fire (else
    the redundancy-gate assertion below would be vacuous -- proving nothing competed)."""
    X, y = _mixed_modular_and_gate_fixture()
    m = MRMR(**_kw())
    m.fit(X, y)

    pairwise_modular = [c for c in _proposed_by_family(m, "periodic") if c.startswith("pmod_")]
    conditional_gate = _proposed_by_family(m, "conditional_gate")
    assert pairwise_modular, "pairwise_modular must propose at least one candidate on its own modular-arithmetic signal component"
    assert conditional_gate, "conditional_gate must propose at least one candidate on its own threshold-gated signal component"


def test_redundancy_gate_does_not_select_near_duplicate_variants_within_a_family():
    """PRIMARY GATE: each family proposes several near-duplicate variants of its one signal component; MRMR's redundancy gate must
    keep at most one variant per family in the final support rather than double-count it. One column from EACH family is allowed:
    the families carry complementary components of y (see the module docstring)."""
    X, y = _mixed_modular_and_gate_fixture()
    m = MRMR(**_kw())
    m.fit(X, y)

    final_names = set(str(s) for s in m.get_feature_names_out())
    selected_modular = final_names & {c for c in _proposed_by_family(m, "periodic") if c.startswith("pmod_")}
    selected_gate = final_names & set(_proposed_by_family(m, "conditional_gate"))

    assert len(selected_gate) <= 1, f"MRMR selected near-duplicate conditional_gate variants: {sorted(selected_gate)}; support={sorted(final_names)}"
    assert len(selected_modular) <= 1, f"MRMR selected near-duplicate pairwise_modular variants: {sorted(selected_modular)}; support={sorted(final_names)}"
    # the fit must still have selected SOMETHING useful (not an empty/degenerate support as a cheap
    # way to vacuously satisfy the assertions above).
    assert len(final_names) >= 1
