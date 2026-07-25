"""Contract: ``fe_max_steps=0`` disables EVERY feature-engineering family, unconditionally.

A family's own ``fe_*_enable`` flag can only enable it WITHIN the FE budget -- it can never buy its way past
``fe_max_steps=0``. Before this was enforced centrally, only the hybrid-orth / univariate-basis pair honoured
the budget; ~30 other default-ON families fired regardless, so "no feature engineering" actually meant "no FE
except the ones you did not know were on", and fits that had explicitly asked for none still got engineered
columns. The discrete-structural operators additionally carried a deliberate carve-out that is now gone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR

# A broad spread of families across the different gate styles (orth basis, discrete-structural operators,
# mi-greedy, wavelet, hinge, higher-arity crosses) -- all switched ON explicitly by the caller.
_ALL_FAMILIES_ON = dict(
    fe_hybrid_orth_enable=True,
    fe_hybrid_orth_triplet_enable=True,
    fe_hybrid_orth_quadruplet_enable=True,
    fe_pairwise_modular_enable=True,
    fe_integer_lattice_enable=True,
    fe_row_argmax_enable=True,
    fe_conditional_gate_enable=True,
    fe_modular_enable=True,
    fe_hinge_enable=True,
    fe_mi_greedy_enable=True,
    fe_wavelet_enable=True,
)

_ROSTERS = (
    "hybrid_orth_features_",
    "hybrid_orth_candidates_",
    "mi_greedy_features_",
    "modular_features_",
    "wavelet_features_",
    "temporal_agg_features_",
)


def _fixture(seed: int = 0, n: int = 2000):
    """Frame carrying quadratic, ratio and integer structure, so every family has something to find."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    rev = np.exp(rng.normal(0.0, 1.0, size=n))
    cost = np.exp(rng.normal(0.0, 1.0, size=n))
    X = pd.DataFrame(
        {
            "a": a,
            "b": rng.standard_normal(n),
            "ints": rng.integers(0, 50, size=n),
            "rev": rev,
            "cost": cost,
            "noise": rng.standard_normal(n),
        }
    )
    y = pd.Series(((((a * a - 1.0) > 0) ^ ((rev / cost) > 1.0))).astype(int), name="y")
    return X, y


def _engineered(support) -> list[str]:
    """Support entries that are engineered rather than raw input columns."""
    return [str(c) for c in support if any(tok in str(c) for tok in ("(", "__", "*"))]


def test_fe_max_steps_zero_engineers_nothing_even_with_every_family_enabled():
    """With every family explicitly ON, fe_max_steps=0 must still produce a raw-only fit."""
    X, y = _fixture()
    m = MRMR(fe_max_steps=0, verbose=0, random_seed=0, n_workers=1, quantization_nbins=10, **_ALL_FAMILIES_ON)
    m.fit(X, y)

    support = list(m.get_feature_names_out())
    engineered = _engineered(support)
    assert not engineered, f"fe_max_steps=0 must engineer nothing; got {engineered} in support={support}"
    assert set(support) <= set(X.columns), f"support must be raw-only; got {support}"

    populated = {r: list(getattr(m, r, []) or []) for r in _ROSTERS if getattr(m, r, None)}
    assert not populated, f"fe_max_steps=0 must leave every FE roster empty; got {populated}"


def test_fe_budget_of_one_lets_the_same_families_fire():
    """Control: the identical config with fe_max_steps=1 DOES engineer -- so the test above pins the budget
    rule, not a frame on which no family could have found anything."""
    X, y = _fixture()
    m = MRMR(fe_max_steps=1, verbose=0, random_seed=0, n_workers=1, quantization_nbins=10, **_ALL_FAMILIES_ON)
    m.fit(X, y)

    produced = list(getattr(m, "hybrid_orth_candidates_", []) or [])
    engineered = _engineered(list(m.get_feature_names_out()))
    assert produced or engineered, "with a budget of 1 the same families must produce candidates"


@pytest.mark.parametrize("flag", sorted(_ALL_FAMILIES_ON))
def test_single_family_cannot_opt_past_a_zero_budget(flag):
    """Each family individually: its own flag must not override fe_max_steps=0."""
    X, y = _fixture(n=800)
    m = MRMR(fe_max_steps=0, verbose=0, random_seed=0, n_workers=1, quantization_nbins=10, **{flag: True})
    m.fit(X, y)
    engineered = _engineered(list(m.get_feature_names_out()))
    assert not engineered, f"{flag}=True must not engineer anything at fe_max_steps=0; got {engineered}"
