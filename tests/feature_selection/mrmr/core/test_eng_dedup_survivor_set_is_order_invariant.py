"""Which engineered columns survive dedup is a function of the columns, not of the order they were appended in.

Near-duplicate at a 0.99 rank correlation is not a transitive relation: A~B and B~C with A not~C is routine at that threshold. The scan
compares each candidate only against what is currently kept and drops or evicts immediately, so with an intransitive cluster the surviving
SET depended on emission order: A,B,C keeps {A, C} while B,A,C keeps {B} alone. That order is whatever the upstream families happened to
append, and it shifts whenever one of their ``top_k`` changes, so the same data could produce different survivors across fits.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._mrmr_fit_impl._eng_dedup_scan import scan_engineered_duplicates


@pytest.fixture
def intransitive():
    """Three columns where A~B and B~C are above the dedup threshold but A~C is not."""
    rng = np.random.default_rng(0)
    n = 600
    base = rng.normal(size=n)
    # B is the hub: A and C each sit close to it, in independent directions, so they are further from each other than either is from B.
    frame = pd.DataFrame({"A": base + 0.10 * rng.normal(size=n), "B": base, "C": base + 0.10 * rng.normal(size=n)})
    ranks = frame.rank(method="average")
    corr = ranks.corr(method="pearson").abs()
    if not (corr.loc["A", "B"] >= 0.99 and corr.loc["B", "C"] >= 0.99 and corr.loc["A", "C"] < 0.99):
        pytest.skip(f"fixture is not intransitive at 0.99: {corr.to_dict()}")
    return frame


def _keep_for(frame, order, mi):
    """Survivor set for one emission order, preferring the higher-MI column of a colliding pair."""
    keep, _drop, _arrs, _ranks = scan_engineered_duplicates(
        frame,
        list(order),
        set(),
        lambda a, b: mi.get(a, 0.0) > mi.get(b, 0.0),
    )
    return frozenset(keep)


def test_the_survivor_set_is_the_same_for_every_emission_order(intransitive):
    """All six permutations of an intransitive triple must return one survivor set."""
    mi = {"A": 0.9, "B": 0.5, "C": 0.3}
    sets = {order: _keep_for(intransitive, order, mi) for order in itertools.permutations(["A", "B", "C"])}
    distinct = set(sets.values())
    assert len(distinct) == 1, f"survivor set depends on emission order: {sets}"


def test_the_strongest_column_always_survives(intransitive):
    """Whatever the order, the highest-MI column of a colliding cluster is never the one evicted."""
    for winner in ("A", "B", "C"):
        mi = {nm: (1.0 if nm == winner else 0.2) for nm in ("A", "B", "C")}
        for order in itertools.permutations(["A", "B", "C"]):
            keep = _keep_for(intransitive, order, mi)
            assert winner in keep, f"{winner} (highest MI) was dropped for order {order}: {sorted(keep)}"


def test_an_unscored_cluster_keeps_the_previous_first_appended_behaviour(intransitive):
    """With no MI to prefer on, the comparator reports equal and the scan falls back to emission order, as before."""
    keep_abc = _keep_for(intransitive, ["A", "B", "C"], {})
    assert "A" in keep_abc, "the first-appended column should still win an unscored cluster"


def test_independent_columns_are_all_kept():
    """The ordering change must not make the scan drop anything it had no reason to drop."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({nm: rng.normal(size=400) for nm in ("p", "q", "r")})
    keep = _keep_for(frame, ["p", "q", "r"], {"p": 0.5, "q": 0.4, "r": 0.3})
    assert keep == frozenset({"p", "q", "r"})
