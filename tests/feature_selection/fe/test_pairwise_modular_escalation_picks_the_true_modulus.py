"""``escalate_modulus`` returns the true period, not a multiple of it.

A multiple of the true modulus (14 or 21 for 7, 72 for 12) splits the same classes into more residues: its population MI is the same and its plug-in MI is higher by the Miller-Madow bias of the larger
table, so taking the largest plug-in MI picked a multiple in most seeds (the true modulus was found in 2 of 12 seeds for 7, 0 of 12 for 11 and for 12), and a linear model on a residue by 14 or 21 loses most of
what the residue by 7 carries. The moduli are now compared on the MI net of that bias and a larger one replaces a smaller only when it gains several standard deviations of its extra cells.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._pairwise_modular_fe import cheap_modular_scan, escalate_modulus

SEEDS = range(6)


def _escalated(true_modulus: int, seed: int):
    """The modulus the escalation returns for ``y = ((a + b) mod m < m / 2)`` with 10 % flipped labels (``None`` when the cheap scan does not respond)."""
    r = np.random.default_rng(seed)
    n = 2100
    a, b = r.integers(0, 60, n), r.integers(0, 60, n)
    y = (((a + b) % true_modulus) < true_modulus // 2).astype(int)
    y = np.where(r.random(n) < 0.1, 1 - y, y)
    X = pd.DataFrame({"a": a.astype(float), "b": b.astype(float)})
    for i in range(4):
        X[f"n{i}"] = r.integers(0, 60, n).astype(float)
    hits = [h for h in cheap_modular_scan(X, y, None) if h.responded]
    return escalate_modulus(X, y, hits[0])[0] if hits else None


@pytest.mark.parametrize("true_modulus", [7, 11, 30])
def test_the_true_modulus_is_returned_not_a_multiple_of_it(true_modulus):
    """At least five of six seeds return exactly the modulus of the data (it was two of twelve for 7 and none for 11)."""
    got = [_escalated(true_modulus, s) for s in SEEDS]
    assert sum(g == true_modulus for g in got) >= 5, got
