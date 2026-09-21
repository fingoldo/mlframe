"""Processing order for the FE step's prospective pairs."""

from __future__ import annotations


def order_prospective_pairs(prospective_pairs: dict) -> dict:
    """Return ``prospective_pairs`` ordered by reuse count, then pair MI, both descending.

    Keys are ``(raw_vars_pair, pair_mi)`` and values are the operands' reuse counter (a cache-locality score). Sorting on the counter alone
    kept the insertion order for ties, and pairs are inserted in ascending-MI order, so the weakest pair of every tie group went first and
    got the expensive operator search ahead of stronger pairs under any truncation. Equal (counter, MI) keep their insertion order.
    """
    return dict(sorted(prospective_pairs.items(), key=lambda kv: (kv[1], float(kv[0][1])), reverse=True))
