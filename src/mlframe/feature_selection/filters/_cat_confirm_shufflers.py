"""IPF and within-strata conditional shufflers for the permutation confirmation in ``_cat_confirm_permutation`` (re-imported there by name)."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _full_conditional_shuffle_ipf(
    classes_x2_safe: np.ndarray,
    classes_x1: np.ndarray,
    classes_y: np.ndarray,
    n_x1_classes: int,
    n_y_classes: int,
    base_seed: int,
) -> None:
    """Full conditional permutation via IPF-style double-stratification. Shuffle X2 within strata of (X1, Y) so that P(X2 | X1, Y) is preserved on average across
    permutations - the strictest synergy null that holds both marginals fixed while breaking X1-X2 conditional dependence.

    Reference: Patefield 1981 (R x C contingency tables with fixed marginals); Anderson & ter Braak 2003 (multi-factorial permutation).

    Implementation: for each unique (X1, Y) stratum, Fisher-Yates shuffle classes_x2_safe restricted to rows in that stratum.
    The Fisher-Yates RNG is a seeded inline LCG (same PCG-style step as ``_count_nfailed_joint_indep_prange``) so the permutation is reproducible at a fixed
    ``base_seed`` and never touches numpy's process-global RNG.
    """
    n = len(classes_y)
    state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(1)
    for cx1 in range(n_x1_classes):
        for cy in range(n_y_classes):
            # Gather indices in this (X1, Y) stratum
            positions = np.empty(n, dtype=np.int64)
            pos_count = 0
            for i in range(n):
                if classes_x1[i] == cx1 and classes_y[i] == cy:
                    positions[pos_count] = i
                    pos_count += 1
            # Fisher-Yates within stratum
            for idx in range(pos_count - 1, 0, -1):
                state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                j = int(state >> np.uint64(33)) % (idx + 1)
                a = positions[idx]
                b = positions[j]
                tmp = classes_x2_safe[a]
                classes_x2_safe[a] = classes_x2_safe[b]
                classes_x2_safe[b] = tmp


@njit(cache=True)
def _conditional_shuffle_within_strata(
    classes_x2_safe: np.ndarray,
    classes_y: np.ndarray,
    n_y_classes: int,
    base_seed: int,
) -> None:
    """Conditional permutation: shuffle ``classes_x2_safe`` IN PLACE, restricting the shuffle to within each stratum of ``classes_y``.

    This is the correct null distribution for testing ``H0: X1 ⊥ X2 | Y`` (no synergy beyond marginals). Plain shuffle-Y tests ``H0: Y ⊥ (X1, X2)`` (joint independence) instead.

    Implementation: for each Y-stratum, collect the indices where ``classes_y[i] == c``, Fisher-Yates shuffle the corresponding slice of ``classes_x2_safe`` in place.
    Per Anderson & ter Braak 2003 ("Permutation tests for multi-factorial analysis of variance").

    Preserves ``P(X2 | Y)`` - so each marginal ``I(X2; Y)`` is
    unchanged under the shuffle, but the conditional ``I(X1; X2 | Y)``
    is broken. The orchestrator combines this with three-MI calls
    above to compute II_perm under the conditional null.

    The Fisher-Yates RNG is a seeded inline LCG (same PCG-style step as ``_count_nfailed_joint_indep_prange``) so the permutation is reproducible at a fixed
    ``base_seed`` and never touches numpy's process-global RNG.
    """
    n = len(classes_y)
    state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(1)
    # For each Y class, gather positions and Fisher-Yates within the slice.
    for c in range(n_y_classes):
        # Collect indices manually (numba doesn't support boolean masks
        # for in-place writes the same way numpy does).
        positions = np.empty(n, dtype=np.int64)
        pos_count = 0
        for i in range(n):
            if classes_y[i] == c:
                positions[pos_count] = i
                pos_count += 1
        # Fisher-Yates shuffle within the stratum
        for idx in range(pos_count - 1, 0, -1):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            j = int(state >> np.uint64(33)) % (idx + 1)
            a = positions[idx]
            b = positions[j]
            tmp = classes_x2_safe[a]
            classes_x2_safe[a] = classes_x2_safe[b]
            classes_x2_safe[b] = tmp
