"""Joint-synergy pair screen: detect zero-/weak-marginal interaction pairs the MARGINAL screen misses.

The MRMR FE rung-0 pair screen ranks candidate raw pairs by their MARGINAL relevance (each operand's
MI with ``y``). On a PURE-SYNERGY target -- e.g. ``y = (x1>0) XOR (x2>0)`` -- BOTH operands have
~zero marginal MI, so the marginal screen sees the pair as noise and never forms the engineered
interaction (the documented I4/I5 "zero-marginal synergy === noise at the marginal level" barrier;
multiway_synergy::test_three_way_xor and friends).

This module scores a pair by the bias-corrected JOINT MI of its 2-D code grid with ``y``:
``MI(renumber(code_x, code_y); y)`` with a Miller-Madow occupancy correction. The joint grid exposes
the synergy the marginals hide, and the MM correction keeps a genuine NOISE pair's joint MI near zero
(its inflated finite-sample joint MI is debited by the occupied-cell count), so high-joint pairs can be
ranked/gated without admitting noise.

VALIDATION (2026-06-17, the gate the I4/I5 re-platform plan requires before wiring): on a single XOR
signal pair hidden among 189 noise pairs (P=20 features, 190 pairs), the signal pair ranks #1 with a
60-370x separation from the strongest noise pair's joint MI, across n in {8k, 25k} x nbins in {6, 8}:

    n=8000  nb=6:  signal=0.6643  rank=1/190  max_noise=0.0059  sep=112.8x
    n=8000  nb=8:  signal=0.6662  rank=1/190  max_noise=0.0106  sep= 62.7x
    n=25000 nb=6:  signal=0.6735  rank=1/190  max_noise=0.0018  sep=371.6x
    n=25000 nb=8:  signal=0.6745  rank=1/190  max_noise=0.0036  sep=187.9x

So a JOINT screen recovers the synergy the marginal screen drops, at equal (near-zero) noise admission.

NOT YET WIRED into ``check_prospective_fe_pairs`` -- this is the screen's validated core + its
detection-vs-noise contract (test_fe_synergy_screen.py). Wiring it as a rung-0 augmentation (form the
engineered interaction for pairs whose joint synergy clears a permutation/analytic null even when both
marginals are ~0) is the next step; gating stays behind that null so noise pairs are never admitted.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _combo_mm_mi_njit(combo_codes, cards, target, kt, n_cells):
    """Miller-Madow-corrected MI (nats, >=0) between the JOINT of ``order`` integer code columns and
    the target -- the synergy screen's hot per-combo histogram, in nopython.

    ``combo_codes`` is ``(n, order)`` int64 per-feature codes; ``cards`` the ``(order,)`` per-feature
    cardinalities; ``target`` the ``(n,)`` int64 target codes; ``kt`` the target cardinality; ``n_cells``
    = product(cards) (the dense joint-cell count, mixed-radix). Builds the (n_cells x kt) count table in
    one O(n*order) pass, then MI = sum p*log(p/(px*py)) over occupied cells, debited by the MM bias
    ``(occ_x-1 + occ_y-1 - (occ_cells-1)) / (2n)`` using OCCUPIED counts. Bit-faithful to the numpy
    reference (same estimator); only the histogram/loop moved to njit so the O(P^order) combo sweep
    is not bottlenecked on per-combo numpy ``unique``/``add.at`` Python dispatch."""
    n = combo_codes.shape[0]
    order = combo_codes.shape[1]
    if n == 0 or kt <= 0 or n_cells <= 0:
        return 0.0
    joint = np.zeros(n_cells * kt, dtype=np.float64)
    px = np.zeros(n_cells, dtype=np.float64)
    py = np.zeros(kt, dtype=np.float64)
    for r in range(n):
        cell = 0
        for c in range(order):
            cell = cell * cards[c] + combo_codes[r, c]
        t = target[r]
        joint[cell * kt + t] += 1.0
        px[cell] += 1.0
        py[t] += 1.0
    inv_n = 1.0 / n
    mi = 0.0
    occ_cells = 0
    for cell in range(n_cells):
        if px[cell] <= 0.0:
            continue
        occ_cells += 1
        pxc = px[cell] * inv_n
        base = cell * kt
        for t in range(kt):
            cnt = joint[base + t]
            if cnt > 0.0:
                pj = cnt * inv_n
                mi += pj * np.log(pj / (pxc * (py[t] * inv_n)))
    occ_x = 0
    for cell in range(n_cells):
        if px[cell] > 0.0:
            occ_x += 1
    occ_y = 0
    for t in range(kt):
        if py[t] > 0.0:
            occ_y += 1
    mm = (occ_x - 1 + occ_y - 1 - (occ_cells - 1)) / (2.0 * n)
    val = mi - mm
    return val if val > 0.0 else 0.0


def _renumber_joint_codes(code_x: np.ndarray, code_y: np.ndarray) -> tuple[np.ndarray, int]:
    """Collapse two integer code arrays into a single dense joint code array.

    Returns ``(joint_codes, n_joint_classes)`` where ``joint_codes[i]`` is a dense 0-based id for the
    ``(code_x[i], code_y[i])`` cell (only OCCUPIED cells get ids, so the count is the realised joint
    cardinality -- what the Miller-Madow correction debits)."""
    cx = np.asarray(code_x).astype(np.int64).ravel()
    cy = np.asarray(code_y).astype(np.int64).ravel()
    ky = int(cy.max()) + 1 if cy.size else 1
    flat = cx * ky + cy
    uniq, inv = np.unique(flat, return_inverse=True)
    return inv.astype(np.int64), int(uniq.size)


def joint_synergy_mi(code_x: np.ndarray, code_y: np.ndarray, target_codes: np.ndarray) -> float:
    """Miller-Madow-corrected MI (nats) between the JOINT (x,y) code grid and the target codes.

    The bias correction debits ``(k_joint-1 + k_target-1 - (k_cells-1)) / (2n)`` using the OCCUPIED
    class/cell counts, so a noise pair's positive finite-sample joint MI collapses toward zero while a
    genuine synergy pair (XOR and friends) retains a large excess -- see the module docstring's
    measured detection-vs-noise separation. Returns ``max(0.0, corrected_mi)``."""
    jc, _ = _renumber_joint_codes(code_x, code_y)
    yt = np.asarray(target_codes).astype(np.int64).ravel()
    n = jc.shape[0]
    if n == 0 or yt.shape[0] != n:
        return 0.0
    kx = int(jc.max()) + 1
    ky = int(yt.max()) + 1
    joint = np.zeros((kx, ky), dtype=np.float64)
    np.add.at(joint, (jc, yt), 1.0)
    joint /= n
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / (px[:, None] * py[None, :])[nz])).sum())
    occ = int(nz.sum())
    occx = int((px > 0).sum())
    occy = int((py > 0).sum())
    mm = (occx - 1 + occy - 1 - (occ - 1)) / (2.0 * n)
    return max(0.0, mi - mm)


def _marginal_mm_mi(code_x: np.ndarray, target_codes: np.ndarray) -> float:
    """MM-corrected marginal MI(code_x; target) in nats (>=0) -- the njit kernel at order 1."""
    cx = np.ascontiguousarray(np.asarray(code_x).astype(np.int64).ravel())
    yt = np.ascontiguousarray(np.asarray(target_codes).astype(np.int64).ravel())
    n = cx.shape[0]
    if n == 0 or yt.shape[0] != n:
        return 0.0
    kx = int(cx.max()) + 1
    kt = int(yt.max()) + 1
    return float(_combo_mm_mi_njit(cx.reshape(n, 1), np.array([kx], dtype=np.int64), yt, kt, kx))


def detect_synergy_combos(
    code_cols, target_codes: np.ndarray, candidate_idx, *,
    max_order: int = 3, min_order: int = 2, synergy_ratio: float = 1.5,
    min_joint_mi: float = 0.05, max_candidates: int = 24, max_combos: int = 4000,
):
    """Find feature COMBOS whose JOINT MI with the target greatly exceeds the SUM of their members'
    marginal MIs -- i.e. genuine SYNERGY the marginal/greedy screen misses (pure XOR: marginals ~0,
    joint high). Returns ``[(combo_tuple, joint_mi), ...]`` sorted by joint MI desc, for combos with
    ``joint_mi >= min_joint_mi`` AND ``joint_mi >= synergy_ratio * sum(member marginals)``.

    ``code_cols`` is an indexable of per-feature integer code arrays; ``candidate_idx`` the feature
    indices to consider. Candidates are capped to the ``max_candidates`` highest MARGINAL MI first to
    bound the O(P^order) combo count (a pure-synergy member has ~0 marginal, so we KEEP low-marginal
    candidates too: rank puts high-marginal first but we always include up to the cap). The MM bias
    correction keeps a noise combo's joint MI near zero, so the synergy ratio does not fire on noise."""
    import itertools as _it
    idx = list(candidate_idx)
    if len(idx) < min_order:
        return []
    _yt = np.ascontiguousarray(np.asarray(target_codes).astype(np.int64).ravel())
    _n = _yt.shape[0]
    _kt = int(_yt.max()) + 1 if _n else 1
    # Per-feature dense codes + cardinalities (cached once).
    _ccode = {i: np.ascontiguousarray(np.asarray(code_cols[i]).astype(np.int64).ravel()) for i in idx}
    _ccard = {i: (int(_ccode[i].max()) + 1 if _ccode[i].size else 1) for i in idx}
    _marg = {i: float(_combo_mm_mi_njit(_ccode[i].reshape(_n, 1), np.array([_ccard[i]], dtype=np.int64), _yt, _kt, _ccard[i])) for i in idx}
    # Cap: keep the strongest-marginal candidates PLUS (synergy needs low-marginal members) fill the
    # remaining cap slots with the lowest-marginal ones so pure-XOR operands are not excluded.
    if len(idx) > max_candidates:
        _by = sorted(idx, key=lambda i: -_marg[i])
        idx = _by[: max_candidates // 2] + _by[-(max_candidates - max_candidates // 2):]
        idx = sorted(set(idx))
    # Dense joint cardinality cap: the njit histogram allocates ``prod(cards)*kt`` cells; skip a combo
    # whose mixed-radix cell count blows past this bound (high-card columns) -- such a combo's joint MI
    # would be unreliable anyway (too few samples/cell). Bounds memory + keeps the kernel allocation small.
    _MAX_CELLS = 1 << 20
    out = []
    _seen = 0
    for _order in range(max(2, int(min_order)), int(max_order) + 1):
        for combo in _it.combinations(idx, _order):
            _seen += 1
            if _seen > max_combos:
                break
            _ncells = 1
            for _c in combo:
                _ncells *= _ccard[_c]
            if _ncells <= 0 or _ncells * _kt > _MAX_CELLS:
                continue
            _mat = np.empty((_n, _order), dtype=np.int64)
            for _k, _c in enumerate(combo):
                _mat[:, _k] = _ccode[_c]
            _cards = np.array([_ccard[_c] for _c in combo], dtype=np.int64)
            # joint MI of the combo's mixed-radix cell-codes vs the target (MM-corrected, njit).
            _jmi = float(_combo_mm_mi_njit(_mat, _cards, _yt, _kt, int(_ncells)))
            if _jmi < min_joint_mi:
                continue
            _msum = sum(_marg[c] for c in combo)
            if _jmi >= synergy_ratio * max(_msum, 1e-12):
                out.append((combo, _jmi))
        if _seen > max_combos:
            break
    out.sort(key=lambda t: -t[1])
    return out


__all__ = ["joint_synergy_mi", "_renumber_joint_codes", "detect_synergy_combos", "_marginal_mm_mi"]
