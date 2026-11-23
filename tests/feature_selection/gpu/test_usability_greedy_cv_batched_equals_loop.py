"""The batched regression CV of the resident usability greedy gives the per-candidate loop's fold errors and a fallback signal on a singular border."""

import logging

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _usability_greedy_gpu_resident as ug


def _setup(seed=0, n=600, p=9, nf=4):
    """Random design, target and fold masks of the resident greedy, and the batched closure."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, p))
    y = V[:, 0] * 1.5 - V[:, 3] + 0.3 * rng.standard_normal(n)
    fold = rng.permutation(n) % nf
    va = [cp.asarray(fold == f) for f in range(nf)]
    tr = [cp.asarray(fold != f) for f in range(nf)]
    return nf, tr, va, cp.asarray(y), cp.asarray(V)


def cv_candidates_loop_reference(cp, Vdev, tr_masks, va_masks, ydev, nf, sel_idx, cand_list) -> dict:
    """Per-candidate reference of the batched regression CV (the loop the production block replaces): kept here, out of the
    production module, because only the equivalence test calls it."""
    # Precompute, ONCE per step per fold, the centered selected design pieces (resident).
    per_fold = []
    for fo in range(nf):
        tr, va = tr_masks[fo], va_masks[fo]
        ytr = ydev[tr]
        ybar = ytr.mean()
        yc = ytr - ybar
        yva = ydev[va]  # fold-invariant across candidates; gather ONCE per fold
        if sel_idx:
            Str = Vdev[tr][:, sel_idx]
            Sva = Vdev[va][:, sel_idx]
            mu = Str.mean(axis=0)
            Sc_tr = Str - mu
            Sc_va = Sva - mu
            Gs = Sc_tr.T @ Sc_tr
            bs = Sc_tr.T @ yc
        else:
            Sc_tr = Sc_va = mu = Gs = bs = None
        per_fold.append((tr, va, ybar, yc, yva, Sc_tr, Sc_va, Gs, bs))

    # Accumulate EVERY candidate's per-fold MAE row resident, then do ONE D2H for the whole
    # shortlist (a (n_cand, nf) stacked pull) instead of one .get() per candidate - the values
    # are unchanged, just coalesced into a single device->host sync per round.
    errs_rows = []  # resident (nf,) row per candidate, in cand_list order
    cand_keys = []
    # Round-1 (empty selection) singular-border guard, kept RESIDENT: the per-candidate-per-fold
    # cc_tr.cc_tr was previously pulled back as a float() scalar PER CANDIDATE PER FOLD (a per-candidate
    # scalar D2H that contradicts this round's "one coalesced D2H" contract). Instead accumulate each
    # candidate's MINIMUM fold cc_tr.cc_tr resident and coalesce ALL of them into the SAME single D2H
    # as the fold-MAE matrix below; the border is then checked host-side once per round. Same verdict:
    # if any round-1 candidate-fold has cc_tr.cc_tr <= 1e-12 the round raises ug._ResidentFallbackError (the
    # whole resident path then refits exactly on CPU), identical to the prior mid-loop break.
    mind_rows = []  # resident scalar (min fold d) per candidate when Sc_tr is None; else None
    for i in cand_list:
        ci = Vdev[:, i]
        # Intentional float64: statistic vector (MI / score / fold error), exact ties decide the selection
        errs_dev = cp.empty(nf, dtype=cp.float64)  # accumulate fold MAEs resident; ONE D2H/candidate
        singular = False
        cand_mind = None  # resident running-min cc_tr.cc_tr over folds (round-1 border)
        for fo in range(nf):
            tr, va, ybar, yc, yva, Sc_tr, Sc_va, Gs, bs = per_fold[fo]
            ctr = ci[tr]
            cva = ci[va]
            cmu = ctr.mean()
            cc_tr = ctr - cmu
            cc_va = cva - cmu
            if Sc_tr is None:
                d = cp.dot(cc_tr, cc_tr)  # resident; border checked in the coalesced D2H below
                cand_mind = d if cand_mind is None else cp.minimum(cand_mind, d)
                # cc_tr.cc_tr == 0 only on a constant train operand; the divide then yields inf/nan,
                # the fold MAE is nan, and the host-side border check below converts that to a fallback.
                beta = cp.dot(cc_tr, yc) / d
                pred = ybar + cc_va * beta
            else:
                g = Sc_tr.T @ cc_tr
                d = cp.dot(cc_tr, cc_tr)
                bnew = cp.dot(cc_tr, yc)
                k = int(Gs.shape[0])
                # Intentional float64: normal equations solved in double, as the host solve this device path mirrors
                G = cp.empty((k + 1, k + 1), dtype=cp.float64)
                G[:k, :k] = Gs
                G[:k, k] = g
                G[k, :k] = g
                G[k, k] = d
                # Intentional float64: normal equations solved in double, as the host solve this device path mirrors
                rhs = cp.empty(k + 1, dtype=cp.float64)
                rhs[:k] = bs
                rhs[k] = bnew
                try:
                    beta = cp.linalg.solve(G, rhs)
                except Exception as e:
                    ug.log_throttle(ug.logger, "usability_greedy_gpu_resident.hot_loop_warning", logging.WARNING, "cp.linalg.solve failed (likely a singular Gram matrix), stopping the greedy: %s", e)
                    singular = True
                    break
                pred = ybar + Sc_va @ beta[:k] + cc_va * beta[k]
            errs_dev[fo] = cp.mean(cp.abs(yva - pred))
        if singular:
            raise ug._ResidentFallbackError()
        errs_rows.append(errs_dev)
        cand_keys.append(i)
        mind_rows.append(cand_mind)
    out: dict = {}
    if errs_rows:
        errs_host = cp.asnumpy(cp.stack(errs_rows, axis=0))  # ONE D2H for the whole shortlist
        if not np.isfinite(errs_host).all():
            raise ug._ResidentFallbackError("a bordered solve returned a non-finite fold error")  # cupy's solve does not raise on a singular matrix: it returns inf / nan
        # Coalesced round-1 border check: pull the per-candidate min cc_tr.cc_tr back in the SAME
        # round (one stacked D2H), then fall back exactly if any candidate sits on the singular border.
        if any(m is not None for m in mind_rows):
            mind_host = cp.asnumpy(cp.stack([m for m in mind_rows if m is not None]))
            if float(mind_host.min()) <= 1e-12:
                raise ug._ResidentFallbackError()
        for r, i in enumerate(cand_keys):
            out[i] = errs_host[r]
    return out


@pytest.mark.parametrize("sel", [[], [0], [0, 3, 5]])
def test_batched_cv_equals_loop(sel):
    """Fold errors agree with the loop for an empty, a one-column and a three-column selection."""
    nf, tr, va, y, V = _setup()
    cv = ug._usability_greedy_gpu_step3_def_cv_candidates(nf, tr, va, y, V, cp)
    cands = [i for i in range(V.shape[1]) if i not in sel][:6]
    new = cv(sel, cands)
    ref = cv_candidates_loop_reference(cp, V, tr, va, y, nf, sel, cands)
    assert list(new) == list(ref)
    for i in cands:
        np.testing.assert_allclose(new[i], ref[i], rtol=1e-9, atol=1e-12)


def test_singular_border_is_not_silently_scored():
    """A candidate that duplicates a selected column gives a singular border: the batched form raises the fallback signal instead of returning garbage errors (the loop relies on cupy raising, which it does not)."""
    nf, tr, va, y, V = _setup()
    V2 = cp.concatenate([V, V[:, :1]], axis=1)
    cv = ug._usability_greedy_gpu_step3_def_cv_candidates(nf, tr, va, y, V2, cp)
    with pytest.raises(ug._ResidentFallbackError):
        cv([0], [V2.shape[1] - 1, 4])
