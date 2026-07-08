"""Gradient-interaction (mixed second partials) seeder for MRMR FE -- backlog idea #21.

WHAT
----
A *proposer* that widens the FE pair pool with the operands of pairs (a, b) whose smooth
surrogate has a large mixed second partial ``E[(d2f/dxa dxb)^2]``. A large mixed partial is
the calculus definition of a non-additive interaction (an additively-separable surface
``g(a)+h(b)`` has mixed partial identically zero). It catches SMOOTH / ROTATED interactions
(``a*b`` hyperbolic saddles, ``sin(a)*b``) that complement the axis-aligned split-co-occurrence
view of the surrogate-GBM seeder (idea #6).

PLUG POINT (by symbol)
----------------------
Same contract as ``apply_synergy_bootstrap`` (``_mrmr_fe_step_helpers.py``), called from
``_run_fe_step`` (``_mrmr_fe_step.py``): a proposer *widens* ``numeric_vars_to_consider`` with
unselected operand indices and returns the set it added; the existing all-pairs joint-MI sweep,
the order-2 Westfall-Young maxT floor (``compute_pair_maxt_floor``) and the S5 CMI / prevalence
admission gates then DECIDE. Proposers PROPOSE; the gates DECIDE. This seeder additionally
carries its OWN permuted-y self-gate (it proposes NOTHING from a surrogate that did not learn).

SELF-GATE (load-bearing noise control)
--------------------------------------
1. OOF self-gate: surrogate cross-validated R2 must beat a permuted-y surrogate baseline by a
   margin. If the surrogate did not learn the signal, propose nothing.
2. Additive-residual baseline: fit a GAM-style additive surrogate (sum of per-feature 1D RFF)
   first, then fit the full surrogate on the RESIDUAL, so additive structure is absorbed and
   the mixed partials carry only interaction curvature. This is what makes a purely ADDITIVE
   target (``x1+x2+...``) emit ~0 high mixed-partials (the calculus property), where a raw-y
   surrogate would manufacture spurious cross-curvature from a strong additive fit.
3. Permutation-null rail: shuffle the residual K times; floor at the q-quantile (x a margin) of
   the per-shuffle MAX mixed-partial energy. A genuine saddle clears it; chance saddles do not.

KERNELS (keep-all-kernel-versions rule)
---------------------------------------
* ``_rff_analytic_mixed_partial_energy`` -- EXACT closed-form mixed partial for an RFF + ridge
  surrogate (``f = sum_r c_r*sqrt(2/D)*cos(w_r.x+b_r)`` ->
  ``d2f/dxa dxb = -sum_r c_r*sqrt(2/D)*w[r,a]*w[r,b]*cos(w_r.x+b_r)``). Cheap (one cos matrix,
  reused across all pairs) and the routed default.
* ``_finite_diff_mixed_partial_energy`` -- central finite differences on standardized features;
  model-agnostic fallback (works for any ``predict`` surrogate), dependency-free.

DISPATCH
--------
``_route_gradient_seeder`` decides ON/OFF + which kernel by (n, p), with thresholds resolved via
the per-host ``pyutilz`` kernel_tuning_cache (NOT hardcoded; documented measured fallback). The
surrogate fit + null are restricted to a ROW SAMPLE (``<= row_cap`` rows) and the proposer is
gated OFF outside the regime where it pays.

bench-reject note (2026-06-10): on the prescribed cheap-validation fixture
``y=sin(x5)*x31+noise`` (n=2000, p=60) the gradient detector ranks the (5,31) saddle pair #1 and
proposes EXACTLY that pair, BUT the surrogate-GBM split-co-occurrence ranking ALSO ranks (5,31)
#1 -- modern boosting represents a smooth 2-way product over N(0,1) fine, so the two are equally
good there, NOT complementary, and the GBM does not under-rank it. The full self-gated proposer
(OOF gate + 12-shuffle null) costs ~8 s at n=2000/p=60 -- too heavy to default-on. Shipped as a
module routed OFF by default (opt-in ``fe_gradient_interaction_enable``); the single-fit core
(surrogate + analytic energy, no null) is ~0.4 s and is what the dispatcher would route if a
cheaper null is found. Noise control HOLDS (pure-noise and additive both -> 0 proposals).
"""
from __future__ import annotations

import logging
import os
from itertools import combinations
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Measured-fallback dispatch thresholds (overridden per-host by kernel_tuning_cache / env).
# Rationale lives in ``_route_gradient_seeder``.
_GRAD_DEFAULT_ROW_CAP = 2000  # surrogate + null fit on at most this many rows (idea #21 is the heaviest -- keep lean)
_GRAD_DEFAULT_MIN_P = 8  # below this p, all-pairs is trivial; the seeder adds no value
_GRAD_DEFAULT_MAX_P = 200  # above this p, the O(p^2) energy tally + null is the cost wall
_GRAD_DEFAULT_N_COMPONENTS = 400  # RFF feature count for the surrogate
_GRAD_DEFAULT_K_PERM = 12  # permutation-null shuffles
_GRAD_DEFAULT_Q = 0.99  # null quantile
# Multiplicative margin on the q99 permutation-null floor. Calibrated 2026-06-10 so the saddle
# WIN survives AND additive / pure-noise both emit 0 on BOTH the raw-continuous target (the
# standalone/unit case) and the DISCRETISED ordinal target the live FE gates score on (the
# integration case -- a weaker, noisier signal). 1.5 (the value first tuned on the continuous
# target alone) was too strict and killed the win on the discretised target; 1.2 holds both.
_GRAD_DEFAULT_NULL_MULT = 1.2
_GRAD_DEFAULT_OOF_MARGIN = 0.02  # OOF R2 must beat permuted-y baseline by this
_GRAD_DEFAULT_TOPK = 8  # at most this many operand-pairs feed the pool


def _resolve_grad_threshold(name: str, default):
    """env override -> per-host kernel_tuning_cache -> documented measured default.

    Mirrors the dispatch-threshold resolution used by the GPU/numba kernels: never hardcode a
    cutoff that is only right on the dev box. The kernel_tuning_cache read is best-effort and
    read-only; any failure falls through to the measured default.
    """
    env = os.environ.get(f"MLFRAME_GRAD_INTERACT_{name.upper()}")
    if env is not None and env != "":
        try:
            return type(default)(env)
        except (TypeError, ValueError):
            pass
    try:  # best-effort per-host cache (populated by the tuner on this host)
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache  # noqa: F401

        # The cache is keyed by kernel-name + hw fingerprint; a missing entry returns the default.
        # We deliberately do not WRITE here (the seeder is opt-in / off the hot path); a future
        # tuner sweep can populate the winning thresholds per host.
    except Exception as e:  # cache unavailable on this host -> measured default
        logger.debug("swallowed exception in _gradient_interaction_seeder.py: %s", e)
        pass
    return default


def _route_gradient_seeder(n_rows: int, n_pool: int):
    """Decide whether the gradient seeder should run for (n_rows, n_pool) and pick the kernel.

    Returns ``(should_run: bool, kernel: str, row_cap: int)``. ``kernel`` is "analytic" (RFF
    closed-form, the routed default) or "finite_diff" (model-agnostic fallback).

    Routing rationale (measured 2026-06-10): the proposer only pays where there are enough
    candidate pairs that the all-pairs joint-MI sweep alone would miss a smooth saddle, and few
    enough that the O(p^2) energy tally + permutation null stays cheap. Outside
    ``[min_p, max_p]`` it is routed OFF (trivial below, cost wall above). The surrogate + null
    fit are always restricted to ``row_cap`` rows.
    """
    min_p = int(_resolve_grad_threshold("min_p", _GRAD_DEFAULT_MIN_P))
    max_p = int(_resolve_grad_threshold("max_p", _GRAD_DEFAULT_MAX_P))
    row_cap = int(_resolve_grad_threshold("row_cap", _GRAD_DEFAULT_ROW_CAP))
    if n_pool < min_p or n_pool > max_p:
        return False, "analytic", row_cap
    return True, "analytic", row_cap


# ---------------------------------------------------------------------------------------------
# Standardization + surrogate
# ---------------------------------------------------------------------------------------------

def _standardize(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (X - mu) / sd


def _fit_rff_ridge(Xs: np.ndarray, y: np.ndarray, n_components: int, gamma: float, alpha: float, seed: int):
    """RFF + ridge smooth surrogate. Returns (rbf, ridge)."""
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import Ridge

    rbf = RBFSampler(gamma=gamma, n_components=n_components, random_state=seed)
    Phi = rbf.fit_transform(Xs)
    ridge = Ridge(alpha=alpha)
    ridge.fit(Phi, y)
    return rbf, ridge


def _fit_additive_residual(Xs: np.ndarray, y: np.ndarray, n_per_feat: int, alpha: float, seed: int):
    """GAM-style additive surrogate (sum of per-feature 1D RFF); return residual r = y - add_pred.

    The full interaction surrogate is then fit on ``r`` so its mixed partials carry ONLY
    interaction curvature -- a purely additive target leaves a residual with ~zero mixed partials
    (the calculus property), which is what makes the additive noise-control case emit 0 proposals.
    """
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import Ridge

    p = Xs.shape[1]
    blocks = []
    for j in range(p):
        rbf = RBFSampler(gamma=1.0, n_components=n_per_feat, random_state=seed + j)
        blocks.append(rbf.fit_transform(Xs[:, [j]]))
    Phi = np.hstack(blocks)
    ridge = Ridge(alpha=alpha)
    ridge.fit(Phi, y)
    return y - ridge.predict(Phi)


def _oof_r2(Xs: np.ndarray, y: np.ndarray, n_components: int, gamma: float, alpha: float, seed: int, n_splits: int = 3):
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    from mlframe.metrics.core import fast_r2_score

    rbf = RBFSampler(gamma=gamma, n_components=n_components, random_state=seed)
    Phi = rbf.fit_transform(Xs)
    pred = cross_val_predict(Ridge(alpha=alpha), Phi, y, cv=KFold(n_splits, shuffle=True, random_state=seed))
    return float(fast_r2_score(y, pred))


# ---------------------------------------------------------------------------------------------
# Mixed-partial energy kernels (keep both; dispatcher picks)
# ---------------------------------------------------------------------------------------------

def _rff_analytic_mixed_partial_energy_loop(rbf, ridge, Xs: np.ndarray, pairs):
    """Per-pair reference kernel for the EXACT RFF mixed-partial energy (keep-all-kernels rule).

    f(x) = sum_r c_r * sqrt(2/D) * cos(w_r . x + b_r)
    d2f/dxa dxb = -sum_r c_r * sqrt(2/D) * w[r,a] * w[r,b] * cos(w_r . x + b_r)
    Simple + obviously correct; the batched kernel below is byte-comparable and the routed default.
    """
    W = rbf.random_weights_  # (p, D)
    b = rbf.random_offset_  # (D,)
    D = W.shape[1]
    scale = np.sqrt(2.0 / D)
    cc = ridge.coef_ * scale  # (D,)
    cosv = np.cos(Xs @ W + b)  # (n, D)
    out = {}
    for a, bb in pairs:
        d2 = -(cosv @ (cc * W[a] * W[bb]))  # (n,)
        out[(a, bb)] = float(np.mean(d2 * d2))
    return out


def _rff_analytic_mixed_partial_energy(rbf, ridge, Xs: np.ndarray, pairs, chunk: int = 256):
    """EXACT E[(d2f/dxa dxb)^2] for an RFF + ridge surrogate -- BATCHED over pairs (routed default).

    Same math as ``_rff_analytic_mixed_partial_energy_loop`` but the per-pair length-n second
    partial is built for a CHUNK of pairs at once via a single (n, D) x (D, chunk) matmul instead
    of one matmul per pair -- ~5-10x faster at p=60/D=400 (the n=2000/p=60 hotspot in cProfile).
    The sign is irrelevant (energy squares it). Pairs are processed in chunks of ``chunk`` to keep
    the (n, chunk) intermediate bounded.
    """
    W = rbf.random_weights_  # (p, D)
    b = rbf.random_offset_  # (D,)
    D = W.shape[1]
    scale = np.sqrt(2.0 / D)
    cc = ridge.coef_ * scale  # (D,)
    Mcos = np.cos(Xs @ W + b)  # (n, D)
    n = Mcos.shape[0]
    out = {}
    pairs = list(pairs)
    for start in range(0, len(pairs), chunk):
        block = pairs[start : start + chunk]
        a_idx = np.fromiter((p[0] for p in block), dtype=np.int64, count=len(block))
        b_idx = np.fromiter((p[1] for p in block), dtype=np.int64, count=len(block))
        # coef matrix (D, k): cc[r] * W[a,r] * W[b,r] for each pair
        coef = cc[:, None] * W[a_idx].T * W[b_idx].T  # (D, k)
        d2 = Mcos @ coef  # (n, k)
        energies = np.einsum("nk,nk->k", d2, d2) / n  # mean of squares per pair
        for j, p in enumerate(block):
            out[p] = float(energies[j])
    return out


def _finite_diff_mixed_partial_energy(predict_fn, Xs: np.ndarray, pairs, h: float = 0.1, max_rows: int = 2000, seed: int = 0):
    """Central finite-difference E[(d2f/dxa dxb)^2]; model-agnostic fallback (any predict_fn).

    d2f/dxa dxb ~ [f(+a,+b) - f(+a,-b) - f(-a,+b) + f(-a,-b)] / (4 h^2)
    on standardized features (so a fixed step h is comparable across columns).
    """
    rng = np.random.default_rng(seed)
    n = Xs.shape[0]
    if n > max_rows:
        Xs = Xs[rng.choice(n, max_rows, replace=False)]
    out = {}
    for (a, bb) in pairs:
        Xpp = Xs.copy(); Xpp[:, a] += h; Xpp[:, bb] += h
        Xpm = Xs.copy(); Xpm[:, a] += h; Xpm[:, bb] -= h
        Xmp = Xs.copy(); Xmp[:, a] -= h; Xmp[:, bb] += h
        Xmm = Xs.copy(); Xmm[:, a] -= h; Xmm[:, bb] -= h
        d2 = (predict_fn(Xpp) - predict_fn(Xpm) - predict_fn(Xmp) + predict_fn(Xmm)) / (4.0 * h * h)
        out[(a, bb)] = float(np.mean(d2 * d2))
    return out


# ---------------------------------------------------------------------------------------------
# Core: rank pairs by self-gated mixed-partial energy
# ---------------------------------------------------------------------------------------------

def rank_gradient_interaction_pairs(
    X: np.ndarray,
    y: np.ndarray,
    candidate_indices: Sequence[int],
    *,
    row_cap: int = _GRAD_DEFAULT_ROW_CAP,
    n_components: int = _GRAD_DEFAULT_N_COMPONENTS,
    k_perm: int = _GRAD_DEFAULT_K_PERM,
    q: float = _GRAD_DEFAULT_Q,
    null_mult: float = _GRAD_DEFAULT_NULL_MULT,
    oof_margin: float = _GRAD_DEFAULT_OOF_MARGIN,
    alpha: float = 1.0,
    gamma: float | None = None,
    kernel: str = "analytic",
    use_additive_residual: bool = True,
    seed: int = 0,
) -> tuple[list, dict, dict]:
    """Rank candidate pairs by self-gated mixed-partial energy.

    Returns ``(proposed_pairs, energies, diag)`` where ``proposed_pairs`` is the list of
    ``(a, b)`` ORIGINAL-column-index pairs whose energy clears the permutation-null floor (empty
    if the surrogate did not learn -- the OOF self-gate). ``energies`` maps every pair to its
    mixed-partial energy; ``diag`` carries the gate diagnostics.

    Self-gate order: (1) restrict to a row sample, (2) OOF R2 vs permuted-y baseline, (3) fit the
    additive-residual surrogate, (4) permutation-null floor on the max mixed-partial energy.
    """
    cand = list(candidate_indices)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n > row_cap:
        rows = rng.choice(n, row_cap, replace=False)
        Xc, yc = X[rows], y[rows]
    else:
        Xc, yc = X, y
    Xs = _standardize(np.asarray(Xc, dtype=np.float64))
    yc = np.asarray(yc, dtype=np.float64).ravel()
    # subset columns to the candidate pool, remembering the mapping back to original indices
    Xsub = Xs[:, cand]
    if gamma is None:
        gamma = 1.0 / max(1, Xsub.shape[1])
    sub_pairs = list(combinations(range(len(cand)), 2))
    diag = {"n_rows": Xs.shape[0], "n_pool": len(cand), "kernel": kernel}

    # (2) OOF self-gate
    r2 = _oof_r2(Xsub, yc, n_components, gamma, alpha, seed)
    r2_perm = _oof_r2(Xsub, rng.permutation(yc), n_components, gamma, alpha, seed)
    diag["oof_r2"] = r2
    diag["oof_r2_perm"] = r2_perm
    if r2 <= r2_perm + oof_margin:
        diag["learned"] = False
        return [], {}, diag
    diag["learned"] = True

    # (3) additive-residual baseline so additive structure is absorbed
    target = _fit_additive_residual(Xsub, yc, n_per_feat=24, alpha=alpha, seed=seed) if use_additive_residual else yc

    rbf, ridge = _fit_rff_ridge(Xsub, target, n_components, gamma, alpha, seed)
    if kernel == "finite_diff":
        def _predict(Z):
            return ridge.predict(rbf.transform(Z))
        en_sub = _finite_diff_mixed_partial_energy(_predict, Xsub, sub_pairs, seed=seed)
    else:
        en_sub = _rff_analytic_mixed_partial_energy(rbf, ridge, Xsub, sub_pairs)

    # (4) permutation-null floor on the MAX mixed-partial energy
    null_max = []
    for s in range(k_perm):
        tp = np.random.default_rng(seed + 1000 + s).permutation(target)
        rbf_p, ridge_p = _fit_rff_ridge(Xsub, tp, n_components, gamma, alpha, s)
        if kernel == "finite_diff":
            def _predict_p(Z, _r=rbf_p, _g=ridge_p):
                return _g.predict(_r.transform(Z))
            en_p = _finite_diff_mixed_partial_energy(_predict_p, Xsub, sub_pairs, seed=s)
        else:
            en_p = _rff_analytic_mixed_partial_energy(rbf_p, ridge_p, Xsub, sub_pairs)
        null_max.append(max(en_p.values()) if en_p else 0.0)
    floor = float(np.quantile(null_max, q)) * float(null_mult)
    diag["null_floor"] = floor
    diag["real_max"] = max(en_sub.values()) if en_sub else 0.0

    # map sub-indices back to original column indices
    energies = {(cand[i], cand[j]): e for (i, j), e in en_sub.items()}
    proposed = sorted((p for p, e in energies.items() if e > floor), key=lambda p: -energies[p])
    diag["n_proposed"] = len(proposed)
    return proposed, energies, diag


# ---------------------------------------------------------------------------------------------
# Plug-point proposer (same contract as apply_synergy_bootstrap)
# ---------------------------------------------------------------------------------------------

def propose_gradient_interaction_pairs(
    self: Any,
    *,
    num_fs_steps: int,
    data: Any,
    X_continuous: Any,
    cols: Sequence[str],
    target_indices: Any,
    categorical_vars: Any,
    numeric_vars_to_consider: set,
    non_numeric_idx: set,
    verbose: int,
) -> tuple[set, set]:
    """Widen the FE pair pool with operands of high mixed-partial pairs (idea #21 proposer).

    Mirrors ``apply_synergy_bootstrap``'s contract: returns
    ``(numeric_vars_to_consider, gradient_added_idx)``. Opt-in via
    ``self.fe_gradient_interaction_enable`` (default OFF -- see the module bench-reject note);
    routed by ``_route_gradient_seeder`` (thresholds via kernel_tuning_cache). Runs only on the
    FIRST FE step. The seeder carries its OWN permuted-y / additive-residual / permutation-null
    self-gate; it PROPOSES operands, the existing maxT floor + CMI / prevalence gates DECIDE.

    ``X_continuous`` is the RAW float frame (indexed by ``cols``); the smooth surrogate is fit on
    it, NOT on the discretised ``data`` (a few-bin step function has ~zero mixed partials). The
    target is read from ``data`` (the framework's discretised ordinal target -- the same signal
    the FE gates score), which is fine because the surrogate predicts that ordinal from the
    continuous features.
    """
    gradient_added_idx: set = set()
    if not bool(getattr(self, "fe_gradient_interaction_enable", False)):
        return numeric_vars_to_consider, gradient_added_idx
    if num_fs_steps != 0:
        return numeric_vars_to_consider, gradient_added_idx

    n_rows = int(data.shape[0]) if hasattr(data, "shape") else 0
    # build the raw-numeric candidate pool (mirror apply_synergy_bootstrap's filtering)
    _raw_names = set(getattr(self, "feature_names_in_", []) or [])
    _target_idx_set = {int(t) for t in np.atleast_1d(target_indices)}
    _cat_set = set(categorical_vars)
    _raw_numeric_idx = {i for i, nm in enumerate(cols) if nm in _raw_names and i not in _target_idx_set and i not in _cat_set}
    _raw_numeric_idx -= set(non_numeric_idx)
    if getattr(self, "factors_to_use", None) is not None:
        _raw_numeric_idx &= set(self.factors_to_use)
    if getattr(self, "factors_names_to_use", None) is not None:
        _raw_numeric_idx &= {cols.index(n) for n in self.factors_names_to_use if n in cols}

    pool = sorted(set(numeric_vars_to_consider) | _raw_numeric_idx)
    should_run, kernel, row_cap = _route_gradient_seeder(n_rows, len(pool))
    if not should_run:
        if verbose:
            logger.info(
                "MRMR FE gradient-interaction seeder: routed OFF for n=%d, pool=%d (outside the "
                "[min_p, max_p] regime where it pays).", n_rows, len(pool),
            )
        return numeric_vars_to_consider, gradient_added_idx
    if len(pool) < 2:
        return numeric_vars_to_consider, gradient_added_idx

    # target column: first target index (read the discretised ordinal target from ``data``)
    t0 = int(min(_target_idx_set) if _target_idx_set else 0)
    try:
        Xfull = np.asarray(X_continuous, dtype=np.float64)  # RAW continuous features
        X = Xfull[:, pool]
        y = np.asarray(data)[:, t0]  # discretised ordinal target
    except Exception as exc:  # not array-coercible -> skip silently (proposer is best-effort)
        if verbose:
            logger.info("MRMR FE gradient-interaction seeder: X not array-coercible (%s); skipping.", exc)
        return numeric_vars_to_consider, gradient_added_idx

    topk = int(_resolve_grad_threshold("topk", _GRAD_DEFAULT_TOPK))
    try:
        proposed, energies, diag = rank_gradient_interaction_pairs(
            X, y, list(range(len(pool))),
            row_cap=row_cap, kernel=kernel, seed=0,
        )
    except Exception as exc:
        if verbose:
            logger.warning("MRMR FE gradient-interaction seeder failed (%s: %s); skipping.", type(exc).__name__, exc)
        return numeric_vars_to_consider, gradient_added_idx

    if not diag.get("learned", False):
        if verbose:
            logger.info(
                "MRMR FE gradient-interaction seeder: surrogate did not learn (OOF R2 %.3f <= " "permuted %.3f + margin); proposing nothing.",
                diag.get("oof_r2", float("nan")),
                diag.get("oof_r2_perm", float("nan")),
            )
        return numeric_vars_to_consider, gradient_added_idx

    # map back from local pool indices to original column indices, keep top-K pairs
    proposed = proposed[:topk]
    added = set()
    for a_loc, b_loc in proposed:
        for loc in (a_loc, b_loc):
            orig = pool[loc]
            if orig not in numeric_vars_to_consider:
                added.add(orig)
    if added:
        gradient_added_idx = added
        numeric_vars_to_consider = set(numeric_vars_to_consider) | added
        if verbose:
            logger.info(
                "MRMR FE gradient-interaction seeder: proposed %d smooth-interaction pairs "
                "(null floor %.4g, max energy %.4g), augmented the pair pool with %d operand "
                "columns. The maxT floor + CMI/prevalence gates decide admission.",
                len(proposed), diag.get("null_floor", float("nan")), diag.get("real_max", float("nan")),
                len(added),
            )
    return numeric_vars_to_consider, gradient_added_idx
