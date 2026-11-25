"""Judge, oracle and OLD proposers for the B3 validation (agent B3, 2026-11-24).

* ``judge``: the tt_a4_oracle protocol copied for an arbitrary candidate dictionary (same split: permutation seed 10000 + seed, 60 percent fit / 15 percent calibration (smearing residuals) / 25 percent test,
  clip to the training range of f(y) +- 5 percent, 64 residual quantiles, retry without margin when the inverse is not finite). ``sanity_against_a4`` checks that the identity numbers equal tt_a4's.
* ``search_rows``: the rows the recommender may look at = fit + calibration blocks (the 25 percent test block is never seen by any proposer).
* ``old_picks``: the CURRENT mlframe implementation (``filters/_target_transform_search.py``, loaded from its file so the package ``__init__`` is not imported): the equal-width-MI proposer top-1 over
  ``default_transforms()`` (``old_mi``), the same with Yeo-Johnson added as tt_a4 scored it (``old_mi_yj``), and the full ``search_target_transform`` = proposer top-3 + ridge gate (``old_search``).
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from typing import Callable, Dict, Optional, Tuple

import numpy as np

import tt_a4_oracle as O
import tt_b3_reference as R

warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)

_OLD_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "filters", "_target_transform_search.py"))
_old_mod = None


def old_module():
    """The current mlframe proposer module, loaded from its file (no package import)."""
    global _old_mod
    if _old_mod is None:
        spec = importlib.util.spec_from_file_location("tt_old_search", _OLD_PATH)
        _old_mod = importlib.util.module_from_spec(spec)
        import sys

        sys.modules["tt_old_search"] = _old_mod
        spec.loader.exec_module(_old_mod)
    return _old_mod


def split_idx(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.random.default_rng(10_000 + seed).permutation(n)
    return idx[: int(0.6 * n)], idx[int(0.6 * n): int(0.75 * n)], idx[int(0.75 * n):]


def search_rows(n: int, seed: int) -> np.ndarray:
    a, b, _ = split_idx(n, seed)
    return np.sort(np.concatenate([a, b]))


Cand = Tuple[Callable, Callable, Optional[Callable]]  # forward, inverse (argument already clipped), optional (z_train, margin) -> (lo, hi)


def extra_candidates(y_search: np.ndarray, lam_yj: Optional[float]) -> Dict[str, Cand]:
    """Candidates that tt_a4 does not have: signed_square, reciprocal, Yeo-Johnson with the regression-profile lambda ('yj_reg')."""
    out: Dict[str, Cand] = {}
    s = float(np.std(y_search)) + 1e-12
    out["signed_square"] = (lambda t: R._fixed_forward("signed_square", t, s), lambda z: R._fixed_inverse("signed_square", z, s), None)
    if y_search.min() > 0 or y_search.max() < 0:
        out["reciprocal"] = (lambda t: R._fixed_forward("reciprocal", t, s), lambda z: R._fixed_inverse("reciprocal", z, s), lambda zt, m: R._safe_bounds("reciprocal", zt, m))
    if lam_yj is not None:
        out["yj_reg"] = (lambda t: R._param_forward("yeo_johnson", t, np.array([lam_yj]))[:, 0], lambda z: R._param_inverse("yeo_johnson", z, lam_yj), lambda zt, m: R._safe_bounds("yeo_johnson", zt, m, lam_yj))
    return out


def judge(X: np.ndarray, y: np.ndarray, model: str, seed: int, extra: Dict[str, Cand], only: Optional[set] = None, margin: float = 0.05, n_smear: int = 64) -> Dict[str, Dict[str, float]]:
    """Original-scale scores of every candidate (a4 set + ``extra``) for ``model`` on the untouched test block; ``only`` restricts the candidate names."""
    n = len(y)
    a, b, t = split_idx(n, seed)
    cands: Dict[str, Cand] = {k: (f, i, None) for k, (f, i) in O.candidate_set(y[a]).items()}
    cands.update(extra)
    make_model = O.models()[model]
    res: Dict[str, Dict[str, float]] = {}
    for name, (f, inv, bound) in cands.items():
        if only is not None and name not in only:
            continue
        z = f(y)
        if not np.all(np.isfinite(z[a])):
            continue
        m = make_model().fit(X[a], z[a])
        za = z[a]
        zb = np.asarray(m.predict(X[b]))
        q = np.quantile(z[b] - zb, (np.arange(n_smear) + 0.5) / n_smear)
        zt = np.asarray(m.predict(X[t]))
        with np.errstate(all="ignore"):
            for mg in (margin, 0.0):
                if bound is None:
                    R_ = float(za.max() - za.min())
                    lo, hi = float(za.min()) - mg * R_, float(za.max()) + mg * R_
                else:
                    lo, hi = bound(za, mg)
                med = inv(np.clip(zt, lo, hi))
                smear = np.mean(inv(np.clip(zt[:, None] + q[None, :], lo, hi)), axis=1)
                if np.all(np.isfinite(med)) and np.all(np.isfinite(smear)):
                    break
        yt = y[t]

        def rmse(p: np.ndarray) -> float:
            return float(np.sqrt(np.mean((p - yt) ** 2))) if np.all(np.isfinite(p)) else float("inf")

        res[name] = {"rmse_smear": rmse(smear), "mae_clip": float(np.mean(np.abs(med - yt))) if np.all(np.isfinite(med)) else float("inf")}
    return res


def sanity_against_a4(kind_data: Tuple[np.ndarray, np.ndarray], model: str = "ridge", seed: int = 0) -> Tuple[float, float]:
    """(my identity rmse_smear, tt_a4's identity rmse_smear) on the same data: must be equal."""
    X, y = kind_data
    mine = judge(X, y, model, seed, {}, only={"identity", "log"})["identity"]["rmse_smear"]
    theirs = O.judge(X, y, model, names=["identity"], seed=seed)["identity"]["rmse_smear"]
    return mine, theirs


def old_picks(X: np.ndarray, y: np.ndarray, seed: int = 0) -> Dict[str, str]:
    """Picks of the current implementation on the rows (X, y): old_mi (proposer top-1 incl. identity), old_mi_yj (+ Yeo-Johnson), old_search (proposer top-3 + ridge gate)."""
    from sklearn.preprocessing import PowerTransformer

    M = old_module()
    pool = M.default_transforms()
    top = M.propose_target_transforms(X, y, pool)[0][0]
    pt = PowerTransformer(method="yeo-johnson", standardize=False).fit(y.reshape(-1, 1))
    yj = M.TargetTransform("yeo_johnson", lambda t: pt.transform(np.asarray(t).reshape(-1, 1)).ravel(), lambda z: pt.inverse_transform(np.asarray(z).reshape(-1, 1)).ravel())
    top_yj = M.propose_target_transforms(X, y, [*pool, yj])[0][0]
    chosen, _ = M.search_target_transform(X, y, pool, seed=seed)
    return {"old_mi": top, "old_mi_yj": top_yj, "old_search": chosen.name}
