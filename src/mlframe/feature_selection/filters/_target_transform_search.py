"""Search for a monotone transform of a regression TARGET that a LINEAR model predicts better (PLAN_target_transform_in_mrmr, step 2: functions only, not wired into MRMR).

Two parts, because the experiment of 2026-11-23 (``_benchmarks/nbins_strategy/target_transform_mi_ext.py``: 5 target kinds x 7 seeds, four model families) showed that the MI alone is a biased judge:

* **proposer** (:func:`propose_target_transforms`): the sum over features of the MI between the feature and ``f(y)`` under EQUAL-WIDTH bins (rank-based bins are invariant to a monotone ``f`` and cannot see it). It
  ranks the candidates; on a plain linear target it still ranks ``signed_sqrt`` first, so the ranking is only a shortlist.
* **gate** (:func:`gate_target_transform`): a ridge model is fitted on ``f(y)`` over one half of the rows, its prediction is inverted and clipped to the range of the training target, and its squared error on the
  ORIGINAL scale over the other half is compared with that of a ridge fitted on ``y`` itself, with a paired bootstrap of the per-row loss difference. ``f`` is accepted only when the whole upper confidence limit of
  (loss(f) - loss(identity)) lies below zero. A tree model does not gain from a target transform (measured), so the gate is a ridge one by design.

:func:`search_target_transform` runs both and returns the accepted transform with the best held-out loss, or the identity.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Sequence

import numpy as np

__all__ = ["TargetTransform", "default_transforms", "propose_target_transforms", "gate_target_transform", "search_target_transform", "GateResult"]


@dataclasses.dataclass(frozen=True)
class TargetTransform:
    """A monotone transform of the target with its inverse; ``needs_positive`` marks one that is defined for a positive target only."""

    name: str
    forward: Callable[[np.ndarray], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray]
    needs_positive: bool = False


@dataclasses.dataclass(frozen=True)
class GateResult:
    """Outcome of the held-out gate for one transform: the mean per-row loss difference (negative: the transform is better), its bootstrap upper limit, and the verdict."""

    name: str
    mean_loss_difference: float
    upper_limit: float
    accepted: bool


def default_transforms() -> "list[TargetTransform]":
    """identity, cbrt, signed square root and log (positive targets only); no unbounded inverse (Yeo-Johnson overflowed on a heavy right tail in the experiment)."""
    return [
        TargetTransform("identity", lambda t: t, lambda z: z),
        TargetTransform("cbrt", np.cbrt, lambda z: z**3),
        TargetTransform("signed_sqrt", lambda t: np.sign(t) * np.sqrt(np.abs(t)), lambda z: np.sign(z) * z**2),
        TargetTransform("log", np.log, np.exp, needs_positive=True),
    ]


def _equal_width_codes(v: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width bin codes of a finite vector (``n_bins`` bins over its range); a constant vector gets one bin."""
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return np.zeros(len(v), dtype=np.int64)
    edges = np.linspace(lo, hi, n_bins + 1)[1:-1]
    return np.searchsorted(edges, v, side="right").astype(np.int64)


def _mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    """Plug-in MI (nats) of two integer code vectors."""
    ka, kb = int(a.max()) + 1, int(b.max()) + 1
    joint = np.bincount(a * kb + b, minlength=ka * kb).astype(np.float64).reshape(ka, kb)  # bincount: np.add.at was 40 percent of the proposer at 1000 columns
    joint /= joint.sum()
    pa = joint.sum(1, keepdims=True)
    pb = joint.sum(0, keepdims=True)
    m = joint > 0
    return float((joint[m] * np.log(joint[m] / (pa @ pb)[m])).sum())


def propose_target_transforms(X: np.ndarray, y: np.ndarray, transforms: Optional[Sequence[TargetTransform]] = None, n_bins: int = 10) -> "list[tuple[str, float]]":
    """``(name, score)`` of every applicable transform, best first; the score is the sum over the columns of ``X`` of MI(column; f(y)) under equal-width bins."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    positive = bool(len(y)) and float(y.min()) > 0.0
    scored: "list[tuple[str, float]]" = []
    cols = [_equal_width_codes(X[:, j], n_bins) for j in range(X.shape[1])]
    for t in transforms if transforms is not None else default_transforms():
        if t.needs_positive and not positive:
            continue
        z = t.forward(y)
        if not np.isfinite(z).all():
            continue
        zc = _equal_width_codes(z, n_bins)
        scored.append((t.name, float(sum(_mutual_information(c, zc) for c in cols))))
    return sorted(scored, key=lambda kv: -kv[1])


def _ridge_loss(X_fit: np.ndarray, z_fit: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray, t: TargetTransform, y_lo: float, y_hi: float, alpha: float) -> np.ndarray:
    """Per-row squared error on the original scale of a ridge fitted on ``t.forward(y)`` (prediction inverted and clipped to the training range)."""
    from sklearn.linear_model import Ridge

    pred_z = Ridge(alpha=alpha).fit(X_fit, z_fit).predict(X_eval)
    with np.errstate(over="ignore", invalid="ignore"):
        pred = np.clip(np.nan_to_num(t.inverse(pred_z), nan=y_hi, posinf=y_hi, neginf=y_lo), y_lo, y_hi)
    return np.asarray((pred - y_eval) ** 2)


def gate_target_transform(
    X: np.ndarray,
    y: np.ndarray,
    transform: TargetTransform,
    *,
    seed: int = 0,
    n_boot: int = 400,
    confidence: float = 0.95,
    alpha: float = 1.0,
    identity: Optional[TargetTransform] = None,
) -> GateResult:
    """Held-out gate of ``transform`` against the identity with a ridge head: accepted when the upper ``confidence`` limit of the bootstrapped mean of (loss(f) - loss(identity)) is negative.

    The rows are split in two halves at random (``seed``); the model is fitted on one half and the loss is measured on the other, the roles are then swapped and the two loss vectors are concatenated, so every row
    is judged out of sample once. The identity itself is never accepted (its difference is zero)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    ident = identity if identity is not None else default_transforms()[0]
    if transform.name == ident.name or len(y) < 40 or (transform.needs_positive and float(y.min()) <= 0.0):
        return GateResult(transform.name, 0.0, 0.0, False)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    halves = (order[: len(y) // 2], order[len(y) // 2 :])
    diffs = []
    for fit_idx, eval_idx in (halves, halves[::-1]):
        lo, hi = float(y[fit_idx].min()), float(y[fit_idx].max())
        z_t = transform.forward(y[fit_idx])
        if not np.isfinite(z_t).all():
            return GateResult(transform.name, 0.0, 0.0, False)
        loss_t = _ridge_loss(X[fit_idx], z_t, X[eval_idx], y[eval_idx], transform, lo, hi, alpha)
        loss_i = _ridge_loss(X[fit_idx], ident.forward(y[fit_idx]), X[eval_idx], y[eval_idx], ident, lo, hi, alpha)
        diffs.append(loss_t - loss_i)
    d = np.concatenate(diffs)
    boots = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        boots[b] = d[rng.integers(0, len(d), size=len(d))].mean()
    upper = float(np.quantile(boots, confidence))
    return GateResult(transform.name, float(d.mean()), upper, bool(upper < 0.0))


def search_target_transform(X: np.ndarray, y: np.ndarray, transforms: Optional[Sequence[TargetTransform]] = None, *, top_k: int = 3, seed: int = 0) -> "tuple[TargetTransform, list[GateResult]]":
    """Propose the ``top_k`` transforms by equal-width MI, gate each, and return the accepted one with the lowest mean loss difference (the identity when none is accepted) with every gate result."""
    pool = list(transforms) if transforms is not None else default_transforms()
    by_name = {t.name: t for t in pool}
    ranked = [name for name, _ in propose_target_transforms(X, y, pool) if name != "identity"][: int(top_k)]
    results = [gate_target_transform(X, y, by_name[name], seed=seed) for name in ranked]
    accepted = [r for r in results if r.accepted]
    if not accepted:
        return by_name.get("identity", default_transforms()[0]), results
    best = min(accepted, key=lambda r: r.mean_loss_difference)
    return by_name[best.name], results
