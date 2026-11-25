"""Out-of-fold regression tests of a pair's winning engineered column: is it an honest one-dimensional summary of the pair, and does the pair carry an interaction at all?

The joint-prevalence gate compares ``MI(best column; y)`` with ``MI(pair; y)``, a 1-D feature binned into ``k`` classes against a 2-D joint binned into about ``k^2`` cells. The ratio is depressed for a column that
loses nothing, partly by the finite-sample bias of the larger table (``(k_x-1)(k_y-1)/2n``, the term the Miller-Madow debias subtracts) and mostly by the resolution of the two partitions, which no bias correction
touches (measured 2026-11-24 on ``y = sin(x1) + x2``, n = 6000: ratio 0.77, Miller-Madow 0.80, permutation-debiased 0.80, against a bar of 0.90, while the winner loses 0.6 % of the out-of-fold R^2 of a 2-D smoother).

This module compares fitted values instead of bin counts. Every model is a ridge regression on a piecewise-linear (hat) basis of RANK-transformed inputs, scored on out-of-fold rows by the MEAN ABSOLUTE ERROR
(the primary metric: R^2 goes negative and misbehaves on heavy-tailed and financial targets; it is reported as a secondary number only). The skill of a model is ``1 - MAE / MAE(median)``:

  * ``g``   a 1-D smooth of the winning candidate column;
  * ``add`` an additive smooth of the two operands (no interaction);
  * ``ten`` a tensor smooth of the two operands (any interaction).

``t_g_vs_ten`` is the paired t statistic of the out-of-fold absolute errors of ``g`` against ``ten``: positive and large means the winner is NOT a sufficient summary of the pair. ``t_add_vs_ten`` is the same for
``add`` against ``ten``: positive and large means the pair has an interaction a sum of two 1-D effects cannot express. Rank transforms make the bases invariant to monotone re-scalings of the inputs, and the paired
statistic on the same rows and folds needs no fixed threshold on the ratio of two plug-in quantities.
"""

from __future__ import annotations

import numpy as np

#: Hat functions per 1-D smooth, and per axis of the tensor smooth.
HATS_1D = 20
HATS_2D = 10
#: Out-of-fold folds and the ridge strength (per row, scaled by the basis width).
FOLDS = 5
RIDGE = 1e-3
#: Rows used at most: a bigger frame is subsampled once (seeded), the statistics do not need more.
MAX_ROWS = 20000
#: A winner is an honest 1-D summary of the pair when its out-of-fold R^2 is within this share of the tensor smooth's (measured 2026-11-24: exact winners lose -0.6 to 1.5 percent, a degraded one 86 percent).
SUFFICIENCY_LOSS_MAX = 0.05
#: A pair has an interaction when the tensor smooth beats the additive one by this paired t statistic of the absolute errors (about four standard errors) AND by this much MAE skill (measured on the weak F2, n = 20000:
#: genuine pairs t 16 to 47 and skill gain 0.03 to 0.12, additive cross pairs and noise t -5 to 0 and skill gain -0.002 to 0; with R^2 the same split, t 15 to 36 against -4.4 to -1.8).
INTERACTION_T_MIN = 5.0
INTERACTION_SKILL_MIN = 0.01
#: The raw MI ratio below which a winner cannot be a summary of the pair, and the test is not run (the regression costs about 50 ms).
MIN_MI_RATIO_TO_TEST = 0.5


def _rank01(x: np.ndarray) -> np.ndarray:
    """Average-free rank of every row mapped into (0, 1); non-finite values count as the median."""
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, np.nanmedian(x) if np.isfinite(x).any() else 0.0)
    ranks = np.empty(x.size)
    ranks[np.argsort(x, kind="stable")] = np.arange(x.size)
    return ((ranks + 0.5) / x.size).astype(np.float32)


def _hats(u: np.ndarray, k: int) -> np.ndarray:
    """``(n, k)`` piecewise-linear hat basis on ``k`` equally spaced knots of [0, 1]."""
    knots = np.linspace(0.0, 1.0, k, dtype=np.float32)
    width = knots[1] - knots[0]
    return np.asarray(np.maximum(0.0, 1.0 - np.abs(u[:, None] - knots[None, :]) / width))


def _oof_errors(basis: np.ndarray, y: np.ndarray, folds: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Per-row out-of-fold ``(absolute error, squared error)`` of a ridge fit of ``y`` on ``basis`` (normal equations per fold, the Gram matrix built once)."""
    n, width = basis.shape
    gram = (basis.T @ basis).astype(np.float64)
    moment = (basis.T @ y).astype(np.float64)
    penalty = RIDGE * n / width * np.eye(width)
    residual = np.empty(n, dtype=np.float64)
    for fold in range(FOLDS):
        test = folds == fold
        held = basis[test]
        weights = np.linalg.solve(gram - (held.T @ held) + penalty, moment - held.T @ y[test])
        residual[test] = y[test] - held @ weights
    return np.abs(residual), residual**2


def _paired_t(worse: np.ndarray, better: np.ndarray) -> float:
    """Paired t statistic of ``mean(worse - better)``; positive when ``better`` has the smaller out-of-fold error."""
    diff = worse - better
    spread = float(diff.std())
    return float(diff.mean() / (spread / np.sqrt(diff.size) + 1e-12))


def pair_regression_stats(winner: np.ndarray, operand_a: np.ndarray, operand_b: np.ndarray, y: np.ndarray, *, seed: int = 0) -> dict:
    """Out-of-fold MAE skill (``skill_g``, ``skill_add``, ``skill_ten``) of the 1-D, additive and tensor smooths, the two paired statistics of the absolute errors (``t_g_vs_ten``, ``t_add_vs_ten``) and the secondary
    R^2 of the same models (``r2_*``); ``{}`` when the inputs cannot be modelled (fewer than 200 rows, a constant target).

    ``winner``, ``operand_a``, ``operand_b`` and ``y`` are equally long vectors of continuous values; ``y`` is the continuous target (a class code works as a regression target for a binary problem)."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 200 or not np.isfinite(y).all() or float(y.std()) <= 0.0:
        return {}
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(n, MAX_ROWS, replace=False)) if n > MAX_ROWS else slice(None)
    y = y[rows]
    y = ((y - np.median(y)) / y.std()).astype(np.float32)
    folds = rng.permutation(y.size) % FOLDS
    u_win, u_a, u_b = (_rank01(np.asarray(v)[rows]) for v in (winner, operand_a, operand_b))
    one = np.ones((y.size, 1), dtype=np.float32)
    hats_a, hats_b = _hats(u_a, HATS_2D), _hats(u_b, HATS_2D)
    bases = {
        "g": np.hstack([one, _hats(u_win, HATS_1D)]),
        "add": np.hstack([one, _hats(u_a, HATS_1D), _hats(u_b, HATS_1D)]),
        "ten": np.einsum("ni,nj->nij", hats_a, hats_b).reshape(y.size, -1),
    }
    errors = {name: _oof_errors(basis, y, folds) for name, basis in bases.items()}
    mae_null = float(np.abs(y.astype(np.float64)).mean())
    se_null = float((y.astype(np.float64) ** 2).mean())
    out = {}
    for name, (absolute, squared) in errors.items():
        out[f"skill_{name}"] = 1.0 - float(absolute.mean()) / max(mae_null, 1e-300)
        out[f"r2_{name}"] = 1.0 - float(squared.mean()) / max(se_null, 1e-300)
    out["t_g_vs_ten"] = _paired_t(errors["g"][0], errors["ten"][0])
    out["t_add_vs_ten"] = _paired_t(errors["add"][0], errors["ten"][0])
    return out


def interaction_verdict(stats: dict) -> "tuple[bool, str]":
    """Whether the winner is an honest 1-D summary of a pair that has an interaction, and the reason when it is not. ``stats`` is the result of :func:`pair_regression_stats` (MAE skills and t statistics)."""
    if not stats:
        return False, "the pair could not be modelled"
    skill_ten = stats["skill_ten"]
    if skill_ten <= 0.0:
        return False, "the pair explains none of the target"
    loss = (skill_ten - stats["skill_g"]) / skill_ten
    if loss > SUFFICIENCY_LOSS_MAX:
        return False, f"the winner loses {100 * loss:.1f} percent of the pair's out-of-fold MAE skill: not a summary of the pair"
    gain = skill_ten - stats["skill_add"]
    if stats["t_add_vs_ten"] < INTERACTION_T_MIN or gain < INTERACTION_SKILL_MIN:
        return False, f"the pair is additive (interaction t={stats['t_add_vs_ten']:.1f}, MAE skill gain {gain:.3f}): its operands already carry it"
    return True, f"interaction t={stats['t_add_vs_ten']:.1f}, MAE skill gain {gain:.3f}, winner loss {100 * loss:.1f} percent"
