"""Feature-base target mix: candidate features found by a second pass on ``T = y - beta * u``, ``u`` a column of the current selection.

A dominant term of the selection fills the rank bins of y, and a weaker term of other columns is invisible to a search that scores candidates one pair at a time against y (the residual rescue
measured MI 0.07 against 1.85 for such a term). Subtracting the dominant term turns ``y ~ f(x_1) + g(x_2, x_3)`` into ``T ~ g(x_2, x_3)``, a pair-level dependence the ordinary search finds. The base
is not an outside column (that is ``training/composite``) but any raw or engineered column the selector already holds, so the base is varied through the selector's own feature transforms.

* bases: the ``max_bases`` columns of the design that track y best, each only if a one-column fit lowers the held-out squared error significantly (otherwise ``T`` is ``y``); ``beta`` is the OLS slope on the
  training rows;
* second pass: a clone of the estimator (every stage that could recurse switched off) fitted on the EVEN positions of the scan rows against ``T``; its engineered recipes are the candidates;
* acceptance: none here. The candidates join the pool of the residual rescue, whose held-out test fits the design on the same even positions and scores the odd ones, so a candidate chosen on the even rows
  is judged on rows no selection step has seen, with the family-wise level over the whole pool. A candidate is an ordinary engineered feature with a recipe that replays from X alone (the mix is a
  search device, ``T`` never reaches transform time).

A placebo (a permuted base) gives the right null for this construction: it keeps the signal of y and adds noise of the same scale, and what it adds is the price of a second pass, not of the mix
(``_benchmarks/target_lens/exp_target_mix.py``, arm D).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["choose_bases", "second_pass_recipes"]


def choose_bases(design: Mapping[str, np.ndarray], y: np.ndarray, max_bases: int, *, min_relative_gain: float) -> "list[tuple[str, np.ndarray, float]]":
    """``(name, values, beta)`` of the design columns to mix with y, best first.

    A column qualifies when the OLS fit of y on it alone lowers the held-out squared error significantly (one-sided 5%) and by at least ``min_relative_gain`` of the target variance; ``beta`` is the slope
    fitted on the even positions."""
    from scipy.stats import norm

    from ._pair_residual_screen import FAMILY_ALPHA
    from ._residual_rescue_fe import linear_oos_gain

    z = float(norm.isf(FAMILY_ALPHA))
    even = np.arange(len(y)) % 2 == 0
    scored = []
    for name, u in design.items():
        u = np.nan_to_num(np.asarray(u), nan=0.0, posinf=0.0, neginf=0.0)
        if float(u.std()) <= 0.0:
            continue
        t_stat, rel = linear_oos_gain({}, u, y)
        if t_stat > z and rel >= min_relative_gain:
            u_even, y_even = u[even], y[even]
            beta = float(np.dot(u_even - u_even.mean(), y_even - y_even.mean()) / max(float(np.dot(u_even - u_even.mean(), u_even - u_even.mean())), 1e-12))
            scored.append((rel, name, u, beta))
    scored.sort(key=lambda r: -r[0])
    return [(name, u, beta) for _rel, name, u, beta in scored[: max(int(max_bases), 0)]]


def _child(estimator: Any):
    """A clone of the estimator for the second pass: quiet, and without the mix itself (no recursion). Its residual rescue stays on: the pair tables it builds on the residual of ``T`` are among the
    best candidates the second pass has (a weak interaction under a dominant term is recovered there, not by the operator search)."""
    from sklearn.base import clone

    child = clone(estimator)
    child.set_params(fe_target_mix_enable=False, fe_max_steps=1, verbose=0)  # one FE step: the second step re-searches the pool the first step enlarged, which is most of the cost on a structured base
    child._lean_second_pass_ = True  # skips the linear-usability retention of the support assembly: the child only proposes candidates
    return child


def second_pass_recipes(estimator: Any, X: "pd.DataFrame", y: np.ndarray, design: Mapping[str, np.ndarray], *, max_bases: int, scan_rows: int, min_relative_gain: float) -> "tuple[dict[str, Any], list[dict]]":
    """Engineered recipes the second pass found on ``T = y - beta u`` for each base, by name (the first base that finds a name wins), and the provenance of the mix.

    The provenance is one entry per base that was used, ``{"base", "beta", "features"}``: the design column ``u`` (by its design name), the OLS slope on the even positions, and the candidate names
    that base produced. It is the only record that the target was mixed; the candidates themselves are ordinary features replayable from ``X``.

    Fitted on the even positions of ``X`` (at most ``scan_rows`` of them), so the held-out test of the rescue, which fits the design on the same positions, scores the candidates on the odd ones."""
    n = len(y)
    bases = choose_bases(design, y, max_bases, min_relative_gain=min_relative_gain)
    if not bases:
        return {}, []
    train = np.arange(0, n, 2)
    if len(train) > scan_rows:
        train = train[np.linspace(0, len(train) - 1, int(scan_rows)).astype(np.int64)]
    X_train = X.iloc[train].reset_index(drop=True)
    found: "dict[str, Any]" = {}
    provenance: "list[dict]" = []
    for name, u, beta in bases:
        try:
            child = _child(estimator)
            child.fit(X_train, pd.Series(y[train] - beta * u[train], name="y"))
        except Exception:
            logger.warning("MRMR target-mix second pass on the base %r failed; skipped.", name, exc_info=True)
            continue
        own = []
        for feature, recipe in zip(child._engineered_features_ or [], child._engineered_recipes_ or []):
            if str(feature) not in found:
                found[str(feature)] = recipe
                own.append(str(feature))
        provenance.append({"base": str(name), "beta": float(beta), "features": own})
    return found, provenance
