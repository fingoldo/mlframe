"""An arm that ranks features by voting, so "combining rankings beats the best one" becomes measurable.

The claim shows up constantly and is almost never tested: ensemble the rankings of several cheap scorers
and you beat any of them alone. It is plausible -- the scorers fail on different structures, so their
errors are not the same errors -- and it is also exactly the kind of claim that survives on plausibility
because nobody put it in a leaderboard next to its own components.

This arm makes it a row. Several base scorers each rank every feature; a social-choice rule aggregates
those rankings into one; the aggregate is the arm's score. The components are already in the roster, so
the comparison is direct: if the aggregate does not beat `skb-f` and `univariate-mi` on the beds where
those two disagree, the claim is not true here.

The voting rules come from this repository's own ``votenrank``, which is already used in production to
aggregate RFECV's per-fold importances -- the same problem one level down. Three are exposed, chosen
because they disagree in ways that matter rather than to have three:

* **Borda** -- weighted rank points. Sensitive to WHERE a feature sits in each ranking, so one scorer
  placing a feature last drags it down even if the others place it first.
* **Dowdall** -- reciprocal rank. Cares almost entirely about the top of each ranking and barely
  distinguishes the bottom, which is the right shape when the budget is small.
* **Copeland** -- pairwise majority. Ignores rank distance entirely: only who beats whom, and by how many
  scorers. Immune to one scorer's extreme placement in a way neither of the others is.

A rule building a pairwise majority graph is quadratic in features, so Copeland is capped by width and the
arm says which rule it actually used rather than silently substituting one.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from ._arms import BaseArm

logger = logging.getLogger(__name__)

__all__ = ["VOTING_RULES", "COPELAND_MAX_FEATURES", "base_score_table", "aggregate_ranks", "RankAggregationArm"]

#: The rules this arm can use, and the order it falls back through when a rule is unavailable at this width.
VOTING_RULES: Tuple[str, ...] = ("borda", "dowdall", "copeland")

#: Copeland builds a features x features majority graph. Beyond this width that is the dominant cost of the
#: whole cell, so the arm falls back and records that it did rather than quietly running for an hour.
COPELAND_MAX_FEATURES = 500


def _f_statistic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-feature ANOVA F, the parametric scorer."""
    from sklearn.feature_selection import f_classif

    values, _p = f_classif(np.nan_to_num(x, nan=0.0), y)
    return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)


def _mutual_information(x: np.ndarray, y: np.ndarray, random_state: int) -> np.ndarray:
    """Per-feature mutual information, the non-parametric scorer that sees non-monotone structure."""
    from sklearn.feature_selection import mutual_info_classif

    return np.asarray(mutual_info_classif(np.nan_to_num(x, nan=0.0), y, random_state=random_state), dtype=np.float64)


def _abs_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-feature absolute rank correlation: monotone, and unaffected by the heavy tails that move Pearson."""
    from scipy import stats

    labels = np.asarray(y, dtype=np.float64)
    out = np.empty(x.shape[1], dtype=np.float64)
    for column in range(x.shape[1]):
        values = np.nan_to_num(x[:, column], nan=0.0)
        if np.all(values == values[0]):
            out[column] = 0.0
            continue
        out[column] = abs(float(stats.spearmanr(values, labels).statistic))
    return np.nan_to_num(out, nan=0.0)


def _tree_importance(x: np.ndarray, y: np.ndarray, random_state: int) -> np.ndarray:
    """Per-feature impurity importance from a small forest: the only base scorer that is multivariate."""
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=64, random_state=random_state, n_jobs=1)
    model.fit(np.nan_to_num(x, nan=0.0), y)
    return np.asarray(model.feature_importances_, dtype=np.float64)


def base_score_table(X: pd.DataFrame, y: np.ndarray, random_state: int = 0) -> pd.DataFrame:
    """Return the ``features x scorers`` table the voting rules aggregate.

    Each column is one base scorer's per-feature score, higher meaning better. They are deliberately
    unlike each other: a parametric test, a binned mutual information, a rank correlation and a tree
    importance fail on different structures, and a vote over four scorers that agree everywhere is a
    slower copy of any one of them.
    """
    values = np.asarray(X.to_numpy(dtype=np.float64))
    labels = np.asarray(y)
    scorers: Dict[str, Callable[[], np.ndarray]] = {
        "f_test": lambda: _f_statistic(values, labels),
        "mutual_info": lambda: _mutual_information(values, labels, random_state),
        "spearman": lambda: _abs_spearman(values, labels),
        "tree": lambda: _tree_importance(values, labels, random_state),
    }
    columns: Dict[str, np.ndarray] = {}
    for name, scorer in scorers.items():
        try:
            columns[name] = scorer()
        except Exception as exc:
            # A scorer that cannot run on this bed is dropped from the vote and NAMED, never replaced by
            # zeros: a column of zeros is a scorer that ranks every feature equally, which silently dilutes
            # every other scorer's opinion rather than abstaining.
            logger.warning("base scorer %r could not run on this bed and is excluded from the vote: %s", name, exc)
    if not columns:
        raise RuntimeError("no base scorer could run on this bed, so there is nothing to aggregate")
    return pd.DataFrame(columns, index=[str(column) for column in X.columns])


def aggregate_ranks(table: pd.DataFrame, rule: str = "borda") -> Tuple[pd.Series, str]:
    """Aggregate a ``features x scorers`` table into one ranking, returning ``(scores, rule actually used)``.

    Higher is better in the returned series, matching every other arm's score convention.

    Raises:
        ValueError: On an unknown rule. A typo falling back to a default would report one aggregation
            under another's name, which is the whole thing this arm exists to compare.
    """
    if rule not in VOTING_RULES:
        raise ValueError(f"unknown voting rule {rule!r}; expected one of {VOTING_RULES}")

    from mlframe.votenrank import Leaderboard

    used = rule
    if rule == "copeland" and table.shape[0] > COPELAND_MAX_FEATURES:
        logger.info("copeland needs a %d x %d majority graph; falling back to borda at this width", table.shape[0], table.shape[0])
        used = "borda"

    board = Leaderboard(table)
    if used == "borda":
        return board.borda_ranking(), used
    if used == "dowdall":
        return board.dowdall_ranking(), used
    return board.copeland_ranking(), used


class RankAggregationArm(BaseArm):
    """Rank every feature with four unlike scorers and let a social-choice rule settle the disagreements.

    Declares ``score_kind='continuous'``: the aggregate is a real number per feature with full coverage,
    which is what makes it comparable to the filters it is built from rather than only to other set
    selectors.
    """

    name = "rank-vote"
    score_kind = "continuous"

    def __init__(self, k: int = 10, rule: str = "borda", random_state: int = 0):
        self.k = int(k)
        self.rule = str(rule)
        self.random_state = int(random_state)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Build the score table, aggregate it, and keep the top ``k``."""
        names: List[str] = [str(column) for column in X.columns]
        table = base_score_table(X, y, random_state=self.random_state)
        scores, used_rule = aggregate_ranks(table, self.rule)

        aligned = np.asarray([float(scores.get(name, float(scores.min()))) for name in names], dtype=np.float64)
        budget = max(1, min(self.k, len(names)))
        keep = np.argsort(-aligned)[:budget]
        support = np.zeros(len(names), dtype=bool)
        support[keep] = True
        return {
            "support": support,
            "score": aligned,
            # One fit, from the tree scorer; the other three fit no model at all.
            "n_model_fits": 1,
            "provenance": {"rule_requested": self.rule, "rule_used": used_rule, "scorers": list(table.columns), "k": budget},
        }
