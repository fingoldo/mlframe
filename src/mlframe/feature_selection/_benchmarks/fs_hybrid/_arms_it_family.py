"""The information-theoretic family, one arm per scorer, so "MRMR" stops standing for all of them.

The roster had one `mrmr` arm. That arm is a particular implementation -- Fleuret's CMIM-style relevance,
its own binning, its own synergy prefilter and gates -- and every conclusion drawn from it was being read
as a conclusion about the whole family. Brown et al. (JMLR 2012) showed the family is a set of distinct
criteria that disagree exactly where it matters: on redundancy, on synergy, and on how a candidate is
judged against the set already chosen. So the family is measured one criterion at a time.

Two halves.

**A shared greedy driver**, `InformationGreedyArm`, with the scorer as its only free choice: MIM (relevance
alone, the no-redundancy control), CMIM (`min_j I(X; Y | Z_j)`), JMIM (`min_j I(X, Z_j; Y)`) and RelaxMRMR
(Vinh 2016, with the three-way interaction term). Every one of them bins the same way, starts from the same
empty set and breaks ties the same way, so a difference between two of these arms is a difference between
two CRITERIA and not between two implementations. JMIM and RelaxMRMR use this repository's own kernels;
CMIM is derived exactly from the JMIM kernel by the chain rule, `I(X; Y | Z) = I(X, Z; Y) - I(Z; Y)`,
rather than written a third time.

**The MRMR class's own variants**, which are the other half of the plan's question -- MRMR as an
ALGORITHM versus MRMR as this repository's IMPLEMENTATION with its gates. The `pld` relevance path, the
RelaxMRMR term switched on inside MRMR, the tree-rescued subclass, the cluster-medoid `GroupAwareMRMR`
wrapper with expansion on and off, and `StabilityMRMR`.

Each arm's `score_kind` was set by running it and looking at what it exposes, not by reading its
docstring. The greedy variants return their selection order. `GroupAwareMRMR` returns its support SORTED
BY INDEX -- a set, not an order -- so it declares `none`; publishing that as a selection order would score
column position. `StabilityMRMR` exposes a selection probability for every column, which is a genuine
continuous score.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ._arms import BaseArm, MRMRArm, _feature_names, _integer_bins, _mask_from_names

__all__ = [
    "IT_SCORERS",
    "InformationGreedyArm",
    "MRMRVariantArm",
    "GroupAwareMRMRArm",
    "StabilityMRMRArm",
]

#: The criteria the shared driver implements, in increasing order of how much of the chosen set each consults.
IT_SCORERS = ("mim", "cmim", "jmim", "relax")


def _target_codes(y: np.ndarray) -> np.ndarray:
    """Integer-code the target in first-seen order, which is what every kernel here indexes by."""
    return np.asarray(pd.factorize(np.asarray(y))[0], dtype=np.int64)


class InformationGreedyArm(BaseArm):
    """Greedy forward selection under one information-theoretic criterion, on one shared binning.

    The returned order is the order the criterion picked columns in; the support is its first `k`. There is
    no score for a column the greedy loop never reached, so the kind is `selection_order`, and the whole
    order is published rather than only the first `k` so a matched-K comparison at `2k` or `5k` still has
    something to read.
    """

    score_kind = "selection_order"

    def __init__(self, scorer: str, k: int, n_bins: int = 10, relax_alpha: float = 1.0, max_order: Optional[int] = None):
        if scorer not in IT_SCORERS:
            raise ValueError(f"unknown information scorer {scorer!r}; choose from {IT_SCORERS}")
        self.scorer = scorer
        self.k = int(k)
        self.n_bins = int(n_bins)
        self.relax_alpha = float(relax_alpha)
        # How far the greedy loop runs. Past `k` only to feed the wider matched-K rows, and capped because
        # RelaxMRMR's cost grows with the square of the chosen set.
        self.max_order = None if max_order is None else int(max_order)
        self.name = f"it-{scorer}"

    def _relevance(self, codes: List[np.ndarray], y_codes: np.ndarray, nbins: List[int], nbins_y: int) -> np.ndarray:
        """Return `I(X_j; Y)` for every column, from the same kernel the conditional scores use."""
        from mlframe.feature_selection.filters._jmim_scorer import _joint_mi_3d_njit

        zeros = np.zeros(y_codes.size, dtype=np.int64)
        return np.asarray([float(_joint_mi_3d_njit(c, zeros, y_codes, nb, 1, nbins_y)) for c, nb in zip(codes, nbins)], dtype=np.float64)

    def _score(
        self,
        j: int,
        chosen: Sequence[int],
        codes: List[np.ndarray],
        y_codes: np.ndarray,
        nbins: List[int],
        nbins_y: int,
        relevance: np.ndarray,
    ) -> float:
        """Return candidate `j`'s criterion value given the columns already chosen."""
        if not chosen or self.scorer == "mim":
            return float(relevance[j])
        selected = [codes[i] for i in chosen]
        selected_bins = [nbins[i] for i in chosen]
        if self.scorer == "jmim":
            from mlframe.feature_selection.filters._jmim_scorer import jmim_score

            return float(jmim_score(codes[j], selected, y_codes, nbins[j], selected_bins, nbins_y))
        if self.scorer == "relax":
            from mlframe.feature_selection.filters._relaxmrmr_3d import relax_mrmr_score

            return float(relax_mrmr_score(codes[j], selected, y_codes, nbins[j], selected_bins, nbins_y, alpha=self.relax_alpha, selected_prechecked=True))
        # CMIM by the chain rule on the JMIM kernel: I(X; Y | Z) = I(X, Z; Y) - I(Z; Y). The second term is
        # the already-computed relevance of the conditioning column, so each pair costs one kernel call.
        from mlframe.feature_selection.filters._jmim_scorer import _joint_mi_3d_njit

        worst = float("inf")
        for i in chosen:
            conditional = float(_joint_mi_3d_njit(codes[j], codes[i], y_codes, nbins[j], nbins[i], nbins_y)) - float(relevance[i])
            worst = min(worst, conditional)
        return max(0.0, worst)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Bin every column once, then pick greedily under the arm's criterion."""
        names = _feature_names(X)
        codes = [_integer_bins(X.iloc[:, j].to_numpy(), self.n_bins) for j in range(len(names))]
        nbins = [int(c.max()) + 1 if c.size else 1 for c in codes]
        y_codes = _target_codes(y)
        nbins_y = int(y_codes.max()) + 1 if y_codes.size else 1
        relevance = self._relevance(codes, y_codes, nbins, nbins_y)

        limit = len(names) if self.max_order is None else min(len(names), max(self.k, self.max_order))
        chosen: List[int] = []
        remaining = list(range(len(names)))
        while remaining and len(chosen) < limit:
            values = np.asarray([self._score(j, chosen, codes, y_codes, nbins, nbins_y, relevance) for j in remaining], dtype=np.float64)
            # argmax takes the FIRST maximum, so ties break by column position -- identically for every
            # scorer, which is what keeps a tie from reading as a difference between criteria.
            pick = remaining[int(np.argmax(values))]
            chosen.append(pick)
            remaining.remove(pick)

        support = np.zeros(len(names), dtype=bool)
        support[chosen[: self.k]] = True
        return {
            "support": support,
            "ranked_prefix": tuple(chosen),
            "n_model_fits": 0,
            "provenance": {"scorer": self.scorer, "n_bins": self.n_bins, "order_length": len(chosen), "relax_alpha": self.relax_alpha},
        }


class MRMRVariantArm(MRMRArm):
    """The MRMR class with one of its own alternative criteria switched on, read exactly as the base arm is.

    Subclassing rather than re-implementing keeps the support and order extraction identical to `mrmr`, so
    the only thing that differs between this arm and that one is the keyword it was built with.
    """

    def __init__(self, name: str, extra: Dict[str, Any], max_runtime_mins: float = 2.0, random_seed: int = 0, tree_rescued: bool = False):
        super().__init__(fe=False, max_runtime_mins=max_runtime_mins, random_seed=random_seed)
        self.name = name
        self.extra = dict(extra)
        self.tree_rescued = bool(tree_rescued)

    def _build_model(self) -> Any:
        """Build the MRMR (or its tree-rescued subclass) with the variant's keywords over the base arm's."""
        from mlframe.feature_selection.filters import MRMR
        from mlframe.feature_selection.filters._mrmr_tree_rescue import MRMRTreeRescued

        cls = MRMRTreeRescued if self.tree_rescued else MRMR
        kwargs: Dict[str, Any] = {"verbose": 0, "fe_max_steps": 0, "n_jobs": -1, "random_seed": self.random_seed, "max_runtime_mins": self.max_runtime_mins}
        kwargs.update(self.extra)
        return cls(**kwargs)


def _bare_mrmr(max_runtime_mins: float, random_seed: int) -> Any:
    """The same bare MRMR the `mrmr` arm fits, to be wrapped -- so a wrapper arm differs only by its wrapper."""
    from mlframe.feature_selection.filters import MRMR

    return MRMR(verbose=0, fe_max_steps=0, n_jobs=-1, random_seed=random_seed, max_runtime_mins=max_runtime_mins)


class GroupAwareMRMRArm(BaseArm):
    """MRMR inside the cluster-medoid `GroupAwareMRMR` wrapper, with cluster expansion on or off.

    With `expand=True` a chosen medoid drags its whole cluster back in -- measured on a probe bed, it
    returned the medoid AND its near-copy. That is the registry's default and the reason this arm exists
    in both settings: whether expansion helps or merely widens is a question only a pair of arms answers.

    Its support comes back sorted by column index, which carries no order, so the kind is `none`.
    """

    score_kind = "none"

    def __init__(self, expand: bool, corr_threshold: float = 0.9, max_runtime_mins: float = 2.0, random_seed: int = 0):
        self.expand = bool(expand)
        self.corr_threshold = float(corr_threshold)
        self.max_runtime_mins = float(max_runtime_mins)
        self.random_seed = int(random_seed)
        self.name = "mrmr-grouped-expand" if self.expand else "mrmr-grouped"

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Fit the wrapped MRMR and read the final support as a set."""
        from mlframe.feature_selection.filters.group_aware import GroupAwareMRMR

        names = _feature_names(X)
        model = GroupAwareMRMR(_bare_mrmr(self.max_runtime_mins, self.random_seed), corr_threshold=self.corr_threshold, expand=self.expand)
        model.fit(X, pd.Series(np.asarray(y)))
        selected = [str(c) for c in model.get_feature_names_out() if str(c) in set(names)]
        return {
            "support": _mask_from_names(names, selected),
            "provenance": {"expand": self.expand, "corr_threshold": self.corr_threshold, "reduction": float(getattr(model, "reduction_", float("nan")))},
        }


class StabilityMRMRArm(BaseArm):
    """MRMR under bootstrap stability selection, scored by each column's selection probability.

    `selection_probabilities_` covers every column, including those no bootstrap ever picked, so this is a
    genuine continuous score rather than one synthesised from the support.
    """

    name = "mrmr-stability"
    score_kind = "continuous"

    def __init__(self, n_bootstraps: int = 12, support_threshold: float = 0.6, max_runtime_mins: float = 1.0, random_seed: int = 0):
        self.n_bootstraps = int(n_bootstraps)
        self.support_threshold = float(support_threshold)
        self.max_runtime_mins = float(max_runtime_mins)
        self.random_seed = int(random_seed)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Fit the stability wrapper and publish the per-column selection probability."""
        from mlframe.feature_selection.filters.stability import StabilityMRMR

        names = _feature_names(X)
        model = StabilityMRMR(
            _bare_mrmr(self.max_runtime_mins, self.random_seed),
            n_bootstraps=self.n_bootstraps,
            support_threshold=self.support_threshold,
            random_state=self.random_seed,
        )
        model.fit(X, pd.Series(np.asarray(y)))
        probabilities = np.asarray(getattr(model, "selection_probabilities_"), dtype=np.float64)
        if probabilities.shape[0] != len(names):
            raise ValueError(f"StabilityMRMR returned {probabilities.shape[0]} selection probabilities for {len(names)} columns")
        selected = [str(c) for c in model.get_feature_names_out() if str(c) in set(names)]
        return {
            "support": _mask_from_names(names, selected),
            "score": np.nan_to_num(probabilities, nan=0.0),
            "n_model_fits": 0,
            "provenance": {"n_bootstraps": self.n_bootstraps, "support_threshold": self.support_threshold, "pfer_bound": float(getattr(model, "pfer_bound_", float("nan")))},
        }
