"""The full arm roster, assembled in one place, with the facts every other module needs about it.

`build_arm_roster` in `_arms` delegates here. The roster had grown to 21 arms against a plan of roughly
forty-five, and the missing half was not obscure: the whole information-theoretic family beyond one MRMR
setting, every wrapper, every stability and bandit method, and the reference lines that bound the board.
Assembling it in its own module keeps `_arms` -- already close to a thousand lines -- from being the place
that has to grow.

Names are part of the results format. A cell is keyed by its arm's name, so an existing name here is
never changed; new arms get new names.

`WRAPPER_INTERNAL_ESTIMATOR` lives here for the same reason. It used to be a hand-written dict in
`run_experiment` keyed `rfecv_lgbm` and `rfecv_logit`, while the arm is named `rfecv` -- so the guard that
stops a wrapper being scored on its own objective looked up a name that never occurs and returned `None`
("not a wrapper") for every real arm. It was decorative. Keyed from this module, and pinned by a test that
every key is a name the roster actually builds, it cannot drift away from the roster again.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

from ._arms import (
    ACEArm,
    AllFeaturesArm,
    BaseArm,
    BorutaArm,
    BorutaShapArm,
    KnockoffArm,
    LarsPathArm,
    MRMRArm,
    RandomSelectionArm,
    RFECVArm,
    SelectFromModelArm,
    ShapProxiedArm,
    SklearnScoreArm,
    UnivariateMIArm,
    VarianceSortArm,
)

__all__ = ["build_full_roster", "WRAPPER_INTERNAL_ESTIMATOR", "CONTROL_ARMS", "PREREGISTERED_2E_ARMS", "is_control_arm"]

#: Arms that fit a model to decide, and the model they fit. The guard in `run_cell` refuses a wrapper whose
#: internal model is the ONLY panel member, since it would then be scored on its own objective.
WRAPPER_INTERNAL_ESTIMATOR: Dict[str, Optional[str]] = {
    "rfecv": "lightgbm",
    "rfecv-registry": "lightgbm",
    "sfm-lgbm": "lightgbm",
    "forward-select": "lightgbm",
    "greedy-backward": "lightgbm",
    "zero-importance": "lightgbm",
    "noise-floor": "lightgbm",
    "unanimous-permutation": "lightgbm",
    "bandit": "lightgbm",
    "bandit-ensemble": "lightgbm",
    "cascade": "lightgbm",
    "cascade-stable": "lightgbm",
    "null-importance": "lightgbm",
    "permutation-topk": "lightgbm",
}

#: Arms that are references rather than methods. They answer "what would doing nothing, or doing it by
#: chance, or knowing the answer, look like", and the pre-registration's rule that every arm be predicted to
#: break on at least two beds does not apply to them -- a control is not expected to break, it is expected
#: to hold still so the others can be read against it.
CONTROL_ARMS = ("all-features", "variance-sort", "oracle-informative", "all-except-informative")


#: The methods whose break predictions were added in section 2e of the pre-registration, before any of them ran
#: on a bed. The `predictions` tier runs exactly these; a test keeps this list and that section's table identical.
PREREGISTERED_2E_ARMS = (
    "bandit",
    "bandit-ensemble",
    "boruta-shap-registry",
    "cascade",
    "cascade-stable",
    "catboost-loss",
    "catboost-predictions",
    "catboost-shap",
    "forward-select",
    "greedy-backward",
    "hetero-vote",
    "it-cmim",
    "it-jmim",
    "it-mim",
    "it-relax",
    "ksg-mi",
    "mrmr-grouped",
    "mrmr-grouped-expand",
    "mrmr-pld",
    "mrmr-relax",
    "mrmr-stability",
    "mrmr-tree-rescued",
    "near-noise-auc",
    "noise-floor",
    "null-importance",
    "permutation-topk",
    "relevance-table",
    "rfecv-registry",
    "ridge-prefilter",
    "shap-proxied",
    "unanimous-permutation",
    "unsupervised-prescreen",
    "zero-importance",
)


def is_control_arm(name: str) -> bool:
    """Whether an arm is a reference rather than a method; `random-<k>` is one, whatever its `k`."""
    return str(name) in CONTROL_ARMS or str(name).startswith("random-")


def build_full_roster(
    n_features: int,
    *,
    k: Optional[int] = None,
    random_state: int = 0,
    relevant: Optional[Sequence[str]] = None,
) -> Dict[str, Callable[[], BaseArm]]:
    """Return `name -> zero-argument builder` for every arm, sized to this bed.

    Args:
        n_features: Width of the bed; sizes the fixed-cardinality arms and decides whether the O(d^2)
            wrappers are registered at all.
        k: Cardinality for the fixed-K arms; `max(1, n_features // 4)` when omitted.
        random_state: Seed threaded into every stochastic arm.
        relevant: The bed's declared answer key, or `None` on a bed without one. The two oracle reference
            arms exist only when it is given.

    Returns:
        An ordered mapping. Every builder returns a FRESH, unfitted arm.
    """
    kk = int(k) if k is not None else max(1, int(n_features) // 4)
    rs = int(random_state)
    roster: Dict[str, Callable[[], BaseArm]] = {}

    # The original 21, names unchanged.
    roster["all-features"] = lambda: AllFeaturesArm()
    roster[f"random-{kk}"] = lambda: RandomSelectionArm(k=kk, random_state=rs)
    roster["variance-sort"] = lambda: VarianceSortArm(k=kk)
    roster["univariate-mi"] = lambda: UnivariateMIArm(random_state=rs)
    roster["skb-f"] = lambda: SklearnScoreArm("kbest_f", k=kk, random_state=rs)
    roster["skb-mi"] = lambda: SklearnScoreArm("kbest_mi", k=kk, random_state=rs)
    roster["select-fdr"] = lambda: SklearnScoreArm("fdr_f", random_state=rs)
    roster["sfm-lgbm"] = lambda: SelectFromModelArm(random_state=rs)
    roster["lars-order"] = lambda: LarsPathArm(max_features=kk)
    roster["boruta"] = lambda: BorutaArm(random_state=rs)
    roster["ace"] = lambda: ACEArm(random_state=rs)
    roster["knockoffs"] = lambda: KnockoffArm(random_state=rs)
    roster["mrmr"] = lambda: MRMRArm(random_seed=rs)
    roster["rfecv"] = lambda: RFECVArm(random_state=rs)
    roster["boruta-shap"] = lambda: BorutaShapArm(random_state=rs)
    roster["shap-proxied"] = lambda: ShapProxiedArm(random_state=rs)

    from ._arms_byproduct import ByProductEnsembleArm
    from ._arms_external import CatBoostSelectArm, catboost_available
    from ._arms_rank_aggregation import RankAggregationArm

    roster["rank-vote"] = lambda: RankAggregationArm(k=kk, rule="borda", random_state=rs)
    roster["byproduct-ensemble"] = lambda: ByProductEnsembleArm(k=kk, random_state=rs)
    if catboost_available():
        # Three arms rather than one: collapsing the elimination criteria would report whichever happened
        # to be the default as "CatBoost", and a method's internal knobs are not a detail when they move
        # the result.
        for algorithm in ("shap", "loss", "predictions"):
            roster[f"catboost-{algorithm}"] = (lambda algo: lambda: CatBoostSelectArm(algorithm=algo, k=kk, random_state=rs))(algorithm)

    _add_information_family(roster, kk, rs)
    _add_wrappers(roster, int(n_features), kk, rs)
    _add_references(roster, kk, rs, relevant)
    return roster


def _add_information_family(roster: Dict[str, Callable[[], BaseArm]], kk: int, rs: int) -> None:
    """One arm per information-theoretic criterion, plus the MRMR class's own variants."""
    from ._arms_it_family import IT_SCORERS, GroupAwareMRMRArm, InformationGreedyArm, MRMRVariantArm, StabilityMRMRArm

    # The greedy order runs to 5k so the widest matched-K row has an order to read, not only the first k.
    for scorer in IT_SCORERS:
        roster[f"it-{scorer}"] = (lambda s: lambda: InformationGreedyArm(scorer=s, k=kk, max_order=5 * kk))(scorer)
    roster["mrmr-pld"] = lambda: MRMRVariantArm("mrmr-pld", {"mrmr_relevance_algo": "pld"}, random_seed=rs)
    roster["mrmr-relax"] = lambda: MRMRVariantArm("mrmr-relax", {"relaxmrmr_alpha": 1.0}, random_seed=rs)
    roster["mrmr-tree-rescued"] = lambda: MRMRVariantArm("mrmr-tree-rescued", {}, random_seed=rs, tree_rescued=True)
    roster["mrmr-grouped"] = lambda: GroupAwareMRMRArm(expand=False, random_seed=rs)
    roster["mrmr-grouped-expand"] = lambda: GroupAwareMRMRArm(expand=True, random_seed=rs)
    roster["mrmr-stability"] = lambda: StabilityMRMRArm(random_seed=rs)


def _add_wrappers(roster: Dict[str, Callable[[], BaseArm]], n_features: int, kk: int, rs: int) -> None:
    """Wrappers, stability, bandits, cascades and the registry-built RFECV and BorutaShap."""
    from ._arms_wrappers import (
        WRAPPER_MAX_WIDTH,
        BanditArm,
        BanditEnsembleArm,
        CascadeArm,
        CascadeStableArm,
        ForwardSelectArm,
        GreedyBackwardArm,
        HeteroVoteArm,
        NoiseFloorArm,
        NullImportanceArm,
        RegistryWrappedArm,
        RidgePrefilterArm,
        UnanimousPermutationArm,
        ZeroImportanceArm,
    )

    roster["rfecv-registry"] = lambda: RegistryWrappedArm("rfecv", random_state=rs)
    roster["boruta-shap-registry"] = lambda: RegistryWrappedArm("boruta_shap", random_state=rs)
    roster["noise-floor"] = lambda: NoiseFloorArm(random_state=rs)
    roster["unanimous-permutation"] = lambda: UnanimousPermutationArm(random_state=rs)
    roster["bandit"] = lambda: BanditArm(subset_size=kk, random_state=rs)
    roster["bandit-ensemble"] = lambda: BanditEnsembleArm(subset_size=kk, random_state=rs)
    roster["cascade"] = lambda: CascadeArm(forward_max_features=kk, random_state=rs)
    roster["cascade-stable"] = lambda: CascadeStableArm(forward_max_features=kk, random_state=rs)
    roster["hetero-vote"] = lambda: HeteroVoteArm(random_state=rs)
    roster["null-importance"] = lambda: NullImportanceArm(random_state=rs)
    roster["ridge-prefilter"] = lambda: RidgePrefilterArm(k=kk, random_state=rs)
    if n_features <= WRAPPER_MAX_WIDTH:
        # O(d^2) model fits: absent past the cap rather than present and timing out, which would read as a
        # reliability failure of the method instead of a decision about the tier.
        roster["forward-select"] = lambda: ForwardSelectArm(max_features=2 * kk, random_state=rs)
        roster["greedy-backward"] = lambda: GreedyBackwardArm(random_state=rs)
        roster["zero-importance"] = lambda: ZeroImportanceArm(random_state=rs)


def _add_references(roster: Dict[str, Callable[[], BaseArm]], kk: int, rs: int, relevant: Optional[Sequence[str]]) -> None:
    """The univariate filters, and the oracle pair when the bed declares an answer key."""
    from ._arms_baselines import (
        AllExceptInformativeArm,
        KSGArm,
        NearNoiseAucArm,
        OracleInformativeArm,
        PermutationTopKArm,
        RelevanceTableArm,
        UnsupervisedPrescreenArm,
    )

    roster["permutation-topk"] = lambda: PermutationTopKArm(k=kk, random_state=rs)
    roster["unsupervised-prescreen"] = lambda: UnsupervisedPrescreenArm()
    roster["near-noise-auc"] = lambda: NearNoiseAucArm(random_state=rs)
    roster["ksg-mi"] = lambda: KSGArm(k=kk, random_state=rs)
    roster["relevance-table"] = lambda: RelevanceTableArm(random_state=rs)
    if relevant:
        key = [str(c) for c in relevant]
        roster["oracle-informative"] = lambda: OracleInformativeArm(key)
        roster["all-except-informative"] = lambda: AllExceptInformativeArm(key)
