"""The names of the ``nbins_strategy`` values: ONE name per strategy, no aliases.

``mdlp`` (the minimum-description-length criterion of Fayyad and Irani 1993; the MRMR default) and its significance-checked twin ``mdlp_validated``; ``optimal_joint`` (the bin edges that maximise the
joint information of the column with the target, scored on held-out folds); ``freedman_diaconis`` (the Freedman-Diaconis width rule); ``sturges``; ``quantile_spacing`` (Gupta et al. 2021);
``uniform``; ``mah`` (Marx 2021 adaptive histogram); and the research-only ``knuth`` and ``bayesian_blocks``.

Each name was chosen over its former synonyms (``fayyad_irani``, ``cv``, ``fd``, ``auto``, ``qs``, ``quantile``, ``gupta``, ``blocks``, ``bb`` and the hyphenated spellings) because it says what the strategy
IS, is spelled out in full, and cannot be read as another option (``quantile`` is also a ``quantization_method``; ``cv`` is a cross-validation argument; ``auto`` hid which rule ran). The former names are
rejected with a message that names the replacement (:func:`unknown_nbins_strategy_message`).
"""

from __future__ import annotations

#: Every accepted strategy name (``None`` is also accepted by ``MRMR`` and means fixed ``quantization_nbins`` bins).
NBINS_STRATEGIES = ("sturges", "freedman_diaconis", "quantile_spacing", "uniform", "knuth", "bayesian_blocks", "mdlp", "mdlp_validated", "optimal_joint", "mah")
#: The strategies that fit their cut points on the target.
SUPERVISED_NBINS_STRATEGIES = ("mdlp", "mdlp_validated", "optimal_joint", "mah")
#: The strategies of the MDLP family (the honest null of these re-runs the tree on a permuted target).
MDLP_NBINS_STRATEGIES = ("mdlp", "mdlp_validated")
#: Strategies that stay selectable for research and reproduction and warn.
DEMOTED_NBINS_STRATEGIES = ("knuth", "bayesian_blocks")
#: Names that existed before and are gone, with the name that replaces each.
REMOVED_NBINS_STRATEGY_NAMES = {
    "auto": "freedman_diaconis",
    "fd": "freedman_diaconis",
    "freedman-diaconis": "freedman_diaconis",
    "fayyad_irani": "mdlp",
    "fayyad-irani": "mdlp",
    "fayyad_irani_validated": "mdlp_validated",
    "fayyad-irani-validated": "mdlp_validated",
    "cv": "optimal_joint",
    "qs": "quantile_spacing",
    "quantile": "quantile_spacing",
    "gupta": "quantile_spacing",
    "blocks": "bayesian_blocks",
    "bb": "bayesian_blocks",
}


def unknown_nbins_strategy_message(name: object) -> str:
    """The error text for a strategy name that is not accepted: the replacement when it is a removed name, else the list of accepted ones."""
    if isinstance(name, str) and name in REMOVED_NBINS_STRATEGY_NAMES:
        return f"nbins_strategy={name!r} no longer exists; use {REMOVED_NBINS_STRATEGY_NAMES[name]!r}"
    return f"unknown nbins_strategy={name!r}; expected one of {list(NBINS_STRATEGIES)} (or None)"
