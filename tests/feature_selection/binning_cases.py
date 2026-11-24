"""Named binning cases for the tests that sweep several binnings: a case name maps to the ``binning_scheme`` / ``nbins_rule`` pair (MRMR keywords) and to the :class:`BinningSpec` that
``categorize_dataset`` takes. The names are test case ids only; the library has no such names."""

from __future__ import annotations

from mlframe.feature_selection.filters._binning_spec import BinningSpec, make_binning_spec

#: case name -> (binning_scheme, nbins_rule)
CASES = {
    "rice_capped": ("quantile", "rice_capped"),
    "sturges": ("quantile", "sturges"),
    "freedman_diaconis": ("quantile", "freedman_diaconis"),
    "uniform": ("uniform", "freedman_diaconis"),
    "knuth": ("uniform", "knuth"),
    "fixed": ("quantile", "fixed"),
    "mdlp": ("mdlp", None),
    "mdlp_validated": ("mdlp_validated", None),
    "optimal_joint": ("optimal_joint", None),
    "mah": ("mah", None),
    "bayesian_blocks": ("bayesian_blocks", None),
}


def binning_kwargs(case: "str | None") -> dict:
    """The MRMR keyword arguments of a case (``None`` is the fixed-count discretiser)."""
    scheme, rule = CASES["fixed" if case is None else case]
    out = {"binning_scheme": scheme}
    if rule is not None:
        out["nbins_rule"] = rule
    return out


def binning_spec(case: "str | None") -> BinningSpec:
    """The :class:`BinningSpec` of a case, for ``categorize_dataset(binning=...)``."""
    scheme, rule = CASES["fixed" if case is None else case]
    return make_binning_spec(scheme, rule, warn=False)


def edges_rice_capped(x, alpha: float = 0.30, cap: int = 48):
    """Inner edges of one column: equal-frequency bins with the ``rice_capped`` count."""
    from mlframe.feature_selection.filters._adaptive_nbins import edges_partition

    return edges_partition(x, "quantile", "rice_capped", cap=cap, alpha=alpha)


def edges_sturges(x, base: str = "quantile"):
    """Inner edges of one column: the Sturges count on the ``base`` partition."""
    from mlframe.feature_selection.filters._adaptive_nbins import edges_partition

    return edges_partition(x, base, "sturges")


def edges_call(case: str) -> dict:
    """The keyword arguments of ``per_feature_edges`` for a case: ``method`` plus, for the partition schemes, the partition, the count rule and the cap."""
    scheme, rule = CASES[case]
    spec = make_binning_spec(scheme, rule, warn=False)
    return {"method": spec.edge_method, **spec.edge_kwargs()}
