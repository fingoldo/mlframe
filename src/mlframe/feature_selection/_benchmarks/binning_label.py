"""Short labels of binning setups for the benchmark scripts and their environment variables (``BENCH_NBINS`` and the like), mapped to the MRMR keywords ``binning_scheme`` / ``nbins_rule``.

The library has no such labels: MRMR takes ``binning_scheme``, ``nbins_rule`` and ``nbins_cap``. A label here is a command-line convenience only."""

from __future__ import annotations

#: label -> (binning_scheme, nbins_rule)
BINNING_LABELS = {
    "quantile": ("quantile", None),
    "rice_capped": ("quantile", "rice_capped"),
    "sturges": ("quantile", "sturges"),
    "freedman_diaconis": ("quantile", "freedman_diaconis"),
    "uniform": ("uniform", None),
    "knuth": ("uniform", "knuth"),
    "fixed": ("quantile", "fixed"),
    "mdlp": ("mdlp", None),
    "mdlp_validated": ("mdlp_validated", None),
    "optimal_joint": ("optimal_joint", None),
    "mah": ("mah", None),
    "bayesian_blocks": ("bayesian_blocks", None),
}


def binning_params(label: str, nbins_cap: "int | None" = None) -> dict:
    """The MRMR keyword arguments of a binning label; ``nbins_cap`` is added when given and the scheme counts its own bins through a rule."""
    if label not in BINNING_LABELS:
        raise ValueError(f"unknown binning label {label!r}; expected one of {sorted(BINNING_LABELS)}")
    scheme, rule = BINNING_LABELS[label]
    out: dict = {"binning_scheme": scheme}
    if rule is not None:
        out["nbins_rule"] = rule
    if nbins_cap is not None:
        out["nbins_cap"] = int(nbins_cap)
    return out


def edge_params(label: str) -> dict:
    """The keyword arguments of ``per_feature_edges`` for a binning label: ``method`` plus, for the partition schemes, the partition, the count rule and the cap."""
    from mlframe.feature_selection.filters._binning_spec import make_binning_spec

    scheme, rule = BINNING_LABELS[label]
    spec = make_binning_spec(scheme, rule, warn=False)
    return {"method": spec.edge_method, **spec.edge_kwargs()}
