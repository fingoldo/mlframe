"""How MRMR bins a column, as two independent choices plus a cap: the PARTITION (``binning_scheme``) and the COUNT RULE (``nbins_rule``).

The former single ``nbins_strategy`` mixed the two (``quantile_spacing`` named the partition into equal-frequency bins but hid the rule that picks how many; ``sturges`` and ``freedman_diaconis`` were count
rules; ``uniform`` was a partition with a hidden count rule). Here they are separate, and there are no aliases and no compatibility with the old names.

``binning_scheme`` (WHERE the cut points go):
  * ``quantile``          equal-frequency bins; the count comes from ``nbins_rule``.
  * ``uniform``           equal-width bins; the count comes from ``nbins_rule``.
  * ``mdlp``              supervised minimum-description-length cuts of Fayyad and Irani 1993, with a significance check on the splits; picks its own count.
  * ``mdlp_validated``    the held-out-validated twin of ``mdlp``; picks its own count.
  * ``optimal_joint``     supervised: the count among a few candidates that maximises the joint information with the target on held-out folds.
  * ``mah``               supervised adaptive histogram (Marx 2021); picks its own count.
  * ``bayesian_blocks``   research only (warns): Bayesian blocks of Scargle; picks its own count.

``nbins_rule`` (HOW MANY bins; only for the ``quantile`` and ``uniform`` schemes):
  * ``rice_capped``       ``min(round(0.3 n), round(2 n^(1/3)), nbins_cap)`` bins, at least 3 (Gupta et al. 2021 for small n, the Rice rule for larger n).
  * ``sturges``           ``ceil(1 + log2 n)`` bins, at most ``nbins_cap``.
  * ``freedman_diaconis`` the Freedman-Diaconis width rule, at most ``nbins_cap`` bins.
  * ``knuth``             Knuth's Bayesian optimum, at most ``nbins_cap`` bins; research only (warns).
  * ``fixed``             exactly ``quantization_nbins`` bins.
``None`` (the default) takes the scheme's own rule: ``rice_capped`` for ``quantile``, ``freedman_diaconis`` for ``uniform``. A rule given with a scheme that picks its own count is an error.

``nbins_cap`` bounds the count of every rule above except ``fixed`` (default 48: measured 2026-11-23, 100000 x 100 with every family wall 50.3 s at 64 against 37.1 s at 48; out-of-fold 80000 x 50 binary AUC
0.9491 at 64, 0.9507 at 48, 0.9305 at 32 where a fold failed).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping

#: Every accepted ``binning_scheme``.
BINNING_SCHEMES = ("quantile", "uniform", "mdlp", "mdlp_validated", "optimal_joint", "mah", "bayesian_blocks")
#: The schemes whose count comes from ``nbins_rule``.
PARTITION_SCHEMES = ("quantile", "uniform")
#: The schemes that fit their cut points on the target.
SUPERVISED_SCHEMES = ("mdlp", "mdlp_validated", "optimal_joint", "mah")
#: The MDLP family (the honest null of these re-runs the tree on a permuted target).
MDLP_SCHEMES = ("mdlp", "mdlp_validated")
#: Schemes and rules that stay selectable for research and reproduction, and warn.
DEMOTED_SCHEMES = ("bayesian_blocks",)
#: Every accepted ``nbins_rule``.
NBINS_RULES = ("rice_capped", "sturges", "freedman_diaconis", "knuth", "fixed")
DEMOTED_RULES = ("knuth",)
#: The rule a partition scheme takes when ``nbins_rule`` is ``None``.
DEFAULT_RULE_OF_SCHEME = {"quantile": "rice_capped", "uniform": "freedman_diaconis"}
#: Default of ``nbins_cap``.
DEFAULT_NBINS_CAP = 48
#: Smallest bin count a rule may return for a column that has enough distinct values.
MIN_NBINS = 3


@dataclass(frozen=True)
class BinningSpec:
    """A validated binning choice: the partition, the (resolved) count rule, the cap and the fixed count.

    ``rule`` is ``None`` for a scheme that picks its own count. Built by :func:`make_binning_spec`; never constructed with unchecked values."""

    scheme: str
    rule: "str | None"
    cap: int
    fixed_nbins: int

    @property
    def supervised(self) -> bool:
        """Whether the cut points are fitted on the target."""
        return self.scheme in SUPERVISED_SCHEMES

    @property
    def mdlp_family(self) -> bool:
        """Whether this is ``mdlp`` or ``mdlp_validated``."""
        return self.scheme in MDLP_SCHEMES

    @property
    def is_fixed(self) -> bool:
        """Whether every column gets ``fixed_nbins`` bins (the unsupervised discretiser path, not the per-column edge dispatcher)."""
        return self.rule == "fixed"

    @property
    def edge_method(self) -> str:
        """The name the per-column edge dispatcher knows: ``partition`` for the quantile / uniform schemes, else the scheme."""
        return "partition" if self.scheme in PARTITION_SCHEMES else self.scheme

    def edge_kwargs(self) -> dict:
        """The keyword arguments the ``partition`` edge builder reads (empty for the schemes that pick their own count)."""
        if self.scheme not in PARTITION_SCHEMES:
            return {}
        return {"partition": self.scheme, "count_rule": self.rule, "nbins_cap": int(self.cap), "fixed_nbins": int(self.fixed_nbins)}

    def describe(self) -> str:
        """A short label for logs and ``column_stats_`` metadata, e.g. ``quantile/rice_capped(cap 48)``."""
        if self.rule is None:
            return self.scheme
        return f"{self.scheme}/{self.rule}" + ("" if self.rule == "fixed" else f"(cap {self.cap})")


def make_binning_spec(binning_scheme: str = "quantile", nbins_rule: "str | None" = None, nbins_cap: int = DEFAULT_NBINS_CAP, quantization_nbins: int = 10, *, warn: bool = True) -> BinningSpec:
    """Validate the three constructor values and the fixed count and return the resolved :class:`BinningSpec`.

    Raises ``ValueError`` for an unknown scheme or rule, a rule given to a scheme that picks its own count, and a cap below 3. A research-only scheme or rule warns once."""
    if binning_scheme not in BINNING_SCHEMES:
        raise ValueError(f"unknown binning_scheme={binning_scheme!r}; expected one of {list(BINNING_SCHEMES)}")
    if nbins_rule is not None and nbins_rule not in NBINS_RULES:
        raise ValueError(f"unknown nbins_rule={nbins_rule!r}; expected one of {list(NBINS_RULES)} (or None for the scheme's own rule)")
    if isinstance(nbins_cap, bool) or not isinstance(nbins_cap, (int,)) or int(nbins_cap) < MIN_NBINS:
        raise ValueError(f"nbins_cap must be an integer >= {MIN_NBINS}, got {nbins_cap!r}")
    if binning_scheme in PARTITION_SCHEMES:
        rule: "str | None" = nbins_rule if nbins_rule is not None else DEFAULT_RULE_OF_SCHEME[binning_scheme]
    else:
        if nbins_rule is not None:
            raise ValueError(f"nbins_rule={nbins_rule!r} does not apply to binning_scheme={binning_scheme!r}, which picks its own number of bins; leave nbins_rule as None")
        rule = None
    if warn:
        if binning_scheme in DEMOTED_SCHEMES or rule in DEMOTED_RULES:
            warnings.warn(f"MRMR: binning_scheme={binning_scheme!r} / nbins_rule={rule!r} is DEMOTED to research-only: kept for research and reproduction, not a recommended setting (the recommended ones are the default quantile scheme and mdlp)", stacklevel=3)
    return BinningSpec(scheme=binning_scheme, rule=rule, cap=int(nbins_cap), fixed_nbins=int(quantization_nbins))


def spec_from_estimator(est: Any, *, warn: bool = False) -> BinningSpec:
    """The :class:`BinningSpec` of an MRMR-like object from its ``binning_scheme``, ``nbins_rule``, ``nbins_cap`` and ``quantization_nbins`` attributes."""
    return make_binning_spec(
        getattr(est, "binning_scheme", "quantile"), getattr(est, "nbins_rule", None), getattr(est, "nbins_cap", DEFAULT_NBINS_CAP), getattr(est, "quantization_nbins", 10), warn=warn,
    )


def spec_as_dict(spec: BinningSpec) -> Mapping[str, Any]:
    """The spec as plain values, for ``column_stats_`` / ``fit_meta_``."""
    return {"binning_scheme": spec.scheme, "nbins_rule": spec.rule, "nbins_cap": spec.cap, "fixed_nbins": spec.fixed_nbins}
