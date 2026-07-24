"""Fit-time warnings for parameter values we KNOW degrade selection accuracy.

Many MRMR knobs accept VALID values that nonetheless make the selection measurably worse -- usually a
``*_enable=False`` that opts OUT of an on-by-default accuracy mechanism "for byte-identical legacy /
replay", or a numeric knob pinned to a documented-bad setting. Those are legitimate (replay parity,
ablation, speed) but a user who set one WITHOUT that intent gets silently worse features.

This module centralises the known-bad values in one registry (``ACCURACY_SUBOPTIMAL``) and emits a
SINGLE consolidated ``UserWarning`` at fit time listing every triggered one with its accuracy cost and
how to restore the better behaviour. It fires only on the explicitly-bad value (every default is the
GOOD value, so a default-config fit is silent), and at most once per fitted estimator.

The same registry is the single source of truth for the unified ``# [ACCURACY-CAVEAT]`` markers in the
constructor signature: each flagged parameter's docstring/inline note points here so the "which values
are bad" knowledge lives in ONE place, not scattered across the 3000-line constructor.
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple

import warnings


class _Caveat(NamedTuple):
    """One accuracy-affecting parameter check: which attr, what makes it "bad", and the cost/restore messages to warn with."""

    attr: str
    is_bad: Callable[[Any], bool]
    cost: str  # what accuracy is lost
    restore: str  # how to get the better behaviour back


def _eq(target: Any) -> Callable[[Any], bool]:
    """Return a predicate that flags a value as "bad" when it equals ``target``."""
    return lambda v: v == target


# The known accuracy-degrading values. Every entry's trigger differs from the parameter's DEFAULT, so a
# default-config fit triggers nothing. Keep this list in sync with the constructor's [ACCURACY-CAVEAT]
# markers. Ordered roughly by impact.
ACCURACY_SUBOPTIMAL: list[_Caveat] = [
    _Caveat(
        "fe_discrete_structural_operators_enable", _eq(False),
        "disables ALL FOUR discrete-structural FE families at once (conditional-gate / pairwise-modular "
        "/ row-argmax / binned-agg) -- modular/parity/regime-switch/threshold interactions smooth bases "
        "cannot fit are never engineered",
        "leave it True (default); opt out only for byte-identical legacy replay",
    ),
    _Caveat(
        "dcd_enable",
        _eq(False),
        "disables denoised cluster-aggregate (DCD) -- redundant near-duplicate column clusters are no " "longer collapsed into a denoised representative",
        "leave it True (default)",
    ),
    _Caveat(
        "fe_accuracy_gate",
        _eq(False),
        "disables the FE accuracy gate -- engineered candidates that do not clear an honest " "relevance/usability bar can enter the selection",
        "leave it True (default)",
    ),
    _Caveat(
        "fe_pairwise_modular_enable", _eq(False),
        "no (a+b) mod m / (a*b) mod m / n-way parity / hidden-period interactions are engineered",
        "leave it True (default); opt out only for byte-identical legacy replay",
    ),
    _Caveat(
        "fe_row_argmax_enable", _eq(False),
        "no row-argmax (which-column-is-largest) categorical interaction is engineered",
        "leave it True (default); opt out only for byte-identical legacy replay",
    ),
    _Caveat(
        "fe_conditional_gate_enable", _eq(False),
        "no conditional-gate (regime switch c>tau ? a : b / masked interaction) feature is engineered",
        "leave it True (default); opt out only for byte-identical legacy replay",
    ),
    _Caveat(
        "fe_confirm_undersample_rows_per_cell", _eq(0.0),
        "forces the STRICT conditional confirmation test everywhere (legacy path) -- on an undersampled "
        "REAL joint at small n the strict test wrongly rejects a genuinely-significant candidate, "
        "collapsing the selection (measured: diabetes n=442 under-selected)",
        "use the default 5.0 (sample-size-aware confirmation)",
    ),
    _Caveat(
        "min_features_fallback",
        _eq(0),
        "removes the never-empty floor -- a fit whose gates reject every candidate returns an EMPTY " "support_ instead of the single strongest column",
        "use the default 1 (or higher) so support_ is never empty",
    ),
    _Caveat(
        "quantization_nbins",
        lambda v: isinstance(v, int) and v < 5,
        "very coarse discretisation -- the plug-in MI estimator loses resolution and both relevance and " "redundancy are estimated poorly",
        "use >= 8 (default 10) unless n is tiny",
    ),
]


def warn_accuracy_suboptimal_params(estimator: Any) -> None:
    """Emit ONE consolidated UserWarning listing every accuracy-degrading parameter value set on
    ``estimator``. Silent on a default config. Never raises -- a missing attribute is simply skipped.

    USABILITY_A-14 fix (mrmr_audit_2026-07-22): the guard used to be a plain one-shot latch
    (``_accuracy_caveats_warned_``), so a ``set_params()`` call that degraded a param AFTER the first
    fit (e.g. flipping ``dcd_enable`` from True to False between two ``fit()`` calls on the same
    instance) never re-fired the warning. Now the latch stores WHICH attrs last triggered, and re-warns
    whenever the current triggered set differs from that -- including growing, shrinking, or changing."""
    triggered = []
    for c in ACCURACY_SUBOPTIMAL:
        try:
            if not hasattr(estimator, c.attr):
                continue
            val = getattr(estimator, c.attr)
            if c.is_bad(val):
                triggered.append((c, val))
        except Exception:  # nosec B112 - best-effort path (hasattr/getattr on a raising property must not propagate)
            continue
    triggered_attrs = frozenset(c.attr for c, _val in triggered)
    if triggered_attrs == getattr(estimator, "_accuracy_caveats_warned_attrs_", frozenset()):
        return
    estimator._accuracy_caveats_warned_attrs_ = triggered_attrs
    if not triggered:
        return
    lines = [f"  * {c.attr}={val!r}: {c.cost}. Better: {c.restore}." for c, val in triggered]
    warnings.warn(
        "MRMR.fit: %d parameter value(s) are valid but known to DEGRADE selection accuracy:\n%s\n"
        "(These are intentional only for byte-identical legacy replay / ablation / speed; otherwise "
        "prefer the defaults. Silence by restoring the defaults above.)" % (len(triggered), "\n".join(lines)),
        UserWarning,
        stacklevel=3,
    )


__all__ = ["ACCURACY_SUBOPTIMAL", "warn_accuracy_suboptimal_params", "_Caveat"]
