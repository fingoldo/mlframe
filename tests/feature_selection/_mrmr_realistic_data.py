"""Realistic, RANDOM data generator for the MRMR end-to-end invariant layer.

WHY THIS EXISTS
---------------
The existing MRMR/FE suite is dense in *component* tests on CLEAN, DESIGNED
fixtures (a single hand-built target term, a fixed seed, a known winning recipe).
Every one of the four 2026-06 production bugs (BUG1 raw-redundancy, BUG2 recipe
survival, BUG3 poly escalation, prewarp slice replay) was an END-TO-END CONTRACT
violation that a clean-fixture unit test structurally could not see -- it only
showed up on the user's first *real* task: uniform[0,1] features, a HIDDEN
confounder term in ``y`` (a variable that drives the target but is NOT a column
of ``X``), a pure-noise feature, a MULTI-TERM ADDITIVE target, ``fe_max_steps=2``.

This module produces that *class* of data -- not one fixture -- parametrised over
distributions, target families, and a confounder, so the invariant layer can
fuzz the contracts the bugs violated.

KEY DESIGN PROPERTIES (each maps to a bug the layer must catch)
  * additive multi-term targets ``y = sum_i term_i + confounder + noise`` -- so a
    raw operand can be FULLY captured by an engineered term (BUG1) OR carry a
    genuine PRIVATE additive term (the must-not-over-drop converse of BUG1).
  * a hidden confounder ``f`` (in ``y``, never in ``X``) -- irreducible residual,
    exactly the user's repro, the thing clean fixtures omit.
  * a pure-noise feature ``e`` (in ``X``, never in ``y``) -- a relevance
    true-negative that must never be selected.
  * varied operand distributions (uniform / normal / lognormal / heavy-tail) so
    binning / prewarp / replay are exercised off the tidy uniform[0,1] grid.
  * both regression and classification targets.

Every generator returns ``(df, y, meta)`` where ``meta`` carries the structural
ground truth the invariants assert against (which raws are subsumed, which raws
are private, the confounder/noise names, the task).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# operand distributions -- a raw feature is a draw from one of these. Names map
# 1:1 to the ``distribution`` axis the fuzzer parametrises over.
# ---------------------------------------------------------------------------
def _draw(rng: np.random.Generator, kind: str, n: int) -> np.ndarray:
    """Helper that draw."""
    if kind == "uniform":
        return rng.uniform(0.2, 1.2, n)  # strictly positive -> log/div safe
    if kind == "uniform_signed":
        return rng.uniform(-2.5, 2.5, n)
    if kind == "normal":
        return rng.normal(0.0, 1.0, n)
    if kind == "lognormal":
        return rng.lognormal(0.0, 0.6, n)
    if kind == "heavytail":  # Student-t df=3 -> heavy tails
        return rng.standard_t(3, n)
    raise ValueError(f"unknown distribution {kind!r}")


def _positive(x: np.ndarray) -> np.ndarray:
    """Map any real array into a strictly-positive range for log/div operands
    without collapsing its rank structure (shift + small floor)."""
    lo = np.nanmin(x)
    return (x - lo) + 0.5


@dataclass
class CaseMeta:
    """Groups tests covering CaseMeta."""
    task: str  # "regression" | "classification"
    feature_names: list  # columns of X, in order
    confounder_name: str  # hidden var in y, NOT a column
    noise_feature: str  # pure-noise column in X, NOT in y
    subsumed_raws: list  # raws FULLY captured by an engineered term -> must be droppable
    private_raws: list  # raws with a genuine PRIVATE additive term -> must be kept
    recoverable: bool  # True if FE can recover genuine engineered structure (I5)
    distribution: str
    target_family: str
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# target families -- multi-term additive structures. Each builds a continuous
# signal from a frame of raw operands + a hidden confounder draw, and reports
# which raws are SUBSUMED (appear only inside one fused engineered term and
# carry no private additive term) vs PRIVATE (also enter y additively alone).
# ---------------------------------------------------------------------------
def _family_ratio_plus_trig(cols: dict, conf: np.ndarray):
    """y = w*a**2/b + g/k + log(c)*sin(d) + confounder.

    The user's CASE family. ``a`` is SUBSUMED: it appears only inside the fused
    composite term ``a**2/b`` (no private additive ``a``). ``c``/``d`` enter only
    via ``log(c)*sin(d)``. This is the exact shape of BUG1 (raw ``a`` wrongly
    kept) and BUG3 (the (a,b) ratio interaction the synergy gate blocked).
    """
    a, b, c, d = cols["a"], cols["b"], cols["c"], cols["d"]
    k = cols["k"]
    g = cols["g"]
    ap, bp, cp, kp = _positive(a), _positive(b) + 0.3, _positive(c), _positive(k) + 0.3
    sig = 0.6 * (ap**2) / bp + 0.5 * _positive(g) / kp + np.log(cp) * np.sin(d)
    sig = sig + conf
    return sig, ["a", "b", "c", "d", "g", "k"], []  # all subsumed, none private


def _family_subsumed_plus_private(cols: dict, conf: np.ndarray):
    """y = a**2/b + 3*a + log(c)*sin(d) + confounder.

    The CONVERSE control for BUG1: ``a`` enters BOTH the fused ``a**2/b`` term AND
    a genuine PRIVATE additive ``3*a`` -> ``a`` carries an independent residual and
    MUST be kept. ``b``/``c``/``d`` are subsumed.
    """
    a, b, c, d = cols["a"], cols["b"], cols["c"], cols["d"]
    ap, bp, cp = _positive(a), _positive(b) + 0.3, _positive(c)
    sig = 0.6 * (ap**2) / bp + 3.0 * a + np.log(cp) * np.sin(d)
    sig = sig + conf
    return sig, ["b", "c", "d"], ["a"]


def _family_smooth_interaction(cols: dict, conf: np.ndarray):
    """y = w*a*b + g/k + confounder. A clean bilinear product + a separate ratio
    additive term -- the smooth low-raw-MI interaction the escalation must route
    (BUG3 prevalence-gate territory)."""
    a, b, g, k = cols["a"], cols["b"], cols["g"], cols["k"]
    kp = _positive(k) + 0.3
    sig = 1.5 * a * b + 0.5 * _positive(g) / kp + conf
    return sig, ["a", "b", "g", "k"], []


_FAMILIES: dict[str, Callable] = {
    "ratio_plus_trig": _family_ratio_plus_trig,
    "subsumed_plus_private": _family_subsumed_plus_private,
    "smooth_interaction": _family_smooth_interaction,
}


# raws each family CONSUMES (must exist as columns of X).
_FAMILY_OPERANDS = {
    "ratio_plus_trig": ["a", "b", "c", "d", "g", "k"],
    "subsumed_plus_private": ["a", "b", "c", "d"],
    "smooth_interaction": ["a", "b", "g", "k"],
}


def make_realistic_case(
    seed: int,
    n: int = 8000,
    distribution: str = "uniform",
    target_family: str = "ratio_plus_trig",
    task: str = "regression",
    noise_scale: float = 0.05,
) -> tuple[pd.DataFrame, pd.Series, CaseMeta]:
    """Build ONE realistic random case.

    The frame always contains a PURE-NOISE column ``e`` (in X, not in y); ``y``
    always contains a HIDDEN CONFOUNDER ``f`` (in y, not in X). The operand
    columns the chosen ``target_family`` needs are drawn from ``distribution``;
    any leftover canonical columns are added as extra noise features so the
    frame width / config noise stays realistic.
    """
    rng = np.random.default_rng(seed)
    operands = _FAMILY_OPERANDS[target_family]

    cols: dict[str, np.ndarray] = {}
    for name in operands:
        cols[name] = _draw(rng, distribution, n)

    # hidden confounder f -- drives y, never a column of X.
    conf_scale = 0.4
    f = conf_scale * _draw(rng, "normal", n)

    family = _FAMILIES[target_family]
    sig, subsumed, private = family(cols, f)

    # additive observation noise on top of the additive signal.
    sig = np.asarray(sig, dtype=float)
    sd = np.nanstd(sig)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    sig = sig + rng.normal(0.0, noise_scale * sd, n)

    # assemble X: the operands + a pure-noise feature e.
    data = {name: cols[name] for name in operands}
    data["e"] = _draw(rng, distribution, n)  # pure noise, IN X
    df = pd.DataFrame(data)

    if task == "classification":
        # median split -> balanced binary target; the additive structure is
        # preserved through the threshold so FE can still recover it via AUC.
        thresh = np.nanmedian(sig)
        y = pd.Series((sig > thresh).astype(int), name="y")
    else:
        y = pd.Series(sig, name="y")

    meta = CaseMeta(
        task=task,
        feature_names=list(df.columns),
        confounder_name="f",
        noise_feature="e",
        subsumed_raws=subsumed,
        private_raws=private,
        recoverable=True,
        distribution=distribution,
        target_family=target_family,
    )
    return df, y, meta


# A compact, representative grid for the fuzzer. Kept small enough that each case
# can be fit ISOLATED in its own subprocess within the RAM/time budget, yet wide
# enough to vary every axis the four bugs lived on.
def default_fuzz_grid() -> list[dict]:
    """Default fuzz grid."""
    grid = []
    base_seeds = [101, 202, 303]
    distros = ["uniform", "normal", "lognormal", "heavytail"]
    families = ["ratio_plus_trig", "subsumed_plus_private", "smooth_interaction"]
    # primary sweep: every family x a rotating distribution x a fresh seed.
    for i, fam in enumerate(families):
        for j, seed in enumerate(base_seeds):
            grid.append(
                dict(
                    seed=seed + 7 * i + j,
                    distribution=distros[(i + j) % len(distros)],
                    target_family=fam,
                    task="regression",
                )
            )
    # classification coverage on the two recoverable families.
    grid.append(dict(seed=909, distribution="uniform", target_family="ratio_plus_trig", task="classification"))
    grid.append(dict(seed=910, distribution="normal", target_family="smooth_interaction", task="classification"))
    return grid
