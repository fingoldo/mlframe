"""Pass/fail contracts of the four-pillar bench, shared by the bench and its tests.

A contract maps one fitted selector (and the case that produced the data) to a verdict dict. A panel of seeds is judged by one-sided Clopper-Pearson bounds, so a single lucky
seed never counts: power is claimed only when the lower bound of the success rate clears its gate, a false-positive rate only when the upper bound of the rate stays under its
ceiling. Names are matched by exact column reference, not by a whole-identifier regex, so ``a_dup`` is not mistaken for ``a``.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.stats import beta, spearmanr


def cp_upper(k: int, m: int, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound of a rate observed as ``k`` events in ``m`` trials."""
    return 1.0 if k >= m else float(beta.ppf(conf, k + 1, m - k))


def cp_lower(k: int, m: int, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson lower bound of a rate observed as ``k`` events in ``m`` trials."""
    return 0.0 if k == 0 else float(beta.ppf(1 - conf, k, m - k + 1))


def refs(name: object, cols: Iterable[str]) -> set:
    """Columns of ``cols`` that ``name`` references as a whole identifier (``a`` is not found inside ``a_dup`` or ``sqrt``)."""
    return {c for c in cols if re.search(rf"(?<![A-Za-z0-9_]){re.escape(c)}(?![A-Za-z0-9_])", str(name))}


def selected_names(sel) -> list:
    """Output feature names of a fitted selector as strings."""
    return [str(s) for s in sel.get_feature_names_out()]


def f2_verdict(sel, gen: Callable, seed: int, noise_cols: Sequence[str] = ("e",), n: int = 3000, min_rho: float = 0.90, max_cols: int = 3) -> dict:
    """F2 contract on a fresh draw: the selected columns must rank the noise-free signal as well as the oracle does, reference no noise column and stay within ``max_cols``.

    ``gen(seed, n)`` returns ``(X, y, truth)``. The linear read-out is fitted on the fit-seed draw and scored by Spearman correlation with ``truth`` on an independent draw, so a column
    of the wrong functional form that merely mentions the right operands fails.
    """
    Xa, ya, ta = gen(seed + 10_000, n)
    Xt, yt, _ = gen(seed, n)
    names = selected_names(sel)
    Ft = np.asarray(sel.transform(Xt), dtype=np.float64)
    Fa = np.asarray(sel.transform(Xa), dtype=np.float64)
    if Ft.ndim == 1:
        Ft, Fa = Ft[:, None], Fa[:, None]
    ok = np.isfinite(Ft).all(1)
    w = np.linalg.lstsq(np.c_[np.ones(ok.sum()), Ft[ok]], np.asarray(yt, dtype=np.float64)[ok], rcond=None)[0]
    oka = np.isfinite(Fa).all(1)
    pred = np.c_[np.ones(oka.sum()), Fa[oka]] @ w
    rho = float(spearmanr(pred, np.asarray(ta)[oka]).correlation)
    oracle = float(spearmanr(np.asarray(ya)[oka], np.asarray(ta)[oka]).correlation)
    bad_noise = [m for m in names if refs(m, noise_cols)]
    ok_all = bool(rho >= min_rho and rho >= 0.95 * oracle and not bad_noise and len(names) <= max_cols)
    return {"ok": ok_all, "rho": rho, "oracle": oracle, "noise": bad_noise, "n_forbidden": len(bad_noise), "n_sel": len(names)}


def null_verdict(sel) -> dict:
    """Pure-noise contract: nothing is selected (the flag alone is not enough, the output must be empty or the single fallback column of a degenerate fit)."""
    names = selected_names(sel)
    return {"ok": len(names) == 0, "selected": names, "n_forbidden": len(names), "n_sel": len(names)}


def cover_verdict(
    sel, required: Sequence[Sequence[str]], forbidden_prefixes: Sequence[str] = ("noise_",), max_cols: int = 4, forbidden_exact: Sequence[str] = (), normalize: Sequence[str] = ()
) -> dict:
    """Interaction contract by token coverage.

    ``required`` is a list of operand groups. Each group must be covered by at least one selected feature (a feature covers a group when it references every operand of it). No selected
    feature may reference a forbidden column (a name starting with one of ``forbidden_prefixes`` or listed in ``forbidden_exact``), and at most ``max_cols`` features are selected.
    Each suffix of ``normalize`` is stripped from a token first, so a monotone twin such as ``x1_exp`` counts as ``x1``.
    """
    names = selected_names(sel)
    toks = {m: {_strip(t, normalize) for t in _identifiers(m)} for m in names}
    bad = [m for m in names if any(t.startswith(tuple(forbidden_prefixes)) or t in forbidden_exact for t in _identifiers(m))]
    covered = [any(set(group) <= toks[m] for m in names) for group in required]
    ok = bool(all(covered) and not bad and len(names) <= max_cols)
    return {"ok": ok, "covered": covered, "forbidden": bad, "n_forbidden": len(bad), "n_sel": len(names), "tokens": sorted(set().union(*toks.values())) if toks else []}


def _strip(token: str, suffixes: Sequence[str]) -> str:
    """``token`` without the first matching suffix of ``suffixes``."""
    for suffix in suffixes:
        if token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _identifiers(name: object) -> list:
    """Identifier tokens of an engineered feature name (``mul(x1,x2)`` gives ``mul``, ``x1``, ``x2``; ``x1*x2__T1_T1`` gives ``x1``, ``x2``, ``T1``)."""
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(name))


def panel(rows: Sequence[dict], power_gate: float = 0.90, fp_gate: float = 0.10) -> dict:
    """Panel verdict over per-seed verdict dicts: power lower bound against ``power_gate``, false-positive upper bound against ``fp_gate``.

    ``PASS`` needs both. ``INCONCLUSIVE`` means the observed success rate reaches the gate but the panel is too small to claim it. Everything else is ``FAIL``.
    """
    m = len(rows)
    s = sum(bool(r["ok"]) for r in rows)
    f = sum(int(r.get("n_forbidden", 0)) > 0 for r in rows)
    lo, hi = cp_lower(s, m), cp_upper(f, m)
    if lo >= power_gate and hi <= fp_gate:
        verdict = "PASS"
    elif m and s / m >= power_gate and f / m <= fp_gate:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL"
    return {"successes": s, "false_positives": f, "seeds": m, "power_lower": lo, "fp_upper": hi, "verdict": verdict}


def format_panel(name: str, p: dict) -> str:
    """One report line for a panel verdict."""
    return f"{name}: power {p['successes']}/{p['seeds']} (CP95 lower {p['power_lower']:.3f}) | forbidden in {p['false_positives']}/{p['seeds']} (CP95 upper {p['fp_upper']:.3f}) | {p['verdict']}"
