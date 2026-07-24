"""One-call SELECTION-STABILITY / CONFIDENCE report for MRMR (backlog W3, 2026-06-11).

A fitted ``MRMR`` gives users a POINT selection -- a set of survivor features --
with no readout of how CONFIDENT that selection is. A feature that barely cleared
the relevance screen on the one full-data split is indistinguishable, in the
public surface, from one that dominates every resample. This accessor closes that
gap at near-replay cost.

The "replay not refit" trick (#15 precedent)
--------------------------------------------
The cheap cross-fold confirmation vote (#15, ``_fe_stability_vote.confirm_recipes_
cross_fold``) made stability cheap by REPLAYING the already-fitted recipes on row
subsets and recomputing only the cheap plug-in-MI gate statistic -- never refitting
MRMR. Layer 36 (``_stability_fe.StabilityFESelector``) takes the expensive route:
it REFITS the whole MRMR ``n_bootstraps`` times.

This report reuses the #15 philosophy. At fit time MRMR already discretised every
candidate column into integer bins (the ``data`` matrix the screen scores against)
and quantised the target into ``classes_y`` codes. We STORE a compact slice of that
binned screening matrix + the target codes + the per-column selection outcome
(``_stability_replay_state_``). The report then, for each of K bootstrap row
resamples, recomputes the cheap marginal ``MI(column_codes; y_codes)`` for every
candidate -- the EXACT debiased primitive the in-fit screen used
(``_cmi_from_binned(x, y, None)``) -- ranks the candidates, and records which would
have been selected. No MRMR refit, no recipe re-search, no quantile re-fitting:
the bins are frozen from the full fit, only the rows are resampled. Cost is K cheap
MI sweeps over the stored matrix == K screen-replays, NOT K * single-fit-time.

What the report measures
------------------------
* per-feature SELECTION-FREQUENCY: the fraction of the K bootstrap resamples on
  which the feature would have been selected (its replayed relevance MI lands in
  the top-``n_selected`` of the candidate pool, mirroring the in-fit relevance
  ranking). A genuine signal feature clears on nearly every resample (-> ~1.0); a
  noise feature that won the point selection by chance does not (-> low).
* per-recipe SURVIVAL-FREQUENCY: for engineered (``unary_binary``) recipes, the
  fraction of resamples on which the recipe still clears its held-out uplift gate
  (the #15 ``_recipe_clears_fold`` statistic re-used verbatim on the resample).

Both are confidence readouts at replay cost, surfaced by one method call.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def _marginal_mi_codes(x_codes: np.ndarray, y_codes: np.ndarray) -> float:
    """Miller-Madow-debiased plug-in ``MI(X; Y)`` from integer bin codes.

    Identical primitive to the in-fit screen relevance estimator and the #15
    cross-fold vote (``_cmi_from_binned`` with an empty conditioning set), so the
    replayed relevance is on the SAME debiased scale as the production selection.
    """
    from ._mi_greedy_cmi_fe import _cmi_from_binned

    return float(_cmi_from_binned(x_codes, y_codes, None))


def selection_stability_report(
    self,
    n_boot: int = 50,
    *,
    random_state: Optional[int] = None,
    quorum: float = 0.6,
    as_text: bool = True,
    verbose: int = 0,
) -> Any:
    """Per-feature SELECTION-FREQUENCY + per-recipe SURVIVAL-FREQUENCY confidence
    report for a fitted ``MRMR``, computed by REPLAY (no MRMR refit).

    For each of ``n_boot`` bootstrap row resamples of the stored fit-time binned
    screening matrix, the cheap marginal ``MI(column; y)`` is recomputed for every
    candidate column (the exact debiased primitive the in-fit relevance screen
    used), the candidates are ranked, and the top-``n_selected`` are recorded as
    "selected on this resample". The selection-frequency of a feature is the
    fraction of resamples on which it was selected -- a confidence readout that
    separates genuine signal (high frequency) from features that won the single
    point selection by chance (low frequency).

    Parameters
    ----------
    n_boot
        Number of bootstrap resamples (K). Each is one cheap MI screen-replay over
        the stored matrix. Cost is ~K replays, NOT K MRMR refits.
    random_state
        Seed for the bootstrap RNG; falls back to the estimator's ``random_seed``.
    quorum
        Fraction of resamples a recipe must clear to be reported as stable (recipe
        survival uses the #15 held-out uplift gate). Informational only here.
    as_text
        When True (default) return a formatted human-readable table (str). When
        False return the raw ``dict`` (per-feature + per-recipe frequencies).
    verbose
        >0 logs a one-line summary.

    Returns
    -------
    A formatted report string (``as_text=True``) or a dict with keys
    ``feature_selection_frequency`` ({name -> freq}), ``selected_features``
    (the point selection), ``recipe_survival_frequency`` ({name -> freq}),
    ``n_boot``, ``n_selected``, ``n_candidates``. Returns a short notice string /
    empty dict when the replay state was not stored (e.g. a degenerate fit).
    """
    state = getattr(self, "_stability_replay_state_", None)
    if not state:
        msg = "selection_stability_report: no replay state stored (the fit was degenerate " "or pre-dates this accessor); nothing to report."
        if verbose:
            logger.info(msg)
        return msg if as_text else {}

    cand_codes: np.ndarray = state["cand_codes"]  # (n_rows, n_cand) int bins
    y_codes: np.ndarray = np.asarray(state["y_codes"]).ravel()
    cand_names: list = list(state["cand_names"])
    selected_mask: np.ndarray = np.asarray(state["selected_mask"], dtype=bool)
    n_rows = int(cand_codes.shape[0])
    n_cand = int(cand_codes.shape[1])
    n_selected = int(selected_mask.sum())

    # (P1): this used to fall back only to the deprecated ``random_seed`` alias
    # (``getattr(self, "random_seed", 0)``), never to the canonical ``random_state``. ``_fit_body`` writes
    # the resolved seed onto ``self.random_seed`` only for the DURATION of fit() and restores the pre-fit
    # value (None, for a caller who used ``random_state=``) in its finally block, so by the time this
    # post-fit accessor ran, an estimator seeded via ``MRMR(random_state=42)`` silently reseeded its
    # bootstrap at 0 instead of 42. ``_effective_random_seed()`` resolves both aliases correctly.
    seed = random_state if random_state is not None else int(self._effective_random_seed() or 0)
    rng = np.random.default_rng(seed)

    K = max(1, int(n_boot))
    sel_counts = np.zeros(n_cand, dtype=np.int64)

    if n_selected <= 0 or n_cand == 0 or n_rows < 2:
        # Nothing rankable: every feature trivially "selected" 0 times.
        freq = {nm: 0.0 for nm in cand_names}
    else:
        for _ in range(K):
            idx = rng.integers(0, n_rows, size=n_rows)  # bootstrap (with replacement)
            y_b = y_codes[idx]
            # Cheap per-candidate marginal MI on the resample (the screen-replay).
            rel = np.empty(n_cand, dtype=np.float64)
            for c in range(n_cand):
                rel[c] = _marginal_mi_codes(cand_codes[idx, c], y_b)
            # Mirror the in-fit relevance ranking: top-n_selected by relevance MI.
            top = np.argpartition(rel, n_cand - n_selected)[n_cand - n_selected :]
            sel_counts[top] += 1
        freq = {cand_names[c]: float(sel_counts[c]) / float(K) for c in range(n_cand)}

    selected_features = [cand_names[c] for c in range(n_cand) if selected_mask[c]]

    # ---- per-recipe SURVIVAL-FREQUENCY (engineered unary_binary recipes) --------
    recipe_freq: dict = {}
    recipes = getattr(self, "_engineered_recipes_", None) or {}
    # ``_engineered_recipes_`` is normally a {name -> recipe} dict but some fit
    # paths leave it a list of recipe objects (each carrying ``.name``); normalise.
    if isinstance(recipes, (list, tuple)):
        recipes = {getattr(r, "name", str(i)): r for i, r in enumerate(recipes)}
    voted = {nm: r for nm, r in recipes.items() if getattr(r, "kind", None) == "unary_binary"}
    rec_state = state.get("recipe_replay")
    if voted and rec_state is not None:
        recipe_freq = _replay_recipe_survival(
            voted=voted, rec_state=rec_state, y_codes=y_codes,
            n_boot=K, rng=np.random.default_rng(seed + 1),
        )

    result = {
        "feature_selection_frequency": dict(sorted(freq.items(), key=lambda kv: -kv[1])),
        "selected_features": selected_features,
        "recipe_survival_frequency": recipe_freq,
        "n_boot": K,
        "n_selected": n_selected,
        "n_candidates": n_cand,
    }

    if verbose:
        _hi = sum(1 for v in freq.values() if v >= quorum)
        logger.info(
            "selection_stability_report: K=%d replays over %d candidates; %d feature(s) "
            ">= quorum %.2f selection-frequency.", K, n_cand, _hi, quorum,
        )

    if not as_text:
        return result
    return _format_report(result, quorum=quorum)


def _replay_recipe_survival(*, voted, rec_state, y_codes, n_boot, rng) -> dict:
    """Per-recipe held-out survival frequency via the #15 uplift-gate statistic,
    replayed on bootstrap resamples of the stored engineered/source bin codes.

    ``rec_state`` maps engineered_name -> {"eng_codes", "a_codes", "b_codes",
    "alt"} -- all frozen at fit time (the engineered column + its source operands,
    already binned). No recipe re-application, no MRMR refit: pure bin-code replay.
    """
    from ._fe_stability_vote import _recipe_clears_fold

    n = int(y_codes.shape[0])
    out: dict = {}
    for nm in voted:
        rs = rec_state.get(nm)
        if rs is None:
            continue
        eng = np.asarray(rs["eng_codes"])
        a = rs.get("a_codes")
        b = rs.get("b_codes")
        a = None if a is None else np.asarray(a)
        b = None if b is None else np.asarray(b)
        alt = bool(rs.get("alt", False))
        if eng.shape[0] != n:
            continue
        passes = 0
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            if _recipe_clears_fold(
                eng_codes=eng[idx],
                src_a_codes=None if a is None else a[idx],
                src_b_codes=None if b is None else b[idx],
                y_codes=y_codes[idx],
                prevalence=1.0,
                alt_acceptance=alt,
            ):
                passes += 1
        out[nm] = float(passes) / float(max(1, n_boot))
    return out


def _format_report(result: dict, *, quorum: float) -> str:
    """One-screen human-readable table of selection / survival frequencies."""
    lines: list = []
    K = result["n_boot"]
    lines.append(
        f"MRMR selection-stability report  (K={K} bootstrap replays, " f"{result['n_selected']}/{result['n_candidates']} selected, quorum={quorum:.2f})"
    )
    lines.append("-" * 72)
    lines.append(f"{'feature':<40}{'sel.freq':>10}  {'point':>6}  conf")
    sel_set = set(result["selected_features"])
    for nm, fr in result["feature_selection_frequency"].items():
        point = "  *  " if nm in sel_set else "     "
        conf = "HIGH" if fr >= max(quorum, 0.7) else ("low" if fr < 0.3 else "mid")
        lines.append(f"{str(nm)[:40]:<40}{fr:>10.2f}  {point:>6}  {conf}")
    if result["recipe_survival_frequency"]:
        lines.append("-" * 72)
        lines.append("engineered recipe survival-frequency (held-out uplift gate):")
        for nm, fr in sorted(result["recipe_survival_frequency"].items(), key=lambda kv: -kv[1]):
            conf = "HIGH" if fr >= max(quorum, 0.7) else ("low" if fr < 0.3 else "mid")
            lines.append(f"  {str(nm)[:50]:<52}{fr:>8.2f}  {conf}")
    lines.append("-" * 72)
    lines.append(
        "sel.freq = fraction of bootstrap resamples on which the feature's replayed "
        "relevance MI ranked in the selected top-set. * marks the point selection. "
        "Computed by replay (no MRMR refit)."
    )
    return "\n".join(lines)
