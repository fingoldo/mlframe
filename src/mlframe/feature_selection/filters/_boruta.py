"""Boruta-style shadow-feature all-relevant selection.

Distinct from ``null_importance_filter`` (which shuffles the TARGET to build a chance baseline for each
feature's own importance) and from MRMR (a MINIMAL-redundant selector that deliberately drops correlated
features once one representative is kept). Boruta instead answers "is this feature relevant AT ALL" (an
ALL-relevant selector): it appends a per-column-shuffled "shadow" copy of every real feature, fits an
importance function over several iterations, and confirms a real feature only when it repeatedly beats the
best (max) shadow importance more often than chance -- via a two-sided binomial test against p=0.5, per the
original Boruta algorithm (Kursa & Rudnicki 2010).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


def boruta_select(
    X: Any,
    y: np.ndarray,
    importance_fn: Callable[[Any, np.ndarray], np.ndarray],
    feature_names: Optional[Sequence[str]] = None,
    n_iterations: int = 20,
    alpha: float = 0.05,
    random_state: int = 0,
    resolve_tentative: bool = False,
    correction: str = "bonferroni",
    convergence_rounds: Optional[int] = None,
) -> Dict[str, Any]:
    """All-relevant feature selection via repeated shadow-feature importance comparison.

    Parameters
    ----------
    X
        Feature matrix/frame with ``n_features`` columns.
    y
        ``(n_samples,)`` target array.
    importance_fn
        ``importance_fn(X_with_shadows, y) -> (2 * n_features,)`` array of per-column importances for a
        matrix that is ``X``'s columns followed by their shuffled shadow copies, in that order (real column
        ``j`` at index ``j``, its shadow at index ``n_features + j``).
    feature_names
        Names for the real columns; inferred from ``X.columns`` if available, else ``f0, f1, ...``.
    n_iterations
        Number of shadow-shuffle-and-refit rounds. Each round, a real feature "wins" if its importance beats
        the max shadow importance that round.
    alpha
        Two-sided binomial-test significance level for confirming (win rate significantly > 0.5) or rejecting
        (win rate significantly < 0.5) a feature; features that never reach significance in either direction
        after ``n_iterations`` are ``"tentative"``.
    random_state
        Seed for the per-iteration shadow shuffles.
    resolve_tentative
        Opt-in (default ``False``, matching original single-shot behavior bit-for-bit). When ``True``, runs the
        original Boruta multi-round rule instead of a one-shot test at the end: after EVERY round, each
        still-undecided feature is re-tested against its cumulative hit count with a significance threshold
        corrected for the repeated testing across rounds (and, for ``"bonferroni"``, across the remaining
        undecided features too) -- so a feature can be confirmed/rejected as soon as it is genuinely decided,
        rather than only once at round ``n_iterations``, and the correction keeps the per-round repeated testing
        from inflating the false-positive rate the way a naive per-round alpha=0.05 test would.
    correction
        Multiple-testing correction used when ``resolve_tentative=True``: ``"bonferroni"`` (per-round alpha
        divided by rounds-so-far times still-undecided features) or ``"bh"`` (Benjamini-Hochberg step-up across
        the undecided features' p-values each round). Ignored when ``resolve_tentative=False``.
    convergence_rounds
        Opt-in early stop (only takes effect when ``resolve_tentative=True``): stop iterating once the confirmed
        feature set has been unchanged for this many consecutive rounds AND every feature has a decision (no
        remaining tentative), reclaiming the trailing ``importance_fn`` refits that no longer change the outcome.
        ``None`` (default) runs the full ``n_iterations``.

    Returns
    -------
    dict
        ``hit_counts`` ``(n_features,)`` int (rounds where the real feature beat max-shadow),
        ``win_rate`` ``(n_features,)`` float, ``decision`` (list of ``"confirmed"``/``"rejected"``/``"tentative"``
        per feature), ``feature_names`` (list), ``n_rounds_run`` (int, equals ``n_iterations`` unless
        ``convergence_rounds`` triggered an early stop).
    """
    from scipy.stats import binomtest

    if hasattr(X, "columns"):
        cols = list(X.columns)
        n_features = len(cols)
        names = list(feature_names) if feature_names is not None else cols
        is_frame = True
    else:
        X_arr = np.asarray(X)
        n_features = X_arr.shape[1]
        names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_features)]
        is_frame = False

    rng = np.random.default_rng(random_state)
    hit_counts = np.zeros(n_features, dtype=np.int64)

    resolved: List[Optional[str]] = [None] * n_features
    prev_confirmed: Optional[frozenset] = None
    stable_rounds = 0
    rounds_run = 0

    for it in range(n_iterations):
        rounds_run = it + 1
        if is_frame:
            import pandas as pd

            # ``Generator.permuted(x, axis=0)`` shuffles each column INDEPENDENTLY along axis 0 in one
            # vectorised call -- bit-identical to the per-column ``rng.permutation(col)`` loop
            # (``.apply(..., axis=0)`` dispatches one Python-level call per column under the hood; verified
            # both consume the SAME rng draw sequence per column) without the ``n_features``-call overhead.
            real_cols = X[cols].to_numpy()
            shadow_arr = rng.permuted(real_cols, axis=0)
            shadow = pd.DataFrame(shadow_arr, columns=[f"{c}__shadow" for c in cols])
            X_shadowed = pd.concat([X[cols].reset_index(drop=True), shadow.reset_index(drop=True)], axis=1)
        else:
            X_arr = np.asarray(X)
            shadow_arr = rng.permuted(X_arr, axis=0)
            X_shadowed = np.concatenate([X_arr, shadow_arr], axis=1)

        importances = np.asarray(importance_fn(X_shadowed, y), dtype=np.float64)
        real_importances = importances[:n_features]
        max_shadow_importance = float(np.max(importances[n_features:]))
        hit_counts += (real_importances > max_shadow_importance).astype(np.int64)

        if resolve_tentative:
            undecided = [j for j in range(n_features) if resolved[j] is None]
            if undecided:
                if correction == "bonferroni":
                    # Correct for both the repeated per-round testing AND the simultaneous per-feature testing --
                    # a naive uncorrected alpha=0.05 test run every round would inflate the false-confirm rate.
                    corrected_alpha = alpha / (rounds_run * len(undecided))
                    for j in undecided:
                        result = binomtest(int(hit_counts[j]), rounds_run, p=0.5, alternative="two-sided")
                        if result.pvalue < corrected_alpha:
                            resolved[j] = "confirmed" if hit_counts[j] / rounds_run > 0.5 else "rejected"
                elif correction == "bh":
                    pvals = np.array(
                        [binomtest(int(hit_counts[j]), rounds_run, p=0.5, alternative="two-sided").pvalue for j in undecided]
                    )
                    order = np.argsort(pvals)
                    m = len(pvals)
                    crit = alpha * np.arange(1, m + 1) / m
                    below = pvals[order] <= crit
                    if below.any():
                        k_max = int(np.max(np.where(below)[0]))
                        for rank in range(k_max + 1):
                            j = undecided[order[rank]]
                            resolved[j] = "confirmed" if hit_counts[j] / rounds_run > 0.5 else "rejected"
                else:
                    raise ValueError(f"Unknown correction {correction!r}, expected 'bonferroni' or 'bh'")

            if convergence_rounds is not None:
                confirmed_set = frozenset(j for j in range(n_features) if resolved[j] == "confirmed")
                if confirmed_set == prev_confirmed:
                    stable_rounds += 1
                else:
                    stable_rounds = 1
                    prev_confirmed = confirmed_set
                all_decided = all(d is not None for d in resolved)
                if all_decided and stable_rounds >= convergence_rounds:
                    break

    win_rate = hit_counts / rounds_run
    decisions: List[str] = []
    for j, count in enumerate(hit_counts):
        if resolve_tentative:
            # Any feature not resolved by the corrected per-round rule above stays "tentative" -- unlike the
            # single-shot mode, this never falls back to an UNcorrected end-of-run test, so it does not
            # reintroduce the family-wise error-rate inflation (n_features simultaneous alpha=0.05 tests) that
            # the correction across rounds AND features exists to control.
            resolved_j = resolved[j]
            decisions.append(resolved_j if resolved_j is not None else "tentative")
            continue
        result = binomtest(int(count), rounds_run, p=0.5, alternative="two-sided")
        if result.pvalue < alpha and count / rounds_run > 0.5:
            decisions.append("confirmed")
        elif result.pvalue < alpha and count / rounds_run < 0.5:
            decisions.append("rejected")
        else:
            decisions.append("tentative")

    return {
        "hit_counts": hit_counts,
        "win_rate": win_rate,
        "decision": decisions,
        "feature_names": names,
        "n_rounds_run": rounds_run,
    }


__all__ = ["boruta_select"]
