"""Boruta-style shadow-feature all-relevant selection.

Distinct from ``null_importance_filter`` (which shuffles the TARGET to build a chance baseline for each
feature's own importance) and from MRMR (a MINIMAL-redundant selector that deliberately drops correlated
features once one representative is kept). Boruta instead answers "is this feature relevant AT ALL" (an
ALL-relevant selector): it appends a per-column-shuffled "shadow" copy of every real feature, fits an
importance function over several iterations, and confirms a real feature only when it repeatedly beats the
best (max) shadow importance more often than chance - via a two-sided binomial test against p=0.5, per the
original Boruta algorithm (Kursa & Rudnicki 2010).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from mlframe.feature_selection._selection_log import logs_selection
from mlframe.feature_selection.filters._lazy_host_codes import shape_of


def _summ_boruta(res, b):
    """Selection-summary fragment of ``boruta`` for the end-of-fit log."""
    names = res["feature_names"]
    kept = [n for n, d in zip(names, res["decision"]) if d == "confirmed"]
    n_tent = sum(1 for d in res["decision"] if d == "tentative")
    return kept, len(names), f"{n_tent} tentative, {res['n_rounds_run']} round(s)"


def _resolve_undecided_features_for_round(resolved, hit_counts, rounds_run, n_features, alpha, correction):
    """Confirm or reject the still-undecided features for this round under the corrected per-round rule (mutates ``resolved`` in place)."""
    from scipy.stats import binomtest

    undecided = [j for j in range(n_features) if resolved[j] is None]
    if undecided:
        if correction == "bonferroni":
            # Correct for both the repeated per-round testing AND the simultaneous per-feature testing -
            # a naive uncorrected alpha=0.05 test run every round would inflate the false-confirm rate.
            corrected_alpha = alpha / (rounds_run * len(undecided))
            for j in undecided:
                result = binomtest(int(hit_counts[j]), rounds_run, p=0.5, alternative="two-sided")
                if result.pvalue < corrected_alpha:
                    resolved[j] = "confirmed" if hit_counts[j] / rounds_run > 0.5 else "rejected"
        elif correction == "bh":
            # The BH step-up alone only controls the FDR
            # across the SIMULTANEOUS per-feature tests within a single round; run every round at the
            # nominal alpha and the repeated-testing-across-rounds inflation the "bonferroni" branch
            # above explicitly guards against (dividing by rounds_run) goes uncorrected here. Scale the
            # target alpha down by rounds_run the same way, so both branches control the same overall
            # error budget across the whole multi-round procedure.
            pvals = np.array([binomtest(int(hit_counts[j]), rounds_run, p=0.5, alternative="two-sided").pvalue for j in undecided])
            order = np.argsort(pvals)
            m = len(pvals)
            round_alpha = alpha / rounds_run
            crit = round_alpha * np.arange(1, m + 1) / m
            below = pvals[order] <= crit
            if below.any():
                k_max = int(np.max(np.where(below)[0]))
                for rank in range(k_max + 1):
                    j = undecided[order[rank]]
                    resolved[j] = "confirmed" if hit_counts[j] / rounds_run > 0.5 else "rejected"
        else:
            raise ValueError(f"Unknown correction {correction!r}, expected 'bonferroni' or 'bh'")


def _resolve_feature_layout(X: Any, feature_names: Optional[Sequence[str]]) -> tuple:
    """Column names, feature count, reported names and pandas-ness of ``X``; returns ``(cols, n_features, names, is_pandas_frame)``."""
    import pandas as pd

    cols: list = []
    if hasattr(X, "columns"):
        cols = list(X.columns)
        n_features = len(cols)
        names = list(feature_names) if feature_names is not None else cols
        # ``hasattr(X, "columns")`` also matches polars.DataFrame, which has no ``.reset_index()``
        # / isn't accepted by ``pd.concat`` - only route through the pandas-specific shadow
        # construction below for an actual pandas.DataFrame; every other columned frame type
        # (polars, etc.) falls through to the dtype-agnostic numpy path, which needs no index.
        is_pandas_frame = isinstance(X, pd.DataFrame)
    else:
        n_features = shape_of(X)[1]
        names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_features)]
        is_pandas_frame = False
    return cols, n_features, names, is_pandas_frame


def _final_decisions(hit_counts: np.ndarray, rounds_run: int, resolved: list, resolve_tentative: bool, alpha: float) -> List[str]:
    """Per-feature ``confirmed`` / ``rejected`` / ``tentative`` of :func:`boruta_select` after the last round."""
    from scipy.stats import binomtest

    decisions: List[str] = []
    for j, count in enumerate(hit_counts):
        if resolve_tentative:
            # Any feature not resolved by the corrected per-round rule above stays "tentative" - unlike the
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
    return decisions


@logs_selection("boruta_select", _summ_boruta)
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
        undecided features too) - so a feature can be confirmed/rejected as soon as it is genuinely decided,
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

    import pandas as pd

    # n_iterations<1 left the loop never running, producing a
    # 0/0 NaN win_rate followed by an unhelpful `binomtest(count, 0, ...)` ValueError from scipy internals.
    if n_iterations < 1:
        raise ValueError(f"boruta_select: n_iterations must be >= 1; got {n_iterations}")

    cols, n_features, names, is_pandas_frame = _resolve_feature_layout(X, feature_names)

    rng = np.random.default_rng(random_state)
    hit_counts = np.zeros(n_features, dtype=np.int64)

    resolved: List[Optional[str]] = [None] * n_features
    prev_confirmed: Optional[frozenset] = None
    stable_rounds = 0
    rounds_run = 0
    # Hoisted out of the per-iteration loop below: ``X`` never changes across rounds, so recasting it ``n_iterations``
    # times (up to a few hundred) on the non-pandas path repeated a no-op-or-copy on every round for nothing.
    X_arr = None if is_pandas_frame else np.asarray(X)
    # One (n, 2p) buffer for the whole run: the real half is written once and each round refills only the shadow half in place, instead of building a
    # permuted copy plus a concatenated copy of the whole frame every round. Same draws as ``rng.permuted(real, axis=0)``, so the result is unchanged.
    # A pandas frame whose columns differ in dtype keeps the per-round concat, which preserves each real column's dtype for ``importance_fn``.
    shadow_buf: Optional[np.ndarray] = None
    shadow_frame = None
    if X_arr is not None:
        shadow_buf = np.empty((X_arr.shape[0], 2 * X_arr.shape[1]), dtype=X_arr.dtype)
        shadow_buf[:, :n_features] = X_arr
    elif X[cols].dtypes.nunique() == 1:
        real_block = X[cols].to_numpy()
        shadow_buf = np.empty((real_block.shape[0], 2 * n_features), dtype=real_block.dtype)
        shadow_buf[:, :n_features] = real_block
        del real_block
        shadow_frame = pd.DataFrame(shadow_buf, columns=list(cols) + [f"{c}__shadow" for c in cols], copy=False)

    for it in range(n_iterations):
        rounds_run = it + 1
        if shadow_buf is not None:
            # ``Generator.permuted(x, axis=0)`` shuffles each column INDEPENDENTLY along axis 0 in one vectorised call (bit-identical to the
            # per-column ``rng.permutation(col)`` loop, one Python-level call per column avoided).
            rng.permuted(shadow_buf[:, :n_features], axis=0, out=shadow_buf[:, n_features:])
            X_shadowed = shadow_frame if shadow_frame is not None else shadow_buf
        else:
            real_cols = X[cols].to_numpy()
            shadow_arr = rng.permuted(real_cols, axis=0)
            shadow = pd.DataFrame(shadow_arr, columns=[f"{c}__shadow" for c in cols])
            X_shadowed = pd.concat([X[cols].reset_index(drop=True), shadow.reset_index(drop=True)], axis=1)

        importances = np.asarray(importance_fn(X_shadowed, y), dtype=np.float64)  # Intentional float64: importance statistic compared exactly against the shadow maximum (ties decide hits)
        real_importances = importances[:n_features]
        max_shadow_importance = float(np.max(importances[n_features:]))
        hit_counts += (real_importances > max_shadow_importance).astype(np.int64)

        if resolve_tentative:
            _resolve_undecided_features_for_round(resolved, hit_counts, rounds_run, n_features, alpha, correction)

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
    decisions = _final_decisions(hit_counts, rounds_run, resolved, resolve_tentative, alpha)

    return {
        "hit_counts": hit_counts,
        "win_rate": win_rate,
        "decision": decisions,
        "feature_names": names,
        "n_rounds_run": rounds_run,
    }


__all__ = ["boruta_select"]
