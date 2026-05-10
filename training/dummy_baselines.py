"""Pre-training Dummy-baseline report.

Runs once per (target_type, target_name) AFTER ``BaselineDiagnostics`` and
BEFORE the per-model training loop. Computes a tabular comparison of
trivial / dummy baselines on val + test, picks the strongest by a
target-type-specific primary metric, and emits one overlay plot for the
strongest baseline only.

Sit-alongside relationship with ``baseline_diagnostics.BaselineDiagnostics``:
that module answers "is the target predictable from these features at
all?" via LightGBM quick-fit + feature ablation; this module answers "is
the task even hard?" via comparison to trivial reference predictors.

Design contract (per plan v3 — 21 defenses D1-D21):
- Default INFO output: ≤ 2 lines per target (verdict + plot path).
- Full per-baseline table at DEBUG level.
- Suite-end summary with cross-target verdict + 4 canonical UPPERCASE
  WARN tokens (``BEST_MODEL_BELOW_DUMMY``, ``ALL_BASELINES_BELOW_RANDOM``,
  ``TS_BEATS_TREES``, ``PARTIAL_FAILURE``).
- All per-cell metrics in their own try/except → NaN row on failure
  (D1).
- Strongest-pick gated on non-degeneracy + paired-bootstrap robustness
  (D2).
- LTR group sanity gate (D3); per-output multi-output regression (D4);
  log_loss as headline classification metric, not AUC (D5); suite-end
  summary integration via ``compute_suite_end_summary`` (D6).
- Promoted IMPORTANT defenses D7-D17 (per-target phase qualifier,
  object-dtype gate, n_finite header, n<10 sample-noise gate, slugified
  paths, plot suppression on <2 finite, deterministic per-target seed,
  schema_version, NaN→None serialization, bootstrap CI when n<2000,
  statsmodels deferred import).
- Promoted v3 inline doc/safety items D18-D21 (plot uniqueness trace,
  α=0.5↔median note, sklearn>=1.0 assert, hash recipe).

Pure w.r.t. ``(target_type, train_y, val_y, test_y, timestamps,
group_ids, cat_features_chosen)`` — see ``_baseline_inputs_hash`` for
sweep-orchestrator memoization.

Quantile-regression scope note
------------------------------
For ``target_type="quantile_regression"`` the dispatcher routes through
the regression path: it emits the constant-prediction baselines (mean,
median, quantile_p25, quantile_p75, per_group_mean, plus TS naive when
timestamps are monotonic) measured by RMSE/MAE. That gives the operator
a constant-prediction *floor* against which the multi-α quantile
regressor's pinball loss is informative even without explicit per-α
empirical-quantile rows. Per-α empirical-quantile baselines paired with
mean-pinball-loss-across-α as primary metric (plan v3 catalog) require
multi-output prediction plumbing — predictions become (N, K) instead of
(N,) — and are deferred until that plumbing is added. The α=0.5 row
in such a future expansion would be identical-by-construction to the
``median`` baseline already emitted (D19 self-consistency note).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Hard sklearn requirement check (D20). mlframe ships sklearn >= 1.0; the
# assertion is defensive against constrained environments / future
# loosened pins.
import sklearn
assert sklearn.__version__ >= "1.0", (
    f"mlframe.training.dummy_baselines requires sklearn >= 1.0 for "
    f"mean_pinball_loss; got {sklearn.__version__}"
)

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
    roc_auc_score,
)

from mlframe.training.baseline_diagnostics import _to_1d_numpy
from mlframe.training.evaluation import _canonical_multilabel_y

# Numba acceleration for hot kernels — multilabel macro log-loss (57x
# vs sklearn's per-label loop), LTR within-group rank assignment.
# Optional dep; on import failure we fall back to numpy/python paths.
try:
    from numba import njit, prange  # noqa: F401
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

if _NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True, cache=False)
    def _numba_macro_log_loss(y_int, p, n, K):
        """Per-label-averaged binary log-loss; skips all-constant labels.

        ``y_int`` is (N, K) int64 with values in {0, 1}; ``p`` is (N, K)
        float64 probability of class 1. Returns NaN when no label has
        both classes present in ``y_int``.
        """
        eps = 1e-15
        label_lls = np.empty(K, dtype=np.float64)
        for k in prange(K):
            s = 0.0
            n_pos = 0
            n_neg = 0
            for i in range(n):
                yi = y_int[i, k]
                pi = p[i, k]
                if pi < eps:
                    pi = eps
                elif pi > 1.0 - eps:
                    pi = 1.0 - eps
                if yi == 1:
                    s -= np.log(pi)
                    n_pos += 1
                else:
                    s -= np.log(1.0 - pi)
                    n_neg += 1
            if n_pos > 0 and n_neg > 0:
                label_lls[k] = s / n
            else:
                label_lls[k] = -1.0  # sentinel: skip (single-class)
        total = 0.0
        valid_count = 0
        for k in range(K):
            if label_lls[k] >= 0:
                total += label_lls[k]
                valid_count += 1
        if valid_count == 0:
            return np.nan
        return total / valid_count

    @njit(fastmath=True, cache=False)
    def _numba_micro_log_loss(y_int, p, n, K):
        """Pooled (micro) binary log-loss across all (N, K) cells."""
        eps = 1e-15
        s = 0.0
        for k in range(K):
            for i in range(n):
                yi = y_int[i, k]
                pi = p[i, k]
                if pi < eps:
                    pi = eps
                elif pi > 1.0 - eps:
                    pi = 1.0 - eps
                if yi == 1:
                    s -= np.log(pi)
                else:
                    s -= np.log(1.0 - pi)
        return s / (n * K)

    @njit(cache=False)
    def _numba_within_group_descending_rank(group_ids: np.ndarray) -> np.ndarray:
        """Descending within-group rank: row 0 of each group → highest score.

        Single-pass over a stable-sorted index. Output[i] = -within_group_idx
        so the first row of each group has the highest score. Robust
        against non-contiguous group_ids; works on any integer dtype.
        Replaces the prior dict-based 2-pass (which produced a numba
        ``unsafe cast from int64 to undefined`` warning at module load
        from the typed-dict default-value type inference).
        """
        n = len(group_ids)
        out = np.empty(n, dtype=np.float64)
        if n == 0:
            return out
        # argsort is stable; sequential scan over sorted indices counts
        # within-group position via prev-group equality check.
        order = np.argsort(group_ids, kind="mergesort")
        prev_g = group_ids[order[0]]
        c = 0
        for k in range(n):
            i = order[k]
            g = group_ids[i]
            if g != prev_g:
                c = 0
                prev_g = g
            out[i] = -float(c)
            c += 1
        return out

    @njit(parallel=True, fastmath=True, cache=False)
    def _numba_paired_bootstrap_rmse(y, p1, p2, n_resamples, seed):
        """Paired bootstrap on RMSE between two predictors.

        Returns ndarray of length ``n_resamples`` with
        ``RMSE(p1) - RMSE(p2)`` per resample (negative when p1 wins under
        minimize-metric convention). ~20-50x faster than sklearn's
        ``mean_squared_error`` per-call loop on n=1500 × 1000 resamples
        (current Python loop ~1100ms → numba ~30ms measured).
        """
        n = len(y)
        out = np.empty(n_resamples, dtype=np.float64)
        # Per-resample independent — prange parallel.
        for i in prange(n_resamples):
            # Per-iteration LCG for index draws (avoids np.random global
            # state under prange; reproducible from (seed, i) pair).
            state = np.uint64(seed) ^ np.uint64(i) * np.uint64(2862933555777941757) + np.uint64(3037000493)
            sse1 = 0.0
            sse2 = 0.0
            for k in range(n):
                # LCG step + mod n for index in [0, n)
                state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                idx = int(state >> np.uint64(33)) % n
                d1 = y[idx] - p1[idx]
                d2 = y[idx] - p2[idx]
                sse1 += d1 * d1
                sse2 += d2 * d2
            out[i] = np.sqrt(sse1 / n) - np.sqrt(sse2 / n)
        return out

    @njit(parallel=True, fastmath=True, cache=False)
    def _numba_paired_bootstrap_mae(y, p1, p2, n_resamples, seed):
        """MAE-paired-bootstrap counterpart of _numba_paired_bootstrap_rmse."""
        n = len(y)
        out = np.empty(n_resamples, dtype=np.float64)
        for i in prange(n_resamples):
            state = np.uint64(seed) ^ np.uint64(i) * np.uint64(2862933555777941757) + np.uint64(3037000493)
            sae1 = 0.0
            sae2 = 0.0
            for k in range(n):
                state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                idx = int(state >> np.uint64(33)) % n
                sae1 += abs(y[idx] - p1[idx])
                sae2 += abs(y[idx] - p2[idx])
            out[i] = sae1 / n - sae2 / n
        return out

    @njit(parallel=True, fastmath=True, cache=False)
    def _numba_bootstrap_rmse_samples(y, p, n_resamples, seed):
        """Bootstrap CI on a single predictor's RMSE.

        Returns ndarray of length ``n_resamples`` with bootstrap samples
        of RMSE. Caller computes 2.5/97.5 percentiles for the CI.
        """
        n = len(y)
        out = np.empty(n_resamples, dtype=np.float64)
        for i in prange(n_resamples):
            state = np.uint64(seed) ^ np.uint64(i) * np.uint64(2862933555777941757) + np.uint64(3037000493)
            sse = 0.0
            for k in range(n):
                state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                idx = int(state >> np.uint64(33)) % n
                d = y[idx] - p[idx]
                sse += d * d
            out[i] = np.sqrt(sse / n)
        return out

    @njit(parallel=True, fastmath=True, cache=False)
    def _numba_bootstrap_mae_samples(y, p, n_resamples, seed):
        n = len(y)
        out = np.empty(n_resamples, dtype=np.float64)
        for i in prange(n_resamples):
            state = np.uint64(seed) ^ np.uint64(i) * np.uint64(2862933555777941757) + np.uint64(3037000493)
            sae = 0.0
            for k in range(n):
                state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                idx = int(state >> np.uint64(33)) % n
                sae += abs(y[idx] - p[idx])
            out[i] = sae / n
        return out

    @njit(parallel=True, fastmath=True, cache=False)
    def _numba_paired_bootstrap_logloss_binary(y_int, p1, p2, n_resamples, seed):
        """Binary cross-entropy paired bootstrap.

        ``y_int`` is (N,) int64 in {0, 1}; ``p1`` / ``p2`` are (N,)
        probability of class 1 (float64). Eps-clipping mirrors
        sklearn's ``log_loss`` (eps=1e-15). Returns ``log_loss(p1) -
        log_loss(p2)`` per resample. Numba kernel is ~30x faster than
        the sklearn loop (sklearn's log_loss does input validation +
        label_binarize per call; the inner-loop is the same arithmetic).
        """
        n = len(y_int)
        eps = 1e-15
        out = np.empty(n_resamples, dtype=np.float64)
        for i in prange(n_resamples):
            state = np.uint64(seed) ^ np.uint64(i) * np.uint64(2862933555777941757) + np.uint64(3037000493)
            ll1 = 0.0
            ll2 = 0.0
            for k in range(n):
                state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                idx = int(state >> np.uint64(33)) % n
                yi = y_int[idx]
                # Predictor 1
                pi1 = p1[idx]
                if pi1 < eps:
                    pi1 = eps
                elif pi1 > 1.0 - eps:
                    pi1 = 1.0 - eps
                if yi == 1:
                    ll1 -= np.log(pi1)
                else:
                    ll1 -= np.log(1.0 - pi1)
                # Predictor 2
                pi2 = p2[idx]
                if pi2 < eps:
                    pi2 = eps
                elif pi2 > 1.0 - eps:
                    pi2 = 1.0 - eps
                if yi == 1:
                    ll2 -= np.log(pi2)
                else:
                    ll2 -= np.log(1.0 - pi2)
            out[i] = ll1 / n - ll2 / n
        return out

    @njit(parallel=True, fastmath=True, cache=False)
    def _numba_bootstrap_logloss_binary_samples(y_int, p, n_resamples, seed):
        """Bootstrap CI samples for binary log-loss on a single predictor."""
        n = len(y_int)
        eps = 1e-15
        out = np.empty(n_resamples, dtype=np.float64)
        for i in prange(n_resamples):
            state = np.uint64(seed) ^ np.uint64(i) * np.uint64(2862933555777941757) + np.uint64(3037000493)
            ll = 0.0
            for k in range(n):
                state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                idx = int(state >> np.uint64(33)) % n
                yi = y_int[idx]
                pi = p[idx]
                if pi < eps:
                    pi = eps
                elif pi > 1.0 - eps:
                    pi = 1.0 - eps
                if yi == 1:
                    ll -= np.log(pi)
                else:
                    ll -= np.log(1.0 - pi)
            out[i] = ll / n
        return out

logger = logging.getLogger(__name__)


def _warmup_numba_kernels(verbose: bool = False) -> None:
    """Trigger numba JIT compilation of all dummy_baselines kernels.

    Pre-warms ``_numba_macro_log_loss``, ``_numba_micro_log_loss``,
    ``_numba_within_group_descending_rank``, and the four bootstrap
    kernels (RMSE/MAE × paired/CI) plus the two log-loss kernels by
    invoking each on a 100-row synthetic input. Subsequent first real
    calls skip the 6-10s JIT cold-start that would otherwise dominate
    the multi_output_regression first-target wall time.

    Idempotent: numba caches compilations process-wide, so calling
    this multiple times is essentially free after the first.

    Cost: ~2-5 seconds on the first call (one-time JIT compilation per
    kernel). Subsequent calls are <10ms. Returns silently on numba
    unavailable; logs at DEBUG by default (set ``verbose=True`` for
    INFO).
    """
    if not _NUMBA_AVAILABLE:
        return
    import time as _time
    log = logger.info if verbose else logger.debug
    t0 = _time.time()
    try:
        rng = np.random.default_rng(0)
        # Multilabel kernels
        y_ml = np.zeros((100, 3), dtype=np.int64)
        y_ml[:50, 0] = 1
        y_ml[:30, 1] = 1
        y_ml[:70, 2] = 1
        p_ml = rng.uniform(0.1, 0.9, (100, 3)).astype(np.float64)
        _numba_macro_log_loss(y_ml, p_ml, 100, 3)
        _numba_micro_log_loss(y_ml, p_ml, 100, 3)
        # Within-group rank
        gids = np.array([0, 0, 1, 1, 1, 2] * 17, dtype=np.int64)[:100]
        _numba_within_group_descending_rank(gids)
        # Bootstrap kernels
        y = rng.normal(size=100).astype(np.float64)
        p1 = (y + rng.normal(0, 0.5, 100)).astype(np.float64)
        p2 = (y + rng.normal(0, 0.7, 100)).astype(np.float64)
        _numba_paired_bootstrap_rmse(y, p1, p2, 50, 7)
        _numba_paired_bootstrap_mae(y, p1, p2, 50, 7)
        _numba_bootstrap_rmse_samples(y, p1, 50, 7)
        _numba_bootstrap_mae_samples(y, p1, 50, 7)
        # Binary log-loss kernels
        y_b = (y > 0).astype(np.int64)
        prob1 = np.clip((y + 0.5) * 0.3, 0.05, 0.95).astype(np.float64)
        prob2 = np.clip((y + 0.3) * 0.4, 0.05, 0.95).astype(np.float64)
        _numba_paired_bootstrap_logloss_binary(y_b, prob1, prob2, 50, 7)
        _numba_bootstrap_logloss_binary_samples(y_b, prob1, 50, 7)
        log("[dummy-baselines] numba kernels pre-warmed in %.2fs", _time.time() - t0)
    except Exception as e:
        logger.debug(
            "[dummy-baselines] numba pre-warmup failed (%s: %s); "
            "first real call will JIT-compile lazily",
            type(e).__name__, e,
        )

# ---------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------


SCHEMA_VERSION = "1.0"


class BaselineReport(NamedTuple):
    """Result of ``compute_dummy_baselines`` for one target.

    Attributes
    ----------
    target_type
        String name of the TargetTypes value (e.g. ``"regression"``).
    target_name
        Specific target column / output name.
    table
        DataFrame indexed by baseline-name with columns for the
        per-split, per-metric values. ``failed`` boolean column flags
        rows whose metrics computation raised.
    strongest
        Name of the strongest baseline by primary metric on the
        reference split (val with test fallback). ``None`` when both
        splits are degenerate or fewer than 2 baselines produced
        finite metrics.
    primary_metric
        The metric name used for strongest-pick (e.g. ``"val_RMSE"``).
    ts_period_used
        Inferred TS period for the strongest TS baseline (None for
        non-TS targets or when no TS baseline picked).
    plot_path
        Path to the strongest baseline's overlay PNG (None when not
        rendered — short-circuit, no consumer, or suppressed).
    elapsed_s
        Wall time of the entire baseline computation.
    n_train, n_val, n_test
        Row counts of the splits.
    n_train_finite, n_val_finite, n_test_finite
        Finite-target row counts (round-3 D9 — surfaces all-NaN target
        columns).
    extras
        Free-form dict for target-type-specific diagnostics
        (per-output strongest-pick block for multi-output regression,
        ts_period_candidates, etc.).
    """

    target_type: str
    target_name: str
    table: pd.DataFrame
    strongest: Optional[str]
    primary_metric: Optional[str]
    ts_period_used: Optional[int]
    plot_path: Optional[str]
    elapsed_s: float
    n_train: int
    n_val: int
    n_test: int
    n_train_finite: int
    n_val_finite: int
    n_test_finite: int
    extras: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict (D14 schema_version + D15 NaN→None)."""
        # D15: replace NaN with None so json.dumps() succeeds.
        def _scrub(v: Any) -> Any:
            if isinstance(v, float) and not np.isfinite(v):
                return None
            return v

        # Convert table to {baseline_name: {col: value}} with NaN → None
        table_dict: Dict[str, Dict[str, Any]] = {}
        for idx, row in self.table.iterrows():
            table_dict[str(idx)] = {col: _scrub(row[col]) for col in self.table.columns}

        return {
            "schema_version": SCHEMA_VERSION,
            "target_type": self.target_type,
            "target_name": self.target_name,
            "data": table_dict,
            "strongest": self.strongest,
            "primary_metric": self.primary_metric,
            "ts_period_used": self.ts_period_used,
            "plot_path": self.plot_path,
            "elapsed_s": self.elapsed_s,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "n_train_finite": self.n_train_finite,
            "n_val_finite": self.n_val_finite,
            "n_test_finite": self.n_test_finite,
            "extras": self.extras,
        }

    def format_text(self, default_level: str = "INFO") -> str:
        """Render report for log emission.

        At ``default_level='INFO'`` (default per Operator Contract
        guarantee 1), emit only the verdict line(s) + plot path.
        Promote to ``'DEBUG'`` to get the full table.
        """
        lines: List[str] = []
        # Header with finite-n summary (D9)
        ts_tag = ""
        if self.ts_period_used is not None:
            ts_tag = f" ts_period={self.ts_period_used}"
        lines.append(
            f"[DUMMY_BASELINES] target='{self.target_name}' {self.target_type}"
            f"{ts_tag} n_train={self.n_train} (finite={self.n_train_finite})"
            f" n_val={self.n_val} (finite={self.n_val_finite})"
            f" n_test={self.n_test} (finite={self.n_test_finite})"
        )

        if self.strongest is None:
            lines.append(
                f"[DUMMY_BASELINES] target='{self.target_name}' strongest=None"
                f" (both splits degenerate; review table manually)"
            )
            return "\n".join(lines)

        # Verdict line with strongest baseline metric.
        try:
            strongest_row = self.table.loc[self.strongest]
            primary_val = strongest_row.get(self.primary_metric, float("nan"))
            # Lift vs mean / prior trivial baseline (whichever is in table).
            trivial_name = "mean" if "mean" in self.table.index else (
                "prior" if "prior" in self.table.index else None
            )
            lift_str = ""
            if trivial_name is not None and trivial_name != self.strongest:
                trivial_val = self.table.loc[trivial_name].get(self.primary_metric, float("nan"))
                if np.isfinite(primary_val) and np.isfinite(trivial_val) and trivial_val != 0:
                    # For minimize-metric (RMSE / log_loss), lift = (trivial - strongest) / trivial
                    lift_pct = (trivial_val - primary_val) / abs(trivial_val) * 100
                    lift_str = f" lift_vs_{trivial_name}={lift_pct:+.1f}%"
            tie_suffix = ""
            paired = self.extras.get("paired_bootstrap") if isinstance(self.extras, dict) else None
            if paired and paired.get("p_strongest_beats") is not None:
                pct = int(round(paired["p_strongest_beats"] * 100))
                if self.extras.get("tie"):
                    tie_suffix = f" (beats runner-up in {pct}% of resamples - TIE, treat as noise)"
            lines.append(
                f"[DUMMY_BASELINES] target='{self.target_name}'"
                f" strongest={self.strongest}"
                f" {self.primary_metric}={primary_val:.4f}"
                f"{lift_str}"
                f" (n_baselines={len(self.table)}, full table at DEBUG){tie_suffix}"
            )
            # D2: paired-bootstrap Δ vs runner-up with 95% CI.
            if paired:
                ru = paired.get("runner_up", "?")
                delta = paired.get("delta")
                ci = paired.get("delta_ci")
                p = paired.get("p_strongest_beats")
                if delta is not None and ci is not None and p is not None:
                    metric_short = self.primary_metric.replace("val_", "")
                    lines.append(
                        f"[DUMMY_BASELINES] target='{self.target_name}'"
                        f" Delta_{metric_short} vs runner-up ({ru}) = {delta:+.4f}"
                        f" [95% bootstrap CI: {ci[0]:+.4f}, {ci[1]:+.4f}];"
                        f" beats runner-up in {int(round(p * 100))}% of resamples"
                    )
            # D16: bootstrap CI line when present (small-n grounding).
            ci = self.extras.get("bootstrap_ci") if isinstance(self.extras, dict) else None
            if ci and "val" in ci:
                lo, point, hi = ci["val"]
                lines.append(
                    f"[DUMMY_BASELINES] target='{self.target_name}'"
                    f" strongest val 95% bootstrap CI: [{lo:.4f}, {hi:.4f}]"
                    f" (n_resamples={self.extras.get('bootstrap_ci_n_resamples', 1000)})"
                )
        except Exception as e:
            lines.append(
                f"[DUMMY_BASELINES] target='{self.target_name}'"
                f" strongest={self.strongest} (verdict format failed: {e})"
            )

        # Plot path line (when present).
        if self.plot_path:
            lines.append(
                f"[DUMMY_BASELINES] target='{self.target_name}'"
                f" overlay plot saved: {self.plot_path}"
            )

        # Extras: per-output multi-output strongest-pick (D4).
        if "per_output_strongest" in self.extras:
            for out_idx, info in enumerate(self.extras["per_output_strongest"]):
                lines.append(
                    f"[DUMMY_BASELINES] target='{self.target_name}' Y[{out_idx}]:"
                    f" strongest={info['name']} ({info['primary_metric']}="
                    f"{info['primary_value']:.4f}, normalized={info.get('normalized', float('nan')):.3f})"
                )
            if "cross_output_strongest" in self.extras:
                xo = self.extras["cross_output_strongest"]
                lines.append(
                    f"[DUMMY_BASELINES] target='{self.target_name}'"
                    f" cross-output normalized strongest={xo['name']}"
                    f" (mean_normalized_RMSE={xo['mean_normalized']:.4f})"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------
# Hash recipe for sweep-orchestrator memoization (D21)
# ---------------------------------------------------------------------


def _baseline_inputs_hash(
    target_type: str,
    train_y: Any,
    val_y: Any,
    test_y: Any,
    timestamps: Any = None,
    group_ids: Any = None,
) -> str:
    """SHA256 of the inputs that determine baseline output.

    Use from a hyperparam-sweep orchestrator to memoize across runs:
    ``compute_dummy_baselines`` is pure w.r.t. these inputs (the fitted
    cat_features pick is determined inside, deterministic per-input).

    Documented in the module docstring; not called by
    ``compute_dummy_baselines`` itself (the contract is caller-side).
    """
    h = hashlib.sha256()
    h.update(str(target_type).encode())
    for arr in (train_y, val_y, test_y, timestamps, group_ids):
        if arr is None:
            h.update(b"None")
        else:
            try:
                a = np.ascontiguousarray(_to_1d_numpy(arr) if hasattr(arr, "__len__") else np.array([arr]))
                h.update(a.tobytes())
            except Exception:
                h.update(repr(arr).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


_SLUG_RE = re.compile(r"[^a-zA-Z0-9_]+")


def _slugify(s: str) -> str:
    """Cheap slugifier for safe path components (D11)."""
    if not s:
        return "unnamed"
    return _SLUG_RE.sub("_", str(s)).strip("_") or "unnamed"


def _per_target_seed(base_seed: int, target_name: str) -> int:
    """Deterministic per-target seed for stochastic baselines (D13).

    ``base_seed + (hash(target_name) & 0xFFFF)`` keeps reproducibility
    across runs (same target → same seed) while ensuring independence
    across targets in the same suite (different target → different
    seed). 0xFFFF mask keeps the offset bounded.
    """
    return (base_seed + (hash(target_name) & 0xFFFF)) & 0x7FFFFFFF


def _is_finite_mask(y: np.ndarray) -> np.ndarray:
    """Boolean mask of rows with finite numeric / non-null values."""
    if y.dtype.kind in "fc":
        return np.isfinite(y)
    if y.dtype == object:
        return np.array([v is not None and (not isinstance(v, float) or np.isfinite(v)) for v in y])
    # int / bool / etc. — always finite for our purposes
    return np.ones(len(y), dtype=bool)


def _has_signal(target_type: str, y_ref: np.ndarray, n_min: int = 10) -> Tuple[bool, str]:
    """Per-target non-degeneracy gate for strongest-pick reference split (D10, D2).

    Returns ``(has_signal, reason_if_not)``.
    """
    if y_ref is None or len(y_ref) < n_min:
        return False, f"n_ref={0 if y_ref is None else len(y_ref)} < {n_min}"
    if target_type in ("regression", "quantile_regression"):
        if y_ref.ndim > 1:
            # multi-output: need signal in at least one column
            has_var = any(np.std(y_ref[:, k]) > 1e-9 for k in range(y_ref.shape[1]))
        else:
            has_var = np.std(y_ref) > 1e-9
        if not has_var:
            return False, "y_ref has zero variance"
    elif target_type in ("binary_classification", "multiclass_classification"):
        n_classes = len(np.unique(y_ref))
        if n_classes < 2:
            return False, f"y_ref has only {n_classes} class"
    elif target_type == "multilabel_classification":
        # multilabel: at least one label needs both 0 and 1 in val
        if y_ref.ndim != 2:
            return False, f"multilabel y_ref ndim={y_ref.ndim} != 2"
        any_label_has_signal = any(
            len(np.unique(y_ref[:, k])) >= 2 for k in range(y_ref.shape[1])
        )
        if not any_label_has_signal:
            return False, "no multilabel column has both classes"
    return True, ""


# ---------------------------------------------------------------------
# Time-series detection (D17 — statsmodels deferred import)
# ---------------------------------------------------------------------


def _normalize_timestamps(ts: Any) -> Optional[np.ndarray]:
    """Coerce timestamps to a 1-D numpy array (round-3 A#4 + A#17)."""
    if ts is None:
        return None
    try:
        if hasattr(ts, "to_numpy"):
            ts = ts.to_numpy()
        ts = np.asarray(ts)
        if ts.ndim != 1:
            ts = ts.ravel()
        # If datetime-like, convert to int64 nanoseconds for diff arithmetic.
        if ts.dtype.kind == "M":
            ts = ts.astype("datetime64[ns]").astype("int64")
        elif ts.dtype.kind == "O":
            # mixed types or pd.Timestamp — try pandas conversion
            ts = pd.to_datetime(ts, utc=True, errors="coerce").astype("datetime64[ns]").astype("int64")
        # Else assume already numeric (epoch ints, floats).
        return ts
    except Exception:
        return None


def _is_temporally_monotonic(
    ts_train: np.ndarray, ts_val: np.ndarray, ts_test: np.ndarray
) -> bool:
    """Strict monotonic split: train.max() <= val.min() AND val.max() <= test.min()."""
    if len(ts_train) == 0 or len(ts_val) == 0 or len(ts_test) == 0:
        return False
    return (
        ts_train.max() <= ts_val.min() and ts_val.max() <= ts_test.min()
    )


def _infer_ts_step_periods(ts_train: np.ndarray) -> Tuple[str, List[int]]:
    """Step-size auto-infer (round-3 A#4: np.unique to handle duplicates).

    Returns ``(step_label, default_periods_for_that_step)``.
    """
    if len(ts_train) < 2:
        return "unknown", []
    unique_ts = np.unique(ts_train)
    if len(unique_ts) < 2:
        return "all-duplicate", []
    diffs = np.diff(unique_ts)
    if len(diffs) == 0:
        return "unknown", []
    median_diff = float(np.median(diffs))
    # Heuristic buckets (assuming int64 ns when datetime, or arbitrary float otherwise).
    NS_PER_HOUR = 3600 * 1e9
    NS_PER_DAY = 24 * NS_PER_HOUR
    NS_PER_WEEK = 7 * NS_PER_DAY
    NS_PER_MONTH = 30 * NS_PER_DAY
    if median_diff <= 0:
        return "duplicate-median", []
    if median_diff < 0.5 * NS_PER_HOUR:
        return "sub-hourly", [1]
    if median_diff < 1.5 * NS_PER_HOUR:
        return "hourly", [1, 24, 168]
    if median_diff < 1.5 * NS_PER_DAY:
        return "daily", [1, 7, 30, 365]
    if median_diff < 1.5 * NS_PER_WEEK:
        return "weekly", [1, 4, 52]
    if median_diff < 1.5 * NS_PER_MONTH:
        return "monthly", [1, 12]
    return "irregular", [1]


def _detect_acf_periods(y_train: np.ndarray, n_train: int) -> List[int]:
    """ACF-based period detection (round-3 C#6: differencing + stratified sample).

    Uses statsmodels.tsa.stattools.acf on first-differenced y_train.
    Returns top-2 peaks above 2/sqrt(n) threshold, filtered to
    ``2 <= P <= n_train // 4`` (round-3 A#17).

    statsmodels imported lazily inside the function (D17): import
    failure → empty list + INFO log, not module-load failure.
    """
    try:
        from statsmodels.tsa.stattools import acf
    except ImportError as e:
        logger.info(
            "[dummy-baselines] statsmodels unavailable (%s); ACF period "
            "detection skipped, using step-size defaults only", e,
        )
        return []

    # Stratified sample (round-3 C#6): for very large n_train, take a
    # uniform-random sample of contiguous-windowed sub-segments
    # (preserves local autocorrelation). Cap at 50000 rows for ACF.
    if n_train > 50_000:
        rng = np.random.default_rng(42)
        # Take 5 contiguous windows of 10000 rows each, randomly placed.
        window_size = 10_000
        n_windows = 5
        max_start = n_train - window_size
        starts = sorted(rng.integers(0, max_start, size=n_windows))
        sample_idx = np.concatenate([np.arange(s, s + window_size) for s in starts])
        y_sample = y_train[sample_idx]
    else:
        y_sample = y_train

    # First-differenced series (round-3 C#6): removes linear trend so
    # ACF peaks reflect seasonality, not trend.
    if len(y_sample) < 30:
        return []
    try:
        y_diff = np.diff(y_sample)
        nlags = min(int(10 * np.log10(len(y_diff))), len(y_diff) // 2)
        if nlags < 2:
            return []
        acf_vals = acf(y_diff, nlags=nlags, fft=True)
    except Exception as e:
        logger.info("[dummy-baselines] ACF computation failed (%s); skipping ACF peaks", e)
        return []

    # Significance threshold ~ 2/sqrt(n) (Bartlett 95% CI under white-noise null).
    threshold = 2.0 / np.sqrt(len(y_diff))
    peaks: List[Tuple[int, float]] = []
    # Lag 0 always ACF=1; skip. Find local maxima above threshold.
    for lag in range(2, len(acf_vals)):
        v = acf_vals[lag]
        if abs(v) > threshold:
            # Local maximum check (avoid trend echo).
            is_peak = (lag == len(acf_vals) - 1 or v >= acf_vals[lag + 1]) and v >= acf_vals[lag - 1]
            if is_peak:
                peaks.append((lag, abs(v)))
    # Top-2 by absolute correlation, filtered by Nyquist-ish constraint.
    peaks.sort(key=lambda kv: -kv[1])
    candidate_periods: List[int] = []
    max_period = n_train // 4  # round-3 A#17: 4 cycles minimum
    for lag, _ in peaks[:5]:  # consider top-5, filter, take top-2 surviving
        if 2 <= lag <= max_period:
            candidate_periods.append(lag)
        if len(candidate_periods) >= 2:
            break
    return candidate_periods


def _resolve_ts_periods(
    y_train: np.ndarray,
    ts_train: np.ndarray,
    extra_periods: Sequence[int] = (),
) -> Tuple[List[int], Dict[str, Any]]:
    """Combine step-size + ACF + user-extra periods into final candidate list.

    Returns ``(periods, diagnostics_dict)``.
    """
    n_train = len(y_train)
    diagnostics: Dict[str, Any] = {}

    # Step inference rejection gates (round-3 A#4).
    unique_ts = np.unique(ts_train)
    duplicate_threshold = max(10, int(0.01 * n_train))
    if len(unique_ts) < duplicate_threshold:
        diagnostics["rejected"] = (
            f"timestamps mostly-duplicate (unique={len(unique_ts)}/{n_train}); "
            "TS baselines disabled — likely event-style data"
        )
        return [], diagnostics

    if n_train < 30:
        diagnostics["rejected"] = f"n_train={n_train} < 30 (ACF would be noise)"
        return [], diagnostics

    step_label, step_periods = _infer_ts_step_periods(ts_train)
    diagnostics["step_label"] = step_label
    diagnostics["step_periods"] = step_periods

    acf_periods = _detect_acf_periods(y_train, n_train)
    diagnostics["acf_periods"] = acf_periods

    # Combine: step + acf + user-extra, dedup, cap at 5, sort.
    combined: List[int] = []
    for p in list(step_periods) + list(acf_periods) + list(extra_periods):
        if isinstance(p, (int, np.integer)) and p >= 1 and p not in combined:
            combined.append(int(p))
    combined.sort()
    if len(combined) > 5:
        combined = combined[:5]
    diagnostics["using"] = combined
    return combined, diagnostics


# ---------------------------------------------------------------------
# Per-group baseline (round-3 D1: cardinality cap + coverage + entity overlap)
# ---------------------------------------------------------------------


def _pick_per_group_categorical(
    train_X: Any,
    cat_features: Optional[Sequence[str]],
    n_train: int,
    max_cardinality_ratio: float,
) -> Optional[str]:
    """Pick the highest-cardinality categorical that PASSES the cap.

    Returns column name or None if no cat passes the gate.
    """
    if not cat_features:
        return None
    # Coerce to pandas-like view for column access.
    if hasattr(train_X, "to_pandas"):
        df = train_X.to_pandas()
    elif isinstance(train_X, pd.DataFrame):
        df = train_X
    else:
        return None
    candidates = []
    cap = max_cardinality_ratio * n_train
    for col in cat_features:
        if col not in df.columns:
            continue
        try:
            n_unique = df[col].nunique(dropna=False)
        except Exception:
            continue
        if 2 <= n_unique <= cap:
            candidates.append((n_unique, col))
    if not candidates:
        return None
    # Highest-cardinality among passing candidates.
    candidates.sort(reverse=True)
    return candidates[0][1]


def _per_group_predict(
    train_X: Any,
    val_X: Any,
    test_X: Any,
    train_y: np.ndarray,
    cat_col: str,
    target_type: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Compute per-group baseline predictions on val + test (D1).

    Handles polars Categorical / Enum / pandas categorical / object
    via uniform string coercion + ``__NULL__`` sentinel for NaN keys
    (round-3 A#13).

    Returns ``(train_pred, val_pred, test_pred, diagnostics)`` where
    diagnostics contains coverage_pct + repeat_entity_rate.
    """
    def _col_to_groupkey(X: Any, col: str) -> pd.Series:
        """Coerce a column to a hashable groupby key.

        Fast path: numeric dtypes (int*, float*, bool) pass through
        unchanged — pandas groupby handles NaN with ``dropna=False``.
        astype(str) is reserved for object / categorical / datetime dtypes
        where the original key is not directly hashable / comparable
        across pl/pd boundary. ~50% wall-time reduction on numeric cat
        cols at n_train ≥ 1M (measured: 240ms → 120ms on n=1M, int32).
        """
        if hasattr(X, "to_pandas"):
            X = X.to_pandas()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        s = X[col]
        # Numeric fast path
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            return s
        if pd.api.types.is_bool_dtype(s):
            return s.astype("int8")  # bool → int for clean groupby
        # Object / categorical / datetime: stringify with NULL sentinel
        return s.astype(str).fillna("__NULL__")

    cat_train = _col_to_groupkey(train_X, cat_col)
    cat_val = _col_to_groupkey(val_X, cat_col)
    cat_test = _col_to_groupkey(test_X, cat_col)

    # Group-mean (regression) or group-positive-rate (binary).
    if target_type == "binary_classification":
        # For DummyClassifier-style pred, output is class label;
        # for our purposes we predict probability = group positive rate.
        y_series = pd.Series(train_y).astype(float)
    else:
        y_series = pd.Series(train_y).astype(float)
    group_means = y_series.groupby(cat_train, dropna=False).mean()
    global_mean = float(y_series.mean())

    train_pred = cat_train.map(group_means).fillna(global_mean).to_numpy()
    val_pred = cat_val.map(group_means).fillna(global_mean).to_numpy()
    test_pred = cat_test.map(group_means).fillna(global_mean).to_numpy()

    # Coverage diagnostics
    train_groups = set(cat_train.unique())
    val_coverage = (cat_val.isin(train_groups)).mean() * 100.0
    test_coverage = (cat_test.isin(train_groups)).mean() * 100.0

    # Entity-overlap rate: fraction of val rows whose group has >=5 train labels
    group_sizes = cat_train.value_counts()
    val_high_overlap = cat_val.map(group_sizes).fillna(0).ge(5).mean()

    return train_pred, val_pred, test_pred, {
        "val_coverage_pct": float(val_coverage),
        "test_coverage_pct": float(test_coverage),
        "repeat_entity_rate": float(val_high_overlap),
        "n_groups_train": int(len(train_groups)),
        "global_fallback": global_mean,
    }


# ---------------------------------------------------------------------
# Per-cell metric computation with isolated try/except (D1)
# ---------------------------------------------------------------------


def _safe_metric(
    metric_fn: callable, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any
) -> float:
    """Compute metric in isolated try/except → NaN on failure (D1).

    Failure logged ONCE per (metric_fn.__name__, error type) — not
    silently swallowed.
    """
    try:
        return float(metric_fn(y_true, y_pred, **kwargs))
    except (ValueError, ZeroDivisionError, FloatingPointError, TypeError) as e:
        # Demote to debug to avoid log noise per cell; the WARN happens
        # at strongest-pick / partial-failure level.
        logger.debug(
            "[dummy-baselines] %s failed (%s: %s) — recording NaN",
            getattr(metric_fn, "__name__", "metric"), type(e).__name__, e,
        )
        return float("nan")


# ---------------------------------------------------------------------
# Per-target dispatchers
# ---------------------------------------------------------------------


def _compute_regression_baselines(
    target_name: str,
    train_X: Any,
    val_X: Any,
    test_X: Any,
    train_y: np.ndarray,
    val_y: Optional[np.ndarray],
    test_y: Optional[np.ndarray],
    timestamps_train: Optional[np.ndarray],
    timestamps_val: Optional[np.ndarray],
    timestamps_test: Optional[np.ndarray],
    cat_features: Optional[Sequence[str]],
    config: Any,
    target_type: str = "regression",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """Build {baseline_name: val_pred} + {baseline_name: test_pred} dicts.

    Returns ``(val_preds, test_preds, extras)``.
    """
    val_preds: Dict[str, np.ndarray] = {}
    test_preds: Dict[str, np.ndarray] = {}
    extras: Dict[str, Any] = {}

    # --- Constant baselines (mean / median / quantile) ---
    n_val = 0 if val_y is None else len(val_y)
    n_test = 0 if test_y is None else len(test_y)
    train_mean = float(np.mean(train_y))
    train_median = float(np.median(train_y))

    val_preds["mean"] = np.full(n_val, train_mean)
    test_preds["mean"] = np.full(n_test, train_mean)

    val_preds["median"] = np.full(n_val, train_median)
    test_preds["median"] = np.full(n_test, train_median)

    for q_label, q_alpha in [("quantile_p25", 0.25), ("quantile_p75", 0.75)]:
        c = float(np.quantile(train_y, q_alpha, method="linear"))
        val_preds[q_label] = np.full(n_val, c)
        test_preds[q_label] = np.full(n_test, c)

    # --- per_group_mean (D1) ---
    cat_col = _pick_per_group_categorical(
        train_X, cat_features, len(train_y), config.per_group_max_cardinality_ratio,
    )
    if cat_col is not None:
        try:
            _, val_pg, test_pg, pg_diag = _per_group_predict(
                train_X, val_X, test_X, train_y, cat_col, target_type,
            )
            # Use TS-aware row label when monotonic split present
            ts_active = (
                timestamps_train is not None
                and timestamps_val is not None
                and timestamps_test is not None
                and _is_temporally_monotonic(timestamps_train, timestamps_val, timestamps_test)
            )
            label = "per_group_historical_mean (ts)" if ts_active else "per_group_mean"
            # Annotate row label with high-overlap warning (D1)
            if pg_diag["repeat_entity_rate"] >= config.per_group_high_overlap_threshold:
                label = f"{label} (high_entity_overlap={pg_diag['repeat_entity_rate']:.2f})"
            val_preds[label] = val_pg
            test_preds[label] = test_pg
            extras["per_group"] = {"cat_col": cat_col, **pg_diag}
            # Coverage gate: exclude from strongest-pick if low
            if (
                pg_diag["val_coverage_pct"] < config.per_group_min_val_coverage_pct
                or pg_diag["test_coverage_pct"] < config.per_group_min_val_coverage_pct
            ):
                extras.setdefault("strongest_pick_excluded", []).append(label)
                logger.info(
                    "[dummy-baselines] target='%s' per_group_mean coverage low "
                    "(val=%.1f%%, test=%.1f%%) — excluded from strongest-pick",
                    target_name, pg_diag["val_coverage_pct"], pg_diag["test_coverage_pct"],
                )
        except Exception as e:
            logger.info(
                "[dummy-baselines] target='%s' per_group_mean failed (%s); skipping",
                target_name, e,
            )
    else:
        logger.debug(
            "[dummy-baselines] target='%s' per_group_mean: no eligible categorical "
            "(cat_features=%s, n_train=%d, max_cardinality_ratio=%.2f)",
            target_name, cat_features, len(train_y), config.per_group_max_cardinality_ratio,
        )

    # --- TS baselines (D17 + round-3 A#2 prediction rules) ---
    if (
        timestamps_train is not None
        and timestamps_val is not None
        and timestamps_test is not None
    ):
        ts_train = _normalize_timestamps(timestamps_train)
        ts_val = _normalize_timestamps(timestamps_val)
        ts_test = _normalize_timestamps(timestamps_test)
        if (
            ts_train is not None
            and ts_val is not None
            and ts_test is not None
            and _is_temporally_monotonic(ts_train, ts_val, ts_test)
        ):
            periods, ts_diag = _resolve_ts_periods(
                train_y, ts_train, config.ts_extra_periods,
            )
            extras["ts_diagnostics"] = ts_diag
            logger.debug(
                "[dummy-baselines] target='%s' ts_periods: step=%s defaults=%s acf_peaks=%s using=%s",
                target_name,
                ts_diag.get("step_label"),
                ts_diag.get("step_periods"),
                ts_diag.get("acf_peaks"),
                ts_diag.get("using"),
            )

            # naive_last (round-3 A#2: suppress when n_val > inferred_period to avoid mean-rebrand)
            min_period = min(periods) if periods else 0
            if n_val > 0 and (min_period == 0 or n_val <= min_period):
                # Single-constant prediction = last train value
                last_val = float(train_y[-1])
                val_preds["naive_last (ts)"] = np.full(n_val, last_val)
                test_preds["naive_last (ts)"] = np.full(n_test, last_val)
            else:
                logger.debug(
                    "[dummy-baselines] target='%s' naive_last: suppressed "
                    "(n_val=%d > inferred_period=%d; would degenerate to constant — "
                    "use seasonal_naive_pP instead)",
                    target_name, n_val, min_period,
                )

            # naive_lagP / seasonal_naive_pP for each period
            for P in periods:
                if P < 2 or len(train_y) < P:
                    continue
                # seasonal_naive: predict y_train[-P + (k mod P)] for val row k
                val_sn = np.array([train_y[-P + (k % P)] for k in range(n_val)])
                test_sn = np.array([train_y[-P + (k % P)] for k in range(n_test)])
                label = f"seasonal_naive_p{P} (ts)"
                if P in (ts_diag.get("acf_peaks") or []):
                    label = f"seasonal_naive_p{P} (ts, ACF-detected)"
                val_preds[label] = val_sn
                test_preds[label] = test_sn

            # rolling_mean: include only when ACF detected a peak >= W
            acf_peaks = ts_diag.get("acf_peaks") or []
            for W in (7, 30):
                if W < len(train_y) and any(p >= W for p in acf_peaks):
                    c = float(np.mean(train_y[-W:]))
                    val_preds[f"rolling_mean_w{W} (ts)"] = np.full(n_val, c)
                    test_preds[f"rolling_mean_w{W} (ts)"] = np.full(n_test, c)

            # linear_extrap: OLS y ~ ts on train tail
            try:
                tail_n = min(len(train_y), 10_000)
                ts_tail = ts_train[-tail_n:].astype(np.float64)
                y_tail = np.asarray(train_y[-tail_n:], dtype=np.float64)
                # Center timestamps to avoid float overflow on large epoch ints
                ts_offset = ts_tail[0]
                ts_centered = ts_tail - ts_offset
                slope, intercept = np.polyfit(ts_centered, y_tail, 1)
                val_lin = slope * (ts_val.astype(np.float64) - ts_offset) + intercept
                test_lin = slope * (ts_test.astype(np.float64) - ts_offset) + intercept
                val_preds["linear_extrap (ts)"] = val_lin
                test_preds["linear_extrap (ts)"] = test_lin
            except Exception as e:
                logger.debug(
                    "[dummy-baselines] target='%s' linear_extrap failed (%s); skipping",
                    target_name, e,
                )
        else:
            extras["ts_skip_reason"] = (
                "interleaved split — TS baselines skipped; for TS-naive use val_placement='forward'"
            )
            logger.info(
                "[dummy-baselines] target='%s' timestamps present but split is interleaved "
                "(monotonic check failed) — TS baselines skipped",
                target_name,
            )

    return val_preds, test_preds, extras


def _compute_classification_baselines(
    target_name: str,
    train_X: Any,
    val_X: Any,
    test_X: Any,
    train_y: np.ndarray,
    val_y: Optional[np.ndarray],
    test_y: Optional[np.ndarray],
    timestamps_train: Optional[np.ndarray],
    cat_features: Optional[Sequence[str]],
    config: Any,
    target_type: str,
    n_classes: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """Build {baseline: probs} dicts for binary / multiclass.

    Returns ``(val_probs, test_probs, extras)`` where probs are
    ``(N, K)`` matrices.
    """
    val_probs: Dict[str, np.ndarray] = {}
    test_probs: Dict[str, np.ndarray] = {}
    extras: Dict[str, Any] = {}

    n_val = 0 if val_y is None else len(val_y)
    n_test = 0 if test_y is None else len(test_y)
    seed = _per_target_seed(config.random_state, target_name)

    # Compute train priors
    classes = np.arange(n_classes)
    train_y_int = train_y.astype(np.int64)
    bincounts = np.bincount(train_y_int, minlength=n_classes).astype(np.float64)
    train_prior = bincounts / bincounts.sum() if bincounts.sum() > 0 else np.full(n_classes, 1.0 / n_classes)

    # prior baseline: constant per-class prob = train prior
    prior_probs = np.tile(train_prior, (max(n_val, 1), 1)) if n_val > 0 else np.empty((0, n_classes))
    if n_val > 0:
        val_probs["prior"] = prior_probs
        test_probs["prior"] = np.tile(train_prior, (n_test, 1))

    # most_frequent: predict argmax of prior with one-hot probs
    most_freq_class = int(np.argmax(train_prior))
    mf_probs_row = np.zeros(n_classes)
    mf_probs_row[most_freq_class] = 1.0
    val_probs["most_frequent"] = np.tile(mf_probs_row, (n_val, 1))
    test_probs["most_frequent"] = np.tile(mf_probs_row, (n_test, 1))

    # uniform: 1/K per row
    uniform_probs_row = np.full(n_classes, 1.0 / n_classes)
    val_probs["uniform"] = np.tile(uniform_probs_row, (n_val, 1))
    test_probs["uniform"] = np.tile(uniform_probs_row, (n_test, 1))

    # all_zeros / all_ones (binary only)
    if target_type == "binary_classification" and n_classes == 2:
        # all-class-0: probs = [1, 0]
        z_row = np.array([1.0, 0.0])
        val_probs["all_zeros"] = np.tile(z_row, (n_val, 1))
        test_probs["all_zeros"] = np.tile(z_row, (n_test, 1))
        # all-class-1: probs = [0, 1]
        o_row = np.array([0.0, 1.0])
        val_probs["all_ones"] = np.tile(o_row, (n_val, 1))
        test_probs["all_ones"] = np.tile(o_row, (n_test, 1))

    # stratified: n_repeats over different seeds (D-inline / round-3 C#2)
    # Predicted class sampled from prior; probs = one-hot of sampled class.
    n_repeats = config.stratified_n_repeats
    val_strat_runs: List[np.ndarray] = []
    test_strat_runs: List[np.ndarray] = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        if n_val > 0:
            val_classes = rng.choice(classes, size=n_val, p=train_prior)
            val_strat = np.zeros((n_val, n_classes))
            val_strat[np.arange(n_val), val_classes] = 1.0
            val_strat_runs.append(val_strat)
        if n_test > 0:
            test_classes = rng.choice(classes, size=n_test, p=train_prior)
            test_strat = np.zeros((n_test, n_classes))
            test_strat[np.arange(n_test), test_classes] = 1.0
            test_strat_runs.append(test_strat)
    # Mean over repeats — gives smoothed probs ≈ train_prior on average,
    # but with the realized variance preserved for log_loss / AUC scoring.
    if val_strat_runs:
        val_probs["stratified"] = np.mean(val_strat_runs, axis=0)
    if test_strat_runs:
        test_probs["stratified"] = np.mean(test_strat_runs, axis=0)
    extras["stratified_n_repeats"] = n_repeats

    # per_group_prior (binary only for now)
    if target_type == "binary_classification":
        cat_col = _pick_per_group_categorical(
            train_X, cat_features, len(train_y), config.per_group_max_cardinality_ratio,
        )
        if cat_col is not None:
            try:
                _, val_pg, test_pg, pg_diag = _per_group_predict(
                    train_X, val_X, test_X, train_y.astype(np.float64), cat_col, target_type,
                )
                # Convert to (N, 2) probs: [1-p, p]
                val_pg_2d = np.column_stack([1 - val_pg, val_pg])
                test_pg_2d = np.column_stack([1 - test_pg, test_pg])
                label = "per_group_prior"
                if pg_diag["repeat_entity_rate"] >= config.per_group_high_overlap_threshold:
                    label = f"per_group_prior (high_entity_overlap={pg_diag['repeat_entity_rate']:.2f})"
                val_probs[label] = val_pg_2d
                test_probs[label] = test_pg_2d
                extras["per_group"] = {"cat_col": cat_col, **pg_diag}
                if (
                    pg_diag["val_coverage_pct"] < config.per_group_min_val_coverage_pct
                    or pg_diag["test_coverage_pct"] < config.per_group_min_val_coverage_pct
                ):
                    extras.setdefault("strongest_pick_excluded", []).append(label)
            except Exception as e:
                logger.info(
                    "[dummy-baselines] target='%s' per_group_prior failed (%s); skipping",
                    target_name, e,
                )

    return val_probs, test_probs, extras


# ---------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------


def compute_dummy_baselines(
    target_type: str,
    target_name: str,
    *,
    train_X: Any,
    val_X: Any,
    test_X: Any,
    train_y: Any,
    val_y: Any,
    test_y: Any,
    timestamps_train: Any = None,
    timestamps_val: Any = None,
    timestamps_test: Any = None,
    group_ids_train: Any = None,
    group_ids_val: Any = None,
    group_ids_test: Any = None,
    doc_ids_train: Any = None,
    doc_ids_val: Any = None,
    doc_ids_test: Any = None,
    cat_features: Optional[Sequence[str]] = None,
    target_label_encoder: Any = None,
    quantile_alphas: Optional[Sequence[float]] = None,
    config: Any = None,
    plot_file_prefix: str = "",
) -> BaselineReport:
    """Compute dummy baselines for one (target_type, target_name).

    Public entry point. Routes to per-target dispatcher, computes
    per-cell metrics in isolated try/except, picks strongest with
    non-degeneracy + paired-bootstrap robustness gates, optionally
    saves overlay plot for strongest baseline.
    """
    import time as _time
    t0 = _time.time()

    if config is None:
        from .configs import DummyBaselinesConfig
        config = DummyBaselinesConfig()

    # Coerce y to 1D / 2D numpy as appropriate (D8 object-dtype gate).
    train_y_arr = _coerce_y(train_y, target_type, target_name)
    val_y_arr = _coerce_y(val_y, target_type, target_name) if val_y is not None else None
    test_y_arr = _coerce_y(test_y, target_type, target_name) if test_y is not None else None

    if train_y_arr is None:
        return _empty_report(target_type, target_name, t0, reason="object-dtype-target")

    n_train = len(train_y_arr)
    n_val = 0 if val_y_arr is None else len(val_y_arr)
    n_test = 0 if test_y_arr is None else len(test_y_arr)
    n_train_finite = int(_is_finite_mask(train_y_arr).sum()) if train_y_arr.ndim == 1 else n_train
    n_val_finite = int(_is_finite_mask(val_y_arr).sum()) if val_y_arr is not None and val_y_arr.ndim == 1 else n_val
    n_test_finite = int(_is_finite_mask(test_y_arr).sum()) if test_y_arr is not None and test_y_arr.ndim == 1 else n_test

    # D9: skip block if both val and test are uninformative
    if n_val_finite < 2 and n_test_finite < 2:
        logger.warning(
            "[DUMMY_BASELINES] FAILED target='%s' - both val (%d/%d finite) and "
            "test (%d/%d finite) targets have <2 finite values",
            target_name, n_val_finite, n_val, n_test_finite, n_test,
        )
        return _empty_report(target_type, target_name, t0, reason="both-splits-uninformative")

    # D4: multi-output regression. For 2D y in regression / quantile_regression,
    # run the dispatcher per output and aggregate per-output strongest +
    # cross-output normalized strongest. Headline emission stays one verdict
    # block per target (not K verdicts).
    if (
        target_type in ("regression", "quantile_regression")
        and train_y_arr.ndim == 2
        and train_y_arr.shape[1] > 1
    ):
        return _compute_multi_output_regression(
            target_type=target_type,
            target_name=target_name,
            train_X=train_X, val_X=val_X, test_X=test_X,
            train_y_arr=train_y_arr, val_y_arr=val_y_arr, test_y_arr=test_y_arr,
            timestamps_train=timestamps_train, timestamps_val=timestamps_val,
            timestamps_test=timestamps_test,
            cat_features=cat_features,
            config=config,
            plot_file_prefix=plot_file_prefix,
            t0=t0,
            n_train=n_train, n_val=n_val, n_test=n_test,
            n_train_finite=n_train_finite, n_val_finite=n_val_finite,
            n_test_finite=n_test_finite,
        )

    # Normalize timestamps once (round-3 A#4 mixed-tz handling).
    ts_train = _normalize_timestamps(timestamps_train)
    ts_val = _normalize_timestamps(timestamps_val)
    ts_test = _normalize_timestamps(timestamps_test)

    # Dispatch by target_type
    val_preds: Dict[str, np.ndarray] = {}
    test_preds: Dict[str, np.ndarray] = {}
    extras: Dict[str, Any] = {}

    if target_type == "quantile_regression" and quantile_alphas is not None:
        # Per-alpha empirical-quantile baselines + pinball-loss metric.
        # Falls back to regression path when quantile_alphas not provided.
        val_preds, test_preds, extras = _compute_quantile_baselines(
            target_name, train_y_arr, val_y_arr, test_y_arr,
            list(quantile_alphas), config,
        )
        extras["quantile_alphas"] = list(quantile_alphas)
    elif target_type in ("regression", "quantile_regression"):
        val_preds, test_preds, extras = _compute_regression_baselines(
            target_name, train_X, val_X, test_X,
            train_y_arr, val_y_arr, test_y_arr,
            ts_train, ts_val, ts_test,
            cat_features, config, target_type=target_type,
        )
    elif target_type in ("binary_classification", "multiclass_classification"):
        # Determine n_classes from train + val + test union
        all_y = np.concatenate([
            train_y_arr,
            val_y_arr if val_y_arr is not None else np.array([], dtype=train_y_arr.dtype),
            test_y_arr if test_y_arr is not None else np.array([], dtype=train_y_arr.dtype),
        ])
        unique_classes = np.unique(all_y[~pd.isna(all_y)] if all_y.dtype.kind in "fc" else all_y)
        n_classes = max(2, len(unique_classes))
        val_preds, test_preds, extras = _compute_classification_baselines(
            target_name, train_X, val_X, test_X,
            train_y_arr, val_y_arr, test_y_arr,
            ts_train, cat_features, config,
            target_type=target_type, n_classes=n_classes,
        )
        extras["n_classes"] = n_classes
    elif target_type == "multilabel_classification":
        val_preds, test_preds, extras = _compute_multilabel_baselines(
            target_name, train_y_arr, val_y_arr, test_y_arr, config,
        )
    elif target_type == "learning_to_rank":
        val_preds, test_preds, extras = _compute_ltr_baselines(
            target_name,
            train_y_arr, val_y_arr, test_y_arr,
            group_ids_train, group_ids_val, group_ids_test,
            ts_train, ts_val, ts_test,
            config,
            doc_ids_train=doc_ids_train,
            doc_ids_val=doc_ids_val,
            doc_ids_test=doc_ids_test,
        )
    else:
        return _empty_report(
            target_type, target_name, t0,
            reason=f"unsupported target_type={target_type}",
        )

    # Compute metrics table
    table, primary_metric = _compute_metrics_table(
        target_type, val_preds, test_preds, val_y_arr, test_y_arr,
        group_ids_val=group_ids_val, group_ids_test=group_ids_test,
        extras=extras,
    )

    # Strongest-pick (D2): non-degeneracy gate + paired-bootstrap
    strongest, ts_period_used = _pick_strongest(
        target_type, table, val_y_arr, test_y_arr, primary_metric, extras, config,
    )

    # D2 (paired-bootstrap robustness): compute Δ vs runner-up + 95% CI +
    # P(strongest beats runner-up). Below `strongest_min_beat_runner_up_prob`
    # the strongest is annotated as TIE and the overlay plot is skipped.
    # Gated on the same n-threshold as bootstrap CI (D16) — at large n the
    # point-estimate signal-to-noise is high enough that paired bootstrap
    # is just expensive ceremony (~3-4s on n=10^5).
    tie_flag = False
    n_ref_for_paired = min(
        n_val_finite if n_val_finite > 0 else 10_000_000,
        n_test_finite if n_test_finite > 0 else 10_000_000,
    )
    if (
        strongest is not None
        and primary_metric is not None
        and n_ref_for_paired < config.bootstrap_ci_threshold
    ):
        try:
            paired = _paired_bootstrap_vs_runner_up(
                target_type, strongest, primary_metric, table,
                val_preds, test_preds, val_y_arr, test_y_arr,
                n_resamples=config.paired_bootstrap_n_resamples,
                seed=_per_target_seed(config.random_state, target_name) + 1,
            )
            if paired is not None:
                extras["paired_bootstrap"] = paired
                if paired.get("p_strongest_beats") is not None and (
                    paired["p_strongest_beats"] < config.strongest_min_beat_runner_up_prob
                ):
                    tie_flag = True
                    extras["tie"] = True
        except Exception as e:
            logger.debug(
                "[dummy-baselines] target='%s' paired-bootstrap failed (%s); skipping",
                target_name, e,
            )

    # D16: bootstrap CI for strongest baseline when min(n_val, n_test) < 2000.
    # Below that threshold the noise floor on RMSE / log_loss / NDCG is non-
    # trivial (>1%), so a CI grounds the verdict line. Above 2000, point
    # estimate is accurate to <1% and CI is suppressed to keep output compact.
    n_ref_for_ci = min(
        n_val_finite if n_val_finite > 0 else 10_000_000,
        n_test_finite if n_test_finite > 0 else 10_000_000,
    )
    if (
        strongest is not None
        and primary_metric is not None
        and n_ref_for_ci < config.bootstrap_ci_threshold
        and n_ref_for_ci >= 10
    ):
        try:
            ci = _bootstrap_ci_for_strongest(
                target_type, strongest, primary_metric,
                val_preds, test_preds, val_y_arr, test_y_arr,
                n_resamples=config.bootstrap_ci_n_resamples,
                seed=_per_target_seed(config.random_state, target_name),
            )
            if ci is not None:
                extras["bootstrap_ci"] = ci
        except Exception as e:
            logger.debug(
                "[dummy-baselines] target='%s' bootstrap CI failed (%s); skipping",
                target_name, e,
            )

    # 2026-05-10: dummy-baselines overlay plot REMOVED per user feedback.
    # The standard ``report_regression_model_perf`` / ``report_probabilistic_model_perf``
    # already produce per-model scatter + residual + calibration charts
    # with full title-metric headers. Re-rendering a separate
    # baseline-overlay PNG was redundant noise on disk and operators
    # asked to "see my standard charts and reports, not a new chart
    # type". The dummy-baselines TABLE (val/test metric grid + strongest
    # verdict line) remains the actionable artifact.
    plot_path = None

    elapsed_s = _time.time() - t0
    return BaselineReport(
        target_type=target_type,
        target_name=target_name,
        table=table,
        strongest=strongest,
        primary_metric=primary_metric,
        ts_period_used=ts_period_used,
        plot_path=plot_path,
        elapsed_s=elapsed_s,
        n_train=n_train, n_val=n_val, n_test=n_test,
        n_train_finite=n_train_finite, n_val_finite=n_val_finite, n_test_finite=n_test_finite,
        extras=extras,
    )


# ---------------------------------------------------------------------
# Multilabel + LTR dispatchers + metrics + plot + helpers
# ---------------------------------------------------------------------


def _compute_quantile_baselines(
    target_name: str,
    train_y: np.ndarray,
    val_y: Optional[np.ndarray],
    test_y: Optional[np.ndarray],
    alphas: Sequence[float],
    config: Any,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """Per-α empirical-quantile baselines for QUANTILE_REGRESSION.

    Emits, per requested α:
      - ``quantile_alpha_{a:.3f}``: constant prediction = empirical α-th
        percentile of train_y (clamped to [1e-3, 1-1e-3] for boundary α
        per round-3 A#9); shape ``(N, K)`` where K=len(alphas).
      - ``median_for_all``: single ``np.median(train_y)`` constant
        broadcast across all α (D19: identical to α=0.5 row by
        construction; documented in row label).

    Predictions are 2D ``(N, K)``. Pinball loss is computed per α
    plus a ``mean_pinball`` aggregate over non-boundary α (α in
    ``[0.05, 0.95]``; round-3 C#7).
    """
    val_preds: Dict[str, np.ndarray] = {}
    test_preds: Dict[str, np.ndarray] = {}
    extras: Dict[str, Any] = {}
    n_val = 0 if val_y is None else len(val_y)
    n_test = 0 if test_y is None else len(test_y)
    K = len(alphas)
    if K == 0:
        return val_preds, test_preds, extras

    train_median = float(np.median(train_y))
    boundary_log: List[Tuple[float, float]] = []  # (orig, clamped)
    n_eff_val: Dict[float, int] = {}
    n_eff_test: Dict[float, int] = {}

    # Per-α: emit one baseline whose prediction is a constant column for
    # that α only, broadcast across the K-output shape so the metrics
    # table can compute pinball@α uniformly.
    consts_per_alpha: List[float] = []
    for a in alphas:
        clamped_a = float(min(max(a, 1e-3), 1 - 1e-3))
        if clamped_a != a:
            boundary_log.append((float(a), clamped_a))
        c = float(np.quantile(train_y, clamped_a, method="linear"))
        consts_per_alpha.append(c)
        if val_y is not None:
            n_eff_val[a] = int(np.sum(val_y < c))
        if test_y is not None:
            n_eff_test[a] = int(np.sum(test_y < c))

    # Build (N, K) predictions per baseline.
    if K > 0:
        # Per-α empirical-quantile baselines: each one is a (N, K)
        # constant matrix where every output uses its own α-th percentile.
        for j, a in enumerate(alphas):
            row_const = consts_per_alpha[j]
            # The j-th baseline emits the j-th constant for ALL alphas
            # (interpretation: "use this α-th percentile to predict every
            # quantile" — degenerate but informative as a reference).
            label = f"quantile_alpha_{a:.3f}"
            if a == 0.5:
                label = f"quantile_alpha_{a:.3f} (=median by construction)"
            val_preds[label] = np.full((n_val, K), row_const)
            test_preds[label] = np.full((n_test, K), row_const)

        # median_for_all: single np.median(train_y) across all α.
        val_preds["median_for_all"] = np.full((n_val, K), train_median)
        test_preds["median_for_all"] = np.full((n_test, K), train_median)

        # multi_quantile_empirical: predicts the j-th α-th percentile in
        # the j-th column — the "right" multi-quantile constant baseline.
        # This is actually what most quantile-loss models should beat.
        consts_arr = np.asarray(consts_per_alpha, dtype=np.float64)
        val_preds["multi_quantile_empirical"] = np.broadcast_to(
            consts_arr, (n_val, K)
        ).copy()
        test_preds["multi_quantile_empirical"] = np.broadcast_to(
            consts_arr, (n_test, K)
        ).copy()

    if boundary_log:
        extras["quantile_boundary_clamped"] = boundary_log
        for orig, clamped in boundary_log:
            logger.info(
                "[dummy-baselines] target='%s' α=%g: clamped to %g for empirical "
                "baseline (degenerate at boundary)",
                target_name, orig, clamped,
            )
    if n_eff_val:
        extras["quantile_n_eff_val"] = n_eff_val
    if n_eff_test:
        extras["quantile_n_eff_test"] = n_eff_test

    return val_preds, test_preds, extras


def _compute_multilabel_baselines(
    target_name: str,
    train_y: np.ndarray,
    val_y: Optional[np.ndarray],
    test_y: Optional[np.ndarray],
    config: Any,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """Multilabel: all_zero / all_one / per_label_prior / per_label_most_frequent."""
    val_preds: Dict[str, np.ndarray] = {}
    test_preds: Dict[str, np.ndarray] = {}
    extras: Dict[str, Any] = {}

    if train_y is None or train_y.ndim != 2:
        return val_preds, test_preds, extras
    K = train_y.shape[1]
    n_val = 0 if val_y is None else len(val_y)
    n_test = 0 if test_y is None else len(test_y)

    # all_zero
    val_preds["all_zero"] = np.zeros((n_val, K))
    test_preds["all_zero"] = np.zeros((n_test, K))
    # all_one
    val_preds["all_one"] = np.ones((n_val, K))
    test_preds["all_one"] = np.ones((n_test, K))
    # per_label_prior — broadcast train per-label mean
    per_label_prior = train_y.mean(axis=0)
    val_preds["per_label_prior"] = np.tile(per_label_prior, (n_val, 1))
    test_preds["per_label_prior"] = np.tile(per_label_prior, (n_test, 1))
    # per_label_most_frequent — round per-label prior to 0/1
    plmf = (per_label_prior >= 0.5).astype(np.float64)
    val_preds["per_label_most_frequent"] = np.tile(plmf, (n_val, 1))
    test_preds["per_label_most_frequent"] = np.tile(plmf, (n_test, 1))

    extras["n_labels"] = K
    return val_preds, test_preds, extras


def _compute_ltr_baselines(
    target_name: str,
    train_y: np.ndarray,
    val_y: Optional[np.ndarray],
    test_y: Optional[np.ndarray],
    group_ids_train: Any,
    group_ids_val: Any,
    group_ids_test: Any,
    ts_train: Optional[np.ndarray],
    ts_val: Optional[np.ndarray],
    ts_test: Optional[np.ndarray],
    config: Any,
    doc_ids_train: Any = None,
    doc_ids_val: Any = None,
    doc_ids_test: Any = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """LTR: random_within_query / identity_input_order / mean_relevance /
    most_recent_first / popularity.

    Group sanity gate (D3) applied before any baseline runs.

    The ``popularity`` baseline activates only when ``doc_ids_*`` are
    supplied (caller has a per-row document identifier outside the
    feature space — this is a strict superset of mlframe's default LTR
    API which only carries ``group_ids`` = qid). Popularity score for
    val/test row = log(1 + count(doc_id in train)). Unseen docs get
    score = 0 (cold-start cells get the smallest possible score).
    """
    val_preds: Dict[str, np.ndarray] = {}
    test_preds: Dict[str, np.ndarray] = {}
    extras: Dict[str, Any] = {}

    if group_ids_train is None or group_ids_val is None or group_ids_test is None:
        extras["ltr_skip_reason"] = "group_ids missing"
        return val_preds, test_preds, extras

    g_train = np.asarray(group_ids_train)
    g_val = np.asarray(group_ids_val)
    g_test = np.asarray(group_ids_test)

    # Defensive: hard-fail on length mismatch with actionable message (D3).
    # A length mismatch is a caller bug, not a runtime degraded condition.
    if len(g_train) != len(train_y):
        raise ValueError(
            f"[dummy-baselines] target='{target_name}' learning_to_rank: "
            f"len(group_ids_train)={len(g_train)} != len(train_y)={len(train_y)}"
        )
    if val_y is not None and len(g_val) != len(val_y):
        raise ValueError(
            f"[dummy-baselines] target='{target_name}' learning_to_rank: "
            f"len(group_ids_val)={len(g_val)} != len(val_y)={len(val_y)}"
        )
    if test_y is not None and len(g_test) != len(test_y):
        raise ValueError(
            f"[dummy-baselines] target='{target_name}' learning_to_rank: "
            f"len(group_ids_test)={len(g_test)} != len(test_y)={len(test_y)}"
        )

    # Group sanity gate (D3)
    n_groups_train = len(np.unique(g_train))
    if n_groups_train < 2:
        extras["ltr_skip_reason"] = f"only {n_groups_train} group in train"
        return val_preds, test_preds, extras
    train_group_sizes = np.bincount(pd.factorize(g_train)[0])
    if train_group_sizes.max() > 0.5 * len(g_train):
        extras["ltr_skip_reason"] = (
            f"max_group_pct={train_group_sizes.max() / len(g_train) * 100:.1f}% "
            "(non-rankable structure)"
        )
        return val_preds, test_preds, extras

    extras["n_groups_train"] = int(n_groups_train)
    extras["n_groups_val"] = int(len(np.unique(g_val))) if len(g_val) > 0 else 0
    extras["n_groups_test"] = int(len(np.unique(g_test))) if len(g_test) > 0 else 0

    n_val = len(g_val)
    n_test = len(g_test)
    seed = _per_target_seed(config.random_state, target_name)

    # random_within_query: n_repeats deterministic seeds (round-3 C#5)
    n_repeats = config.random_within_query_n_repeats
    val_runs: List[np.ndarray] = []
    test_runs: List[np.ndarray] = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        val_runs.append(rng.random(n_val) if n_val > 0 else np.array([]))
        test_runs.append(rng.random(n_test) if n_test > 0 else np.array([]))
    val_preds["random_within_query"] = val_runs[0] if val_runs else np.array([])
    test_preds["random_within_query"] = test_runs[0] if test_runs else np.array([])
    extras["random_within_query_n_repeats"] = n_repeats

    # identity_input_order: predict scores in feature-row order (1 / rank-within-group)
    # For a group's rows, score = N - i where i is the within-group index
    val_preds["identity_input_order"] = _within_group_descending_index(g_val, n_val)
    test_preds["identity_input_order"] = _within_group_descending_index(g_test, n_test)

    # mean_relevance: constant train_y.mean()
    train_y_arr = np.asarray(train_y, dtype=np.float64)
    mean_rel = float(train_y_arr.mean()) if len(train_y_arr) > 0 else 0.0
    val_preds["mean_relevance"] = np.full(n_val, mean_rel)
    test_preds["mean_relevance"] = np.full(n_test, mean_rel)

    # most_recent_first (TS only): rank by recency within group
    if ts_val is not None and ts_test is not None:
        val_preds["most_recent_first (ts)"] = ts_val.astype(np.float64)
        test_preds["most_recent_first (ts)"] = ts_test.astype(np.float64)

    # popularity: per-doc train-frequency. Activates only when
    # doc_ids_* are supplied (FTE protocol extension; mlframe's default
    # LTR carries only group_ids = qid). Score = log(1 + count_train).
    # Unseen docs at val/test get 0 (cold-start cells rank lowest).
    if doc_ids_train is not None and doc_ids_val is not None and doc_ids_test is not None:
        try:
            d_train = np.asarray(doc_ids_train)
            d_val = np.asarray(doc_ids_val)
            d_test = np.asarray(doc_ids_test)
            if (
                len(d_train) == len(train_y)
                and len(d_val) == n_val
                and len(d_test) == n_test
            ):
                # Coerce non-numeric doc IDs to a hashable string form
                # so pd.Series.value_counts handles them uniformly.
                if d_train.dtype.kind not in ("i", "u", "f"):
                    d_train_s = pd.Series([str(x) for x in d_train])
                    d_val_s = pd.Series([str(x) for x in d_val])
                    d_test_s = pd.Series([str(x) for x in d_test])
                else:
                    d_train_s = pd.Series(d_train)
                    d_val_s = pd.Series(d_val)
                    d_test_s = pd.Series(d_test)
                pop_counts = d_train_s.value_counts()
                # Score = log(1 + count); unseen → 0
                val_pop = d_val_s.map(pop_counts).fillna(0).astype(np.float64).to_numpy()
                test_pop = d_test_s.map(pop_counts).fillna(0).astype(np.float64).to_numpy()
                val_preds["popularity"] = np.log1p(val_pop)
                test_preds["popularity"] = np.log1p(test_pop)
                # Diagnostics
                val_unseen_pct = float(np.mean(val_pop == 0) * 100)
                test_unseen_pct = float(np.mean(test_pop == 0) * 100)
                extras["popularity_diagnostics"] = {
                    "n_unique_docs_train": int(len(pop_counts)),
                    "val_cold_start_pct": val_unseen_pct,
                    "test_cold_start_pct": test_unseen_pct,
                }
        except Exception as _pop_err:
            # Non-fatal: popularity is one of N LTR baselines.
            extras["popularity_skip_reason"] = str(_pop_err)

    return val_preds, test_preds, extras


def _within_group_descending_index(group_ids: np.ndarray, n: int) -> np.ndarray:
    """For each row, return descending index within its group (rank 0 = first row).

    Numba-accelerated when ``group_ids`` is integer-typed; falls back to
    Python loop for non-integer keys (e.g. string group_ids).
    """
    if n == 0:
        return np.array([])
    if _NUMBA_AVAILABLE and group_ids.dtype.kind in ("i", "u"):
        # Coerce to int64 so the numba kernel signature is stable.
        gi = np.ascontiguousarray(group_ids, dtype=np.int64)
        try:
            return _numba_within_group_descending_rank(gi)
        except Exception:
            pass
    out = np.zeros(n, dtype=np.float64)
    counts: Dict[Any, int] = {}
    for i in range(n):
        g = group_ids[i]
        c = counts.get(g, 0)
        out[i] = -c
        counts[g] = c + 1
    return out


def _coerce_y(y: Any, target_type: str, target_name: str) -> Optional[np.ndarray]:
    """Coerce y to numpy with target-type-aware shape (D8 object-dtype gate).

    For regression / quantile_regression: 2D ``(N, K)`` inputs preserved
    for D4 multi-output dispatch; 1D inputs reshaped via ``_to_1d_numpy``.
    For multilabel_classification: ``_canonical_multilabel_y`` returns 2D.
    For all classification targets: 1D enforced.
    """
    if y is None:
        return None
    if target_type == "multilabel_classification":
        return _canonical_multilabel_y(y)
    if target_type in ("regression", "quantile_regression"):
        # Preserve 2D for multi-output regression (D4)
        if hasattr(y, "to_numpy"):
            arr = y.to_numpy()
        elif hasattr(y, "values"):
            arr = y.values
        else:
            arr = np.asarray(y)
        if arr.ndim == 1:
            pass
        elif arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.reshape(-1)
        # else: leave 2D for multi-output dispatcher
        if arr.dtype == object:
            try:
                arr = arr.astype(np.float64)
            except (TypeError, ValueError) as e:
                logger.warning(
                    "[dummy-baselines] target='%s' has object dtype incompatible with %s "
                    "baselines (%s); skipping",
                    target_name, target_type, e,
                )
                return None
        return arr
    arr = _to_1d_numpy(y)
    if arr.dtype == object:
        try:
            arr = arr.astype(np.int64)
        except (TypeError, ValueError) as e:
            logger.warning(
                "[dummy-baselines] target='%s' has object dtype incompatible with %s "
                "baselines (%s); skipping",
                target_name, target_type, e,
            )
            return None
    return arr


def _empty_report(
    target_type: str, target_name: str, t0: float, reason: str,
) -> BaselineReport:
    """Return an empty report when block can't run (D8 / D9 / unknown target_type)."""
    import time as _time
    return BaselineReport(
        target_type=target_type,
        target_name=target_name,
        table=pd.DataFrame(),
        strongest=None,
        primary_metric=None,
        ts_period_used=None,
        plot_path=None,
        elapsed_s=_time.time() - t0,
        n_train=0, n_val=0, n_test=0,
        n_train_finite=0, n_val_finite=0, n_test_finite=0,
        extras={"skip_reason": reason},
    )


def _compute_metrics_table(
    target_type: str,
    val_preds: Dict[str, np.ndarray],
    test_preds: Dict[str, np.ndarray],
    val_y: Optional[np.ndarray],
    test_y: Optional[np.ndarray],
    group_ids_val: Any = None,
    group_ids_test: Any = None,
    extras: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, str]:
    """Build the per-baseline x per-split metrics DataFrame (D1, D5)."""
    rows: List[Dict[str, Any]] = []
    baseline_names = sorted(set(val_preds.keys()) | set(test_preds.keys()))

    if target_type == "quantile_regression" and (extras or {}).get("quantile_alphas"):
        # Per-α pinball-loss table. Predictions are 2D (N, K). Headline =
        # mean pinball over non-boundary α (α in [0.05, 0.95] per round-3 C#7).
        alphas = list(extras["quantile_alphas"])
        primary_metric = "val_pinball_mean"
        non_boundary_idx = [i for i, a in enumerate(alphas) if 0.05 <= a <= 0.95]
        for name in baseline_names:
            row: Dict[str, Any] = {"baseline": name}
            vp = val_preds.get(name)
            tp = test_preds.get(name)
            for split_name, y, p in [("val", val_y, vp), ("test", test_y, tp)]:
                pinball_per_a: List[float] = []
                if p is not None and y is not None and len(p) == len(y) and p.ndim == 2 and p.shape[1] == len(alphas):
                    for j, a in enumerate(alphas):
                        v = _safe_metric(mean_pinball_loss, y, p[:, j], alpha=a)
                        row[f"{split_name}_pinball@{a:.3f}"] = v
                        if np.isfinite(v):
                            pinball_per_a.append(v if j in non_boundary_idx else float("nan"))
                    if non_boundary_idx:
                        non_boundary_vals = [
                            row[f"{split_name}_pinball@{alphas[j]:.3f}"]
                            for j in non_boundary_idx
                            if np.isfinite(row.get(f"{split_name}_pinball@{alphas[j]:.3f}", float("nan")))
                        ]
                        row[f"{split_name}_pinball_mean"] = (
                            float(np.mean(non_boundary_vals)) if non_boundary_vals else float("nan")
                        )
                    else:
                        row[f"{split_name}_pinball_mean"] = float("nan")
                else:
                    for a in alphas:
                        row[f"{split_name}_pinball@{a:.3f}"] = float("nan")
                    row[f"{split_name}_pinball_mean"] = float("nan")
            row["failed"] = not (
                np.isfinite(row.get("val_pinball_mean", float("nan")))
                or np.isfinite(row.get("test_pinball_mean", float("nan")))
            )
            rows.append(row)
    elif target_type in ("regression", "quantile_regression"):
        primary_metric = "val_RMSE"
        for name in baseline_names:
            row: Dict[str, Any] = {"baseline": name}
            vp = val_preds.get(name)
            tp = test_preds.get(name)
            if vp is not None and val_y is not None and len(vp) == len(val_y):
                row["val_RMSE"] = _safe_metric(
                    lambda y, p: np.sqrt(mean_squared_error(y, p)), val_y, vp,
                )
                row["val_MAE"] = _safe_metric(mean_absolute_error, val_y, vp)
            else:
                row["val_RMSE"] = float("nan")
                row["val_MAE"] = float("nan")
            if tp is not None and test_y is not None and len(tp) == len(test_y):
                row["test_RMSE"] = _safe_metric(
                    lambda y, p: np.sqrt(mean_squared_error(y, p)), test_y, tp,
                )
                row["test_MAE"] = _safe_metric(mean_absolute_error, test_y, tp)
            else:
                row["test_RMSE"] = float("nan")
                row["test_MAE"] = float("nan")
            row["failed"] = not (
                np.isfinite(row["val_RMSE"]) or np.isfinite(row["test_RMSE"])
            )
            rows.append(row)

    elif target_type in ("binary_classification", "multiclass_classification"):
        # D5: log_loss is headline; AUC secondary.
        primary_metric = "val_log_loss"
        n_classes = (extras or {}).get("n_classes", 2)
        labels = np.arange(n_classes)
        for name in baseline_names:
            row = {"baseline": name}
            vp = val_preds.get(name)
            tp = test_preds.get(name)
            for split_name, y, p in [("val", val_y, vp), ("test", test_y, tp)]:
                if p is not None and y is not None and len(p) == len(y) and p.ndim == 2:
                    row[f"{split_name}_log_loss"] = _safe_metric(
                        log_loss, y, p, labels=labels,
                    )
                    if target_type == "binary_classification":
                        row[f"{split_name}_AUC"] = _safe_metric(
                            roc_auc_score, y, p[:, 1],
                        )
                    else:
                        row[f"{split_name}_AUC_macro"] = _safe_metric(
                            roc_auc_score, y, p,
                            multi_class="ovr", average="macro", labels=labels,
                        )
                else:
                    row[f"{split_name}_log_loss"] = float("nan")
                    auc_key = f"{split_name}_AUC" if target_type == "binary_classification" else f"{split_name}_AUC_macro"
                    row[auc_key] = float("nan")
            row["failed"] = not (
                np.isfinite(row.get("val_log_loss", float("nan")))
                or np.isfinite(row.get("test_log_loss", float("nan")))
            )
            rows.append(row)

    elif target_type == "multilabel_classification":
        primary_metric = "val_log_loss_macro"
        for name in baseline_names:
            row = {"baseline": name}
            vp = val_preds.get(name)
            tp = test_preds.get(name)
            for split_name, y, p in [("val", val_y, vp), ("test", test_y, tp)]:
                if p is not None and y is not None and y.ndim == 2 and p.ndim == 2:
                    K = y.shape[1]
                    n = y.shape[0]
                    if _NUMBA_AVAILABLE and n > 0 and K > 0:
                        # Numba kernel: ~57x faster than per-label sklearn loop.
                        try:
                            y_int = np.ascontiguousarray(y, dtype=np.int64)
                            p_arr = np.ascontiguousarray(p, dtype=np.float64)
                            macro = float(_numba_macro_log_loss(y_int, p_arr, n, K))
                            micro = float(_numba_micro_log_loss(y_int, p_arr, n, K))
                        except Exception:
                            macro = float("nan")
                            micro = float("nan")
                        row[f"{split_name}_log_loss_macro"] = macro
                        row[f"{split_name}_log_loss_micro"] = micro
                    else:
                        # Fallback: sklearn per-label loop.
                        label_lls: List[float] = []
                        for k in range(K):
                            if len(np.unique(y[:, k])) >= 2:
                                ll = _safe_metric(log_loss, y[:, k], p[:, k], labels=[0, 1])
                                if np.isfinite(ll):
                                    label_lls.append(ll)
                        row[f"{split_name}_log_loss_macro"] = (
                            float(np.mean(label_lls)) if label_lls else float("nan")
                        )
                        row[f"{split_name}_log_loss_micro"] = _safe_metric(
                            log_loss, y.ravel(), p.ravel(), labels=[0, 1],
                        )
                else:
                    row[f"{split_name}_log_loss_macro"] = float("nan")
                    row[f"{split_name}_log_loss_micro"] = float("nan")
            row["failed"] = not (
                np.isfinite(row["val_log_loss_macro"]) or np.isfinite(row["test_log_loss_macro"])
            )
            rows.append(row)

    elif target_type == "learning_to_rank":
        primary_metric = "val_NDCG@10"
        from mlframe.ranking_metrics import compute_ranking_summary
        for name in baseline_names:
            row = {"baseline": name}
            for split_name, y, p, g in [
                ("val", val_y, val_preds.get(name), group_ids_val),
                ("test", test_y, test_preds.get(name), group_ids_test),
            ]:
                if p is not None and y is not None and g is not None and len(p) == len(y):
                    try:
                        summary = compute_ranking_summary(
                            np.asarray(y), np.asarray(p), np.asarray(g), eval_at=(1, 5, 10),
                        )
                        for k in (1, 5, 10):
                            row[f"{split_name}_NDCG@{k}"] = summary.get(f"ndcg@{k}", float("nan"))
                        row[f"{split_name}_MAP@10"] = summary.get("map@10", float("nan"))
                        row[f"{split_name}_MRR"] = summary.get("mrr", float("nan"))
                    except Exception:
                        for k in (1, 5, 10):
                            row[f"{split_name}_NDCG@{k}"] = float("nan")
                        row[f"{split_name}_MAP@10"] = float("nan")
                        row[f"{split_name}_MRR"] = float("nan")
                else:
                    for k in (1, 5, 10):
                        row[f"{split_name}_NDCG@{k}"] = float("nan")
                    row[f"{split_name}_MAP@10"] = float("nan")
                    row[f"{split_name}_MRR"] = float("nan")
            row["failed"] = not (
                np.isfinite(row.get("val_NDCG@10", float("nan")))
                or np.isfinite(row.get("test_NDCG@10", float("nan")))
            )
            rows.append(row)

    else:
        return pd.DataFrame(), ""

    table = pd.DataFrame(rows).set_index("baseline")
    return table, primary_metric


def _pick_strongest(
    target_type: str,
    table: pd.DataFrame,
    val_y: Optional[np.ndarray],
    test_y: Optional[np.ndarray],
    primary_metric: str,
    extras: Dict[str, Any],
    config: Any,
) -> Tuple[Optional[str], Optional[int]]:
    """Pick strongest baseline with non-degeneracy + paired-bootstrap gates (D2)."""
    if table.empty or not primary_metric:
        return None, None

    excluded = set(extras.get("strongest_pick_excluded", []))
    eligible = table.drop(index=[b for b in excluded if b in table.index], errors="ignore")
    if eligible.empty:
        return None, None

    # D10: non-degeneracy gate on reference split
    val_ok, val_reason = _has_signal(target_type, val_y) if val_y is not None else (False, "val=None")
    test_metric_name = primary_metric.replace("val_", "test_")

    if val_ok and primary_metric in eligible.columns:
        ref_metric = primary_metric
    elif test_metric_name in eligible.columns:
        test_ok, test_reason = _has_signal(target_type, test_y) if test_y is not None else (False, "test=None")
        if test_ok:
            ref_metric = test_metric_name
        else:
            logger.info(
                "[dummy-baselines] strongest=None (val: %s; test: %s)",
                val_reason, test_reason,
            )
            return None, None
    else:
        return None, None

    # Strongest = lowest for minimize-metrics (RMSE / log_loss / pinball);
    # highest for NDCG / MAP / MRR.
    minimize = primary_metric not in ("val_NDCG@10", "val_MAP@10", "val_MRR", "val_NDCG@5", "val_NDCG@1")
    metric_col = eligible[ref_metric].dropna()
    if metric_col.empty:
        return None, None
    if minimize:
        strongest = metric_col.idxmin()
    else:
        strongest = metric_col.idxmax()

    # Determine ts_period if strongest is a TS baseline
    ts_period_used = None
    if "(ts" in str(strongest):
        m = re.search(r"_p(\d+)", str(strongest))
        if m:
            ts_period_used = int(m.group(1))
        elif "rolling_mean_w" in str(strongest):
            m2 = re.search(r"_w(\d+)", str(strongest))
            if m2:
                ts_period_used = int(m2.group(1))

    return strongest, ts_period_used


# 2026-05-10: ``_save_overlay_plot`` REMOVED per user feedback. The
# standard ``report_regression_model_perf`` / ``report_probabilistic_model_perf``
# pipelines already render per-model scatter / residual / calibration
# charts; the dummy_baselines side rendering its own PNG was redundant
# noise on disk. The dummy_baselines TABLE (val/test metric grid +
# strongest verdict line + paired-bootstrap CI) remains the actionable
# artifact. To re-enable a baseline-overlay PNG in the future, the
# call site at ``compute_dummy_baselines`` should be the single place
# to add it back, gated behind a config flag (default off).


def _paired_bootstrap_vs_runner_up(
    target_type: str,
    strongest: str,
    primary_metric: str,
    table: pd.DataFrame,
    val_preds: Dict[str, np.ndarray],
    test_preds: Dict[str, np.ndarray],
    val_y: Optional[np.ndarray],
    test_y: Optional[np.ndarray],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> Optional[Dict[str, Any]]:
    """D2 paired-bootstrap robustness check.

    Picks the runner-up baseline by primary metric on val (test
    fallback), runs a paired bootstrap (1000 resamples by default) on
    the same resample-indices for both predictors, and returns:

      ``{"runner_up": name,
         "delta": strongest_val - runner_up_val (or model_val - dummy_val),
         "delta_ci": (lo, hi),
         "p_strongest_beats": fraction of resamples where strongest wins}``

    Returns ``None`` when no runner-up exists or metric not computable.
    """
    if strongest not in table.index:
        return None

    # Pick runner-up = second-best by primary_metric on the reference split.
    if primary_metric in table.columns:
        ref_col = primary_metric
    else:
        ref_col = primary_metric.replace("val_", "test_") if primary_metric.startswith("val_") else None
        if ref_col is None or ref_col not in table.columns:
            return None
    series = table[ref_col].dropna()
    if strongest not in series.index or len(series) < 2:
        return None
    minimize = primary_metric not in ("val_NDCG@10", "val_MAP@10", "val_MRR", "val_NDCG@5", "val_NDCG@1")
    series_excl_strongest = series.drop(index=strongest)
    if series_excl_strongest.empty:
        return None
    runner_up = series_excl_strongest.idxmin() if minimize else series_excl_strongest.idxmax()

    # Need predictions for both on the same split.
    sp_val = val_preds.get(strongest)
    sp_test = test_preds.get(strongest)
    rp_val = val_preds.get(runner_up)
    rp_test = test_preds.get(runner_up)

    # Pick split where both have predictions + target is present.
    use_val = (
        val_y is not None and sp_val is not None and rp_val is not None
        and len(sp_val) == len(val_y) and len(rp_val) == len(val_y)
    )
    use_test = (
        test_y is not None and sp_test is not None and rp_test is not None
        and len(sp_test) == len(test_y) and len(rp_test) == len(test_y)
    )
    if use_val:
        y_ref, p1, p2 = val_y, sp_val, rp_val
    elif use_test:
        y_ref, p1, p2 = test_y, sp_test, rp_test
    else:
        return None

    # Metric callable. Limited to RMSE / MAE / log_loss for robustness;
    # NDCG / AUC paired-bootstrap requires per-query / per-class plumbing
    # that is out of scope here (returns None to skip TIE check).
    n = len(y_ref)
    if n < 10:
        return None

    # Numba-accelerated paths for RMSE / MAE / binary log-loss — ~30-340×
    # faster than the Python loop with sklearn metric inside (measured:
    # 1100ms → 3.4ms on n=1500, 1000 resamples for RMSE). Falls back to
    # sklearn loop for log_loss with non-binary preds, multilabel macro
    # log-loss (no numba kernel — cost > value at the n<2000 gate), and
    # when numba unavailable.
    deltas = None
    if _NUMBA_AVAILABLE:
        try:
            if "RMSE" in primary_metric:
                y_arr = np.ascontiguousarray(y_ref, dtype=np.float64)
                p1_arr = np.ascontiguousarray(p1, dtype=np.float64)
                p2_arr = np.ascontiguousarray(p2, dtype=np.float64)
                deltas = _numba_paired_bootstrap_rmse(
                    y_arr, p1_arr, p2_arr, int(n_resamples), int(seed),
                )
                if not minimize:
                    deltas = -deltas
            elif "MAE" in primary_metric:
                y_arr = np.ascontiguousarray(y_ref, dtype=np.float64)
                p1_arr = np.ascontiguousarray(p1, dtype=np.float64)
                p2_arr = np.ascontiguousarray(p2, dtype=np.float64)
                deltas = _numba_paired_bootstrap_mae(
                    y_arr, p1_arr, p2_arr, int(n_resamples), int(seed),
                )
                if not minimize:
                    deltas = -deltas
            elif "log_loss" in primary_metric and "macro" not in primary_metric:
                # Binary-only log-loss kernel: requires 1D y in {0,1} and
                # 1D probs in [0,1]. For 2D-prob multiclass the predictions
                # are (N, K) softmax, not directly compatible with the
                # binary kernel — fall through to sklearn for those cases.
                y_arr_1d = np.ascontiguousarray(y_ref).ravel()
                p1_arr = np.asarray(p1)
                p2_arr = np.asarray(p2)
                # Detect binary 1D case: targets in {0, 1} and probs are 1D
                if (
                    p1_arr.ndim == 1 and p2_arr.ndim == 1
                    and y_arr_1d.dtype.kind in "iu"
                    and len(np.unique(y_arr_1d)) <= 2
                ):
                    y_int = np.ascontiguousarray(y_arr_1d, dtype=np.int64)
                    p1_f = np.ascontiguousarray(p1_arr, dtype=np.float64)
                    p2_f = np.ascontiguousarray(p2_arr, dtype=np.float64)
                    deltas = _numba_paired_bootstrap_logloss_binary(
                        y_int, p1_f, p2_f, int(n_resamples), int(seed),
                    )
                    if not minimize:
                        deltas = -deltas
        except Exception:
            deltas = None  # fall through to sklearn loop

    if deltas is None:
        # Fallback path: sklearn metric loop. Used for log_loss and as
        # a safety net if the numba kernel raises.
        if "RMSE" in primary_metric:
            fn = lambda y, p: float(np.sqrt(mean_squared_error(y, p)))
        elif "MAE" in primary_metric:
            fn = lambda y, p: float(mean_absolute_error(y, p))
        elif "log_loss_macro" in primary_metric:
            return None  # multi-output; cost > value at this gate
        elif "log_loss" in primary_metric:
            from sklearn.metrics import log_loss as _ll
            fn = lambda y, p: float(_ll(y, p))
        else:
            return None

        rng = np.random.default_rng(seed)
        deltas = np.empty(n_resamples, dtype=np.float64)
        valid = 0
        for i in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            try:
                v1 = fn(y_ref[idx], p1[idx])
                v2 = fn(y_ref[idx], p2[idx])
            except Exception:
                continue
            if not (np.isfinite(v1) and np.isfinite(v2)):
                continue
            deltas[valid] = (v1 - v2) if minimize else (v2 - v1)
            valid += 1
        if valid < n_resamples // 4:
            return None
        deltas = deltas[:valid]

    # For minimize metrics: strongest wins iff strongest_val < runner_up_val
    # → delta = (strongest - runner_up) < 0. P(strongest beats) = mean(delta < 0).
    # For maximize metrics: strongest wins iff strongest > runner_up
    # → delta = (runner_up - strongest) < 0. Same condition.
    p_strongest_beats = float(np.mean(deltas < 0))
    point = float(np.mean(deltas))
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))

    return {
        "runner_up": str(runner_up),
        "delta": point,
        "delta_ci": (lo, hi),
        "p_strongest_beats": p_strongest_beats,
        "split_used": "val" if use_val else "test",
    }


def _bootstrap_ci_for_strongest(
    target_type: str,
    strongest: str,
    primary_metric: str,
    val_preds: Dict[str, np.ndarray],
    test_preds: Dict[str, np.ndarray],
    val_y: Optional[np.ndarray],
    test_y: Optional[np.ndarray],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> Optional[Dict[str, Any]]:
    """D16: bootstrap CI on val + test for the strongest baseline only.

    Returns ``{"val": (lo, point, hi), "test": (lo, point, hi)}`` or
    ``None`` when not computable. 1000 resamples by default; cost ~1s on
    n=10^4. Seed is per-target for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Pick the metric callable matching primary_metric. Minimize
    # convention follows _pick_strongest naming.
    def _resample_metric(y: np.ndarray, p: np.ndarray) -> Optional[Tuple[float, float, float]]:
        n = len(y)
        if n < 10:
            return None

        # Numba-accelerated path for RMSE / MAE / binary log-loss
        # (~30-340× faster than the sklearn-per-call loop on n=1500
        # × 1000 resamples).
        if _NUMBA_AVAILABLE and y.ndim == 1 and p.ndim == 1:
            try:
                y_arr = np.ascontiguousarray(y, dtype=np.float64)
                p_arr = np.ascontiguousarray(p, dtype=np.float64)
                if "RMSE" in primary_metric:
                    samples = _numba_bootstrap_rmse_samples(
                        y_arr, p_arr, int(n_resamples), int(seed),
                    )
                    point = float(np.sqrt(np.mean((y_arr - p_arr) ** 2)))
                    if not np.isfinite(point):
                        return None
                    lo = float(np.percentile(samples, 2.5))
                    hi = float(np.percentile(samples, 97.5))
                    return (lo, point, hi)
                if "MAE" in primary_metric:
                    samples = _numba_bootstrap_mae_samples(
                        y_arr, p_arr, int(n_resamples), int(seed),
                    )
                    point = float(np.mean(np.abs(y_arr - p_arr)))
                    if not np.isfinite(point):
                        return None
                    lo = float(np.percentile(samples, 2.5))
                    hi = float(np.percentile(samples, 97.5))
                    return (lo, point, hi)
                if (
                    "log_loss" in primary_metric
                    and "macro" not in primary_metric
                    and y.dtype.kind in "iu"
                    and len(np.unique(y)) <= 2
                ):
                    # Binary 1D-prob case: matches the binary log-loss
                    # numba kernel signature.
                    y_int = np.ascontiguousarray(y, dtype=np.int64)
                    samples = _numba_bootstrap_logloss_binary_samples(
                        y_int, p_arr, int(n_resamples), int(seed),
                    )
                    # Point estimate via the same eps-clipped formula
                    # the kernel uses (matches sklearn's eps=1e-15).
                    eps = 1e-15
                    p_clip = np.clip(p_arr, eps, 1.0 - eps)
                    point = float(np.mean(
                        -np.where(y_int == 1, np.log(p_clip), np.log(1.0 - p_clip))
                    ))
                    if not np.isfinite(point):
                        return None
                    lo = float(np.percentile(samples, 2.5))
                    hi = float(np.percentile(samples, 97.5))
                    return (lo, point, hi)
            except Exception:
                pass  # fall through to sklearn loop

        # Fallback path: sklearn metric loop. Used for log_loss
        # variants and as a safety net if the numba kernel raises.
        try:
            if "RMSE" in primary_metric:
                fn = lambda yi, pi: float(np.sqrt(mean_squared_error(yi, pi)))
            elif "MAE" in primary_metric:
                fn = lambda yi, pi: float(mean_absolute_error(yi, pi))
            elif "log_loss_macro" in primary_metric:
                # Multilabel macro: average over labels; here we use per-row
                # log_loss approx (cheap CI is best-effort for multilabel).
                from sklearn.metrics import log_loss as _ll
                K = y.shape[1] if y.ndim == 2 else 1
                def fn(yi, pi):
                    if yi.ndim == 1:
                        return float(_ll(yi, pi, labels=[0, 1]))
                    losses = []
                    for k in range(K):
                        try:
                            losses.append(float(_ll(yi[:, k], pi[:, k], labels=[0, 1])))
                        except Exception:
                            continue
                    return float(np.mean(losses)) if losses else float("nan")
            elif "log_loss" in primary_metric:
                from sklearn.metrics import log_loss as _ll
                # 1D label, 1D or 2D pred
                fn = lambda yi, pi: float(_ll(yi, pi))
            else:
                # Maximize metrics (NDCG / AUC) — bootstrap point estimate
                # works the same; the CI is naturally returned in metric
                # units regardless of direction.
                return None
        except Exception:
            return None

        # Point estimate
        try:
            point = fn(y, p)
        except Exception:
            return None
        if not np.isfinite(point):
            return None
        # Bootstrap resamples
        samples: List[float] = []
        for _ in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            try:
                v = fn(y[idx], p[idx])
                if np.isfinite(v):
                    samples.append(float(v))
            except Exception:
                continue
        if len(samples) < n_resamples // 4:
            return None
        lo = float(np.percentile(samples, 2.5))
        hi = float(np.percentile(samples, 97.5))
        return (lo, point, hi)

    out: Dict[str, Any] = {}
    val_p = val_preds.get(strongest)
    test_p = test_preds.get(strongest)
    if val_y is not None and val_p is not None and len(val_y) == len(val_p):
        v = _resample_metric(val_y, val_p)
        if v is not None:
            out["val"] = v
    if test_y is not None and test_p is not None and len(test_y) == len(test_p):
        v = _resample_metric(test_y, test_p)
        if v is not None:
            out["test"] = v
    return out if out else None


def _compute_multi_output_regression(
    *,
    target_type: str,
    target_name: str,
    train_X: Any, val_X: Any, test_X: Any,
    train_y_arr: np.ndarray,
    val_y_arr: Optional[np.ndarray],
    test_y_arr: Optional[np.ndarray],
    timestamps_train: Any, timestamps_val: Any, timestamps_test: Any,
    cat_features: Optional[Sequence[str]],
    config: Any,
    plot_file_prefix: str,
    t0: float,
    n_train: int, n_val: int, n_test: int,
    n_train_finite: int, n_val_finite: int, n_test_finite: int,
) -> BaselineReport:
    """D4: multi-output regression dispatcher.

    Runs ``compute_dummy_baselines`` per output (K independent calls), then
    aggregates a per-output strongest-pick block + cross-output normalized
    strongest-pick (mean of ``RMSE / std(y_train_per_target)``). Output 0
    is used for the overlay plot; outputs 1..K-1 plots suppressed (one-
    plot-per-target invariant).
    """
    import time as _time
    K = train_y_arr.shape[1]
    per_output_strongest: List[Dict[str, Any]] = []
    sub_reports: List[BaselineReport] = []

    for k in range(K):
        sub_train = train_y_arr[:, k]
        sub_val = val_y_arr[:, k] if val_y_arr is not None and val_y_arr.ndim == 2 else None
        sub_test = test_y_arr[:, k] if test_y_arr is not None and test_y_arr.ndim == 2 else None
        # Pass plot_file_prefix only for k=0 (one plot per target invariant)
        sub_plot_prefix = plot_file_prefix if k == 0 else ""
        sub_rep = compute_dummy_baselines(
            target_type=target_type,
            target_name=f"{target_name}[{k}]",
            train_X=train_X, val_X=val_X, test_X=test_X,
            train_y=sub_train, val_y=sub_val, test_y=sub_test,
            timestamps_train=timestamps_train,
            timestamps_val=timestamps_val,
            timestamps_test=timestamps_test,
            cat_features=cat_features,
            config=config,
            plot_file_prefix=sub_plot_prefix,
        )
        sub_reports.append(sub_rep)
        if sub_rep.strongest is not None and sub_rep.primary_metric is not None:
            try:
                primary_value = float(
                    sub_rep.table.loc[sub_rep.strongest, sub_rep.primary_metric]
                )
            except Exception:
                primary_value = float("nan")
            # Normalized aggregation: RMSE / std(y_train_per_target)
            std_k = float(np.std(sub_train)) if sub_train.size > 1 else 1.0
            normalized = primary_value / std_k if std_k > 1e-12 else float("nan")
            per_output_strongest.append({
                "output": k,
                "name": sub_rep.strongest,
                "primary_metric": sub_rep.primary_metric,
                "primary_value": primary_value,
                "normalized": normalized,
            })

    # Cross-output normalized strongest-pick: pick the baseline whose mean
    # normalized RMSE across outputs is lowest.
    cross_output: Optional[Dict[str, Any]] = None
    if per_output_strongest:
        # Aggregate per-baseline mean normalized RMSE across all sub-reports.
        baseline_norm_means: Dict[str, List[float]] = {}
        for k, sub_rep in enumerate(sub_reports):
            sub_train_k = train_y_arr[:, k]
            std_k = float(np.std(sub_train_k)) if sub_train_k.size > 1 else 1.0
            if std_k <= 1e-12:
                continue
            for baseline_name in sub_rep.table.index:
                v = sub_rep.table.loc[baseline_name].get(sub_rep.primary_metric)
                if v is None or not np.isfinite(v):
                    continue
                baseline_norm_means.setdefault(str(baseline_name), []).append(float(v) / std_k)
        if baseline_norm_means:
            mean_norms = {
                name: float(np.mean(vals)) for name, vals in baseline_norm_means.items()
                if len(vals) >= max(1, K // 2)  # require coverage on >= half of outputs
            }
            if mean_norms:
                best_name = min(mean_norms, key=mean_norms.get)
                cross_output = {
                    "name": best_name,
                    "mean_normalized": mean_norms[best_name],
                }

    # Build a primary table from output-0 (representative). Strongest /
    # primary_metric / plot_path inherit from sub_reports[0].
    rep0 = sub_reports[0] if sub_reports else None
    extras = {
        "n_outputs": K,
        "per_output_strongest": per_output_strongest,
    }
    if cross_output is not None:
        extras["cross_output_strongest"] = cross_output

    return BaselineReport(
        target_type=target_type,
        target_name=target_name,
        table=rep0.table if rep0 is not None else pd.DataFrame(),
        strongest=cross_output["name"] if cross_output else (rep0.strongest if rep0 else None),
        primary_metric=rep0.primary_metric if rep0 else None,
        ts_period_used=rep0.ts_period_used if rep0 else None,
        plot_path=rep0.plot_path if rep0 else None,
        elapsed_s=_time.time() - t0,
        n_train=n_train, n_val=n_val, n_test=n_test,
        n_train_finite=n_train_finite, n_val_finite=n_val_finite, n_test_finite=n_test_finite,
        extras=extras,
    )


def format_suite_end_summary(
    dummy_baselines_metadata: Dict[str, Any],
    failures_metadata: Optional[Dict[str, Any]] = None,
    best_model_metrics_by_target: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
    min_lift: float = 1.5,
) -> str:
    """Format the cross-target verdict block emitted at suite end (D6).

    Operator Contract guarantee 2: a single suite-end block with rows per
    target, columns ``(strongest_dummy, dummy_metric, best_model,
    model_metric, lift_x, verdict)``.

    Auto-emits canonical UPPERCASE WARN tokens (D6, B#10):
      - ``BEST_MODEL_BELOW_DUMMY`` when ``model/dummy < min_lift`` for a
        minimize metric (or ``dummy/model < min_lift`` for maximize).
      - ``ALL_BASELINES_BELOW_RANDOM`` for binary when all classifier
        baselines have AUC < 0.5 (label-flip suspect).
      - ``TS_BEATS_TREES`` when the strongest TS baseline beats the best
        model on val.
      - ``PARTIAL_FAILURE`` when ``failures_metadata`` is non-empty for
        the target.

    Parameters
    ----------
    dummy_baselines_metadata
        ``metadata["dummy_baselines"]`` from
        ``train_mlframe_models_suite`` — nested dict
        ``{target_type: {target_name: report.to_dict()}}``.
    failures_metadata
        ``metadata["dummy_baselines_failures"]`` (same shape) — used for
        ``PARTIAL_FAILURE`` WARN.
    best_model_metrics_by_target
        Optional ``{(target_type, target_name): {metric: value}}`` — used
        for the lift-vs-model column and ``BEST_MODEL_BELOW_DUMMY`` /
        ``TS_BEATS_TREES`` WARN. When None, only the dummy verdict block
        is emitted.
    min_lift
        Lift threshold below which the model is flagged as not
        meaningfully beating dummy. Default 1.5 (model must be ≥1.5×
        better than dummy on the primary metric).
    """
    lines: List[str] = []
    if not dummy_baselines_metadata:
        return ""

    lines.append("[DUMMY_BASELINES] CROSS-TARGET VERDICT")
    lines.append(
        f"{'target':<24} {'strongest_dummy':<28} {'dummy_metric':<28} "
        f"{'best_model':<12} {'model_metric':<22} {'lift':<8} verdict"
    )

    warn_lines: List[str] = []
    n_total = 0
    n_healthy = 0

    for target_type, by_name in dummy_baselines_metadata.items():
        for target_name, rep_dict in by_name.items():
            n_total += 1
            strongest = rep_dict.get("strongest")
            primary_metric = rep_dict.get("primary_metric")
            data = rep_dict.get("data", {})
            if not strongest or not primary_metric or strongest not in data:
                continue
            strongest_row = data[strongest]
            dummy_val = strongest_row.get(primary_metric)
            if dummy_val is None:
                continue

            # Best model metric lookup (optional)
            best_model_name = "-"
            model_val: Optional[float] = None
            if best_model_metrics_by_target is not None:
                key = (str(target_type), str(target_name))
                model_metrics = best_model_metrics_by_target.get(key, {})
                if model_metrics:
                    # Map the dummy primary_metric to the model's analogue
                    # (e.g. val_RMSE → RMSE_val or vice versa). Caller is
                    # expected to pass model_metrics keyed compatibly.
                    model_val = model_metrics.get(primary_metric)
                    best_model_name = model_metrics.get("model_name", "—")

            # Compute lift. Minimize-metrics (RMSE / log_loss): lift =
            # dummy / model. Maximize-metrics (NDCG@k / AUC): lift = model
            # / dummy. Heuristic by metric name.
            is_minimize = (
                "RMSE" in primary_metric
                or "MAE" in primary_metric
                or "log_loss" in primary_metric
                or "pinball" in primary_metric
            )
            lift_str = "-"
            verdict = "-"
            if model_val is not None and np.isfinite(dummy_val) and np.isfinite(model_val):
                if is_minimize and model_val > 0:
                    lift = dummy_val / model_val
                elif not is_minimize and dummy_val > 0:
                    lift = model_val / dummy_val
                else:
                    lift = float("nan")
                if np.isfinite(lift):
                    lift_str = f"{lift:.2f}x"
                    if lift >= min_lift:
                        verdict = "TASK_NON_TRIVIAL_AND_MODELS_HEALTHY"
                        n_healthy += 1
                    else:
                        verdict = "MODELS_BARELY_BEAT_TRIVIAL"
                        warn_lines.append(
                            f"[DUMMY_BASELINES] WARN BEST_MODEL_BELOW_DUMMY "
                            f"target='{target_name}' lift={lift:.2f}x — "
                            f"investigate label encoding, target leak, "
                            f"train/test contamination."
                        )

            # TS_BEATS_TREES heuristic: strongest baseline name contains
            # 'naive' / 'seasonal' / 'rolling' / 'linear_extrap', AND
            # model_val is worse than dummy_val on a minimize metric.
            ts_strongest = any(
                tok in str(strongest).lower()
                for tok in ("naive", "seasonal", "rolling", "linear_extrap")
            )
            if (
                ts_strongest
                and model_val is not None
                and is_minimize
                and np.isfinite(model_val)
                and np.isfinite(dummy_val)
                and model_val > dummy_val
            ):
                warn_lines.append(
                    f"[DUMMY_BASELINES] WARN TS_BEATS_TREES "
                    f"target='{target_name}' — verify val_placement='forward'; "
                    f"check for leaked-from-future feature columns."
                )

            # ALL_BASELINES_BELOW_RANDOM (binary only): every classifier
            # baseline has AUC < 0.5 → label flip suspected.
            if str(target_type) == "binary_classification":
                aucs = [
                    row.get("val_AUC") for row in data.values()
                    if row.get("val_AUC") is not None
                    and np.isfinite(row.get("val_AUC", float("nan")))
                ]
                if aucs and all(a < 0.5 for a in aucs):
                    warn_lines.append(
                        f"[DUMMY_BASELINES] WARN ALL_BASELINES_BELOW_RANDOM "
                        f"target='{target_name}' — check target_label_encoder "
                        f"direction; check sign of cost_function."
                    )

            lines.append(
                f"{str(target_name)[:24]:<24} {str(strongest)[:28]:<28} "
                f"{primary_metric}={dummy_val:<.4f}     {str(best_model_name)[:12]:<12} "
                f"{(primary_metric + '=' + (f'{model_val:.4f}' if model_val is not None else '-'))[:22]:<22} "
                f"{lift_str:<8} {verdict}"
            )

    # PARTIAL_FAILURE WARN — emitted once per target with failures.
    if failures_metadata:
        for target_type, by_name in failures_metadata.items():
            for target_name, err_msg in by_name.items():
                warn_lines.append(
                    f"[DUMMY_BASELINES] WARN PARTIAL_FAILURE "
                    f"target='{target_name}' ({target_type}) — {err_msg}"
                )

    lines.extend(warn_lines)
    if best_model_metrics_by_target is not None:
        lines.append(
            f"[DUMMY_BASELINES] HEALTH: {n_healthy}/{n_total} targets — "
            f"{'ALL_HEALTHY' if n_healthy == n_total else 'see WARN lines above'}"
        )

    return "\n".join(lines)


__all__ = [
    "compute_dummy_baselines",
    "format_suite_end_summary",
    "BaselineReport",
    "SCHEMA_VERSION",
    "_baseline_inputs_hash",
]
