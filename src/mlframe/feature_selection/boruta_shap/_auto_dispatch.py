"""Importance-measure AUTO dispatch for BorutaShap.

BorutaShap's ``importance_measure`` picks the per-trial driver:

* ``gini`` -- fast (impurity), the default. Over-credits noise features via split-frequency bias on
  high-noise / low-signal / small-n-relative-to-p data.
* ``permutation`` -- held-out permutation_importance (when train_or_test='test'). ~11x slower but the
  measured noise-control leader: drives accepted-noise to ~0 where gini leaks 1-5 spurious columns.
* ``shap`` -- slowest, dominated on both axes; never auto-selected.

``importance_measure="auto"`` (this module) runs ONE cheap probe on (X, y) at fit start and routes:
permutation on noisy / overfit-prone / small-n/p beds (where gini's bias hurts), gini on clean /
large-n beds (where the 11x permutation cost buys nothing). Explicit gini/permutation/shap stay
recoverable -- auto only acts when the user asks for it.

The probe (one shared RandomForest fit, train+OOB, no extra full BorutaShap pass):

1. ``n/p`` ratio -- small samples-per-feature inflate impurity's split-frequency bias.
2. ``oob_gap`` -- (train R2/acc) - (OOB R2/acc); a large train-vs-OOB gap is direct overfit evidence.
3. ``shadow_gap`` -- fraction of REAL features whose impurity exceeds the MAX shadow (permuted-copy)
   impurity. DIAGNOSTIC ONLY, NOT a routing signal: a clean bed with many true features legitimately
   shows a HIGH fraction here, so it does not discriminate noise (measured: clean n/p=150 -> 0.6).
   Kept for inspection; routing uses only the two overfit signals above.

Routing rule (any one trips -> permutation): n/p < NP_RATIO_THR OR oob_gap > OOB_GAP_THR.
Thresholds are module globals, env-overridable.

KTC note: the kernel_tuning_cache infra is for GPU/size-crossover dispatch fingerprinted by HARDWARE.
This dispatch is a data-statistics heuristic (n/p, overfit, noise) with NO GPU kernel and NO
hardware-relative crossover, so KTC does not apply; thresholds are plain env-overridable constants
per the dispatcher-contract convention instead.
"""

from __future__ import annotations

import os
import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logger = logging.getLogger(__name__)

# Routing thresholds (env-overridable; see module docstring). Calibrated on the fs_hybrid bed so that
# clean / large-n beds stay on gini (no permutation cost) and noisy / small-n/p beds switch to
# permutation (where it controls accepted-noise). Conservative by design: only clear noise trips it.
_NP_RATIO_THR = float(os.environ.get("MLFRAME_BORUTA_AUTO_NP_RATIO", "30.0"))
_OOB_GAP_THR = float(os.environ.get("MLFRAME_BORUTA_AUTO_OOB_GAP", "0.25"))
# Probe RF is intentionally small/cheap (this is a router, not the selector); bounded rows keep it ~O(50ms).
_PROBE_N_ESTIMATORS = int(os.environ.get("MLFRAME_BORUTA_AUTO_PROBE_TREES", "80"))
_PROBE_MAX_ROWS = int(os.environ.get("MLFRAME_BORUTA_AUTO_PROBE_ROWS", "2000"))


def _probe_signals(X, y, classification: bool, random_state: int) -> dict:
    """Compute the cheap noise/overfit signals from a single small RandomForest fit (with OOB)."""
    Xv = np.asarray(X, dtype=np.float64)
    yv = np.asarray(y).ravel()
    n, p = Xv.shape

    # Bound rows: the router must be cheap relative to a full BorutaShap fit.
    if n > _PROBE_MAX_ROWS:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=_PROBE_MAX_ROWS, replace=False)
        Xv, yv = Xv[idx], yv[idx]
        n = _PROBE_MAX_ROWS

    np_ratio = float(n) / max(1, p)

    if classification:
        rf = RandomForestClassifier(
            n_estimators=_PROBE_N_ESTIMATORS, oob_score=True, bootstrap=True,
            random_state=random_state, n_jobs=-1,
        )
    else:
        rf = RandomForestRegressor(
            n_estimators=_PROBE_N_ESTIMATORS, oob_score=True, bootstrap=True,
            random_state=random_state, n_jobs=-1,
        )

    # Shadow columns: a permuted copy of each real feature (same construction the gate uses), so the
    # real-vs-shadow impurity gap mirrors what BorutaShap will see -- a cheap preview of the leak.
    rng = np.random.default_rng(random_state)
    shadow = Xv.copy()
    for j in range(p):
        rng.shuffle(shadow[:, j])
    Xext = np.hstack([Xv, shadow])

    try:
        rf.fit(Xext, yv)
        oob = float(getattr(rf, "oob_score_", np.nan))
        train = float(rf.score(Xext, yv))
    except Exception as exc:  # degenerate probe (e.g. single-class subsample) -> neutral signals
        logger.debug("BorutaShap auto-probe RF fit failed (%s); routing to gini default.", exc)
        return {"np_ratio": np_ratio, "oob_gap": 0.0, "shadow_gap": 0.0, "probe_ok": False}

    oob_gap = train - oob if np.isfinite(oob) else 0.0

    imp = np.abs(rf.feature_importances_)
    real_imp = imp[:p]
    shadow_imp = imp[p:]
    max_shadow = float(shadow_imp.max()) if shadow_imp.size else 0.0
    shadow_gap = float(np.mean(real_imp > max_shadow)) if p else 0.0

    return {
        "np_ratio": np_ratio,
        "oob_gap": float(oob_gap),
        "shadow_gap": shadow_gap,
        "probe_ok": True,
    }


def resolve_auto_importance_measure(X, y, classification: bool, random_state: int) -> tuple[str, dict]:
    """Return (resolved_measure, diagnostics). resolved_measure in {'gini', 'permutation'}.

    permutation when ANY noise/overfit signal trips its threshold; gini otherwise (clean/large-n,
    where permutation's ~11x cost buys nothing). diagnostics carries the signals + reason for logging
    and unit tests."""
    sig = _probe_signals(X, y, classification, random_state)

    reasons = []
    if sig["np_ratio"] < _NP_RATIO_THR:
        reasons.append(f"n/p={sig['np_ratio']:.1f}<{_NP_RATIO_THR}")
    if sig["oob_gap"] > _OOB_GAP_THR:
        reasons.append(f"oob_gap={sig['oob_gap']:.3f}>{_OOB_GAP_THR}")

    measure = "permutation" if reasons else "gini"
    diag = dict(sig)
    diag["resolved_measure"] = measure
    diag["reasons"] = reasons
    diag["thresholds"] = {"np_ratio": _NP_RATIO_THR, "oob_gap": _OOB_GAP_THR}
    return measure, diag
