"""Helpers carved out of ``pick_best_calibrator`` (``policy.py``) to keep that function under the function-length gate.

Every helper resolves its ``policy`` collaborators lazily so ``policy`` can import this module at module scope.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, cast

import numpy as np

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger("mlframe.calibration.policy")


def warn_same_oof_selection() -> None:
    """Warn that ``selection='same_oof'`` fits and scores candidates on the same OOF rows."""
    # same_oof fits AND ECE-scores every candidate on the SAME OOF rows: a flexible calibrator
    # (Isotonic) interpolates its in-sample ECE toward ~0 and is selected ~100% of the time, with a
    # reported ECE optimistic by ~0.006 vs fresh data. Prefer selection="inner_cv" for an honest
    # held-out estimate; same_oof is kept only for replay / A-B against the legacy path.
    logger.warning(
        "pick_best_calibrator: selection='same_oof' fits and scores candidates on the same OOF rows "
        "-- the reported ECE is optimistically biased (Isotonic-favouring). Use selection='inner_cv' "
        "for an honest held-out estimate."
    )


def prepare_oof_inputs(oof_probs: np.ndarray, oof_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Coerce the OOF probabilities to the positive-class column and validate row counts; returns ``(p_pos, y, n_oof)``."""
    oof_p = np.asarray(oof_probs, dtype=np.float64)
    if oof_p.ndim == 2 and oof_p.shape[1] >= 2:
        oof_p_pos = oof_p[:, 1]
    else:
        oof_p_pos = oof_p.ravel()
    oof_y_arr = np.asarray(oof_y).ravel()
    n_oof = int(oof_y_arr.shape[0])
    if oof_p_pos.shape[0] != n_oof:
        raise ValueError(f"pick_best_calibrator: oof_probs rows ({oof_p_pos.shape[0]}) do not match oof_y ({n_oof})")
    if n_oof < 4:
        raise ValueError(f"pick_best_calibrator: need at least 4 OOF rows; got n_oof={n_oof}")
    return oof_p_pos, oof_y_arr, n_oof


def resolve_candidate_names(candidates: Optional[Iterable[str]]) -> tuple[str, ...]:
    """Filter the requested candidate names down to the known ``CANDIDATE_NAMES`` (warning on unknown ones)."""
    from mlframe.calibration.policy import CANDIDATE_NAMES

    cand_names = tuple(candidates) if candidates is not None else CANDIDATE_NAMES
    unknown = [c for c in cand_names if c not in CANDIDATE_NAMES]
    if unknown:
        logger.warning("pick_best_calibrator: unknown candidate(s) ignored: %s", unknown)
    cand_names = tuple(c for c in cand_names if c in CANDIDATE_NAMES)
    if not cand_names:
        raise ValueError(f"pick_best_calibrator: no valid candidate names; allowed={CANDIDATE_NAMES}")
    return cand_names


def build_selection_resamples(
    selection: str,
    oof_y_arr: np.ndarray,
    n_oof: int,
    n_bootstrap: int,
    inner_cv_splits: int,
    random_state: Optional[int],
) -> tuple[Optional[list[np.ndarray]], Optional[np.ndarray]]:
    """Build the inner-CV folds (``inner_cv``) or the shared bootstrap resample-index matrix (``same_oof``); returns ``(folds, idx_matrix)``."""
    from mlframe.calibration.policy import _build_resample_indices, _stratified_inner_folds

    classes = np.unique(oof_y_arr)
    stratify = oof_y_arr if classes.size == 2 else None
    inner_folds: Optional[list[np.ndarray]] = None
    if selection == "inner_cv":
        inner_folds = _stratified_inner_folds(oof_y_arr, max(2, int(inner_cv_splits)), random_state)

    # Only the same-OOF path reads the bootstrap. Under the default ``selection="inner_cv"`` the held-out block
    # in ``score_candidates`` replaces BOTH the point estimate and the interval, so building the matrix and running 1000 resamples
    # plus a jackknife per candidate produced a number nothing read -- and the build itself raises MemoryError
    # once ``4 * n_bootstrap * n_oof`` passes the 1 GiB ceiling, killing a call whose result was going to be
    # discarded (1.2 GiB at n_oof=300k with the defaults).
    #
    # Build the matrix ONCE for the path that does read it: every candidate shares the same n / stratify / seed
    # and differs only in calibrated y_pred, so a per-candidate ``bootstrap_metric`` would regenerate identical
    # resamples. One matrix reused across candidates -> same indices -> bit-identical CIs. Indices mirror
    # bootstrap_metric's RNG order.
    idx_matrix: Optional[np.ndarray] = None if inner_folds is not None else _build_resample_indices(n_oof, n_bootstrap, stratify, random_state)
    return inner_folds, idx_matrix


def score_candidates(
    cand_names: tuple[str, ...],
    oof_p_pos: np.ndarray,
    oof_y_arr: np.ndarray,
    inner_folds: Optional[list[np.ndarray]],
    idx_matrix: Optional[np.ndarray],
    n_bins: int,
    alpha: float,
) -> dict[str, dict[str, Any]]:
    """Fit every candidate calibrator on the OOF rows and score it; returns ``{name: {ece_mean, ece_ci, calibrated_probs}}``."""
    from mlframe.calibration.policy import _bootstrap_ece_with_indices, _ece_score, _fit_calibrator, _heldout_ece_ci, _heldout_ece_inner_cv

    results: dict[str, dict[str, Any]] = {}

    def metric_fn(_y: np.ndarray, _p: np.ndarray, _nb: int = n_bins) -> float:
        """ECE metric closure over the outer ``n_bins`` default, passed to the bootstrap/jackknife machinery."""
        return _ece_score(_y, _p, n_bins=_nb)

    for name in cand_names:
        apply_fn = _fit_calibrator(name, oof_p_pos, oof_y_arr)
        if apply_fn is None:
            continue
        try:
            cal_oof = np.asarray(apply_fn(oof_p_pos), dtype=np.float64).ravel()
            cal_oof = np.clip(cal_oof, 0.0, 1.0)
        except Exception as exc:
            log_throttle(logger, "calib_policy_oof_predict_failed", logging.WARNING, "pick_best_calibrator: %s.predict on OOF failed: %s", name, exc)
            continue
        # ``rank_ece`` drives selection: held-out (honest) for inner_cv, same-OOF (legacy) otherwise.
        rank_ece: float
        ece_ci: tuple
        if inner_folds is not None:
            ho = _heldout_ece_inner_cv(name, oof_p_pos, oof_y_arr, inner_folds, n_bins)
            if ho is None:
                continue
            rank_ece, fold_eces = ho
            # CI from the SAME held-out resampling as the point estimate, so the reported interval brackets the
            # reported number. An in-sample bootstrap CI paired with a held-out point would not.
            ece_ci = _heldout_ece_ci(rank_ece, fold_eces, alpha)
        else:
            try:
                ci = _bootstrap_ece_with_indices(oof_y_arr, cal_oof, cast(np.ndarray, idx_matrix), metric_fn, alpha, n_bins=n_bins)
            except Exception as exc:
                log_throttle(logger, "calib_policy_bootstrap_failed", logging.WARNING, "pick_best_calibrator: bootstrap on %s failed: %s", name, exc)
                continue
            rank_ece = float(ci["point"])
            ece_ci = (float(ci["lo"]), float(ci["hi"]))
        results[name] = {
            "ece_mean": rank_ece,
            "ece_ci": ece_ci,
            "calibrated_probs": cal_oof,
        }

    if not results:
        raise RuntimeError("pick_best_calibrator: every candidate calibrator failed; check optional deps " "(betacal, ml_insights) and OOF input shape.")
    return results


def choose_calibrator(results: dict[str, dict[str, Any]], selection: str, n_oof: int) -> tuple[str, str]:
    """Pick the winning candidate and the selection-rule label; returns ``(chosen_name, selection_rule)``."""
    from mlframe.calibration.policy import SMALL_SAMPLE_THRESHOLD, _cis_overlap

    ranked = sorted(results.items(), key=lambda kv: kv[1]["ece_mean"])
    chosen_name = ranked[0][0]
    selection_rule = "lowest_ece"
    if selection == "inner_cv":
        # Held-out ECE already removes the same-data optimism, so the lowest held-out ECE is the honest
        # winner; the same-OOF bootstrap CI (still reported) would mis-tie Isotonic, so no CI tiebreak here.
        selection_rule = "lowest_heldout_ece"
    elif len(ranked) > 1:
        top_ci = ranked[0][1]["ece_ci"]
        runner_ci = ranked[1][1]["ece_ci"]
        if _cis_overlap(top_ci, runner_ci):
            default_choice = "Isotonic" if n_oof >= SMALL_SAMPLE_THRESHOLD else "Beta"
            tied = [name for name, info in ranked if _cis_overlap(top_ci, info["ece_ci"])]
            if default_choice in tied:
                chosen_name = default_choice
                selection_rule = "default_isotonic" if default_choice == "Isotonic" else "default_beta"
            else:
                # Default candidate isn't in the tied subset; fall back to the lowest-mean.
                selection_rule = "lowest_ece_ci_overlap"
        else:
            selection_rule = "lowest_ece_ci_separated"
    return chosen_name, selection_rule
