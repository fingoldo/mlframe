"""Quality-gate source-selection + catastrophic-MAE early-drop helpers carved out of ``mlframe.models.score``.

Three helpers, each behavioural-equivalent to the original inline block:

1. ``select_gate_source_split`` -- chooses the gate-source (OOF preferred, val/test/train fallback) and applies coarse-gate threshold flips.
2. ``catastrophic_drop_kn`` -- K>2 absolute-target-MAE catastrophic drop.
3. ``catastrophic_drop_k2`` -- K=2 single-member dropout w/ early-return signal.

All three preserve the original side-effects: logger.warning / info lines on the parent module's logger, in-place ``res`` dict stamping, list rebinding.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("mlframe.models.ensembling")

# Matches the "[member+tags]" label bracket inside an ensemble_name, e.g. "[cb+xgb] level2".
_ENSEMBLE_LABEL_RE = re.compile(r"\[[^\]]+\]")


def select_gate_source_split(
    *,
    level_models_and_predictions,
    require_oof_for_gate: bool,
    coarse_gate_max_mae_relative: float,
    coarse_gate_max_std_relative: float,
    max_mae: float,
    max_std: float,
    max_mae_relative: float,
    max_std_relative: float,
    verbose: bool,
) -> Tuple[Optional[List[np.ndarray]], Optional[str], bool, float, float, float, float]:
    """Pick the gate-source (OOF preferred) and apply coarse-gate threshold flips.

    Returns ``(_gate_preds_for_check, _gate_source_split, _coarse_gate_active, max_mae, max_std, max_mae_relative, max_std_relative)``. The four threshold floats are returned because the coarse-gate path mutates them in place in the original.
    """
    _gate_source_split: Optional[str] = None
    _gate_preds_for_check: Optional[List[np.ndarray]] = None
    # GATE-DOUBLE-DIP / GATE-NO-OOF: prefer oof_* exclusively; fall back to val/test/train only when require_oof_for_gate is False. When True and any member lacks OOF we WARN and skip the gate entirely (better to run all members than to gate on the same surface the early-stopper / test-set selector burned). The "all members must share a split" condition stays the same -- mixing splits across members would compare incomparable rows.
    _candidate_attrs: Tuple[Tuple[str, str], ...] = (
        ("oof_preds", "oof"),
        ("oof_probs", "oof"),
    )
    if not require_oof_for_gate:
        _candidate_attrs = _candidate_attrs + (
            ("val_preds", "val"),
            ("test_preds", "test"),
            ("train_preds", "train"),
            ("val_probs", "val"),
            ("test_probs", "test"),
            ("train_probs", "train"),
        )
    for _attr, _label in _candidate_attrs:
        _candidate = [getattr(m, _attr, None) for m in level_models_and_predictions]
        # MagicMock test doubles fabricate any attribute access, so ``p is not None`` would always pass; require an actual numpy array to gate the source-split selection.
        if all(isinstance(p, np.ndarray) for p in _candidate):
            _gate_preds_for_check = _candidate  # type: ignore[assignment]  # narrowed to list[np.ndarray] by the all(isinstance(...)) check above
            _gate_source_split = _label
            break
    # COARSE-GATE-FALLBACK: when OOF is unavailable AND require_oof_for_gate=True, the fine-grained 2.5x gate is intentionally skipped to avoid double-dipping on val. But that lets catastrophic outliers (R^2=-4.75 alongside R^2=0.99 members) survive into the ensemble. Run a SECOND coarse-threshold pass against the val/test/train fallback chain to catch only the disasters. Marked separately in logs (split=val-coarse) so it can't be confused with the strict gate.
    _coarse_gate_active = False
    if require_oof_for_gate and _gate_preds_for_check is None and (coarse_gate_max_mae_relative > 0.0 or coarse_gate_max_std_relative > 0.0):
        _coarse_fallback_attrs = (
            ("val_preds", "val-coarse"),
            ("test_preds", "test-coarse"),
            ("train_preds", "train-coarse"),
            ("val_probs", "val-coarse"),
            ("test_probs", "test-coarse"),
            ("train_probs", "train-coarse"),
        )
        for _attr, _label in _coarse_fallback_attrs:
            _candidate = [getattr(m, _attr, None) for m in level_models_and_predictions]
            if all(isinstance(p, np.ndarray) for p in _candidate):
                _gate_preds_for_check = _candidate  # type: ignore[assignment]  # narrowed to list[np.ndarray] by the all(isinstance(...)) check above
                _gate_source_split = _label
                _coarse_gate_active = True
                # Coarse pass replaces fine thresholds for the single call below; the embedded per-flavor filter has already been zeroed before this block runs (or will be zeroed in the kept-survivors branch below), so this only governs the single compute_member_quality_gate invocation that follows.
                max_mae = 0.0
                max_std = 0.0
                max_mae_relative = float(coarse_gate_max_mae_relative)
                max_std_relative = float(coarse_gate_max_std_relative)
                break
        if verbose:
            if _coarse_gate_active:
                logger.warning(
                    "[ensemble] OOF unavailable; running COARSE gate on %s at %.1fx median MAE / %.1fx median STD (catches catastrophic outliers only; theoretical val double-dip risk acknowledged).",
                    _gate_source_split, max_mae_relative, max_std_relative,
                )
            else:
                logger.warning(
                    "[ensemble] require_oof_for_gate=True but at least one member lacks OOF preds AND no val/test/train fallback available; skipping quality gate entirely."
                )
    elif require_oof_for_gate and _gate_preds_for_check is None and verbose:
        logger.warning(
            "[ensemble] require_oof_for_gate=True but at least one member lacks OOF preds; coarse-gate disabled (coarse_gate_max_mae_relative<=0); skipping quality gate."
        )
    return (
        _gate_preds_for_check,
        _gate_source_split,
        _coarse_gate_active,
        max_mae,
        max_std,
        max_mae_relative,
        max_std_relative,
    )


def catastrophic_drop_kn(
    *,
    level_models_and_predictions,
    _gate_preds_for_check: Optional[List[np.ndarray]],
    _gate_source_split: Optional[str],
    _ensemble_member_tags: List[str],
    _ensemble_short_tags: List[str],
    train_target_arr,
    val_target_arr,
    test_target_arr,
    k2_catastrophic_mae_ratio: float,
    verbose: bool,
    res: dict,
) -> Tuple[Any, Optional[List[np.ndarray]], List[str], List[str]]:
    """K>2 absolute target-MAE catastrophic drop.

    The legacy peer-median gate at line 2106 below uses median MAE across members for K>2 but the median IS THE GROUP being judged -- if 2 of 4 members are catastrophic the median MAE is HALF-catastrophic and the relative threshold (2.5x median) may still let the disasters through. This helper runs FIRST when target is available for the gate-source split: any member whose MAE-to-target exceeds the catastrophic ratio relative to the BEST member is removed before the peer-median gate sees it.

    Mutates ``res`` in place (matches the legacy contract). Returns possibly-sliced ``(level_models_and_predictions, _gate_preds_for_check, _ensemble_member_tags, _ensemble_short_tags)``.
    """
    if not (_gate_preds_for_check is not None and len(_gate_preds_for_check) > 2 and k2_catastrophic_mae_ratio > 1.0):
        return level_models_and_predictions, _gate_preds_for_check, _ensemble_member_tags, _ensemble_short_tags
    _gate_target_arr_kn = None
    if _gate_source_split in ("test", "test-coarse"):
        _gate_target_arr_kn = test_target_arr
    elif _gate_source_split in ("val", "val-coarse"):
        _gate_target_arr_kn = val_target_arr
    elif _gate_source_split in ("train", "train-coarse"):
        _gate_target_arr_kn = train_target_arr
    elif _gate_source_split == "oof":
        _gate_target_arr_kn = train_target_arr
    if not (isinstance(_gate_target_arr_kn, np.ndarray) and _gate_target_arr_kn.size > 0):
        return level_models_and_predictions, _gate_preds_for_check, _ensemble_member_tags, _ensemble_short_tags
    try:
        _t_kn = _gate_target_arr_kn.reshape(-1).astype(np.float64)
        _maes = []
        for _p in _gate_preds_for_check:
            _p_arr = np.asarray(_p).reshape(-1).astype(np.float64)
            if _p_arr.shape == _t_kn.shape:
                _maes.append(float(np.nanmean(np.abs(_p_arr - _t_kn))))
            else:
                _maes.append(float("nan"))
        _maes_arr = np.asarray(_maes)
        # Defensive: if every per-member MAE was NaN (shape mismatch on every member, or t_kn itself NaN), ``np.nanmin`` emits ``RuntimeWarning: All-NaN slice encountered`` and returns NaN. The subsequent ``> 0.0`` short-circuits the diagnostic block so the warning is the only observable -- but it pollutes the log with no actionable signal. Short-circuit cleanly when all-NaN: skip the blowout diagnostic block.
        if np.all(np.isnan(_maes_arr)):
            _best_mae = float("nan")
        else:
            _best_mae = float(np.nanmin(_maes_arr))
        if _best_mae > 0.0 and math.isfinite(_best_mae):
            _ratios = _maes_arr / _best_mae
            # E4.3 (2026-05-21): surface a borderline-member diagnostic for operators. Any member whose target-MAE exceeds 0.5 * catastrophic_ratio (default 10x best) but stays below the catastrophic_ratio (20x best) is "borderline" -- not dropped, but worth knowing in metadata for the next-session tune of model selection.
            _blowout_floor = 0.5 * float(k2_catastrophic_mae_ratio)
            _blowout_idx = [
                i for i in range(len(_gate_preds_for_check)) if math.isfinite(_ratios[i]) and _blowout_floor <= _ratios[i] < float(k2_catastrophic_mae_ratio)
            ]
            if _blowout_idx:
                res["_diagnostic_mae_blowout"] = {
                    "split": _gate_source_split,
                    "borderline_idx": _blowout_idx,
                    "borderline_member_tags": [_ensemble_member_tags[i] for i in _blowout_idx],
                    "per_member_target_mae": _maes,
                    "best_mae": _best_mae,
                    "borderline_floor_ratio": _blowout_floor,
                    "catastrophic_ratio_threshold": float(k2_catastrophic_mae_ratio),
                }
                if verbose:
                    logger.warning(
                        "[ensemble] K>%d borderline-MAE diagnostic (split=%s): %d member(s) "
                        "between %.1fx and %.1fx best MAE -- not dropped, but worth review: %s",
                        2, _gate_source_split, len(_blowout_idx),
                        _blowout_floor, float(k2_catastrophic_mae_ratio),
                        [_ensemble_member_tags[i] for i in _blowout_idx],
                    )

            _drop_mask = (_ratios >= float(k2_catastrophic_mae_ratio)) | ~np.isfinite(_maes_arr)
            _kept_idx_kn = [i for i in range(len(_gate_preds_for_check)) if not _drop_mask[i]]
            _excl_kn = [(i, f"target_mae={_maes[i]:.4f} vs best={_best_mae:.4f} ratio={_ratios[i]:.1f}x") for i in range(len(_gate_preds_for_check)) if _drop_mask[i]]
            # E4.2 (2026-05-21): all-members-catastrophic sentinel. If every member's target-MAE either NaNs or exceeds the catastrophic threshold relative to the best one (rare edge case: best is barely-finite, others are way worse), the peer-median fallback would just keep everyone. Detect explicitly and signal the suite to skip ensembling for this target.
            if len(_kept_idx_kn) < 2 and len(_excl_kn) >= 1:
                # Either 1 or 0 survivors -- treat as "no honest ensemble possible". Use a separate sentinel so the caller (suite finalize) can decide: ship-best-single-member or fall back to dummy baseline.
                res["_all_members_catastrophic"] = {
                    "split": _gate_source_split,
                    "n_survivors": len(_kept_idx_kn),
                    "n_members": len(_gate_preds_for_check),
                    "best_mae": _best_mae,
                    "per_member_target_mae": _maes,
                    "ratio_threshold": float(k2_catastrophic_mae_ratio),
                }
                if verbose:
                    logger.warning(
                        "[ensemble] K>2 all-members-catastrophic (split=%s): only %d/%d "
                        "members within %.1fx of best MAE=%.4f. Caller should skip "
                        "ensembling and ship the single survivor (or dummy baseline if 0).",
                        _gate_source_split, len(_kept_idx_kn),
                        len(_gate_preds_for_check), float(k2_catastrophic_mae_ratio),
                        _best_mae,
                    )
            if len(_kept_idx_kn) >= 1 and len(_excl_kn) >= 1:
                if verbose:
                    _drop_tags = [_ensemble_member_tags[i] for i, _ in _excl_kn]
                    _kept_tags_log = [_ensemble_member_tags[i] for i in _kept_idx_kn]
                    logger.warning(
                        "[ensemble] K>2 absolute-MAE catastrophic-drop (split=%s): dropping %s; keeping %s (ratio threshold=%.1fx, best MAE=%.4f).",
                        _gate_source_split, _drop_tags, _kept_tags_log,
                        float(k2_catastrophic_mae_ratio), _best_mae,
                    )
                # Slice members + tags down to the survivor set; peer-median gate below then runs on the cleaner pool.
                level_models_and_predictions = [level_models_and_predictions[i] for i in _kept_idx_kn]
                _gate_preds_for_check = [_gate_preds_for_check[i] for i in _kept_idx_kn]
                _ensemble_member_tags = [_ensemble_member_tags[i] for i in _kept_idx_kn]
                _ensemble_short_tags = [_ensemble_short_tags[i] for i in _kept_idx_kn]
                # Stamp diagnostic into the result dict so finalize / metadata reflect the pre-median-gate purge. Key uses ``_`` prefix per the sentinel-key contract (P0 #4 follow-up: callers filter ``_``-prefixed keys when building the per-target model list).
                res["_kn_catastrophic_dropped"] = {
                    "split": _gate_source_split,
                    "dropped_idx": [i for i, _ in _excl_kn],
                    "best_mae": _best_mae,
                    "ratio_threshold": float(k2_catastrophic_mae_ratio),
                    "per_member_target_mae": _maes,
                }
    except Exception as _kn_err:
        if verbose:
            logger.warning("[ensemble] K>2 catastrophic-drop check raised %s; proceeding without drop.", _kn_err)
    return level_models_and_predictions, _gate_preds_for_check, _ensemble_member_tags, _ensemble_short_tags


def catastrophic_drop_k2(
    *,
    level_models_and_predictions,
    _gate_preds_for_check: Optional[List[np.ndarray]],
    _gate_source_split: Optional[str],
    _ensemble_member_tags: List[str],
    _ensemble_short_tags: List[str],
    ensemble_name: str,
    train_target_arr,
    val_target_arr,
    test_target_arr,
    k2_catastrophic_mae_ratio: float,
    verbose: bool,
    res: dict,
) -> Tuple[Any, List[str], List[str], str, bool]:
    """K=2 catastrophic-dropout check.

    When K == 2, the legacy peer-median gate is symmetric (both members equidistant from (a+b)/2). When TARGET is available for the gate-source split, this helper compares per-member MAE-to-target directly and drops the obvious catastrophic outlier.

    Mutates ``res`` in place; returns ``(level_models_and_predictions, _ensemble_member_tags, _ensemble_short_tags, ensemble_name, early_return)``. When ``early_return`` is True the caller must immediately ``return res`` (legacy single-member short-circuit path).
    """
    if not (_gate_preds_for_check is not None and len(_gate_preds_for_check) == 2 and k2_catastrophic_mae_ratio > 1.0):
        return level_models_and_predictions, _ensemble_member_tags, _ensemble_short_tags, ensemble_name, False
    # Match the gate source to its target array. ``oof_*`` aligns with train rows; ``val/test/train`` (and their *-coarse variants) align with the matching target_arr that score_ensemble already produced above.
    _gate_target_arr = None
    if _gate_source_split in ("test", "test-coarse"):
        _gate_target_arr = test_target_arr
    elif _gate_source_split in ("val", "val-coarse"):
        _gate_target_arr = val_target_arr
    elif _gate_source_split in ("train", "train-coarse"):
        _gate_target_arr = train_target_arr
    elif _gate_source_split == "oof":
        _gate_target_arr = train_target_arr
    if not (isinstance(_gate_target_arr, np.ndarray) and _gate_target_arr.size > 0):
        return level_models_and_predictions, _ensemble_member_tags, _ensemble_short_tags, ensemble_name, False
    try:
        _t = _gate_target_arr.reshape(-1).astype(np.float64)
        _p0 = np.asarray(_gate_preds_for_check[0]).reshape(-1).astype(np.float64)
        _p1 = np.asarray(_gate_preds_for_check[1]).reshape(-1).astype(np.float64)
        if _p0.shape == _t.shape and _p1.shape == _t.shape:
            # Use nanmean so a NaN-poisoned member doesn't make the gate skip.
            _mae_a = float(np.nanmean(np.abs(_p0 - _t)))
            _mae_b = float(np.nanmean(np.abs(_p1 - _t)))
            _worse = max(_mae_a, _mae_b)
            _better = min(_mae_a, _mae_b)
            if _better > 0.0 and math.isfinite(_worse / _better) and _worse / _better >= float(k2_catastrophic_mae_ratio):
                # Deterministic tiebreak: when both members are within fp64 noise of each other (``_mae_a == _mae_b``), pick the alphabetically-larger tag as the "worse" so removal is reproducible regardless of BLAS thread count / input ordering. Without this, an exact tie always drops index 1 (silent bias toward whichever member happened to be supplied second).
                if _mae_a > _mae_b:
                    _worse_idx = 0
                elif _mae_b > _mae_a:
                    _worse_idx = 1
                else:
                    _tag_a = _ensemble_member_tags[0]
                    _tag_b = _ensemble_member_tags[1]
                    _worse_idx = 0 if _tag_a > _tag_b else 1
                # Capture the dropped tag BEFORE the slicing below overwrites the tag list.
                _dropped_tag = _ensemble_member_tags[_worse_idx]
                _kept_tag = _ensemble_member_tags[1 - _worse_idx]
                if verbose:
                    logger.warning(
                        "[ensemble] K=2 catastrophic-dropout (split=%s): dropping %s (MAE=%.4f), keeping %s (MAE=%.4f); ratio=%.1fx >= %.1fx threshold.",
                        _gate_source_split,
                        _dropped_tag, _worse,
                        _kept_tag, _better,
                        _worse / _better, float(k2_catastrophic_mae_ratio),
                    )
                level_models_and_predictions = [level_models_and_predictions[1 - _worse_idx]]
                _ensemble_member_tags = [_kept_tag]
                _ensemble_short_tags = [_ensemble_short_tags[1 - _worse_idx]]
                # Refresh ensemble_name to reflect the surviving single member.
                try:
                    _kept_tags = _ensemble_short_tags
                    _re_label = "[" + "+".join(_kept_tags) + "]"
                    if _ENSEMBLE_LABEL_RE.search(ensemble_name):
                        _label_value: str = _re_label

                        def _replace_label(_m, _v: str = _label_value) -> str:
                            return _v

                        ensemble_name = _ENSEMBLE_LABEL_RE.sub(
                            _replace_label,
                            ensemble_name, count=1,
                        )
                    else:
                        ensemble_name = f"{_re_label} {ensemble_name}".rstrip() if ensemble_name else _re_label
                except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                    logger.debug("suppressed in score_gate.py:319: %s", e)
                    pass
                # Drop into single-member short-circuit: with K=1 surviving, there's no ensemble to score. Return the sentinel-only dict exactly as the upstream "single_member" branch does. Operators see "_reason: k2_catastrophic_dropout" in the result and can choose to keep just the surviving member.
                res["_reason"] = "k2_catastrophic_dropout"
                res["_n_members"] = 1
                res["_dropped_member"] = _dropped_tag
                res["_kept_member"] = _kept_tag
                res["_k2_mae_ratio"] = _worse / _better
                return level_models_and_predictions, _ensemble_member_tags, _ensemble_short_tags, ensemble_name, True
    except Exception as _k2_err:
        if verbose:
            logger.warning("[ensemble] K=2 catastrophic-dropout check raised %s; proceeding without drop.", _k2_err)
    return level_models_and_predictions, _ensemble_member_tags, _ensemble_short_tags, ensemble_name, False
