"""Auto-select the best post-hoc calibrator by OOF ECE with bootstrap CI tiebreak.

The ``pick_best_calibrator`` helper benches a small palette of binary calibrators
(Sigmoid / Isotonic / Beta / Spline) on the OOF-train probabilities, computes the
ECE point estimate plus a percentile bootstrap CI (1000 resamples by default), and
returns the calibrator that minimises ECE — with a Kull-2017 default-rule fallback
(Isotonic for n_oof >= 1000, Beta otherwise) when the candidate CIs overlap so the
choice is non-arbitrary on small-sample / nearly-tied OOF fits.

Wire-up: ``post_calibrate_model`` consults ``CalibrationConfig.policy_auto_pick``
(default True) and threads the chosen calibrator into the meta-model fit. The
chosen calibrator + its CI is also stamped into the metadata report so a reviewer
can see at a glance which method the suite picked and how confident the OOF ECE
estimate is.

References:
  - Kull, Filho, Flach (AISTATS 2017) "Beta calibration".
  - Niculescu-Mizil & Caruana ICML 2005 "Predicting good probabilities".
  - Naeini et al. AAAI 2015 (ECE binning).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_ECE_NBINS: int = 15
DEFAULT_N_BOOTSTRAP: int = 1000
DEFAULT_ALPHA: float = 0.05
SMALL_SAMPLE_THRESHOLD: int = 1000

CANDIDATE_NAMES: tuple[str, ...] = ("Sigmoid", "Isotonic", "Beta", "Spline")


try:
    from numba import njit as _njit  # type: ignore
    _HAS_NUMBA = True
except Exception:  # pragma: no cover  -- numba is a hard dep, keep guard for static analysers
    _HAS_NUMBA = False
    def _njit(*args, **kwargs):  # type: ignore
        def _decorator(fn):
            return fn
        return _decorator


@_njit(cache=True, nogil=True)
def _ece_score_numba_serial(y: np.ndarray, p: np.ndarray, n_bins: int) -> float:
    """Single-pass per-bin reduction (12-15x vs numpy bincount on n=2k..1M).

    Streams ``(p[i], y[i])`` once, computing per-bin counts + sums in
    fixed-size float64 accumulators. Parallel variant exists in
    ``profiling/bench_ece_score_variants.py`` but pays prange overhead
    that the per-iter scalar work cannot amortise on n<1M -- the serial
    kernel wins on every size in the bench.
    """
    n = p.size
    sum_p = np.zeros(n_bins, dtype=np.float64)
    sum_y = np.zeros(n_bins, dtype=np.float64)
    n_finite = 0.0
    for i in range(n):
        pi = p[i]
        yi = y[i]
        if not (np.isfinite(pi) and np.isfinite(yi)):
            continue
        b = int(pi * n_bins)
        if b >= n_bins:
            b = n_bins - 1
        elif b < 0:
            b = 0
        sum_p[b] += pi
        sum_y[b] += yi
        n_finite += 1.0
    if n_finite == 0.0:
        return float("nan")
    total = 0.0
    for b in range(n_bins):
        diff = sum_y[b] - sum_p[b]
        if diff < 0.0:
            diff = -diff
        total += diff
    return total / n_finite


def _ece_score(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = DEFAULT_ECE_NBINS) -> float:
    """Equal-width ECE over ``n_bins`` for binary probability ``p_pred[:, 1]`` or 1-D ``p_pred``.

    Standard ECE: ``sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|`` over equal-width
    confidence bins on ``[0, 1]``. Returns nan when ``p_pred`` is empty or all-nan.

    iter309 (2026-05-26): numba single-pass reduction kernel. The
    iter308 ``np.bincount`` rewrite was 3.38x faster than the per-bin
    Python loop; this numba kernel is another ~12-15x faster than the
    bincount path because the per-i computation collapses to one branch
    + one integer cast + three accumulator updates, all inside a tight
    numba loop with no temporary arrays. Bench
    ``profiling/bench_ece_score_variants.py``:
      n=2k:    0.115 ms (numpy)   ->  0.008 ms (numba)  14.7x
      n=20k:   0.758 ms           ->  0.064 ms          11.9x
      n=200k:  9.413 ms           ->  0.636 ms          14.8x
      n=1M:   51.530 ms           ->  3.445 ms          15.0x
    Parallel variant tried and rejected: prange overhead dominates the
    per-iter scalar work; serial wins on every n in the bench.
    Numerical equivalence verified to <1e-12 vs the bincount path.

    Equivalence math: ``sum_b (count_b/n) * |conf_b - acc_b|`` with
    ``conf_b = sum_p_b / count_b`` and ``acc_b = sum_y_b / count_b``
    reduces to ``(1/n) * sum_b |sum_y_b - sum_p_b|`` because the count_b
    cancels between the per-bin weight times the per-bin magnitude.
    """
    p = np.asarray(p_pred, dtype=np.float64)
    if p.ndim == 2 and p.shape[1] >= 2:
        p = p[:, 1]
    p = np.ascontiguousarray(p.ravel())
    # iter598: dropped the unconditional ``dtype=np.float64`` cast on
    # y_true (same pattern as iter595/596/597). The kernel only uses
    # ``yi`` in ``sum_y[b] += yi`` where sum_y is float64; mixed-dtype
    # numba dispatch widens at the accumulator, identical to the upfront
    # cast result. Bench n=100k: int64 1.40x, int8 1.27x, float64 0.99x
    # (no harm); n=25k int64 (bootstrap typical) 1.33x. Bit-equivalent.
    y = np.ascontiguousarray(np.asarray(y_true).ravel())
    if p.size == 0 or y.size != p.size:
        return float("nan")
    return _ece_score_numba_serial(y, p, int(n_bins))


def _fit_calibrator(name: str, calib_p: np.ndarray, calib_y: np.ndarray) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Fit ``name`` on ``(calib_p, calib_y)``; return ``predict_proba_pos(p_test) -> p_test_calibrated`` or ``None``.

    Optional deps (betacal for Beta, ml_insights for Spline) are guarded; absent dep
    silently drops that candidate from the bench so the policy still runs with the
    remaining baseline (Sigmoid / Isotonic via sklearn).
    """
    p = np.asarray(calib_p, dtype=np.float64)
    if p.ndim == 2 and p.shape[1] >= 2:
        p = p[:, 1]
    p = p.reshape(-1, 1)
    y = np.asarray(calib_y).ravel()
    try:
        if name == "Sigmoid":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=1e10, solver="lbfgs")
            clf.fit(p, y)
            def _apply_sigmoid(probs: np.ndarray) -> np.ndarray:
                q = np.asarray(probs, dtype=np.float64)
                if q.ndim == 2 and q.shape[1] >= 2:
                    q = q[:, 1]
                return clf.predict_proba(q.reshape(-1, 1))[:, 1]
            return _apply_sigmoid
        if name == "Isotonic":
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(p.ravel(), y)
            def _apply_iso(probs: np.ndarray) -> np.ndarray:
                q = np.asarray(probs, dtype=np.float64)
                if q.ndim == 2 and q.shape[1] >= 2:
                    q = q[:, 1]
                return iso.predict(q.ravel())
            return _apply_iso
        if name == "Beta":
            try:
                from betacal import BetaCalibration
            except ImportError:
                logger.debug("pick_best_calibrator: betacal not installed; skipping Beta")
                return None
            beta = BetaCalibration(parameters="abm")
            beta.fit(p, y)
            def _apply_beta(probs: np.ndarray) -> np.ndarray:
                q = np.asarray(probs, dtype=np.float64)
                if q.ndim == 2 and q.shape[1] >= 2:
                    q = q[:, 1]
                out = beta.predict(q.reshape(-1, 1))
                return np.asarray(out).ravel()
            return _apply_beta
        if name == "Spline":
            try:
                import ml_insights as mli
            except ImportError:
                logger.debug("pick_best_calibrator: ml_insights not installed; skipping Spline")
                return None
            spline = mli.SplineCalib()
            spline.fit(p.ravel(), y)
            def _apply_spline(probs: np.ndarray) -> np.ndarray:
                q = np.asarray(probs, dtype=np.float64)
                if q.ndim == 2 and q.shape[1] >= 2:
                    q = q[:, 1]
                return np.asarray(spline.predict(q.ravel())).ravel()
            return _apply_spline
    except Exception as exc:
        logger.warning("pick_best_calibrator: %s fit failed: %s", name, exc)
        return None
    return None


def _cis_overlap(ci_a: tuple[float, float], ci_b: tuple[float, float]) -> bool:
    """True if two percentile CIs overlap (closed intervals)."""
    lo_a, hi_a = ci_a
    lo_b, hi_b = ci_b
    return not (hi_a < lo_b or hi_b < lo_a)


def _emit_reliability_plot(
    candidates: Mapping[str, dict[str, Any]],
    oof_probs: np.ndarray,
    oof_y: np.ndarray,
    plot_path: str,
    n_bins: int = DEFAULT_ECE_NBINS,
) -> Optional[str]:
    """Render a reliability diagram for every candidate alongside the raw OOF curve.

    Routed through the shared ``build_reliability_overlay_spec`` + renderer pipeline
    (a multi-series LinePanelSpec: perfect diagonal + raw OOF + per-candidate curves)
    so the reliability diagram has ONE implementation across the suite. Returns the
    absolute path on success, ``None`` if the render dependency is missing or the
    write fails.
    """
    try:
        from mlframe.reporting.charts.calibration import build_reliability_overlay_spec
        from mlframe.reporting.output import parse_plot_output_dsl
        from mlframe.reporting.renderers import render_and_save
    except ImportError as exc:
        logger.warning("pick_best_calibrator: reporting stack unavailable; skipping reliability plot: %s", exc)
        return None
    try:
        os.makedirs(os.path.dirname(os.path.abspath(plot_path)) or ".", exist_ok=True)
    except OSError as exc:
        logger.warning("pick_best_calibrator: could not create plot dir for %s: %s", plot_path, exc)
        return None

    raw_p = np.asarray(oof_probs, dtype=np.float64)
    if raw_p.ndim == 2 and raw_p.shape[1] >= 2:
        raw_p = raw_p[:, 1]
    raw_p = raw_p.ravel()
    y = np.asarray(oof_y, dtype=np.float64).ravel()

    calibrated = {
        name: np.asarray(info["calibrated_probs"]).ravel()
        for name, info in candidates.items()
        if info.get("calibrated_probs") is not None
    }
    labels = {
        name: f"{name} ECE={info['ece_mean']:.4f}"
        for name, info in candidates.items()
        if info.get("calibrated_probs") is not None
    }

    spec = build_reliability_overlay_spec(
        raw_p, y, calibrated_probs=calibrated, series_labels=labels, n_bins=n_bins,
    )

    root, ext = os.path.splitext(plot_path)
    fmt = ext.lstrip(".").lower()
    if fmt not in ("png", "pdf", "svg", "jpg", "jpeg"):
        fmt = "png"
        plot_path = root + ".png"
    try:
        render_and_save(spec, parse_plot_output_dsl(f"matplotlib[{fmt}]"), root, interactive=False)
    except OSError as exc:
        logger.warning("pick_best_calibrator: reliability render failed for %s: %s", plot_path, exc)
        return None
    return os.path.abspath(plot_path)


def pick_best_calibrator(
    probs: Optional[np.ndarray],
    y: Optional[np.ndarray],
    oof_probs: np.ndarray,
    oof_y: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    candidates: Optional[Iterable[str]] = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    n_bins: int = DEFAULT_ECE_NBINS,
    random_state: Optional[int] = 0,
    emit_plot: bool = False,
    plot_path: Optional[str] = None,
) -> dict[str, Any]:
    """Pick the calibrator that minimises OOF ECE (with bootstrap CI tiebreak).

    Parameters
    ----------
    probs, y
        Optional held-out probs / labels for diagnostic-only secondary ECE; not used
        for the selection decision (selection is OOF-only to keep test honest).
    oof_probs, oof_y
        OOF-train probs/labels — the calibrator fit + ECE benchmark surface. Required.
    alpha
        Two-sided coverage; default 0.05 -> 95% percentile CI.
    candidates
        Iterable of calibrator names to try; defaults to ``CANDIDATE_NAMES``. Unknown
        names are skipped with a warning.
    n_bootstrap
        Resample count for the OOF ECE CI; ``DEFAULT_N_BOOTSTRAP=1000``.
    n_bins
        ECE bin count; ``DEFAULT_ECE_NBINS=15`` mirrors the suite's standard report.
    random_state
        Seed for the bootstrap RNG. Pin for reproducibility.
    emit_plot
        When True, render a reliability diagram for every candidate to ``plot_path``.
    plot_path
        Output path; when ``emit_plot=True`` and ``plot_path is None``, defaults to
        ``reports/calibration_<utc_ts>.png`` in the working directory.

    Returns
    -------
    dict
        ``{"chosen": <name>, "ece_mean": ..., "ece_ci": (lo, hi),
           "alternatives": {<name>: {"ece_mean", "ece_ci"}}, "rule": <selection-rule>,
           "n_oof": int, "plot_path": Optional[str]}``.

    Selection rule
    --------------
    1. Bench every candidate; compute OOF ECE + bootstrap CI.
    2. Sort by ECE mean ascending.
    3. If the top candidate's CI does NOT overlap the runner-up's, return top.
    4. Otherwise apply Kull-2017 default: Isotonic when ``n_oof >= 1000``, Beta when
       ``n_oof < 1000``; if the default isn't in the OOF-tied subset, fall through
       to the lowest-ECE candidate.
    """
    from mlframe.evaluation.bootstrap import bootstrap_metric

    oof_p = np.asarray(oof_probs, dtype=np.float64)
    if oof_p.ndim == 2 and oof_p.shape[1] >= 2:
        oof_p_pos = oof_p[:, 1]
    else:
        oof_p_pos = oof_p.ravel()
    oof_y_arr = np.asarray(oof_y).ravel()
    n_oof = int(oof_y_arr.shape[0])
    if oof_p_pos.shape[0] != n_oof:
        raise ValueError(
            f"pick_best_calibrator: oof_probs rows ({oof_p_pos.shape[0]}) do not match oof_y ({n_oof})"
        )
    if n_oof < 4:
        raise ValueError(f"pick_best_calibrator: need at least 4 OOF rows; got n_oof={n_oof}")

    cand_names = tuple(candidates) if candidates is not None else CANDIDATE_NAMES
    unknown = [c for c in cand_names if c not in CANDIDATE_NAMES]
    if unknown:
        logger.warning("pick_best_calibrator: unknown candidate(s) ignored: %s", unknown)
    cand_names = tuple(c for c in cand_names if c in CANDIDATE_NAMES)
    if not cand_names:
        raise ValueError(f"pick_best_calibrator: no valid candidate names; allowed={CANDIDATE_NAMES}")

    results: dict[str, dict[str, Any]] = {}
    classes = np.unique(oof_y_arr)
    stratify = oof_y_arr if classes.size == 2 else None

    metric_fn = lambda _y, _p, _nb=n_bins: _ece_score(_y, _p, n_bins=_nb)

    for name in cand_names:
        apply_fn = _fit_calibrator(name, oof_p_pos, oof_y_arr)
        if apply_fn is None:
            continue
        try:
            cal_oof = np.asarray(apply_fn(oof_p_pos), dtype=np.float64).ravel()
            cal_oof = np.clip(cal_oof, 0.0, 1.0)
        except Exception as exc:
            logger.warning("pick_best_calibrator: %s.predict on OOF failed: %s", name, exc)
            continue
        try:
            ci = bootstrap_metric(
                oof_y_arr,
                cal_oof,
                metric_fn=metric_fn,
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                stratify=stratify,
                random_state=random_state,
            )
        except Exception as exc:
            logger.warning("pick_best_calibrator: bootstrap on %s failed: %s", name, exc)
            continue
        results[name] = {
            "ece_mean": float(ci["point"]),
            "ece_ci": (float(ci["lo"]), float(ci["hi"])),
            "calibrated_probs": cal_oof,
        }

    if not results:
        raise RuntimeError(
            "pick_best_calibrator: every candidate calibrator failed; check optional deps "
            "(betacal, ml_insights) and OOF input shape."
        )

    ranked = sorted(results.items(), key=lambda kv: kv[1]["ece_mean"])
    chosen_name = ranked[0][0]
    selection_rule = "lowest_ece"
    if len(ranked) > 1:
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

    plot_out: Optional[str] = None
    if emit_plot:
        if plot_path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            plot_path = os.path.join("reports", f"calibration_{ts}.png")
        plot_out = _emit_reliability_plot(results, oof_p_pos, oof_y_arr, plot_path, n_bins=n_bins)

    return {
        "chosen": chosen_name,
        "ece_mean": float(results[chosen_name]["ece_mean"]),
        "ece_ci": tuple(results[chosen_name]["ece_ci"]),
        "alternatives": {
            name: {"ece_mean": info["ece_mean"], "ece_ci": info["ece_ci"]}
            for name, info in results.items()
        },
        "rule": selection_rule,
        "n_oof": n_oof,
        "plot_path": plot_out,
    }


@dataclass
class CalibrationConfig:
    """Calibration-policy knobs for ``post_calibrate_model``.

    Currently a single field; kept as a dataclass so further policy knobs (ECE bin
    count override, candidate set override, plot emission toggle) can be added
    without breaking call sites.

    Parameters
    ----------
    policy_auto_pick
        When True (default), ``post_calibrate_model`` invokes
        :func:`pick_best_calibrator` on the OOF probs and uses its decision in
        addition to (not in place of) the legacy meta-model path. The chosen
        calibrator + CI is stamped into the metrics dict under
        ``metadata["calibration_policy"]`` for downstream consumers (honest
        diagnostics report, ops dashboards).
    emit_plot
        When True, the reliability plot is rendered to ``plot_path``.
    plot_path
        Optional explicit path for the reliability plot. ``None`` = auto-generate
        ``reports/calibration_<utc_ts>.png``.
    n_bootstrap
        Bootstrap resample count for the OOF ECE CI (default 1000).
    alpha
        CI coverage; 0.05 -> 95% CI.
    candidates
        Restricted candidate set; ``None`` = ``CANDIDATE_NAMES``.
    """

    policy_auto_pick: bool = True
    emit_plot: bool = False
    plot_path: Optional[str] = None
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    alpha: float = DEFAULT_ALPHA
    candidates: Optional[tuple[str, ...]] = None


__all__ = ["pick_best_calibrator", "CalibrationConfig", "CANDIDATE_NAMES", "DEFAULT_ECE_NBINS"]
