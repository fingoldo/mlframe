"""``train_postcalibrators``, carved out of ``post.py`` (X_EFFICIENCY_ARCHITECTURE-1 fix,
mrmr_audit_2026-07-22) to clear the repo's enforced hard 1000-LOC CI gate (that file was 1005 lines).
Behaviour preserved bit-for-bit; ``post.py`` re-exports this function so every existing
``from mlframe.calibration.post import train_postcalibrators`` import keeps working unchanged.
"""
from __future__ import annotations

import logging
from os.path import join
from typing import Any, Optional, Sequence

import numpy as np
import joblib
from pyutilz.strings import slugify

from mlframe.training import TargetTypes

logger = logging.getLogger("mlframe.calibration.post")  # matches the pre-carve logger name (post.py); preserves log-filter/caplog compatibility for existing callers/tests


def train_postcalibrators(
    models: dict,
    model_name: str,
    models_dir: str,
    target_name: str,
    featureset_name: str,
    task_type: Any = TargetTypes.BINARY_CLASSIFICATION,
    include_patterns: Optional[list] = None,
    max_mae: Optional[float] = None,
    max_std: Optional[float] = None,
    ensembling_method: Optional[str] = None,
    ensure_prob_limits: bool = True,
    uncertainty_quantile: float = 0.0,
    normalize_stds_by_mean_preds: bool = True,
    verbose: int = 1,
    *,
    calib_probs_per_model: Optional[Sequence[np.ndarray]] = None,
    calib_target: Optional[np.ndarray] = None,
    metadata: Optional[dict] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> dict:
    """Fit postcalibrators on a DISJOINT calibration split.

    Calibrators MUST be fit on a calibration split that is disjoint from BOTH the training set
    (where the models were trained) AND the honest test/holdout set (used for honest evaluation
    of calibrated metrics).

    Required parameters (callers MUST supply BOTH):
      - ``calib_probs_per_model``: an iterable of per-model probability arrays aligned to a
        dedicated calibration split. Row i across every member must refer to the SAME row of
        ``calib_target``. This function will NOT fall back to ``model.test_probs``: using the
        honest test set to fit calibrators converts every later test-set calibration metric
        into an in-sample read-out.
      - ``calib_target``: the target vector aligned to ``calib_probs_per_model``.

    The function asserts that ``calib_target`` is NOT identical to any model's ``.test_target``
    (no silent same-set fallback), and refuses if either parameter is missing.

    ``ensembling_method`` defaults to the suite-chosen flavour from
    ``metadata["ensembles_chosen"]`` (per target). Pass an explicit string to override; pass
    ``None`` and supply ``metadata`` to inherit the suite winner.

    ``sample_weight``, when given, is a per-row weight vector aligned to ``calib_target``, forwarded to
    ``compare_postcalibrators`` (see that function's docstring for exactly which wrapped calibrators honor
    it). ``None`` (default) preserves the pre-existing fully-unweighted calibration fitting.

    Returns
    -------
    dict
        ``{"calibrators": {name: fitted_calibrator, ...}, "metrics": {name: calib_set_metrics, ...},
        "failed_calibrators": {name: repr(exception), ...}}``. ``metrics`` is the
        ``compare_postcalibrators`` comparison table on the calib set (evaluated on inner-CV held-out
        rows by default -- see ``compare_postcalibrators``'s ``selection`` parameter -- so it is no
        longer the same rows used to fit the winning calibrator); also logged at INFO level.
        ``failed_calibrators`` lists any candidate that raised during fit/predict and was excluded.
    """
    from .post import _CalibTestOverlapError, _values_overlap_fraction, compare_postcalibrators

    if include_patterns is None:
        include_patterns = [r"SplineCalib", r"pycalib.BetaCalibration"]

    if calib_probs_per_model is None or calib_target is None:
        raise ValueError(
            "train_postcalibrators requires explicit calib_probs_per_model + calib_target "
            "from a DISJOINT calibration split (not the train set, and not the honest test set). "
            "Falling back to model.test_probs would fit calibrators on the same rows used to "
            "report honest test metrics, converting every later test-set calibration read-out "
            "into an in-sample measurement. Pre-allocate a held-out calibration slice in the "
            "suite caller and pass it positionally."
        )

    # Refuse silent same-set reuse: if ``calib_target``/``calib_probs_per_model`` overlap substantially
    # with any model's ``.test_target``/``.test_probs`` we raise. This is the defence against the
    # historical "test == calib" bug -- the caller may not realise they have shared the slot. Beyond
    # the original exact-array-equality check (order-and-shape identical), this also catches: the same
    # rows reshuffled (different order), a strict subset/superset of the same rows (different shape),
    # row-ID/index overlap when an index is available, and probability-row reuse even when the target
    # vectors themselves were reshuffled/relabelled independently.
    OVERLAP_RAISE_THRESHOLD = 0.9
    OVERLAP_WARN_THRESHOLD = 0.3
    _calib_target_np = np.asarray(calib_target)
    _calib_index = calib_target.index if hasattr(calib_target, "index") else None
    for _name, _m in models.items():
        _tt = getattr(_m, "test_target", None)
        if _tt is not None:
            try:
                _tt_np = _tt.values if hasattr(_tt, "values") else np.asarray(_tt)
            except Exception as exc:
                logger.debug(
                    "train_postcalibrators: could not coerce model %r test_target for calib==test safety check; "
                    "skipping this model in the identity guard: %r", _name, exc, exc_info=True
                )
                _tt_np = None
            if _tt_np is not None and _tt_np.shape == _calib_target_np.shape:
                # array_equal handles dtype mismatches and avoids deprecation noise on ndarray==
                try:
                    if np.array_equal(_tt_np, _calib_target_np):
                        raise _CalibTestOverlapError(
                            "train_postcalibrators: calib_target appears to be identical to "
                            f"model '{_name}'.test_target. Refusing to fit calibrators on the "
                            "honest holdout split (in-sample calibration read-out). Use a "
                            "dedicated calibration split disjoint from test."
                        )
                    # Same shape, not equal in order -- check whether it's the SAME multiset of values just
                    # reshuffled (the exact-equality check above is blind to row order).
                    if np.array_equal(np.sort(_tt_np, axis=None), np.sort(_calib_target_np, axis=None)):
                        raise _CalibTestOverlapError(
                            "train_postcalibrators: calib_target has the SAME multiset of values as model "
                            f"'{_name}'.test_target -- same rows, reordered. Refusing to fit calibrators on a "
                            "reshuffled honest holdout. Use a dedicated calibration split disjoint from test."
                        )
                except _CalibTestOverlapError:
                    raise
                except (TypeError, ValueError):
                    pass  # Non-comparable dtypes (e.g. mixed-type pandas Series) -- skip the equality check.

            # Row-ID/index overlap: catches subset/superset reuse (different shape, so the checks above
            # never ran) and reordering even when dtypes prevent a direct value comparison.
            _tt_index = _tt.index if hasattr(_tt, "index") else None
            if _tt_index is not None and _calib_index is not None:
                try:
                    _tt_idx_set = set(_tt_index)
                    _calib_idx_set = set(_calib_index)
                    _smaller = min(len(_tt_idx_set), len(_calib_idx_set))
                    if _smaller > 0:
                        _idx_overlap = len(_tt_idx_set & _calib_idx_set) / _smaller
                        if _idx_overlap >= OVERLAP_RAISE_THRESHOLD:
                            raise ValueError(
                                f"train_postcalibrators: calib_target's row index overlaps {_idx_overlap:.0%} with "
                                f"model '{_name}'.test_target's row index (>= {OVERLAP_RAISE_THRESHOLD:.0%} "
                                "threshold) -- the same underlying rows, reordered or as a subset. Refusing to fit "
                                "calibrators on rows that substantially overlap the honest holdout split."
                            )
                        if _idx_overlap >= OVERLAP_WARN_THRESHOLD:
                            logger.warning(
                                "train_postcalibrators: calib_target's row index overlaps %.0f%% with model %r's "
                                "test_target row index. This may be partial calib/test leakage -- verify the splits "
                                "are disjoint.",
                                _idx_overlap * 100, _name,
                            )
                except TypeError as exc:
                    logger.debug(
                        "train_postcalibrators: could not hash row index for overlap check on model %r: %r",
                        _name, exc, exc_info=True,
                    )

        # Probability-row overlap: cross-checks calib_probs_per_model against model.test_probs directly
        # (not just the target vector), which the pre-fix guard never did at all -- a caller who
        # reshuffled/relabelled y for the calib call while reusing model.test_probs would slip through
        # every target-only check above.
        _test_probs = getattr(_m, "test_probs", None)
        if _test_probs is not None and calib_probs_per_model:
            for _cp in calib_probs_per_model:
                try:
                    _prob_overlap = _values_overlap_fraction(np.asarray(_cp), np.asarray(_test_probs))
                except Exception as exc:
                    logger.debug("train_postcalibrators: probs overlap check failed for model %r: %r", _name, exc, exc_info=True)
                    continue
                if _prob_overlap >= OVERLAP_RAISE_THRESHOLD:
                    raise ValueError(
                        f"train_postcalibrators: a calib_probs_per_model array overlaps {_prob_overlap:.0%} (by row "
                        f"value) with model '{_name}'.test_probs -- the same underlying probability rows are being "
                        "reused for both calibration fitting and honest test evaluation. Use a dedicated calibration "
                        "split disjoint from test."
                    )
                if _prob_overlap >= OVERLAP_WARN_THRESHOLD:
                    logger.warning(
                        "train_postcalibrators: a calib_probs_per_model array overlaps %.0f%% (by row value) with "
                        "model %r.test_probs. This may be partial calib/test leakage -- verify the splits are disjoint.",
                        _prob_overlap * 100, _name,
                    )

    # Resolve the ensemble flavour. Priority: explicit arg >
    # metadata["ensembles_chosen"]["simple"][tt][tname] > "harm" fallback.
    # Arch-3: ensembles_chosen is sub-keyed per family; calibrators always run on the simple
    # per-target ensemble bucket (cross-target ensembles have their own calibration story).
    _resolved_method = ensembling_method
    if _resolved_method is None and metadata is not None:
        _ec = metadata.get("ensembles_chosen") if isinstance(metadata, dict) else None
        if isinstance(_ec, dict):
            _bucket = _ec.get("simple")
            if isinstance(_bucket, dict):
                _by_tt = _bucket.get(task_type) or _bucket.get(str(task_type))
                if isinstance(_by_tt, dict):
                    _resolved_method = _by_tt.get(target_name) or _by_tt.get(str(target_name))
                elif isinstance(_by_tt, str):
                    _resolved_method = _by_tt
    if _resolved_method is None:
        _resolved_method = "harm"
        logger.warning(
            "train_postcalibrators: no ensemble flavour passed and metadata['ensembles_chosen'] "
            "did not expose one for target=%r; defaulting to 'harm'. The deployed predict path "
            "may select a different flavour, leading to a calibration / deployment mismatch.",
            target_name,
        )

    _calib_arrays = [np.asarray(p) for p in calib_probs_per_model]
    if not _calib_arrays:
        raise ValueError("train_postcalibrators: calib_probs_per_model is empty.")
    _n_calib = _calib_arrays[0].shape[0]
    if _calib_target_np.shape[0] != _n_calib:
        raise ValueError(f"train_postcalibrators: calib_target length {_calib_target_np.shape[0]} " f"does not match calib_probs row count {_n_calib}.")
    for _i, _p in enumerate(_calib_arrays):
        if _p.shape[0] != _n_calib:
            raise ValueError(f"train_postcalibrators: calib_probs_per_model[{_i}] has {_p.shape[0]} rows; " f"expected {_n_calib} aligned to calib_target.")

    # Deferred: calibration/ is a lower-level package that models.ensembling/ sits above; this avoids
    # an eager module-scope import for a name only used inside this function body.
    from mlframe.models.ensembling import ensemble_probabilistic_predictions

    # ensemble_probabilistic_predictions returns a 3-tuple (preds, uncertainty, confident_idx).
    # Pre-fix this site unpacked into TWO names -- raising ValueError on every reachable call.
    ensembled_calib_predictions, _uncertainty, _confident_calib_indices = ensemble_probabilistic_predictions(
        *_calib_arrays,
        ensemble_method=_resolved_method,
        max_mae=max_mae if max_mae is not None else 0.0,
        max_std=max_std if max_std is not None else 0.0,
        ensure_prob_limits=ensure_prob_limits,
        uncertainty_quantile=uncertainty_quantile,
        normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
        verbose=bool(verbose),
    )

    first_model = next(iter(models.values()))
    columns = first_model.columns

    calib_test_metrics, test_calibrators, failed_calibrators = compare_postcalibrators(
        model_name=model_name,
        columns=columns,
        calib_probs=ensembled_calib_predictions,
        calib_target=_calib_target_np,
        oos_probs=None,
        oos_target=None,
        calib_type="calib",
        include_patterns=include_patterns,
        sample_weight=sample_weight,
    )
    logger.info("train_postcalibrators: calib-set comparison metrics for %s: %s", model_name, calib_test_metrics)
    if failed_calibrators:
        logger.warning(
            "train_postcalibrators: %d calibrator(s) failed and were excluded from %s: %s",
            len(failed_calibrators),
            model_name,
            failed_calibrators,
        )

    # Raw caller-supplied target/featureset/task/model names plumbed into os.path.join is a
    # path-traversal vector (one absolute component eats the prefix, "../../etc" escapes).
    final_models_dir = join(
        models_dir,
        slugify(target_name),
        slugify(featureset_name),
        slugify(str(task_type)),
        slugify(model_name),
    )

    for calib_name, calibrator in test_calibrators.items():
        ens_name = f"ens_{_resolved_method}"
        calib_fpath = join(final_models_dir, f"{ens_name}_postcalibrator_{slugify(calib_name)}.dump")
        joblib.dump(calibrator, calib_fpath, compress=("lzma", 6))
        # Wave 19 P1: write the .meta.json sidecar so calibrator-loaders
        # surface mlframe-version drift. Calibrator classes
        # (_PerClassIsotonicCalibrator / _PostHocMultiCalibratedModel) carry
        # attributes (n_classes, is_exclusive, _target_type) whose semantics
        # could shift across mlframe versions; predict-time uses getattr
        # blindly.
        try:
            from ..training.io import _write_save_meta_sidecar as _wsms
            _wsms(calib_fpath, durable=False)
        except Exception as _meta_e:
            logger.warning(
                "calibration: failed to write .meta.json sidecar for %s: %s. "
                "Calibrator saved; load-time version validation will fall "
                "through to back-compat.", calib_fpath, _meta_e,
            )

    # Return the fitted calibrator objects, the calib-set comparison metrics that picked them, and any
    # calibrator names that failed to fit/predict -- so a caller can see WHICH calibrator won on
    # calib-set metrics, by how much, and which candidates (if any) did not make it into either dict.
    # No in-repo caller currently unpacks this return value directly (checked), so widening the shape is safe here.
    return {"calibrators": test_calibrators, "metrics": calib_test_metrics, "failed_calibrators": failed_calibrators}
