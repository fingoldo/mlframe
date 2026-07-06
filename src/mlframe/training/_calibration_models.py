"""Post-hoc probability calibration wrappers.

Wave 94 (2026-05-21): split out from `_training_loop.py` to keep that
file below the 1k-line threshold. Behaviour preserved bit-for-bit; every
calibrator symbol is re-exported from `_training_loop` so existing
``from ._training_loop import _PostHocCalibratedModel`` (and the other
four moved names) imports continue to work.

What lives here:
  - ``_SigmoidAdapter`` (LogisticRegression -> Isotonic-style API)
  - ``_PostHocCalibratedModel`` (single-output binary classifier wrapper)
  - ``_PerClassIsotonicCalibrator`` (K-independent isotonic fits, used
    for MULTICLASS / MULTILABEL)
  - ``_PostHocMultiCalibratedModel`` (multi-output classifier wrapper)
  - ``_maybe_apply_posthoc_calibration`` (no-op hook kept for API
    compat; pre-fit CalibratedClassifierCV handles the work)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sklearn.isotonic import IsotonicRegression

if TYPE_CHECKING:
    import numpy as np

    from .configs import TargetTypes

logger = logging.getLogger(__name__)


class _SigmoidAdapter:
    """Thin adapter giving a fitted LogisticRegression an IsotonicRegression-
    style .predict() API that returns positive-class probabilities."""

    def __init__(self, lr):
        self.lr = lr

    def predict(self, x):
        import numpy as _np

        return self.lr.predict_proba(_np.asarray(x).reshape(-1, 1))[:, 1]


def _fit_per_class_calibrator(p: "np.ndarray", y_k: "np.ndarray", method: str):
    """Fit one one-vs-rest calibrator. Returns an object with a ``.predict(col)`` API
    (``_SigmoidAdapter`` or ``IsotonicRegression``). Sigmoid fits a logistic on the logit of the
    column; if the column is degenerate (no spread) it falls back to isotonic so the map is never
    constant. ``method`` is normalised here; unknown values fall back to sigmoid."""
    import numpy as _np
    from sklearn.linear_model import LogisticRegression

    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y_k)
        return iso

    eps = 1e-6
    pc = _np.clip(_np.asarray(p, dtype=_np.float64), eps, 1.0 - eps)
    z = _np.log(pc / (1.0 - pc)).reshape(-1, 1)
    if not _np.isfinite(z).all() or float(_np.ptp(z)) < 1e-12:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y_k)
        return iso
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(z, y_k)
    return _SigmoidLogitAdapter(lr)


class _SigmoidLogitAdapter:
    """Platt adapter for the per-class path: maps a probability column through ``logit -> logistic``.
    ``.predict(col)`` returns calibrated positive-class probabilities; picklable (holds only the
    fitted LogisticRegression), so it satisfies the post-hoc calibrator pickle contract."""

    def __init__(self, lr):
        self.lr = lr

    def predict(self, x):
        import numpy as _np

        eps = 1e-6
        xc = _np.clip(_np.asarray(x, dtype=_np.float64), eps, 1.0 - eps)
        z = _np.log(xc / (1.0 - xc)).reshape(-1, 1)
        return self.lr.predict_proba(z)[:, 1]


class _PostHocCalibratedModel:
    """Transparent wrapper that applies isotonic post-hoc calibration to
    predict_proba outputs of a fitted binary classifier.

    Makes ``prefer_calibrated_classifiers=True`` actually calibrate tree
    classifiers. The naive path only swapped the early-stopping
    eval_metric, which was a no-op when early stopping did not trigger -- so
    calibrated and uncalibrated runs produced bit-identical probabilities.

    The wrapper delegates every attribute to the underlying ``base`` model
    except ``predict_proba``, which runs the base classifier and then maps
    the positive-class probability through a fitted IsotonicRegression.
    """

    def __init__(self, base, calibrator):
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "_calibrator", calibrator)

    def __getattr__(self, name):  # delegate unknown attrs to base
        # During unpickling __getattr__ may fire before __dict__ is populated.
        # Guard against that to avoid infinite recursion / KeyError.
        if name in ("base", "_calibrator", "__setstate__", "__getstate__", "__reduce__", "__reduce_ex__"):
            raise AttributeError(name)
        try:
            base = object.__getattribute__(self, "__dict__")["base"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(base, name)

    def __getstate__(self):
        return {"base": self.base, "_calibrator": self._calibrator}

    def __setstate__(self, state):
        object.__setattr__(self, "base", state["base"])
        object.__setattr__(self, "_calibrator", state["_calibrator"])

    def predict_proba(self, X):
        import numpy as _np

        raw = self.base.predict_proba(X)
        raw = _np.asarray(raw)
        if raw.ndim == 2 and raw.shape[1] == 2:
            p1 = self._calibrator.predict(raw[:, 1])
            p1 = _np.clip(p1, 0.0, 1.0)
            out = _np.column_stack([1.0 - p1, p1])
            return out
        return raw

    def predict(self, X):
        import numpy as _np

        probs = self.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] == 2:
            classes = getattr(self.base, "classes_", _np.array([0, 1]))
            return classes[(probs[:, 1] >= 0.5).astype(int)]
        return self.base.predict(X)


class _PerClassIsotonicCalibrator:
    """Multi-output post-hoc calibrator: K independent IsotonicRegression fits.

    Unblocks calibration on ``MULTICLASS_CLASSIFICATION`` and
    ``MULTILABEL_CLASSIFICATION``
    target types (previously raised ``NotImplementedError`` in
    ``evaluation.post_calibrate_model``).

    Semantics per target_type:
      - **MULTICLASS** (exclusive labels, softmax output): fit one
        isotonic per class on ``probs[:, k]`` vs ``(y_true == k)``.
        At predict time, each column is mapped independently through
        its own isotonic; then re-normalised row-wise so probabilities
        sum to 1 (preserves the exclusive-class invariant).
      - **MULTILABEL** (independent binary outputs, per-label sigmoid):
        fit one isotonic per label on ``probs[:, k]`` vs ``y_true[:, k]``.
        At predict time, each column is mapped independently; no
        re-normalisation (labels are independent).

    Numerical guards:
      - Each per-class isotonic needs >=2 samples of both classes in
        training; if a class is near-constant in the calibration set,
        we skip that class's calibrator (identity mapping applied).
      - Output clipped to [0, 1] post-isotonic (isotonic can over/
        undershoot at boundaries).

    Wrapped in _PostHocCalibratedModel for transparent predict_proba /
    predict delegation. Stored as a dict {class_idx: IsotonicRegression}
    plus a boolean mode flag (exclusive vs independent).
    """

    def __init__(self, calibrators, is_exclusive: bool, n_classes: int):
        """
        calibrators: dict {class_idx: per-class calibrator (.predict(col) API) or None (identity skip)}.
            Each entry is a _SigmoidLogitAdapter (default method) or an IsotonicRegression (method='isotonic').
        is_exclusive: True for MULTICLASS softmax, False for MULTILABEL sigmoid
        n_classes: K
        """
        self.calibrators = calibrators
        self.is_exclusive = is_exclusive
        self.n_classes = n_classes

    @classmethod
    def fit(cls, probs_NK, y_true, target_type, classes=None, method: str = "sigmoid"):
        """Fit K independent one-vs-rest calibrators on the calibration set.

        method : {"sigmoid", "isotonic"}, default "sigmoid"
            Per-class map. ``"sigmoid"`` is a 2-parameter Platt fit (logistic on the column logit);
            ``"isotonic"`` is a free-form monotone step map. Default is sigmoid: on the small per-class
            OOF / calibration slices this path typically sees, isotonic interpolates the training
            calibration and generalises worse on held-out data. Bench (5 scenarios x 3 seeds x calib
            sizes 200/1000/8000): sigmoid wins held-out mean-per-class ECE 15/15, 15/15, 13/15 with
            lower mean ECE at every size (_benchmarks/bench_per_class_method_isotonic_vs_sigmoid.py).
            Pass ``method="isotonic"`` for non-parametric / saturating miscalibration with abundant data.

        Parameters
        ----------
        probs_NK : np.ndarray (N, K)
            Canonical (N, K) probability matrix (use
            ``_canonical_predict_proba_shape`` to coerce first).
        y_true : np.ndarray
            - MULTICLASS: shape (N,) with the raw class labels (may be
              non-0..K-1, e.g. ``[10, 20, 30]`` or strings).
            - MULTILABEL: shape (N, K) binary indicator matrix
        target_type : TargetTypes
        classes : sequence, optional
            The class label for each probability COLUMN (i.e. ``model.classes_``,
            which is what orders ``predict_proba`` columns). Column ``k`` is
            calibrated against ``y_true == classes[k]``. MULTICLASS only. When
            ``None`` it falls back to ``np.unique(y_true)`` (sklearn's sorted
            classes_ order) so callers that omit it stay correct for the common
            "every class appears in calib" case. Passing ``range(K)`` reproduces
            the legacy positional behaviour and is only correct when the labels
            are exactly ``0..K-1``.
        """
        import numpy as _np
        from .configs import TargetTypes

        probs = _np.asarray(probs_NK, dtype=_np.float64)
        K = probs.shape[1]
        is_exclusive = target_type == TargetTypes.MULTICLASS_CLASSIFICATION
        y = _np.asarray(y_true)

        col_labels = None
        if is_exclusive:
            # Map probability COLUMN k -> its class label. predict_proba orders
            # columns by model.classes_ (sorted unique), NOT by integer value, so
            # ``y == k`` is wrong whenever the labels are not exactly 0..K-1.
            if classes is not None:
                col_labels = _np.asarray(classes)
            else:
                col_labels = _np.unique(y)
            if col_labels.shape[0] != K:
                # Fall back to positional only if we truly cannot align (e.g. a
                # class is absent from the calib slice); a misaligned guess would
                # silently calibrate the wrong column, so prefer identity there.
                col_labels = None

        calibrators = {}
        for k in range(K):
            # Per-class binary target
            if is_exclusive:
                y_k = (y == (col_labels[k] if col_labels is not None else k)).astype(_np.int8)
            else:
                # Multilabel: y is (N, K)
                y_k = y[:, k].astype(_np.int8)
            # Guard: skip constant-label calibrators (1-class or near-so)
            n_pos = int(y_k.sum())
            if n_pos < 2 or n_pos >= (len(y_k) - 1):
                calibrators[k] = None  # identity mapping
                continue
            calibrators[k] = _fit_per_class_calibrator(probs[:, k], y_k, method)

        return cls(calibrators, is_exclusive, K)

    def predict_proba(self, probs_NK):
        """Apply per-class isotonic to the (N, K) probability matrix.

        Returns a new (N, K) array with each column independently
        calibrated. For MULTICLASS, row-normalise so rows sum to 1.
        """
        import numpy as _np

        probs = _np.asarray(probs_NK, dtype=_np.float64)
        out = _np.empty_like(probs)
        for k in range(self.n_classes):
            iso = self.calibrators.get(k)
            if iso is None:
                out[:, k] = probs[:, k]  # identity
            else:
                out[:, k] = _np.clip(iso.predict(probs[:, k]), 0.0, 1.0)
        if self.is_exclusive:
            # Softmax-space: re-normalise rows to sum to 1. Guard against
            # all-zero rows (rare but possible after clip).
            row_sums = out.sum(axis=1, keepdims=True)
            row_sums = _np.where(row_sums == 0.0, 1.0, row_sums)
            out = out / row_sums
        return out


class _PostHocMultiCalibratedModel:
    """Multi-output variant of _PostHocCalibratedModel.

    Wraps ``base`` classifier + per-class isotonic calibrator.
    ``predict_proba(X)`` runs the base model and routes through
    ``_PerClassIsotonicCalibrator.predict_proba`` for (N, K) output.

    Uses ``_canonical_predict_proba_shape`` to normalise ``MultiOutputClassifier``'s
    List[(N, 2)] output to (N, K) before calibration.
    """

    def __init__(self, base, calibrator: _PerClassIsotonicCalibrator, target_type, classes_=None):
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "_calibrator", calibrator)
        object.__setattr__(self, "_target_type", target_type)
        object.__setattr__(self, "_classes", classes_)

    def __getattr__(self, name):
        if name in ("base", "_calibrator", "_target_type", "_classes", "__setstate__", "__getstate__", "__reduce__", "__reduce_ex__"):
            raise AttributeError(name)
        try:
            base = object.__getattribute__(self, "__dict__")["base"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(base, name)

    def __getstate__(self):
        return {
            "base": self.base,
            "_calibrator": self._calibrator,
            "_target_type": self._target_type,
            "_classes": self._classes,
        }

    def __setstate__(self, state):
        for k, v in state.items():
            object.__setattr__(self, k, v)

    def predict_proba(self, X):
        from .helpers import _canonical_predict_proba_shape

        raw = self.base.predict_proba(X)
        classes_ = getattr(self.base, "classes_", self._classes)
        probs_NK = _canonical_predict_proba_shape(raw, classes_=classes_)
        return self._calibrator.predict_proba(probs_NK)

    def predict(self, X):
        from .helpers import _predict_from_probs

        probs = self.predict_proba(X)
        return _predict_from_probs(probs, self._target_type, classes_=self._classes)


def _model_probs_are_posthoc_calibrated(model) -> bool:
    """True iff ``model``'s probabilities come from a held-set post-hoc calibrator (``CalibratedClassifierCV`` or one of
    the project's post-hoc wrappers), as opposed to merely being calibration-trained via an eval-metric swap (trees)."""
    from sklearn.calibration import CalibratedClassifierCV

    candidates = [model]
    try:
        if hasattr(model, "steps") and model.steps:
            candidates.append(model.steps[-1][1])
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("suppressed in _calibration_models.py:348: %s", e)
        pass
    for obj in candidates:
        if isinstance(obj, (CalibratedClassifierCV, _PostHocCalibratedModel, _PostHocMultiCalibratedModel)):
            return True
    return False


def _maybe_apply_posthoc_calibration(model, fit_params, model_type_name, verbose=False):
    """Stamp ``model._mlframe_probs_posthoc_calibrated`` to record whether the probabilities are post-hoc calibrated.

    There is NO behaviour change here beyond the metadata flag. The ``prefer_calibrated_classifiers`` flag means two
    different things by model family, which is easy to misread:

    - Linear classifiers are wrapped pre-fit in ``CalibratedClassifierCV`` (internal CV, honest, no test touch); their
      probabilities ARE genuinely held-set post-hoc calibrated -> flag True.
    - Tree models only get their *training objective* tuned toward calibration (the early-stopping / eval metric is
      swapped to ``integral_calibration_error``); no disjoint-set post-hoc calibrator is fit, so the flag is False.
      This is deliberate: fitting an isotonic wrapper on val would burn the early-stopping budget twice / overfit val.
      The honest post-hoc path for trees is ``post_calibrate_model`` on a disjoint calib slice (TrainingSplitConfig.calib_size).

    The historical isotonic-on-val wrapper this hook once applied was disabled for exactly that val-overfitting reason;
    the hook stays as the single place the per-model calibration state is surfaced to downstream metadata.
    """
    is_posthoc = _model_probs_are_posthoc_calibrated(model)
    try:
        setattr(model, "_mlframe_probs_posthoc_calibrated", is_posthoc)
    except (AttributeError, TypeError):
        # Slot-only / read-only estimators refuse new attrs; metadata consumers fall back to getattr(..., None).
        pass
    return model


def calibrate_namespace_model(entry: Any, *, target_type: "TargetTypes | None" = None) -> bool:
    """Fit a post-hoc isotonic calibrator on a per-target model's DISJOINT calib slice and wrap it.

    ``entry`` is a per-target model namespace object as built by the trainer: it carries ``.model`` plus
    ``.calib_probs`` / ``.calib_target`` (the base model's predict_proba on the carved calib slice +
    aligned labels) when ``TrainingSplitConfig.calib_size > 0``. The calib slice is leakage-free by
    construction: the splitter carves it from train and the base model is fit on train-minus-calib.

    Leakage-safety re-checked here (raise, not warn): the calib slice must NOT equal the model's stamped
    test slice -- reuses the same ``np.array_equal`` guard the disjoint paths use. Fits binary isotonic
    (or per-class isotonic for multi-output), wraps ``entry.model`` in the matching post-hoc wrapper,
    and stamps ``calibrated_<split>_probs`` for the ensembling read-side. Returns True when calibration
    was applied, False when no calib slice was present (no-op).
    """
    import numpy as _np

    calib_probs = getattr(entry, "calib_probs", None)
    calib_target = getattr(entry, "calib_target", None)
    if calib_probs is None or calib_target is None:
        return False

    _cp = _np.asarray(calib_probs)
    _cy = _np.asarray(calib_target)
    if _cp.shape[0] == 0 or _cp.shape[0] != _cy.shape[0]:
        raise ValueError(f"calibrate_namespace_model: calib_probs/calib_target row mismatch {_cp.shape[0]} vs {_cy.shape[0]}")

    # Hard leak guard: calib must not be the honest test slice. The base model never trained on calib
    # (carved from train), so the remaining hazard is a caller wiring calib == test by mistake.
    _test_target = getattr(entry, "test_target", None)
    if _test_target is not None:
        _tt = _np.asarray(_test_target.values if hasattr(_test_target, "values") else _test_target)
        if _tt.shape == _cy.shape and _np.array_equal(_tt, _cy):
            raise ValueError(
                "calibrate_namespace_model: calib_target is identical to the model's test_target; "
                "refusing to fit the calibrator on the honest holdout slice."
            )

    base = getattr(entry, "model", None)
    if base is None:
        return False

    _is_multi = _cp.ndim == 2 and _cp.shape[1] != 2
    if _is_multi:
        from .configs import TargetTypes
        _ttype = target_type if target_type is not None else getattr(entry, "target_type", None)
        if isinstance(_ttype, str):
            _ttype = getattr(TargetTypes, _ttype, None) or (TargetTypes.MULTILABEL_CLASSIFICATION if _cy.ndim == 2 else TargetTypes.MULTICLASS_CLASSIFICATION)
        if _ttype is None:
            _ttype = TargetTypes.MULTILABEL_CLASSIFICATION if _cy.ndim == 2 else TargetTypes.MULTICLASS_CLASSIFICATION
        calibrator = _PerClassIsotonicCalibrator.fit(_cp, _cy, _ttype, classes=getattr(base, "classes_", None))
        wrapped = _PostHocMultiCalibratedModel(base, calibrator, _ttype, classes_=getattr(base, "classes_", None))
    else:
        _pos = _cp[:, 1] if _cp.ndim == 2 else _cp.ravel()
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(_pos, _cy.ravel())
        wrapped = _PostHocCalibratedModel(base, iso)

    # Stamp calibrated probs for the existing ensembling read-side (_select_member_probs).
    for _split in ("val", "test"):
        _raw = getattr(entry, f"{_split}_probs", None)
        if _raw is None:
            continue
        try:
            _cal = wrapped._calibrator.predict_proba(_np.asarray(_raw)) if _is_multi else None
            if not _is_multi:
                _p1 = _np.clip(wrapped._calibrator.predict(_np.asarray(_raw)[:, 1] if _np.asarray(_raw).ndim == 2 else _np.asarray(_raw).ravel()), 0.0, 1.0)
                _cal = _np.column_stack([1.0 - _p1, _p1])
            for _obj in (entry, base):
                try:
                    setattr(_obj, f"calibrated_{_split}_probs", _cal)
                except (AttributeError, TypeError):
                    pass
        except Exception as _stamp_err:
            logger.warning("calibrate_namespace_model: %s-probs stamp failed: %s", _split, _stamp_err)

    try:
        entry.model = wrapped
    except (AttributeError, TypeError):
        logger.warning("calibrate_namespace_model: could not replace entry.model with calibrated wrapper.")
        return False
    return True
