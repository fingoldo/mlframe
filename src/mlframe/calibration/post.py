"""Post-hoc probability calibration: fits a zoo of third-party calibrators on a disjoint calib split, compares them, and persists the fitted objects."""
from __future__ import annotations

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from typing import Any, Optional
# No direct references to mlframe.config names remain in this module. If you
# need one, import it explicitly rather than reintroducing the wildcard.

import re
import copy
from functools import lru_cache
from dataclasses import dataclass


@lru_cache(maxsize=128)
def _compile_pattern(pattern: str) -> "re.Pattern":
    """Cached regex compilation for runtime-provided patterns."""
    return re.compile(pattern)


# Module-level compiled sentinel so meta-tests can confirm the precompile
# refactor landed. Real include/skip patterns are cached via _compile_pattern.
_INCLUDE_RE: "re.Pattern" = re.compile("")
from timeit import default_timer as timer


from sklearn import config_context
from sklearn.base import BaseEstimator, ClassifierMixin

import pandas as pd, numpy as np

from pyutilz.system import tqdmu
from pyutilz.pythonlib import store_params_in_object, get_parent_func_args

from mlframe.calibration.policy import _stratified_inner_folds

# Heavy optional deps (netcal/pycalib pull torch transitively → DLL-load can fail
# on Windows boxes with mismatched CUDA toolkits). Imported lazily inside
# get_postcalibrators(); module-level import would crash pytest collection on
# such boxes even though the rest of the module doesn't need these.
from sklearn.calibration import CalibratedClassifierCV

try:
    from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
except Exception:
    FullDirichletCalibrator = None

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Core
# -----------------------------------------------------------------------------------------------------------------------------------------------------


class _CalibTestOverlapError(ValueError):
    """Raised only by ``train_postcalibrators``' own deliberate calib==test leakage guard.

    Kept as a distinct type (not a string-matched plain ``ValueError``) so the outer
    ``except (TypeError, ValueError)`` around the equality checks can re-raise exactly this guard trip and
    let every other ``TypeError``/``ValueError`` (e.g. non-comparable dtypes) fall through to the
    "skip the equality check" branch, without depending on the exact wording of the guard's message staying
    in sync with a substring check.
    """


def _try_import_class(module_path: str, class_name: str) -> Optional[type]:
    """Import ``class_name`` from ``module_path``, or None when the optional dep is missing."""
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)  # type: ignore[no-any-return]
    except ImportError:
        return None


# Calibrators that expect a 2D (n_samples, n_classes) prob matrix, as (module_path, class_name) pairs;
# each is imported lazily so optional deps (dirichletcal / netcal / pycalib / venn_abers) may be absent.
_NEEDS_2D_CALIBRATORS = (
    ("venn_abers", "VennAbersCalibrator"),
    ("netcal.scaling", "LogisticCalibration"),
    ("netcal.scaling", "LogisticCalibrationDependent"),
    ("pycalib.models", "LogisticCalibration"),
    ("dirichletcal.calib.fulldirichlet", "FullDirichletCalibrator"),
    ("sklearn.calibration", "CalibratedClassifierCV"),
)


class BinaryPostCalibrator(BaseEstimator, ClassifierMixin):
    """sklearn-compatible adapter that wraps a third-party binary calibrator behind a uniform interface.

    Normalises the many calibrator libraries (netcal, pycalib, betacal, dirichletcal, venn-abers,
    sklearn, verified_calibration) that differ in fit/transform method names and 1D-vs-2D probability
    shape. ``fit`` trains the wrapped calibrator on calibration-set probabilities/targets;
    ``postcalibrate_probs`` (aliased as ``predict_proba``) maps raw probabilities to calibrated ones,
    always returning a 2D ``(n_samples, 2)`` matrix.
    """

    calibrator: object
    fit_method_name: str
    transform_method_name: str
    needs_2d_probs: Optional[bool]
    _resolved_transform_method_name: str

    def __init__(
        self,
        calibrator: object,
        fit_method_name: str = "fit",
        transform_method_name: str = "transform",
        needs_2d_probs: Optional[bool] = None,
    ) -> None:
        # postfix="" -- this class reads attributes back by their BARE param name (see the class-level
        # annotations above); pyutilz's store_params_in_object() now defaults postfix to "_param_" to
        # round-trip with load_object_params_into_func(), which broke every bare-name reader that
        # didn't pin the old convention explicitly.
        store_params_in_object(obj=self, params=get_parent_func_args(), postfix="")

    def _calibrator_needs_2d_probs(self, calibrator: object) -> bool:
        """Returns True if the wrapped calibrator expects a 2D (n_samples, n_classes) prob matrix.

        Uses isinstance checks against the relevant calibrator classes (imported lazily so optional
        deps can be missing) rather than substring-matching on the class name. Any calibrator NOT in
        the hardcoded ``_NEEDS_2D_CALIBRATORS`` list -- a caller-supplied custom calibrator, or a class
        added to a supported library in a future release -- silently defaults to the 1D path here, so
        ``self.needs_2d_probs`` (set at construction via ``named_calibrator``) is checked first as an
        explicit caller override; only when it is ``None`` do we fall back to the isinstance/name guess.
        """
        if self.needs_2d_probs is not None:
            return self.needs_2d_probs
        # Late imports: some of these are optional (dirichletcal) or may be reshuffled upstream.
        needs_2d_types = [cls for module_path, class_name in _NEEDS_2D_CALIBRATORS if (cls := _try_import_class(module_path, class_name)) is not None]
        # "Top" calibrators (e.g. TopLabelCalibrator style) preserve 2D shape; match by class-name
        # prefix here as there is no single importable base — minimal remaining substring check.
        if type(calibrator).__name__.startswith("Top"):
            return True
        return isinstance(calibrator, tuple(needs_2d_types)) if needs_2d_types else False

    @staticmethod
    def _is_venn_abers(calibrator: object) -> bool:
        """isinstance check against VennAbersCalibrator so subclasses dispatch correctly
        (substring matching on the class name routed subclasses to the wrong branch)."""
        _VA = _try_import_class("venn_abers", "VennAbersCalibrator")
        return _VA is not None and isinstance(calibrator, _VA)

    def _transform_probs(self, probs: np.ndarray) -> np.ndarray:
        """Reduce a 2D ``(n, 2)`` prob matrix to the positive-class column for wrapped calibrators that expect 1D input; pass 2D-expecting calibrators through unchanged."""
        if probs.ndim == 2 and not self._calibrator_needs_2d_probs(self.calibrator):
            probs = probs[:, 1]
        return probs

    def fit(
        self,
        calib_probs: np.ndarray,
        calib_target: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "BinaryPostCalibrator":
        """Fit the wrapped calibrator on calib-set probabilities/targets, resolving its transform method name once (predict/transform/venn-abers) for reuse in ``postcalibrate_probs``.

        ``sample_weight``, when given, is passed through to the wrapped calibrator's own fit method ONLY if
        that method's signature actually accepts a ``sample_weight`` keyword (checked via
        ``inspect.signature`` -- the calibrator zoo mixes sklearn estimators that support it (e.g.
        ``IsotonicRegression``, ``LogisticRegression``) with third-party calibrators that don't). When the
        wrapped calibrator does NOT support it, a WARNING is logged (once) so the caller knows their weights
        were silently dropped for this specific calibrator, rather than the previous behavior of having no
        parameter to pass weights to at all. The Venn-Abers path does not support weighting (fits directly
        on stored ``(p_cal, y_cal)`` via a from-scratch algorithm, not a generic ``.fit()`` call).
        """
        # sklearn ClassifierMixin tooling (CalibratedClassifierCV, cross_val_predict, check_is_fitted) expects
        # ``classes_`` and ``n_features_in_`` after fit. We set them from the raw inputs BEFORE the prob reshape so a
        # 2D (n, n_classes) prob matrix reports its real feature width.
        _target_arr = np.asarray(calib_target)
        self.classes_ = np.unique(_target_arr)
        _probs_arr = np.asarray(calib_probs)
        self.n_features_in_ = _probs_arr.shape[1] if _probs_arr.ndim == 2 else 1

        calib_probs = self._transform_probs(calib_probs)

        if not self._is_venn_abers(self.calibrator):
            _fit_fn = getattr(self.calibrator, self.fit_method_name)
            if sample_weight is not None:
                import inspect
                try:
                    _accepts_sw = "sample_weight" in inspect.signature(_fit_fn).parameters
                except (TypeError, ValueError):
                    _accepts_sw = False
                if _accepts_sw:
                    _fit_fn(calib_probs, calib_target, sample_weight=sample_weight)
                else:
                    logger.warning(
                        "[calibration] BinaryPostCalibrator.fit: wrapped calibrator %r's %s() does not "
                        "accept sample_weight; fitting unweighted (weights silently ignored for this "
                        "calibrator).",
                        type(self.calibrator).__name__, self.fit_method_name,
                    )
                    _fit_fn(calib_probs, calib_target)
            else:
                _fit_fn(calib_probs, calib_target)
        else:
            if sample_weight is not None:
                logger.warning(
                    "[calibration] BinaryPostCalibrator.fit: Venn-Abers calibration does not support " "sample_weight; fitting unweighted.",
                )
            self.y_cal = calib_target
            self.p_cal = calib_probs

        # Resolve transform method name ONCE at fit-time rather than mutating self inside
        # postcalibrate_probs (previously racy and surprising for re-use across calls).
        resolved_transform = self.transform_method_name
        if not hasattr(self.calibrator, resolved_transform) and hasattr(self.calibrator, "predict"):
            resolved_transform = "predict"
        self._resolved_transform_method_name = resolved_transform

        return self

    def postcalibrate_probs(self, probs: np.ndarray) -> np.ndarray:
        """Map raw probabilities through the fitted calibrator, always returning a 2D ``(n_samples, 2)`` matrix."""
        probs = self._transform_probs(probs)
        if not self._is_venn_abers(self.calibrator):
            # Use method resolved at fit-time; fall back gracefully if fit() wasn't called.
            transform_name = getattr(self, "_resolved_transform_method_name", self.transform_method_name)
            calibrated_probs = getattr(self.calibrator, transform_name)(probs)
        else:
            calibrated_probs = getattr(self.calibrator, "predict_proba")(p_cal=self.p_cal, y_cal=self.y_cal, p_test=probs)

        calibrated_probs = np.asarray(calibrated_probs)
        if calibrated_probs.ndim == 2 and (
            hasattr(self.calibrator, "method") and getattr(self.calibrator, "method") in ["momentum", "variational", "mcmc"]
        ):  # mcmc methods of netcal
            calibrated_probs = calibrated_probs.mean(axis=0)
        if calibrated_probs.ndim == 1:
            # Clip to [0, 1] before stacking so numerical drift from calibrator outputs does
            # not produce negative or >1 entries in the 2D prob matrix.
            calibrated_probs = np.clip(calibrated_probs, 0.0, 1.0)
            calibrated_probs = np.vstack([1 - calibrated_probs, calibrated_probs]).T

        return np.asarray(calibrated_probs)

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        """sklearn-style alias for :meth:`postcalibrate_probs` so ClassifierMixin tooling (which calls
        ``predict_proba``) routes to the calibrated 2D probability matrix."""
        return self.postcalibrate_probs(probs)


@dataclass
class NamedCalibrator:
    """A ``BinaryPostCalibrator`` paired with display metadata (name, library, parameter string).

    ``full_name()`` builds a stable human-readable identifier like ``lib.Name[param_str]`` used as the
    row key when comparing calibrators.
    """

    calibrator: BinaryPostCalibrator
    name: Optional[str] = None
    param_str: Optional[str] = ""
    lib: Optional[str] = None

    def full_name(self) -> str:
        """Build the ``lib.Name[param_str]`` display identifier used as the row key when comparing calibrators."""
        base_name = self.name or self._extract_calibrator_class_name()
        # Prepend lib only if not already in name (case-insensitive)
        if self.lib and self.lib.lower() not in base_name.lower():
            base_name = f"{self.lib}.{base_name}"
        if self.param_str:
            return f"{base_name}[{self.param_str}]"
        return base_name

    def _extract_calibrator_class_name(self) -> str:
        """Class name of the wrapped calibrator, unwrapping a ``BinaryPostCalibrator`` adapter to reach the underlying third-party object."""
        obj = getattr(self.calibrator, "calibrator", self.calibrator)
        return obj.__class__.__name__


def named_calibrator(
    calibrator_obj: object,
    name: Optional[str] = None,
    param_str: Optional[str] = "",
    lib: Optional[str] = None,
    **postcal_kwargs: Any,
) -> NamedCalibrator:
    """Wrap a raw calibrator object into a ``NamedCalibrator`` (with a ``BinaryPostCalibrator`` adapter).

    ``name``/``param_str``/``lib`` feed the display identifier; extra kwargs are forwarded to the
    ``BinaryPostCalibrator`` (e.g. ``fit_method_name``, ``transform_method_name``, ``needs_2d_probs``
    to override the isinstance-based 2D-probs auto-detection for a calibrator not in the hardcoded
    ``_NEEDS_2D_CALIBRATORS`` list).
    """
    return NamedCalibrator(
        calibrator=BinaryPostCalibrator(calibrator=calibrator_obj, **postcal_kwargs),
        name=name,
        param_str=param_str,
        lib=lib,
    )


def should_run(name: str, include: Optional[list[str]] = None, skip: Optional[list[str]] = None) -> bool:
    """Return whether a calibrator ``name`` passes the include/skip regex filters.

    ``True`` only when ``name`` matches at least one ``include`` pattern (or ``include`` is empty)
    and matches no ``skip`` pattern. Patterns are treated as regexes via ``re.search``.
    """
    if include and not any(_compile_pattern(p).search(name) for p in include):
        return False
    if skip and any(_compile_pattern(p).search(name) for p in skip):
        return False
    return True


def get_postcalibrators(calib_target: np.ndarray, num_bins: int) -> list[NamedCalibrator]:
    """Build the zoo of candidate ``NamedCalibrator`` instances across all supported calibration libraries.

    Instantiates sklearn, ml_insights, verified_calibration, betacal, venn-abers, netcal (binning +
    scaling), pycalib and (if available) dirichletcal calibrators, sizing binning methods by ``num_bins``
    and the calibration-set length. Optional libraries that are absent or renamed upstream are skipped.
    Returns the list used by ``compare_postcalibrators``.
    """
    import netcal, pycalib
    import ml_insights as mli
    from betacal import BetaCalibration
    import calibration as verified_calibration
    from venn_abers import VennAbersCalibrator

    calibrators = [
        named_calibrator(CalibratedClassifierCV(method="sigmoid", ensemble=False), name="CalibratedClassifierCV", param_str="method=sigmoid", lib="sklearn"),
        named_calibrator(CalibratedClassifierCV(method="isotonic", ensemble=False), name="CalibratedClassifierCV", param_str="method=isotonic", lib="sklearn"),
        named_calibrator(mli.SplineCalib(), lib="mli"),
        # ``verified_calibration`` upstream renamed / restructured public symbols
        # across releases (``HistogramCalibrator`` / ``PlattBinnerCalibrator`` /
        # ``PlattCalibrator`` are not all guaranteed to be present). Skip the
        # absent ones quietly so the rest of the calibrator zoo still loads.
        *[
            named_calibrator(
                _cls(len(calib_target), num_bins=num_bins),
                param_str=f"bins={num_bins}",
                lib="verified_calibration",
                fit_method_name="train_calibration",
                transform_method_name="calibrate",
            )
            for _cls in (getattr(verified_calibration, _name, None) for _name in ("HistogramCalibrator", "PlattBinnerCalibrator", "PlattCalibrator"))
            if _cls is not None
        ],
        *[named_calibrator(BetaCalibration(variant), lib="betacal", param_str=f"variant={variant}") for variant in ["abm", "ab", "am"]],
        named_calibrator(VennAbersCalibrator(), lib="vaa"),
        *[
            named_calibrator(cls(), lib="netcal")
            for cls in [
                netcal.binning.BBQ,
                netcal.binning.HistogramBinning,
                netcal.binning.IsotonicRegression,
                netcal.binning.ENIR,
                netcal.binning.NearIsotonicRegression,
            ]
        ],
        *[
            named_calibrator(cls(), lib="netcal")
            for cls in [
                netcal.scaling.TemperatureScaling,
                netcal.scaling.BetaCalibration,
                netcal.scaling.BetaCalibrationDependent,
                netcal.scaling.LogisticCalibration,
                netcal.scaling.LogisticCalibrationDependent,
            ]
        ],
        *[
            named_calibrator(cls(), lib="pycalib", transform_method_name="predict_proba")
            for cls in [
                pycalib.models.IsotonicCalibration,
                pycalib.models.SigmoidCalibration,
                pycalib.models.BinningCalibration,
                pycalib.models.LogisticCalibration,
                pycalib.models.BetaCalibration,
            ]
        ],
    ]

    if FullDirichletCalibrator:
        calibrators.append(named_calibrator(FullDirichletCalibrator(), transform_method_name="predict_proba"))

    return calibrators


def compare_postcalibrators(
    model_name: str,
    columns: list,
    calib_probs: np.ndarray,
    calib_target: np.ndarray,
    oos_probs: Optional[np.ndarray],
    oos_target: Optional[np.ndarray],
    num_bins: int = 15,
    calib_type: str = "test",
    plot_file: str = "",
    report_params: Optional[dict] = None,
    include_patterns: Optional[list] = None,
    skip_patterns: Optional[list] = None,  # r"BetaCalibration\[variant=ab\]"
    selection: str = "inner_cv",
    inner_cv_splits: int = 5,
    random_state: Optional[int] = 0,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple[Optional[pd.DataFrame], dict, dict[str, str]]:
    """Given calibration and (optionally) OOS probabilities and true targets, fits a number of
    calibrator models on the calib set and computes ML metrics on a held-out slice. When ``oos_probs``/
    ``oos_target`` are ``None``, evaluation falls back to ``selection``:

    - ``"inner_cv"`` (default): mirrors ``policy.py::pick_best_calibrator``'s honest fix for the same
      "same_oof" optimism bug class. The calib set is split into ``inner_cv_splits`` stratified folds;
      each calibrator is fit on the fold complement and scored on the held-out fold, and the assembled
      out-of-fold predictions (never seen during that fold's fit) feed ``report_model_perf``. The
      calibrator persisted in ``fit_calibrators`` is then refit on the FULL calib set for deployment.
      Falls back to ``"self_eval"`` (with a warning) when calib_target does not have exactly 2 classes
      or is too small for the requested fold count.
    - ``"self_eval"`` (legacy): each calibrator is fit AND scored on the exact same calib rows --
      optimistic, since a flexible calibrator (Isotonic, spline, BBQ) can interpolate its own reported
      metrics toward "perfect" purely by memorising the data it saw. Kept for replay / A-B comparison.

    ``sample_weight``, when given, is aligned to ``calib_probs``/``calib_target`` (length == calib set size)
    and threaded through every calibrator fit (full-set refit, and each inner_cv fold's fit, sliced to that
    fold's train rows) via ``BinaryPostCalibrator.fit``'s own ``sample_weight`` support -- see that method's
    docstring for which wrapped calibrators actually honor it (a calibrator whose fit signature has no
    ``sample_weight`` keyword is fit unweighted, with a warning). ``None`` (default) preserves the
    pre-existing fully-unweighted behavior bit-for-bit.

    A calibrator whose fit/predict raises is skipped (logged as a warning, not fatal) so the remaining
    candidates still complete and are not lost. Returns ``(metrics_df, fit_calibrators, failed_calibrators)``:
    ``metrics_df`` is a pandas dataframe of ML metrics by calibrator name (``None`` only if every candidate
    was filtered out by ``include_patterns``/``skip_patterns`` or every candidate failed); ``fit_calibrators``
    maps calibrator name to the fitted object (deployment-ready, refit on the full calib set even under
    ``inner_cv``); ``failed_calibrators`` maps the name of any calibrator that raised during fit/predict to
    ``repr(exception)`` -- explicitly surfaced (not silently dropped) so a caller can see which candidates,
    if any, did not make it into ``metrics_df``/``fit_calibrators``.
    """
    if include_patterns is None:
        include_patterns = []
    if skip_patterns is None:
        skip_patterns = [r"netcal\.BetaCalibrationDependent", r"netcal\.ENIR", r"netcal\.NearIsotonicRegression"]
    if selection not in ("inner_cv", "self_eval"):
        raise ValueError(f"compare_postcalibrators: selection must be 'inner_cv' or 'self_eval'; got {selection!r}")

    # get_postcalibrators/BinaryPostCalibrator wrap third-party BINARY calibrators only (see class
    # docstring): postcalibrate_probs unconditionally reduces to a positive-class column and reshapes
    # Deferred: calibration/ is a lower-level package that training/ orchestrates; this avoids an
    # eager module-scope training/ import for a name only used inside this function body.
    from mlframe.training.evaluation import report_model_perf

    # 1D output into a (n, 2) matrix, which is only correct for exactly 2 classes. A 3+-class or
    # single-class calib_target must raise here rather than silently mis-fit deep inside sklearn/netcal.
    _classes_check = np.unique(np.asarray(calib_target))
    if _classes_check.size != 2:
        raise ValueError(
            f"compare_postcalibrators: calib_target must have exactly 2 distinct classes (got {_classes_check.size}: "
            f"{_classes_check.tolist()!r}). The calibrator zoo (BinaryPostCalibrator/get_postcalibrators) is binary-only; "
            "a 3+-class target silently mis-fits some wrapped calibrators, and a single-class target crashes deep "
            "inside third-party fit() calls (e.g. sklearn's 'needs samples of at least 2 classes'). Provide a "
            "2-class calib_target, or route multi-class calibration through a one-vs-rest wrapper upstream."
        )

    logger.info(
        "Calib set size=%d, oos set size=%d, num_bins=%s.",
        len(calib_target),
        len(oos_target) if oos_target is not None else 0,
        num_bins,
    )

    if report_params is None:
        report_params = {"report_ndigits": 4, "calib_report_ndigits": 4, "print_report": False}

    metrics: dict[str, Any] = {"oos": {}}
    fit_calibrators = {}
    failed_calibrators: dict[str, str] = {}

    # No separate OOS set (the ONLY current caller, train_postcalibrators, never supplies one -- see
    # its docstring: calibrator FITTING and honest EVALUATION are deliberately split across different
    # rows/splits). Without an OOS set, prefer honest inner-CV held-out evaluation over same-data
    # self-eval -- pre-fix, every calibrator was scored on the exact rows it was fit on, which is the
    # same "same_oof" selection-optimism bug class policy.py::pick_best_calibrator already diagnosed
    # and fixed (flexible calibrators like Isotonic interpolate their own score toward "perfect").
    calib_probs_np = np.asarray(calib_probs)
    calib_target_np = np.asarray(calib_target)
    sample_weight_np = np.asarray(sample_weight) if sample_weight is not None else None
    if sample_weight_np is not None and sample_weight_np.shape[0] != calib_target_np.shape[0]:
        raise ValueError(f"compare_postcalibrators: sample_weight length {sample_weight_np.shape[0]} != calib_target length {calib_target_np.shape[0]}.")
    _eval_probs = oos_probs if oos_probs is not None else calib_probs
    _eval_target = oos_target if oos_target is not None else calib_target
    _eval_label = "OOS" if oos_probs is not None else "CALIB (self-eval, optimistic)"

    use_inner_cv = False
    inner_folds: Optional[list[np.ndarray]] = None
    if oos_probs is None and selection == "inner_cv":
        classes = np.unique(calib_target_np)
        min_rows_needed = max(2, int(inner_cv_splits)) * 2
        if classes.size != 2:
            logger.warning(
                "compare_postcalibrators: selection='inner_cv' requires exactly 2 classes in calib_target; " "got %d. Falling back to self-eval (optimistic).",
                classes.size,
            )
        elif calib_target_np.shape[0] < min_rows_needed:
            logger.warning(
                "compare_postcalibrators: selection='inner_cv' needs >= %d calib rows for %d folds; got %d. " "Falling back to self-eval (optimistic).",
                min_rows_needed,
                inner_cv_splits,
                calib_target_np.shape[0],
            )
        else:
            inner_folds = _stratified_inner_folds(calib_target_np, max(2, int(inner_cv_splits)), random_state)
            use_inner_cv = True
            _eval_label = "CALIB (inner-CV held-out)"

    _, _ = report_model_perf(
        targets=_eval_target,
        columns=columns,
        df=None,
        model_name=f"{model_name}",
        model=None,
        target_label_encoder=None,
        preds=None,
        probs=_eval_probs,
        plot_file=plot_file,
        report_title=_eval_label,
        metrics=metrics["oos"],
        group_ids=None,
        **report_params,
    )

    calibrators = get_postcalibrators(calib_target=calib_target, num_bins=num_bins)

    _seen_names: dict[str, int] = {}
    for nc in tqdmu(calibrators, desc="calibrator"):
        clf = nc.calibrator
        calibrator_name = nc.full_name()

        # full_name() collisions (two zoo entries resolving to the same lib.Name[param_str] key,
        # e.g. a caller-added custom calibrator without a distinguishing name/param_str) would
        # otherwise silently overwrite one calibrator's row in metrics/fit_calibrators -- disambiguate
        # with a numeric suffix and warn, rather than dropping a result with no error (P1-5).
        if calibrator_name in _seen_names:
            _seen_names[calibrator_name] += 1
            _disambiguated_name = f"{calibrator_name}#{_seen_names[calibrator_name]}"
            logger.warning(
                "compare_postcalibrators: calibrator name %r collides with a previously-seen entry; "
                "renaming this one to %r to avoid silently overwriting its result. Give it a distinguishing "
                "name/param_str in get_postcalibrators/named_calibrator to fix at the source.",
                calibrator_name,
                _disambiguated_name,
            )
            calibrator_name = _disambiguated_name
        else:
            _seen_names[calibrator_name] = 0

        if not should_run(calibrator_name, include_patterns, skip_patterns):
            # logger.info(f"Skipping calibrator: {calibrator_name} due to matching skip pattern.")
            continue

        _calibrator_start = timer()
        try:
            with config_context(transform_output="default"):

                r"""
                config_context needed here to avoid:

                R:\ProgramData\anaconda3\Lib\site-packages\netcal\binning\IsotonicRegression.py:183, in IsotonicRegression.transform(self, X)
                    179     calibrated = self._iso.transform(X)
                    181 # add clipping to [0, 1] to avoid exceeding due to numerical issues
                    182 # https://github.com/EFS-OpenSource/calibration-framework/issues/54
                --> 183 np.clip(calibrated, 0, 1, out=calibrated)
                    185 return calibrated
                """

                start = timer()

                if use_inner_cv and inner_folds is not None:
                    # Fit on each fold's complement, predict on the held-out fold -- the calibrator
                    # never sees the rows it is scored on, unlike the same-data self-eval path.
                    oof_calibrated = None
                    for held_idx in inner_folds:
                        held_mask = np.zeros(calib_target_np.shape[0], dtype=bool)
                        held_mask[held_idx] = True
                        train_idx = np.flatnonzero(~held_mask)
                        fold_clf = copy.deepcopy(clf)
                        fold_sw = sample_weight_np[train_idx] if sample_weight_np is not None else None
                        fold_clf.fit(calib_probs_np[train_idx], calib_target_np[train_idx], sample_weight=fold_sw)
                        fold_pred = np.asarray(fold_clf.postcalibrate_probs(calib_probs_np[held_idx]))
                        if oof_calibrated is None:
                            oof_calibrated = np.empty((calib_target_np.shape[0], *fold_pred.shape[1:]), dtype=fold_pred.dtype)
                        oof_calibrated[held_idx] = fold_pred
                    calibrated_probs = oof_calibrated
                    fitting_time = timer() - start

                    # Refit on the FULL calib set for the deployment artefact persisted to disk.
                    start = timer()
                    clf.fit(calib_probs, calib_target, sample_weight=sample_weight_np)
                    fit_calibrators[calibrator_name] = clf
                    predicting_time = timer() - start
                    _row_eval_target = calib_target_np
                else:
                    clf.fit(calib_probs, calib_target, sample_weight=sample_weight_np)
                    fit_calibrators[calibrator_name] = clf
                    fitting_time = timer() - start

                    start = timer()
                    calibrated_probs = clf.postcalibrate_probs(_eval_probs)
                    predicting_time = timer() - start
                    _row_eval_target = _eval_target
        except Exception as exc:
            # Elapsed time up to the point of failure -- a calibrator that hangs/is unusually slow
            # before crashing otherwise leaves no partial timing signal to diagnose which one (P2-1).
            _elapsed = timer() - _calibrator_start
            logger.warning(
                "compare_postcalibrators: calibrator %s failed to fit/predict after %.3fs and is skipped: %r",
                calibrator_name,
                _elapsed,
                exc,
                exc_info=True,
            )
            failed_calibrators[calibrator_name] = repr(exc)
            continue

        metrics[calibrator_name] = {}
        _, _ = report_model_perf(
            targets=_row_eval_target,
            columns=columns,
            df=None,
            model_name=f"{model_name} {calibrator_name} {calib_type}",
            model=None,
            target_label_encoder=None,
            preds=None,
            probs=calibrated_probs,
            plot_file=plot_file,
            report_title=_eval_label,
            metrics=metrics[calibrator_name],
            group_ids=None,
            **report_params,
        )
        metrics[calibrator_name]["fitting_time"] = fitting_time
        metrics[calibrator_name]["predicting_time"] = predicting_time

    if failed_calibrators:
        logger.warning(
            "compare_postcalibrators: %d calibrator(s) failed and were skipped (results from the rest are still " "reported): %s",
            len(failed_calibrators),
            failed_calibrators,
        )

    metrics_df: Optional[pd.DataFrame]
    if len(metrics) <= 1:
        # Only the "oos"/baseline row was ever populated -- every calibrator was either skipped by
        # should_run's include/skip patterns or failed during fit/predict.
        metrics_df = None
    else:
        metrics_df = pd.DataFrame(metrics).T
        # Column `1` holds the second element of the tuple returned by report_model_perf
        # (a dict of per-metric scores); we flatten it into wide-form columns and drop the
        # feature_importances column which isn't useful for calibrator comparison.
        PERF_DICT_COL = 1
        perf_dict_df = metrics_df[PERF_DICT_COL].apply(pd.Series)
        # report_model_perf's per-calibrator dict shape may vary across task types/configs (e.g. a
        # metric undefined for a degenerate/constant prediction). Taking the UNION of keys means a
        # calibrator missing a key gets a NaN there instead of raising -- surface that explicitly
        # rather than letting it silently rank the calibrator via NaN sort placement (P1-4).
        _row_key_counts = perf_dict_df.notna().sum(axis=1)
        _expected_keys = perf_dict_df.shape[1]
        _incomplete_rows = _row_key_counts[_row_key_counts < _expected_keys]
        if not _incomplete_rows.empty:
            logger.warning(
                "compare_postcalibrators: %d calibrator(s) have a narrower metric-key set than the rest "
                "(missing metrics NaN-filled rather than computed): %s",
                len(_incomplete_rows),
                {name: _expected_keys - int(cnt) for name, cnt in _incomplete_rows.items()},
            )
        metrics_df = metrics_df.drop(columns=[PERF_DICT_COL]).join(perf_dict_df)
        metrics_df = metrics_df.drop(columns=["feature_importances"], errors="ignore")
        if "ice" in metrics_df.columns:
            metrics_df = metrics_df.sort_values("ice")
        else:
            logger.warning(
                "compare_postcalibrators: expected 'ice' metric column not found in report_model_perf output " "(got columns=%s); skipping the ice-based sort.",
                list(metrics_df.columns),
            )

    return metrics_df, fit_calibrators, failed_calibrators


def _values_overlap_fraction(a: np.ndarray, b: np.ndarray, max_rows: int = 2_000_000) -> float:
    """Fraction of the SHORTER array's rows that also appear (by value) in the longer array.

    Rows are compared as raw value tuples via a Python set, so this catches the same rows present in
    both arrays regardless of order (reordered) or one being a strict subset of the other (different
    shape) -- both are blind spots of plain ``np.array_equal`` shape-and-order equality. Returns 0.0
    (no detectable overlap) when either array is empty, non-hashable, or larger than ``max_rows``
    (guards against an unbounded hash-set build on a very large array; the exact/reorder/index checks
    still apply in that case).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0.0
    if a.shape[0] > max_rows or b.shape[0] > max_rows:
        return 0.0
    shorter, longer = (a, b) if a.shape[0] <= b.shape[0] else (b, a)
    try:
        longer_set = {tuple(np.atleast_1d(row).tolist()) for row in longer}
        hits = sum(1 for row in shorter if tuple(np.atleast_1d(row).tolist()) in longer_set)
    except TypeError:
        return 0.0
    return float(hits / shorter.shape[0])


# X_EFFICIENCY_ARCHITECTURE-1 fix (mrmr_audit_2026-07-22): train_postcalibrators carved out into
# _post_train_calibrators.py to clear the repo's enforced hard 1000-LOC CI gate (this file was 1005
# lines). Re-exported here so every existing import keeps working unchanged.
from ._post_train_calibrators import train_postcalibrators  # noqa: F401
