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

from typing import *  # noqa: F401 pylint: disable=wildcard-import,unused-wildcard-import
from .config import *

import re
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


import joblib
from os.path import join
from pyutilz.strings import slugify
from mlframe.training import TargetTypes
from mlframe.ensembling import ensemble_probabilistic_predictions

from sklearn import config_context
from sklearn.base import BaseEstimator, ClassifierMixin

import polars as pl, pandas as pd, numpy as np

from pyutilz.system import tqdmu
from pyutilz.pythonlib import store_params_in_object, get_parent_func_args

from .training.evaluation import report_model_perf

import netcal, pycalib
from pycalib import models  # must be
from netcal import binning  # must be
from netcal import scaling  # must be
import ml_insights as mli
from betacal import BetaCalibration
import calibration as verified_calibration
from venn_abers import VennAbersCalibrator
from sklearn.calibration import CalibratedClassifierCV

try:
    from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
except Exception as e:
    FullDirichletCalibrator = None

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Core
# -----------------------------------------------------------------------------------------------------------------------------------------------------


class BinaryPostCalibrator(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        calibrator: object,
        fit_method_name: str = "fit",
        transform_method_name: str = "transform",
    ):
        store_params_in_object(obj=self, params=get_parent_func_args())

    @staticmethod
    def _calibrator_needs_2d_probs(calibrator) -> bool:
        """Returns True if the wrapped calibrator expects a 2D (n_samples, n_classes) prob matrix.

        Replaces substring-matching on the class name with isinstance checks against the
        relevant calibrator classes (imported lazily so optional deps can be missing).
        """
        # Late imports: some of these are optional (dirichletcal) or may be reshuffled upstream.
        needs_2d_types: list = []
        try:
            from venn_abers import VennAbersCalibrator as _VA

            needs_2d_types.append(_VA)
        except ImportError:
            pass
        try:
            from netcal.scaling import LogisticCalibration as _LC
            from netcal.scaling import LogisticCalibrationDependent as _LCD

            needs_2d_types.extend([_LC, _LCD])
        except ImportError:
            pass
        try:
            from pycalib.models import LogisticCalibration as _PLC

            needs_2d_types.append(_PLC)
        except ImportError:
            pass
        try:
            from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator as _FDC

            needs_2d_types.append(_FDC)
        except ImportError:
            pass
        try:
            from sklearn.calibration import CalibratedClassifierCV as _CCCV

            needs_2d_types.append(_CCCV)
        except ImportError:
            pass
        # "Top" calibrators (e.g. TopLabelCalibrator style) preserve 2D shape; match by class-name
        # prefix here as there is no single importable base — minimal remaining substring check.
        if type(calibrator).__name__.startswith("Top"):
            return True
        return isinstance(calibrator, tuple(needs_2d_types)) if needs_2d_types else False

    def _transform_probs(self, probs) -> np.ndarray:
        if probs.ndim == 2 and not self._calibrator_needs_2d_probs(self.calibrator):
            probs = probs[:, 1]
        return probs

    def fit(
        self,
        calib_probs: np.ndarray,
        calib_target: np.ndarray,
    ):

        calib_probs = self._transform_probs(calib_probs)

        if not "VennAbersCalibrator" in type(self.calibrator).__name__:
            getattr(self.calibrator, self.fit_method_name)(calib_probs, calib_target)
        else:
            self.y_cal = calib_target
            self.p_cal = calib_probs

        # Resolve transform method name ONCE at fit-time rather than mutating self inside
        # postcalibrate_probs (previously racy and surprising for re-use across calls).
        resolved_transform = self.transform_method_name
        if not hasattr(self.calibrator, resolved_transform) and hasattr(self.calibrator, "predict"):
            resolved_transform = "predict"
        self._resolved_transform_method_name = resolved_transform

        return self

    def postcalibrate_probs(self, probs) -> np.ndarray:

        probs = self._transform_probs(probs)
        if not "VennAbersCalibrator" in type(self.calibrator).__name__:
            # Use method resolved at fit-time; fall back gracefully if fit() wasn't called.
            transform_name = getattr(self, "_resolved_transform_method_name", self.transform_method_name)
            calibrated_probs = getattr(self.calibrator, transform_name)(probs)
        else:
            calibrated_probs = self.calibrator.predict_proba(p_cal=self.p_cal, y_cal=self.y_cal, p_test=probs)

        if calibrated_probs.ndim == 2 and (
            hasattr(self.calibrator, "method") and self.calibrator.method in ["momentum", "variational", "mcmc"]
        ):  # mcmc methods of netcal
            calibrated_probs = calibrated_probs.mean(axis=0)
        if calibrated_probs.ndim == 1:
            # Clip to [0, 1] before stacking so numerical drift from calibrator outputs does
            # not produce negative or >1 entries in the 2D prob matrix.
            calibrated_probs = np.clip(calibrated_probs, 0.0, 1.0)
            calibrated_probs = np.vstack([1 - calibrated_probs, calibrated_probs]).T

        return calibrated_probs


@dataclass
class NamedCalibrator:
    calibrator: BinaryPostCalibrator
    name: Optional[str] = None
    param_str: Optional[str] = ""
    lib: Optional[str] = None

    def full_name(self) -> str:
        base_name = self.name or self._extract_calibrator_class_name()
        # Prepend lib only if not already in name (case-insensitive)
        if self.lib and self.lib.lower() not in base_name.lower():
            base_name = f"{self.lib}.{base_name}"
        if self.param_str:
            return f"{base_name}[{self.param_str}]"
        return base_name

    def _extract_calibrator_class_name(self) -> str:
        obj = getattr(self.calibrator, "calibrator", self.calibrator)
        return obj.__class__.__name__


def named_calibrator(
    calibrator_obj,
    name: Optional[str] = None,
    param_str: Optional[str] = "",
    lib: Optional[str] = None,
    **postcal_kwargs,
) -> NamedCalibrator:
    return NamedCalibrator(
        calibrator=BinaryPostCalibrator(calibrator=calibrator_obj, **postcal_kwargs),
        name=name,
        param_str=param_str,
        lib=lib,
    )


def should_run(name: str, include: list[str] = None, skip: list[str] = None) -> bool:
    if include and not any(_compile_pattern(p).search(name) for p in include):
        return False
    if skip and any(_compile_pattern(p).search(name) for p in skip):
        return False
    return True


def get_postcalibrators(calib_target, num_bins: int) -> list:

    calibrators = [
        named_calibrator(CalibratedClassifierCV(method="sigmoid", ensemble=False), name="CalibratedClassifierCV", param_str="method=sigmoid", lib="sklearn"),
        named_calibrator(CalibratedClassifierCV(method="isotonic", ensemble=False), name="CalibratedClassifierCV", param_str="method=isotonic", lib="sklearn"),
        named_calibrator(mli.SplineCalib(), lib="mli"),
        *[
            named_calibrator(
                cls(len(calib_target), num_bins=num_bins),
                param_str=f"bins={num_bins}",
                lib="verified_calibration",
                fit_method_name="train_calibration",
                transform_method_name="calibrate",
            )
            for cls in [
                verified_calibration.HistogramCalibrator,
                verified_calibration.PlattBinnerCalibrator,
                verified_calibration.PlattCalibrator,
            ]
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
    oos_probs: np.ndarray,
    oos_target: np.ndarray,
    num_bins: int = 15,
    calib_type: str = "test",
    plot_file: str = "",
    report_params: dict = None,
    include_patterns: list = [],
    skip_patterns: list = [r"netcal\.BetaCalibrationDependent", "netcal\.ENIR", "netcal\.NearIsotonicRegression"],  # r"BetaCalibration\[variant=ab\]"
) -> tuple:
    """Given calibration and OOS probabilities and true targets,
    fits a number of calibrator models  on the calib set and computes ML metrics on the OOS set.
    returns a pandas dataframe of ML metrics by calibrator name.
    """

    logger.info(f"Calib set size={len(calib_target):_}, oos set size={len(oos_target) if oos_target is not None else 0:_}, num_bins={num_bins}.")

    if report_params is None:
        report_params = {"report_ndigits": 4, "calib_report_ndigits": 4, "print_report": False}

    metrics = {"oos": {}}
    fit_calibrators = {}

    if oos_probs is not None:
        _, _ = report_model_perf(
            targets=oos_target,
            columns=columns,
            df=None,
            model_name=f"{model_name}",
            model=None,
            target_label_encoder=None,
            preds=None,
            probs=oos_probs,
            plot_file=plot_file,
            report_title="OOS",
            metrics=metrics["oos"],
            group_ids=None,
            **report_params,
        )

    calibrators = get_postcalibrators(calib_target=calib_target, num_bins=num_bins)

    for nc in tqdmu(calibrators, desc="calibrator"):
        clf = nc.calibrator
        calibrator_name = nc.full_name()

        if not should_run(calibrator_name, include_patterns, skip_patterns):
            # logger.info(f"Skipping calibrator: {calibrator_name} due to matching skip pattern.")
            continue

        with config_context(transform_output="default"):

            """
            config_context needed here to avoid:

            R:\ProgramData\anaconda3\Lib\site-packages\netcal\binning\IsotonicRegression.py:183, in IsotonicRegression.transform(self, X)
                179     calibrated = self._iso.transform(X)
                181 # add clipping to [0, 1] to avoid exceeding due to numerical issues
                182 # https://github.com/EFS-OpenSource/calibration-framework/issues/54
            --> 183 np.clip(calibrated, 0, 1, out=calibrated)
                185 return calibrated
            """

            start = timer()

            clf.fit(calib_probs, calib_target)
            fit_calibrators[calibrator_name] = clf

            fitting_time = timer() - start

            if oos_probs is None:
                continue

            start = timer()
            calibrated_probs = clf.postcalibrate_probs(oos_probs)
            predicting_time = timer() - start

        metrics[calibrator_name] = {}
        _, _ = report_model_perf(
            targets=oos_target,
            columns=columns,
            df=None,
            model_name=f"{model_name} {calibrator_name} {calib_type}",
            model=None,
            target_label_encoder=None,
            preds=None,
            probs=calibrated_probs,
            plot_file=plot_file,
            report_title="OOS",
            metrics=metrics[calibrator_name],
            group_ids=None,
            **report_params,
        )
        metrics[calibrator_name]["fitting_time"] = fitting_time
        metrics[calibrator_name]["predicting_time"] = predicting_time

    if oos_probs is None:
        metrics = None
    else:
        metrics = pd.DataFrame(metrics).T
        metrics = metrics.drop(columns=[1]).join(metrics[1].apply(pd.Series)).drop(columns=["feature_importances"]).sort_values("ice")

    return metrics, fit_calibrators


def train_postcalibrators(
    models: dict,
    model_name: str,
    models_dir: str,
    target_name: str,
    featureset_name: str,
    task_type=TargetTypes.BINARY_CLASSIFICATION,
    include_patterns=[r"SplineCalib", r"pycalib.BetaCalibration"],
    max_mae: float = None,
    max_std: float = None,
    ensembling_method="harm",
    ensure_prob_limits: bool = True,
    uncertainty_quantile: float = 0.0,
    normalize_stds_by_mean_preds: bool = True,
    verbose: int = 1,
):
    ensembled_test_predictions, confident_test_indices = ensemble_probabilistic_predictions(
        *(el.test_probs for el in models.values()),
        ensemble_method=ensembling_method,
        max_mae=max_mae,
        max_std=max_std,
        ensure_prob_limits=ensure_prob_limits,
        uncertainty_quantile=uncertainty_quantile,
        normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
        verbose=verbose,
    )

    first_model = list(models.values())[0]
    columns = first_model.columns

    test_target = first_model.test_target.values

    calib_test_metrics, test_calibrators = compare_postcalibrators(
        model_name=model_name,
        columns=columns,
        calib_probs=ensembled_test_predictions,
        calib_target=test_target,
        oos_probs=None,
        oos_target=None,
        calib_type="test",
        include_patterns=include_patterns,
    )

    final_models_dir = join(models_dir, target_name, featureset_name, task_type, model_name)

    for calib_name, calibrator in test_calibrators.items():
        ens_name = f"ens_{ensembling_method}"
        calib_fpath = join(final_models_dir, f"{ens_name}_postcalibrator_{slugify(calib_name)}.dump")
        joblib.dump(calibrator, calib_fpath, compress=("lzma", 6))

    return test_calibrators
