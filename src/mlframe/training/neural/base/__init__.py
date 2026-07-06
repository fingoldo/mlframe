"""
Base infrastructure for PyTorch Lightning models in mlframe.

This module provides:
- sklearn-compatible estimator wrappers (PytorchLightningEstimator, Regressor, Classifier)
- Callbacks (BestEpochModelCheckpoint, AggregatingValidationCallback, etc.)
- Utilities (MetricSpec, to_tensor_any, to_numpy_safe)
"""

from __future__ import annotations


import logging

logger = logging.getLogger(__name__)

# Keys the suite stashes in ``datamodule_params`` for the estimator's own
# predict-time use that are NOT constructor parameters of the datamodule
# classes (TorchDataModule / RecurrentDataModule). They must be stripped
# before any ``datamodule_class(**params)`` splat. ``predict_batch_size`` is
# read directly off ``self.datamodule_params`` in ``_predict_raw``; passing it
# to the datamodule constructor raises ``unexpected keyword argument``.
_PREDICT_ONLY_DM_PARAM_KEYS = frozenset({"predict_batch_size"})

# Logging filters / MetricSpec / suppression context manager / _rmse_metric: side-effect log-filter attach runs at sibling import time. Re-exports preserve identity for downstream isinstance / hasattr.
from .._base_logging import (  # noqa: F401, E402
    _LightningRankZeroNoiseFilter,
    _LIGHTNING_NOISE_FILTER,
    suppress_lightning_workers_warning,
    _rmse_metric,
    MetricSpec,
)

# stdlib
from typing import Dict, Optional

# third-party
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

# iter189 (2026-05-23): Lightning's _load_external_callbacks scans every
# installed Python distribution via importlib.metadata.entry_points on EACH
# Trainer.fit() / Trainer.predict() invocation -- ~180ms / call on a Windows
# box with a typical anaconda site-packages (5346 dist-info METADATA reads).
# c0065 iter189 profile attributed 1.484s to this across 6 fit calls (~6% of
# the 23.4s wall). Result is process-stable (sys.path + installed dists don't
# change between fits), so cache it.
#
# Mirrors the _PROBE_PRECISION_CACHE pattern in mlp_runtime_defaults.py
# (iter181) and _CB_GPU_USABLE_CACHE in _cb_pool.py. Defensive try/except so
# a Lightning internal-API rename surfaces as a slow-but-correct fallback,
# not an ImportError that crashes mlframe import.
#
# iter259 (2026-05-23) follow-up: patching only ``_lf_registry`` misses the
# real callers. ``callback_connector.py`` and ``fabric.py`` do
# ``from lightning.fabric.utilities.registry import _load_external_callbacks``
# at import time, which binds the ORIGINAL function into the caller's module
# namespace -- mutating ``_lf_registry._load_external_callbacks`` after that
# leaves caller bindings stale. c0119 iter259 profile still attributed 3.73s
# to ``_load_external_callbacks`` (12 calls x 311ms) despite the iter189
# patch. Rebind in every importer to make the cache actually fire.
try:
    from lightning.fabric.utilities import registry as _lf_registry
    if not getattr(_lf_registry, "_mlframe_callback_cache_installed", False):
        _orig_load_external_callbacks = _lf_registry._load_external_callbacks
        _external_callback_cache: Dict[str, list] = {}

        def _load_external_callbacks_cached(group: str) -> list:
            cached = _external_callback_cache.get(group)
            if cached is None:
                cached = _orig_load_external_callbacks(group)
                _external_callback_cache[group] = cached
            return list(cached)  # defensive copy so callers can't mutate cache

        _lf_registry._load_external_callbacks = _load_external_callbacks_cached
        # Modern Lightning ships two distinct package namespaces:
        #   * ``lightning.fabric.utilities.registry``      (umbrella re-export)
        #   * ``lightning_fabric.utilities.registry``      (standalone wheel)
        # They are SEPARATE module objects with their own function objects --
        # patching only the umbrella leaves the standalone path on the slow
        # function. ``lightning_fabric.fabric`` imports the standalone copy
        # via ``from lightning_fabric.utilities.registry import
        # _load_external_callbacks`` -- without the explicit patch its 12
        # per-fit calls slipped past the cache. Mirror the wrap on the
        # standalone module + every importer of the standalone symbol.
        try:
            from lightning_fabric.utilities import registry as _lfs_registry
            if _lfs_registry is not _lf_registry:
                _lfs_registry._load_external_callbacks = _load_external_callbacks_cached
                _lfs_registry._mlframe_callback_cache_installed = True
        except ImportError:
            pass
        # Rebind in every Lightning module that imported the original by name.
        # Each ``from ... import _load_external_callbacks`` creates a local
        # binding that mutating the source module does not affect. Walk
        # sys.modules and rebind every match. Best-effort: a Lightning version
        # that adds a new caller will fall back to the slow path until the
        # next mlframe release, never breaking.
        #
        # Both the umbrella ``lightning.*`` and the standalone ``lightning_fabric.*``
        # namespaces exist in modern Lightning installs (the umbrella re-exports
        # the standalone package); we must cover both prefixes or callers via
        # the standalone path bypass the cache.
        import sys as _sys_for_rebind
        _rebind_prefixes = ("lightning.", "lightning_fabric.", "lightning_pytorch.")
        for _mod_name, _mod in list(_sys_for_rebind.modules.items()):
            if _mod is None:
                continue
            if not (_mod_name == "lightning" or _mod_name == "lightning_fabric" or _mod_name.startswith(_rebind_prefixes)):
                continue
            _local_ref = getattr(_mod, "_load_external_callbacks", None)
            if _local_ref is None or _local_ref is _load_external_callbacks_cached:
                continue
            try:
                _mod._load_external_callbacks = _load_external_callbacks_cached
            except Exception:  # nosec B110 - non-trivial body
                # Frozen / immutable module objects: skip silently.
                pass
        _lf_registry._mlframe_callback_cache_installed = True
except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
    logger.debug("suppressed in __init__.py:119: %s", e)
    pass


from pyutilz.pythonlib import get_parent_func_args, store_params_in_object  # noqa: F401

# Tensor / dataframe helpers carved to sibling. Re-exports preserve identity.
from .._base_tensor_helpers import (  # noqa: F401, E402
    custom_collate_fn,
    to_tensor_any,
    to_numpy_safe,
    _ensure_numpy,
    safe_accelerator,
)

# sklearn get_params/set_params carved to a sibling; bound as methods on the
# estimator class below. Re-import keeps them accessible at module scope.
from .._base_sklearn_params import (  # noqa: F401, E402
    get_params as _sklearn_get_params,
    set_params as _sklearn_set_params,
)

# Loss-builder + input-validation leaves carved to a sibling. Re-exported so
# downstream importers (and the fit mixin) keep resolving them.
from ._base_losses import _make_binary_focal_loss, _validate_no_nan_inf  # noqa: F401, E402

# Fit / predict bodies carved to mixin siblings; mixed into the estimator
# classes below. The mixins lazy-import ``base`` symbols in-body to avoid a
# load-time cycle.
from ._base_fit import _FitMixin  # noqa: E402
from ._base_predict import _PredictMixin, _ClassifierPredictMixin  # noqa: E402


class PytorchLightningEstimator(_FitMixin, _PredictMixin, BaseEstimator):
    """Wrapper that allows Pytorch Lightning model, datamodule and trainer to participate in sklearn pipelines.
    Supports early stopping (via eval_set in fit_params).
    """

    def __getstate__(self) -> dict:
        """F-73b (2026-06-01): drop runtime-only, non-picklable caches on
        serialise. The F-67 prediction-trainer cache holds live
        ``lightning.pytorch.Trainer`` objects which reference a
        ``lightning_utilities.core.rank_zero.WarningCache`` -- a class the
        mlframe save_load ``_SafeUnpickler`` allowlist (correctly) blocks.
        These caches exist purely to skip per-predict Trainer
        re-construction; the next predict() on the restored estimator
        lazily rebuilds the cache. Also null the live ``trainer`` for the
        same reason (it's already nulled after every fit/predict, but a
        mid-lifecycle pickle could still catch a live one).
        """
        state = self.__dict__.copy()
        state.pop("_prediction_trainer_cache", None)
        state["trainer"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # Rebuilt lazily on the next predict(); start clean.
        self._prediction_trainer_cache = {}

    def __init__(
        self,
        model_class: object,
        model_params: dict,
        network_params: dict,
        datamodule_class: object,
        datamodule_params: dict,
        trainer_params: object,
        use_swa: bool = False,
        swa_params: dict = None,
        use_ema: bool = False,
        ema_params: dict = None,
        label_smoothing: float = 0.0,
        focal_loss_gamma: Optional[float] = None,
        focal_loss_alpha: float = 0.25,
        tune_params: bool = False,
        tune_batch_size: bool = False,
        float32_matmul_precision: str = None,
        early_stopping_rounds: int = 100,
        # Monotonic strict-decline overfitting stop, COMPLEMENTARY to ``early_stopping_rounds`` patience:
        # stop once val_<metric> strictly worsens for this many consecutive epochs since the best (a
        # confident-overfitting signal that fires faster than patience). Default-on; None disables.
        monotonic_decline_patience: Optional[int] = 7,
        # Per-epoch full-metric-suite capture into ``iteration_metrics_`` for meta-learning / HPO-from-early-
        # observation. Default-ON for neural (val preds are already concatenated each epoch, so the only marginal
        # cost is the cheap metric kernel). Set False to skip the capture; None also resolves to the ON default.
        capture_iteration_metrics: Optional[bool] = True,
        random_state: Optional[int] = None,
        class_weight=None,
        use_learnable_cat_embeddings: bool = True,
        categorical_embed_dim: Optional[int] = None,
    ):
        # ``random_state``: sklearn-canonical seed parameter (F-06, 2026-05-30).
        # When set to an integer, ``_fit_common`` seeds torch / numpy / Python
        # random + the Lightning DataLoader worker seed at fit() entry, so two
        # ``fit()`` calls on the same data with the same ``random_state``
        # produce bit-identical predictions. ``None`` (the default) preserves
        # the pre-fix non-deterministic behaviour: callers who manage their
        # own seed (e.g. via a higher-level pipeline) are not overridden.
        #
        # ``use_learnable_cat_embeddings`` (default True): when fit() receives ``cat_features`` via fit_params, factorize those raw cat
        # columns to integer codes at the fit boundary and prepend a learnable ``nn.Embedding`` per cat (trained end-to-end) instead of
        # relying on an upstream target encoder. This recovers non-monotone category->target structure a single target-encoded scalar cannot.
        # Set False to disable the in-estimator factorization (the suite's CatBoostEncoder path, or a caller's own encoding, then handles cats).
        # ``categorical_embed_dim``: fixed per-cat embedding width; None uses the fastai heuristic min(50, round(1.6*card**0.56)).
        #
        # Don't modify swa_params here (e.g., `swa_params or {}`) because sklearn's clone() requires constructor parameters not be
        # modified. Handle None later.
        store_params_in_object(obj=self, params=get_parent_func_args())
        # Runtime (non-param) attribute, mirrored in __getstate__/__setstate__. F-67 prediction-trainer caching was reverted 2026-06-02 (Lightning Trainer reuse broke multi-predict fits), so this stays empty -- predict() builds a fresh Trainer per call -- but the attribute must exist so introspection and the pickle-state symmetry don't hit AttributeError on a freshly-constructed (never-pickled) estimator.
        self._prediction_trainer_cache = {}

    # sklearn protocol methods carved to ``_base_sklearn_params`` (monolith
    # split). Bound here so ``clone()`` / ``get_params(deep=True)`` behave
    # identically -- the functions take ``self`` as first arg.
    get_params = _sklearn_get_params
    set_params = _sklearn_set_params


class PytorchLightningRegressor(RegressorMixin, PytorchLightningEstimator):  # RegressorMixin must come first
    _estimator_type = "regressor"


class PytorchLightningClassifier(
    _ClassifierPredictMixin,
    ClassifierMixin,
    PytorchLightningEstimator,
):  # ClassifierMixin must come first
    _estimator_type = "classifier"


# Callback classes carved to ``_base_callbacks.py``. Re-exports preserve class identity so downstream isinstance / Trainer callback-list checks keep working unchanged.
from .._base_callbacks import (  # noqa: F401, E402
    NetworkGraphLoggingCallback,
    AggregatingValidationCallback,
    ValLossDivergenceCallback,
    MonotonicDeclineStopCallback,
    BestEpochModelCheckpoint,
    PeriodicLearningRateFinder,
)
