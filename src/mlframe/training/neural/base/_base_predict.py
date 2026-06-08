"""Predict / score mixins carved out of ``neural.base``.

``_PredictMixin`` holds the batched Lightning-trainer predict path
(``_predict_raw``) plus the base ``predict`` / ``score``;
``_ClassifierPredictMixin`` holds the classifier label / probability
overrides. Both operate purely on ``self`` so the estimator classes mix
them in unchanged.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import lightning as L
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score

from .._base_tensor_helpers import to_numpy_safe

logger = __import__("logging").getLogger("mlframe.training.neural.base")


class _PredictMixin:
    """Batched Lightning predict path + base predict / score for the estimator."""

    def _predict_raw(self, X, device: Optional[str] = None, precision: Optional[str] = None, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Internal method for memory-efficient batched prediction using Lightning's trainer.predict().

        This method processes data in batches to avoid OOM errors, leveraging the existing
        DataModule prediction infrastructure.

        Args:
            X: Input data (numpy array, pandas DataFrame, polars DataFrame, or torch.Tensor)
            device: Optional device string ('cpu' or 'cuda'). If None, uses trainer's device.
            precision: Optional precision mode for inference ('16-mixed', 'bf16-mixed', 'bf16-true', or None).
                       If not provided, falls back to the trainer's precision.
            batch_size: Optional batch size for prediction. If None, uses datamodule's batch_size.
                        Larger batch sizes can speed up prediction but use more memory.

        Returns:
            numpy.ndarray: Model predictions (probabilities for classification, values for regression)
        """
        # Apply the same embedding/text -> numeric encoding fitted at train (the MLP has no native embedding/text
        # layers). No-op when the model trained without such columns (encoder is None) or when X arrives already
        # numeric (encoder skips absent columns). Runs before batch-size probing so the probed width is the encoded one.
        _emb_text_enc = getattr(self, "_emb_text_encoder_", None)
        if _emb_text_enc is not None:
            X = _emb_text_enc.transform(X)

        # Lazy import: the package __init__ imports this mixin at class-definition
        # time, so a module-top ``from . import ...`` would be a cycle.
        from . import _PREDICT_ONLY_DM_PARAM_KEYS

        if not hasattr(self, "model") or self.model is None:
            # Wave 37 P1 fix (2026-05-20): NotFittedError per sklearn.
            from sklearn.exceptions import NotFittedError as _NFE
            raise _NFE("Model has not been fitted yet. Call fit() before predict().")

        if not hasattr(self, "prediction_datamodule") or self.prediction_datamodule is None:
            # Reaches here only when the estimator was reconstructed
            # without ever going through .fit() (e.g. user-constructed
            # bare estimator, or a load() path that bypasses the sklearn
            # lifecycle). The post-fit memory-safety pass now only NULLs
            # the heavy train/val tensors INSIDE the datamodule (see
            # fit() epilogue), keeping the lightweight shell around so
            # predict-after-fit no longer reaches this branch at all.
            logger.warning("No datamodule found from training. Creating temporary datamodule for prediction.")
            # Same predict-only-key strip as the fit-time construction: the
            # temporary datamodule constructor rejects ``predict_batch_size``
            # (it is read off self.datamodule_params at line ~1161 below, not
            # passed to the datamodule).
            datamodule = self.datamodule_class(**{
                k: v for k, v in self.datamodule_params.items()
                if k not in _PREDICT_ONLY_DM_PARAM_KEYS
            })
        else:
            # Pre-fix this else-branch was missing and ``datamodule`` was
            # left unbound; line 522 ``datamodule.setup_predict(...)`` then
            # raised ``UnboundLocalError`` whenever a training-time
            # datamodule was retained on the estimator.
            datamodule = self.prediction_datamodule

        # Determine batch size for prediction. Three layers of precedence:
        #   1. ``batch_size`` arg explicitly passed to predict() - always wins.
        #   2. ``datamodule_params["predict_batch_size"]`` - the suite-level knob plumbed through ``train_mlframe_models_suite``
        #      via ``hyperparams_config.mlp_predict_batch_size``.
        #   3. Adaptive resolver based on free memory + input width.
        # The legacy ``datamodule_params["batch_size"]`` (train batch) is the last-resort fallback when the resolver fails.
        #
        # The legacy fallback was a hardcoded 64, which made 4M-row predict paths spend minutes on DataLoader overhead for
        # microseconds of actual MLP compute. The adaptive resolver picks the biggest batch that fits 25% of free memory at the
        # input width, clamped to ``[64, 16384]``.
        if batch_size is not None:
            pred_batch_size = int(batch_size)
            _batch_source = "predict argument"
        else:
            override = self.datamodule_params.get("predict_batch_size")
            if override is not None:
                pred_batch_size = max(1, int(override))
                _batch_source = "datamodule predict_batch_size"
            else:
                try:
                    from mlframe.training.mlp_runtime_defaults import resolve_mlp_predict_batch_size
                    # Probe input width when possible - cheap on numpy / pandas / polars. shape[1] is the standard width on all three.
                    _n_features: Optional[int] = None
                    try:
                        if hasattr(X, "shape") and len(X.shape) >= 2:
                            _n_features = int(X.shape[1])
                        elif hasattr(X, "columns"):
                            _n_features = int(len(X.columns))
                    except Exception:
                        _n_features = None
                    pred_batch_size = resolve_mlp_predict_batch_size(
                        n_features=_n_features,
                        train_batch_size=self.datamodule_params.get("batch_size"),
                    )
                    _batch_source = f"auto n_features={_n_features if _n_features is not None else 'unknown'}"
                except Exception:
                    # Resolver failed - fall back to the train-time batch size (still vastly better than 64 on production setups).
                    _train_batch_hint = self.datamodule_params.get("batch_size", 1024)
                    if isinstance(_train_batch_hint, str):
                        _train_batch_hint = 1024
                    pred_batch_size = int(_train_batch_hint)
                    _batch_source = "fallback train batch_size"
        logger.info("MLP prediction batch_size=%s (%s)", pred_batch_size, _batch_source)

        datamodule.setup_predict(X, batch_size=pred_batch_size)

        if not hasattr(self, "trainer") or self.trainer is None:
            _cached_acc = getattr(self, "_last_predict_accelerator", None)
            _user_acc = (self.trainer_params or {}).get("accelerator")
            trainer_params = {
                "accelerator": _cached_acc or _user_acc or "auto",
                "devices": 1,
                "logger": False,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
            }
        else:
            trainer_params = {
                "accelerator": (
                    self.trainer.accelerator.__class__.__name__.replace("Accelerator", "").lower() if hasattr(self.trainer, "accelerator") else "auto"
                ),
                "devices": 1,
                "logger": False,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
            }
            # Inherit the live trainer's precision when the caller didn't ask for
            # a specific one; an explicit ``precision`` arg is applied below.
            if precision is None and hasattr(self.trainer, "precision"):
                trainer_params["precision"] = self.trainer.precision

        # Honor explicit predict(device=, precision=) overrides in BOTH paths.
        # These used to live only in the ``else`` (live-trainer) branch, but
        # ``self.trainer`` is reset to None after fit and after every predict, so
        # the normal post-fit predict path always took the other branch and
        # silently dropped the device/precision arguments documented on the
        # public predict()/predict_proba() API.
        if device is not None:
            if device.startswith("cuda"):
                trainer_params["accelerator"] = "cuda"
            elif device == "cpu":
                trainer_params["accelerator"] = "cpu"
        if precision is not None:
            trainer_params["precision"] = precision

        # F-67 prediction-trainer caching REVERTED (2026-06-02): reusing one
        # L.Trainer across multiple predict() calls accumulates Lightning's
        # prediction-loop state -- ``predict_loop`` grows ``max_batches`` by one
        # entry per predict while the (also reused) CombinedLoader keeps a single
        # iterable, so the SECOND+ predict assigns a length-N list to
        # ``combined_loader.limits`` against 1 iterable and Lightning raises
        # "Mismatch in number of limits (N) and number of iterables (1)"
        # (combined_loader.py:333). That silently broke EVERY multi-predict fit:
        # val/test/OOF plus the per-feature permutation-importance loop issue
        # dozens of predicts each, and all but the first failed (each dropped via
        # the per-model resilience catch, so models reported degraded/no
        # importances). Lightning Trainers are NOT safe to reuse across
        # predict() calls -- F-67's "idempotent reset between calls" assumption
        # was wrong -- and the cache's only benefit case (many predicts per fit)
        # is exactly its bug case. Build a fresh Trainer per call; the ~236 ms GC
        # the cache saved is negligible against losing every prediction past the
        # first. ``_prediction_trainer_cache`` (pickle-excluded at __getstate__)
        # is left unused.
        #
        # Multilabel-MLP predict -> CPU (2026-06-02): the multilabel head + the
        # per-feature permutation-importance loop (dozens of predicts per fit)
        # churn the model across devices and intermittently corrupt the CUDA
        # context -- "Expected all tensors on the same device, cpu and cuda:0"
        # escalating to "CUDA illegal memory access" and an outright process
        # crash on some hosts (cu118, observed 2026-06-02). The first GPU attempt
        # is what poisons the context, so retrying-on-CPU after the fact is not
        # enough; route the WHOLE multilabel predict to CPU up front. GPU never
        # gets touched -> the context stays clean -> predictions are stable. GPU
        # acceleration is marginal for a small-MLP predict anyway, and
        # binary/regression MLP predict is unaffected (still GPU). 16-mixed
        # precision is invalid on CPU, so drop it when forcing CPU.
        if getattr(self, "_is_multilabel", False):
            trainer_params["accelerator"] = "cpu"
            trainer_params.pop("precision", None)
        prediction_trainer = L.Trainer(**trainer_params)

        # F-G fix: cache the accelerator the current prediction_trainer
        # was built with so the next _predict_raw call (after
        # ``self.trainer = None`` below) can re-resolve to the same
        # device instead of falling through to accelerator="auto".
        try:
            self._last_predict_accelerator = trainer_params.get(
                "accelerator", "auto",
            )
        except Exception:
            pass

        # Unconditional eval() switch - cheap idempotent op, removes the spurious
        # "Model was in training mode during prediction" warning that fired on
        # every legit predict-after-fit (Lightning's Trainer.fit leaves the model
        # in train mode on exit).
        self.model.eval()
        if hasattr(self.model, "_orig_mod"):
            self.model._orig_mod.eval()

        try:
            predictions = prediction_trainer.predict(
                model=self.model,
                datamodule=datamodule,
            )
        except RuntimeError as e:
            # iter293 (2026-05-26): defensive CPU fallback on CUDA runtime
            # errors. Concurrent CUDA usage by another process on the same
            # GPU (or a CUDA context invalidated by an earlier in-process
            # failure) makes the predict trainer hit ``illegal memory
            # access`` / ``out of memory`` / ``device-side assert`` even
            # when the model and data were fine at fit time. Pre-fix the
            # exception propagated and the whole suite died; that masks
            # genuine training results behind a transient CUDA-context
            # problem. Retry exactly once on CPU so the suite still
            # delivers a usable prediction set + surfaces the underlying
            # CUDA issue as a WARNING.
            #
            # Filter is narrow: only RuntimeError messages containing
            # ``CUDA`` or one of the known CUDA-runtime fingerprints get
            # retried. Other RuntimeError variants (shape mismatch,
            # dataloader misconfig) re-raise immediately.
            _msg = str(e)
            _cuda_fingerprints = (
                "CUDA",
                "cuda runtime error",
                "illegal memory access",
                "device-side assert",
                "out of memory",
                "CUBLAS_STATUS_",
                "CUDNN_STATUS_",
                # Device-placement mismatch (model on cuda:0 but a batch / buffer
                # left on cpu) surfaces as this RuntimeError whose text carries
                # only the lowercase device tag ("cuda:0 and cpu"), so it slipped
                # past the uppercase "CUDA" fingerprint above and propagated as a
                # hard "Prediction failed" instead of triggering this CPU retry.
                # The retry resolves it by placing model + data both on cpu.
                # Observed on the multilabel-MLP GPU predict path (2026-06-02).
                "Expected all tensors to be on the same device",
            )
            _is_cuda = (
                trainer_params.get("accelerator") in ("cuda", "gpu", "auto")
                and any(fp in _msg for fp in _cuda_fingerprints)
            )
            if not _is_cuda:
                logger.error(f"Prediction failed: {e}")
                raise
            logger.warning(
                "Prediction on accelerator=%r failed with CUDA-side error "
                "(%s); retrying on CPU. Common cause: another process "
                "holds the GPU or the in-process CUDA context was "
                "invalidated by an earlier failure. The CPU fallback "
                "produces equivalent numeric results but loses GPU "
                "acceleration for this single predict.",
                trainer_params.get("accelerator"), _msg,
            )
            try:
                # iter333 (2026-05-27) follow-up to iter293: the original
                # CPU fallback re-raised because the CUDA context was still
                # dirty when the CPU trainer touched a still-on-GPU tensor
                # via the model / datamodule reference graph. Explicitly:
                #   1. Move the model to CPU (sub-modules + buffers).
                #   2. Empty the CUDA cache (releases tensor memory).
                #   3. Synchronise to flush any pending GPU ops so the next
                #      torch op doesn't replay the failed kernel.
                #   4. Build a fresh CPU-only Trainer.
                try:
                    self.model.to("cpu")
                    if hasattr(self.model, "_orig_mod"):
                        self.model._orig_mod.to("cpu")
                except Exception:
                    pass  # best-effort: model may already be on CPU
                # Reset CUDA state best-effort. ``empty_cache`` is a no-op
                # when CUDA isn't initialised; ``synchronize`` only fires
                # when CUDA is available. ``ipc_collect`` releases inter-
                # process tensor references on Windows where mmap can hold
                # GPU memory alive across the failed predict.
                try:
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        try:
                            torch.cuda.ipc_collect()
                        except Exception:
                            pass
                except Exception:
                    pass
                _cpu_params = {
                    "accelerator": "cpu",
                    "devices": 1,
                    "logger": False,
                    "enable_checkpointing": False,
                    "enable_progress_bar": False,
                }
                cpu_trainer = L.Trainer(**_cpu_params)
                predictions = cpu_trainer.predict(
                    model=self.model,
                    datamodule=datamodule,
                )
            except Exception as e_cpu:
                # iter341 (2026-05-27): if CPU fallback also fails, the
                # CUDA context is permanently invalidated (verified on
                # real concurrent-GPU contention 2026-05-27 c0014). To
                # let the suite progress without GPU acceleration for
                # the remainder of this process, hard-disable CUDA at
                # the torch module level so subsequent estimators do
                # not even try CUDA. The next predict / fit call then
                # falls through to CPU naturally instead of hitting
                # the same dirty-context error.
                try:
                    _cuda_msg = str(e_cpu)
                    _is_still_cuda = any(
                        fp in _cuda_msg for fp in _cuda_fingerprints
                    )
                except Exception:
                    _is_still_cuda = False
                if _is_still_cuda:
                    logger.error(
                        "CPU fallback after CUDA prediction failure ALSO failed "
                        "with a CUDA-side error: %s. The CUDA context is "
                        "permanently invalidated for this process. Disabling "
                        "CUDA at the torch module level so subsequent "
                        "estimators skip GPU and run on CPU; GPU acceleration "
                        "will resume on the next process restart. Original "
                        "CUDA error: %s",
                        e_cpu, e,
                    )
                    try:
                        # Hide CUDA from torch for the remainder of this
                        # process. The env var only helps if torch hasn't
                        # imported yet; the monkey-patch on
                        # torch.cuda.is_available is the load-bearing piece.
                        os.environ["CUDA_VISIBLE_DEVICES"] = ""
                        torch.cuda.is_available = lambda: False
                    except Exception:
                        pass
                    # iter420 (2026-05-27): explicitly move self.model
                    # parameters to CPU BEFORE the retry. Hiding CUDA at
                    # the module level does NOT relocate tensors that are
                    # already on the (invalidated) GPU; Lightning's
                    # accelerator='cpu' then crashes trying to operate
                    # on GPU-resident weights with a broken context.
                    # Surfaced on c0005 LTR run 2026-05-27: even after
                    # iter341's CUDA-hide, the second CPU retry raised
                    # the same CUDA illegal-memory-access error because
                    # ``self.model.parameters()`` were still cuda:0
                    # tensors. ``.to('cpu')`` reads from GPU memory
                    # which is exactly what's broken -- so do it inside
                    # try/except and continue regardless; if the move
                    # itself fails the Trainer call will still raise
                    # cleanly with the original CUDA error.
                    try:
                        self.model.to("cpu")
                    except Exception as _e_move:
                        logger.error(
                            "Failed to move model parameters off the "
                            "invalidated GPU context (%s); the CPU retry "
                            "below will likely re-raise the CUDA error.",
                            _e_move,
                        )
                    # Retry one more time on CPU now that CUDA is hidden
                    # AND model weights are CPU-resident.
                    try:
                        _cpu_params2 = {
                            "accelerator": "cpu",
                            "devices": 1,
                            "logger": False,
                            "enable_checkpointing": False,
                            "enable_progress_bar": False,
                        }
                        cpu_trainer2 = L.Trainer(**_cpu_params2)
                        predictions = cpu_trainer2.predict(
                            model=self.model,
                            datamodule=datamodule,
                        )
                    except Exception as e_cpu2:
                        logger.error(
                            "Even with CUDA hidden the predict failed: %s. "
                            "Re-raising original CUDA error.",
                            e_cpu2,
                        )
                        raise
                else:
                    logger.error(
                        "CPU fallback after CUDA prediction failure ALSO failed "
                        "with a non-CUDA error: %s. Original CUDA error: %s",
                        e_cpu, e,
                    )
                    raise
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

        self.trainer = None

        if len(predictions) == 0:
            raise RuntimeError("No predictions were generated. Check your data and model.")

        # Handle different return types from predict_step
        if isinstance(predictions[0], torch.Tensor):
            predictions = torch.cat(predictions, dim=0)
            predictions = to_numpy_safe(predictions, cpu=True)
        elif isinstance(predictions[0], np.ndarray):
            predictions = np.concatenate(predictions, axis=0)
        else:
            raise TypeError(f"Unexpected prediction type: {type(predictions[0])}")

        logger.info("Generated predictions with shape %s", predictions.shape)

        return predictions

    def predict(self, X, device: Optional[str] = None, precision: Optional[str] = None, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Predict using the model with memory-efficient batched processing.

        Args:
            X: Input data (numpy array, pandas DataFrame, polars DataFrame, or torch.Tensor)
            device: Optional device string ('cpu' or 'cuda'). If None, uses trainer's device.
            precision: Optional precision mode for inference ('16-mixed', 'bf16-mixed', etc.)
            batch_size: Optional batch size for prediction. Larger values speed up prediction but use more memory.

        Returns:
            numpy.ndarray: Model predictions (class labels for classification, values for regression)
        """
        predictions = self._predict_raw(X, device=device, precision=precision, batch_size=batch_size)

        # For regression, return raw predictions (squeeze to 1D for single-target regression)
        if isinstance(self, RegressorMixin):
            if predictions.ndim == 2 and predictions.shape[1] == 1:
                predictions = predictions.squeeze(axis=1)
            return predictions

        # Base class returns probabilities for classification; PytorchLightningClassifier overrides to return labels.
        return predictions

    def score(self, X, y, sample_weight: Optional[np.ndarray] = None) -> float:
        """Returns the coefficient of determination R^2 for regression or accuracy for classification."""
        y_pred = self.predict(X)
        if isinstance(self, RegressorMixin):
            return r2_score(y, y_pred, sample_weight=sample_weight)
        elif isinstance(self, ClassifierMixin):
            # y_pred is already class labels from PytorchLightningClassifier.predict()
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        else:
            raise TypeError(f"Estimator must be a RegressorMixin or ClassifierMixin, got {type(self).__name__}")


class _ClassifierPredictMixin:
    """Classifier label + probability prediction overrides."""

    def predict(self, X, device: Optional[str] = None, precision: Optional[str] = None, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X: Input data
            device: Optional device string ('cpu' or 'cuda')
            precision: Optional precision mode for inference
            batch_size: Optional batch size for prediction

        Returns:
            numpy.ndarray: Predicted class labels
        """
        raw = self._predict_raw(X, device=device, precision=precision, batch_size=batch_size)
        # Multilabel: each of the K labels is an independent sigmoid, so predict()
        # must return an (N, K) 0/1 indicator matrix (each label thresholded
        # independently), NOT a single argmax label per row. The multilabel head
        # sets _is_multilabel=True and leaves _label_encoder/classes_ unset, so
        # without this guard the code fell through to ``argmax`` -> shape (N,),
        # breaking the predict() contract and degenerating multilabel
        # permutation-importance (predict vs 2-D y mismatch).
        if getattr(self, "_is_multilabel", False):
            return (np.asarray(raw) >= 0.5).astype(np.int64)
        # F-05 binary path: raw has shape (N, 1) and contains P(y=1).
        # Use ``> 0.5`` (strict) so the predict() / predict_proba contract holds:
        # at raw==0.5 a ``>= 0.5`` threshold returned class 1 while
        # ``argmax(predict_proba)`` -> ``argmax([0.5, 0.5])`` returns 0 (numpy
        # tie-break to first index). The strict comparison aligns the two
        # public methods on collapsed / under-trained models where many rows
        # sit at exactly 0.5.
        if getattr(self, "_binary_sigmoid_head", False):
            idx = (raw.reshape(-1) > 0.5).astype(np.int64)
        else:
            idx = np.argmax(raw, axis=1)
        # sklearn convention: ``predict`` returns LABELS (entries of
        # ``classes_``), not argmax INDICES. Pre-fix this returned the bare
        # indices, which the downstream reporting layer band-aided with
        # ``model.classes_[preds]`` (see _reporting_probabilistic.py:266);
        # any direct ``accuracy_score(y, model.predict(X))`` silently
        # miscalled for any y whose value set was not ``{0..K-1}``. F-01 in
        # the 2026-05-30 mlp audit. The ``_label_encoder`` branch is the
        # canonical path; ``classes_`` direct indexing covers estimators
        # loaded from an older pickle that has classes_ but no encoder; the
        # final ``return idx`` covers multilabel / dropped-state cases.
        if getattr(self, "_label_encoder", None) is not None:
            return self._label_encoder.inverse_transform(idx)
        if getattr(self, "classes_", None) is not None:
            return self.classes_[idx]
        return idx

    def predict_proba(self, X, device: Optional[str] = None, precision: Optional[str] = None, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Args:
            X: Input data
            device: Optional device string ('cpu' or 'cuda')
            precision: Optional precision mode for inference
            batch_size: Optional batch size for prediction

        Returns:
            numpy.ndarray: Predicted class probabilities, shape (N, K).
        """
        raw = self._predict_raw(X, device=device, precision=precision, batch_size=batch_size)
        # F-05 binary path: raw has shape (N, 1) with P(y=1); stack
        # [1-p, p] to honour the sklearn (N, 2) ``predict_proba`` contract.
        # The column order matches ``classes_`` (sorted): col 0 = P(class[0]),
        # col 1 = P(class[1]).
        if getattr(self, "_binary_sigmoid_head", False):
            p1 = raw.reshape(-1)
            return np.column_stack([1.0 - p1, p1])
        return raw
