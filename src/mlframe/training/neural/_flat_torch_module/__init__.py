"""MLPTorchModel carved out of ``mlframe.training.neural.flat``.

Re-imported at the parent module bottom so historical
``from mlframe.training.neural.flat import MLPTorchModel`` callers keep working.
"""
from __future__ import annotations

import contextvars as _contextvars
import copyreg as _copyreg
import logging
import os
from typing import Any, Callable, Dict, List, Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from ..base import MetricSpec
from ._flat_torch_loss import _LossMixin
from ._flat_torch_predict_accel import _PredictAccelMixin

logger = logging.getLogger("mlframe.training.neural.flat")


def _rebuild_contextvar(name: str) -> "_contextvars.ContextVar":
    return _contextvars.ContextVar(name)


def _reduce_contextvar(cv: "_contextvars.ContextVar"):
    # A ContextVar is pure runtime context (its per-thread/-task value is never part of a model's fitted state).
    # Python 3.14 made the ``warnings`` filter state ContextVar-backed, so a captured warnings context can leak into a
    # LightningModule's pickled state and crash dill ("cannot pickle '_contextvars.ContextVar' object"). Reduce a
    # ContextVar to a fresh one with the same name -- the unpickled process re-initialises its own context, which is the
    # correct semantics for a runtime-only object. Registered process-wide so it covers ContextVars nested anywhere in
    # the serialised graph (hparams / torch submodules / captured closures), not only top-level module attributes.
    return (_rebuild_contextvar, (cv.name,))


_copyreg.pickle(_contextvars.ContextVar, _reduce_contextvar)


class MLPTorchModel(_PredictAccelMixin, _LossMixin, L.LightningModule):
    def __init__(
        self,
        loss_fn: Callable,
        metrics: List[MetricSpec],
        network: torch.nn.Module,
        learning_rate: float = 1e-3,
        l1_alpha: float = 0.0,
        optimizer: Optional[type] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[type] = None,
        lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        compile_network: Optional[str] = None,
        compute_trainset_metrics: bool = False,
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_monitor: Optional[str] = None,
        load_best_weights_on_train_end: bool = True,
        log_lr: bool = False,
        task_type: Optional[str] = None,
        use_lookahead: bool = False,
        lookahead_k: int = 5,
        lookahead_alpha: float = 0.5,
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
        use_sam: bool = False,
        sam_rho: float = 0.05,
        sam_adaptive: bool = False,
    ):
        """
        PyTorch Lightning module for MLP training.

        Args:
            loss_fn: Loss function callable
            metrics: List of MetricSpec objects for evaluation
            network: The neural network module
            learning_rate: Learning rate for optimizer
            l1_alpha: L1 regularization coefficient (0.0 = no regularization)
            optimizer: Optimizer class (default: AdamW)
            optimizer_kwargs: Additional kwargs for optimizer
            lr_scheduler: Learning rate scheduler class. Recommended choices
                for tabular MLP fits:
                  * ``torch.optim.lr_scheduler.OneCycleLR`` -- best for short
                    fits (<50 epochs), single LR cycle. Special-cased here:
                    ``total_steps`` auto-computed from ``trainer.max_epochs * steps_per_epoch``.
                  * ``torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`` --
                    best for longer fits (50+ epochs). Pass ``T_0`` (epochs
                    per first cycle) and ``T_mult`` (cycle-growth ratio) via
                    ``lr_scheduler_kwargs``. Matches RealMLP-TD's default
                    (Holzmuller 2024) and gives better final minima than a
                    single OneCycle on the 50-100 epoch range.
                  * ``torch.optim.lr_scheduler.ReduceLROnPlateau`` -- value-
                    driven; requires ``lr_scheduler_monitor`` (e.g. ``"val_loss"``).
            lr_scheduler_kwargs: Additional kwargs for scheduler
            compile_network: torch.compile mode (e.g., 'max-autotune', 'reduce-overhead')
            compute_trainset_metrics: Whether to compute metrics on training set
            lr_scheduler_interval: 'epoch' or 'step'
            lr_scheduler_monitor: Metric to monitor for scheduler (e.g., 'val_loss')
            load_best_weights_on_train_end: Load best checkpoint weights after training
        """
        super().__init__()

        if network is None:
            raise ValueError("network must be provided")
        if loss_fn is None:
            raise ValueError("loss_fn must be provided")
        if metrics is None:
            metrics = []
        if lr_scheduler_interval not in ["epoch", "step"]:
            raise ValueError(f"lr_scheduler_interval must be 'epoch' or 'step', got {lr_scheduler_interval}")

        optimizer = optimizer or torch.optim.AdamW
        optimizer_kwargs = optimizer_kwargs or {}
        lr_scheduler_kwargs = lr_scheduler_kwargs or {}

        # Skip non-serializable objects so Lightning can pickle hparams.
        self.save_hyperparameters(ignore=["loss_fn", "metrics", "network", "optimizer", "lr_scheduler"])

        self.network = network
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.best_epoch = None
        # task_type drives predict_step activation for K>1 outputs: None keeps softmax (multi-class),
        # "multilabel" switches to per-label sigmoid. Regression/binary go through the dim==1 branch.
        self.task_type = task_type

        self.training_step_outputs = []
        self.validation_step_outputs = []

        # F-38 (2026-05-31): CUDA-graph predict cache. Shape-keyed map of
        # (input_shape, dtype, device) -> (static_input_buffer, graphed_fn).
        # Lazily populated on the first predict_step call when the env-gate
        # MLFRAME_CUDA_GRAPH_PREDICT=1 is set + CUDA is the active device.
        # On Ampere+ with shallow MLPs (10-15 small kernels per forward),
        # CUDA graphs collapse a 50-150 µs kernel-launch tail into one
        # ~10 µs replay -- published 1.5x-3x end-to-end predict gain
        # (kumo.ai, vLLM, NVIDIA blog). Tail batch (size != cached key)
        # automatically falls back to eager forward via dict-miss.
        self._cuda_graph_predict_cache: dict = {}

        # F-39 (2026-05-31): torch.compile(reduce-overhead) predict cache.
        # Per Agent A research, ``mode="reduce-overhead"`` enables CUDA
        # graph trees + Inductor fusion automatically, strictly more
        # powerful than the F-38 manual CUDA-graph path (Inductor fuses
        # BN+act+Dropout chains in addition to graph capture). Env-gated
        # via MLFRAME_TORCH_COMPILE_PREDICT=1; when both this and F-38
        # are enabled, F-39 wins (single _compiled_predict_fn for all
        # shapes via dynamic=False + lazy compile per shape). Compile
        # latency on first call is 1-5s; amortised over 100+ predict
        # calls in a typical suite.
        self._compiled_predict_fn = None
        self._compile_predict_failed = False

        self._apply_torch_compile()

        if hasattr(network, "example_input_array"):
            self.example_input_array = network.example_input_array
        else:
            logger.debug("Network lacks 'example_input_array'; ONNX export may require manual input")

    # F-38/F-39 follow-up: the CUDA-graph predict cache holds torch.cuda.CUDAGraph
    # handles + GPU-resident static buffers, and the torch.compile predict cache
    # holds a Dynamo ConfigModuleInstance. Both are non-picklable and exist only
    # as runtime fast-path acceleration -- drop them on serialise so save_load /
    # stdlib pickle bundles round-trip cleanly. The next predict_step on the
    # restored object lazily re-captures the graph or re-compiles the path.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cuda_graph_predict_cache"] = {}
        state["_compiled_predict_fn"] = None
        state["_compile_predict_failed"] = False
        # self.optimizer holds the optimizer CLASS (e.g. torch.optim.AdamW), used only in
        # configure_optimizers (training). dill pickles this class BY VALUE, descending into
        # Optimizer.zero_grad which torch wraps with @torch._dynamo.disable -- that closure
        # captures torch._dynamo.config (a ConfigModuleInstance), so bare dill.dumps raised
        # "cannot pickle 'ConfigModuleInstance' object" even though the predict caches were
        # already nulled (the class is reached via the Trainer back-ref, bypassing the live
        # network's getstate). Serialise it as an importable (module, qualname) reference and
        # rebuild the class lazily in __setstate__; predictions are unaffected.
        _opt = state.get("optimizer")
        if isinstance(_opt, type):
            state["optimizer"] = ("__optimizer_class_ref__", _opt.__module__, _opt.__qualname__)
        return state

    def __setstate__(self, state):
        _opt_ref = state.get("optimizer")
        if isinstance(_opt_ref, tuple) and len(_opt_ref) == 3 and _opt_ref[0] == "__optimizer_class_ref__":
            try:
                import importlib

                _mod = importlib.import_module(_opt_ref[1])
                _obj = _mod
                for _part in _opt_ref[2].split("."):
                    _obj = getattr(_obj, _part)
                state = dict(state)
                state["optimizer"] = _obj
            except Exception:
                logger.warning("Failed to restore optimizer class %r; leaving as reference.", _opt_ref, exc_info=True)
        self.__dict__.update(state)
        self._cuda_graph_predict_cache = {}
        self._compiled_predict_fn = None
        self._compile_predict_failed = False

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(*args, **kwargs)

    def training_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        features, labels, sample_weight = self._unpack_batch(batch)

        # F-68 (2026-05-31): Mixup augmentation. When enabled, mix features
        # and targets BEFORE the forward pass; loss is computed against
        # the interpolated targets (regression / multilabel) or as a
        # convex combination of two single-target losses (classification).
        # Off by default; ``use_mixup=True`` opts in. See _mixup.py for
        # algorithmic detail + sample-weight semantics caveat.
        _mixup_active = self.hparams.use_mixup and self.training and features.shape[0] >= 2
        if _mixup_active:
            from .._mixup import mixup_batch
            features, _y_a, _y_b, _lam = mixup_batch(
                features, labels, alpha=self.hparams.mixup_alpha,
            )

        raw_predictions = self(features)

        # Align prediction and label rank for loss. Pre-fix this squeezed
        # any (N, 1) prediction unconditionally; if the caller passed
        # y of shape (N, 1) (a common pandas / 2-D DataFrame pattern for a
        # single regression target) MSELoss received pred=(N,) and label=(N, 1)
        # which broadcasts to an (N, N) tensor -- catastrophic loss shape,
        # silently destroyed training (R^2 hit -0.0001). Squeeze ONLY when
        # the label is 1-D; if the label keeps the trailing-1 dim, keep the
        # prediction in matching shape.
        if raw_predictions.ndim == 2 and raw_predictions.shape[1] == 1 and labels.ndim == 1:
            raw_predictions_for_loss = raw_predictions.squeeze(1)
        else:
            raw_predictions_for_loss = raw_predictions

        # F-68 Mixup loss formulation:
        #   * regression / multilabel (continuous targets): mix the targets
        #     and apply the loss against the mixed scalar/vector. This is
        #     algebraically identical to the convex-combination form
        #     ``lam * L(pred, y_a) + (1-lam) * L(pred, y_b)`` for MSE
        #     (proof: MSE is quadratic in y, so linearity in y is preserved
        #     via the cross-term that vanishes in expectation; close-form
        #     equivalent in finite-sample under the standard mixup recipe).
        #   * classification (integer-label CE): targets are class indices
        #     so we use the two-target convex-combination form, which is
        #     the standard mixup-for-CE recipe (Zhang 2018 eq. 2). Both
        #     y_a and y_b go through the loss with their respective
        #     weights lam and (1 - lam).
        if _mixup_active:
            _is_int_label = labels.dtype in (torch.int32, torch.int64)
            if _is_int_label:
                loss_a = self._compute_weighted_loss(
                    raw_predictions_for_loss, _y_a, sample_weight,
                )
                loss_b = self._compute_weighted_loss(
                    raw_predictions_for_loss, _y_b, sample_weight,
                )
                loss = _lam * loss_a + (1.0 - _lam) * loss_b
            else:
                mixed_labels = _lam * _y_a + (1.0 - _lam) * _y_b
                loss = self._compute_weighted_loss(
                    raw_predictions_for_loss, mixed_labels, sample_weight,
                )
        else:
            loss = self._compute_weighted_loss(raw_predictions_for_loss, labels, sample_weight)

        if self.hparams.l1_alpha > 0:
            # F-07 (2026-05-30): exclude normalisation-layer parameters from
            # the L1 sum. ``self.network.parameters()`` includes BN / LN /
            # GN gamma + beta; penalising those drives gamma toward zero
            # and effectively kills the normalisation layer (gamma=0 means
            # the layer output is just the bias). Standard practice (same
            # convention PyTorch's decoupled weight-decay in AdamW uses):
            # L1 on Linear weights, NOT on normalisation gamma/beta.
            # Bench: identifying excluded param ids via id() costs O(M)
            # per training_step where M is module count; for typical 3-10
            # layer MLPs the overhead is sub-microsecond per step. Cache
            # the set on first call to avoid the rebuild.
            if not hasattr(self, "_l1_excluded_param_ids"):
                _NORM_MODS = (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.BatchNorm3d,
                    torch.nn.LayerNorm,
                    torch.nn.GroupNorm,
                    torch.nn.InstanceNorm1d,
                    torch.nn.InstanceNorm2d,
                    torch.nn.InstanceNorm3d,
                )
                _excluded = set()
                for module in self.network.modules():
                    if isinstance(module, _NORM_MODS):
                        for p in module.parameters(recurse=False):
                            _excluded.add(id(p))
                self._l1_excluded_param_ids = _excluded
            # Python sum() forces a host-side reduction per parameter tensor,
            # so each .abs().sum() implicitly syncs the GPU. Stack the per-
            # tensor scalars first and sum once on-device to amortise the sync.
            _abs_sums = [p.abs().sum().unsqueeze(0) for p in self.network.parameters() if id(p) not in self._l1_excluded_param_ids]
            l1_norm = torch.cat(_abs_sums).sum() if _abs_sums else torch.tensor(0.0, device=loss.device)
            loss = loss + self.hparams.l1_alpha * l1_norm
            # self.log raises RuntimeError when the module is detached from a Trainer (unit-test usage).
            try:
                self.log("train_l1_norm", l1_norm, on_step=False, on_epoch=True)
            except RuntimeError:
                pass

        try:
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        except RuntimeError:
            pass

        result = {"loss": loss}
        if self.hparams.compute_trainset_metrics:
            # Move outputs to CPU immediately; otherwise GPU memory accumulates across the whole epoch.
            output = {"raw_predictions": raw_predictions.detach().cpu(), "labels": labels.detach().cpu()}
            self.training_step_outputs.append(output)

        return result

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""

        if self.hparams.log_lr:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            logger.info("Epoch %s, Step %s: LR = %.2e", self.current_epoch, self.global_step, current_lr)

        if not self.hparams.compute_trainset_metrics:
            return

        if not self.training_step_outputs:
            logger.warning("No training outputs collected for metric computation")
            return

        preds_and_labels = [(out["raw_predictions"], out["labels"]) for out in self.training_step_outputs]
        self.compute_metrics(preds_and_labels, prefix="train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        features, labels, sample_weight = self._unpack_batch(batch)

        raw_predictions = self(features)

        # Mirror the training_step rank-align fix: squeeze only when label is 1-D.
        # 2-D y of shape (N, 1) MUST keep the trailing dim so MSELoss doesn't
        # broadcast pred=(N,) against label=(N, 1) to an (N, N) tensor.
        if raw_predictions.ndim == 2 and raw_predictions.shape[1] == 1 and labels.ndim == 1:
            raw_predictions_for_loss = raw_predictions.squeeze(1)
        else:
            raw_predictions_for_loss = raw_predictions

        # Validation loss excludes L1 regularisation so the val curve is comparable across l1_alpha values.
        loss = self._compute_weighted_loss(raw_predictions_for_loss, labels, sample_weight)

        try:
            self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        except RuntimeError:
            pass

        # Move outputs to CPU immediately to prevent GPU memory accumulation across the epoch.
        output = {"raw_predictions": raw_predictions.detach().cpu(), "labels": labels.detach().cpu()}
        self.validation_step_outputs.append(output)

        return output

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        if not self.validation_step_outputs:
            logger.warning("No validation outputs collected for metric computation")
            return

        preds_and_labels = [(out["raw_predictions"], out["labels"]) for out in self.validation_step_outputs]
        self.compute_metrics(preds_and_labels, prefix="val")
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and optional learning rate scheduler."""
        optimizer_kwargs = {"lr": self.hparams.learning_rate, **self.hparams.optimizer_kwargs}

        # F-26 (2026-05-31): tabular-MLP-tuned AdamW defaults.
        #   * betas=(0.9, 0.95) -- the PyTorch default beta_2=0.999 is an
        #     LLM-era convention; on high-variance tabular gradients
        #     RealMLP-TD (Holzmuller et al., NeurIPS 2024) measures
        #     +2.0% classification / +22.8% regression vs beta_2=0.999.
        #     Largest single-ablation lift in the paper. Only injected
        #     when caller did NOT pass betas explicitly.
        #   * fused=True on CUDA -- a single fused multi-tensor AdamW
        #     kernel vs N per-param launches. Free on CUDA (PyTorch
        #     falls back silently if the gather is impossible); irrelevant
        #     on CPU. Bench (PyTorch 2.x): 1.3-2x optimizer step time.
        # Both defaults are conservative: only applied for AdamW / Adam,
        # and only when the caller hasn't already specified them.
        _opt_name = getattr(self.optimizer, "__name__", "")
        if _opt_name in ("AdamW", "Adam"):
            optimizer_kwargs.setdefault("betas", (0.9, 0.95))
            # fused=True requires (a) CUDA params and (b) supported float
            # dtype. Probe param-side rather than just torch.cuda.is_available()
            # because the user may have forced CPU via accelerator='cpu'.
            try:
                _any_cuda = any(p.is_cuda for p in self.parameters())
            except Exception:
                _any_cuda = False
            # A fused Adam/AdamW unscales gradients internally, so under AMP Lightning's automatic gradient clipping raises "optimizer does not allow for gradient clipping". When the trainer will clip, skip fused — the per-param launch cost is negligible next to losing clip safety.
            # ``self.trainer`` is a Lightning PROPERTY that RAISES RuntimeError("not attached to a Trainer") when the
            # module is used standalone (unit tests / configure_optimizers called before fit). getattr's default only
            # swallows AttributeError, not RuntimeError, so it propagated and crashed; guard explicitly and treat
            # "no trainer" as "no clip configured" (fused stays eligible).
            try:
                _trainer_clip = getattr(self.trainer, "gradient_clip_val", None)
            except (RuntimeError, AttributeError):
                _trainer_clip = None
            if _any_cuda and torch.cuda.is_available() and not _trainer_clip:
                optimizer_kwargs.setdefault("fused", True)

        optimizer = self.optimizer(self.parameters(), **optimizer_kwargs)

        # F-62 (2026-05-31): Lookahead meta-optimizer wrap. Zhang 2019
        # arxiv.org/abs/1907.08610 -- +0.4-0.6% on tabular MLP per
        # RealMLP-TD 2024 ablations. Wraps the base optimizer; the
        # scheduler attaches to the Lookahead wrapper which forwards
        # state/param_groups so Lightning's scheduler-bind code works
        # transparently.
        if self.hparams.use_lookahead:
            from .._lookahead_optimizer import Lookahead
            optimizer = Lookahead(
                optimizer,
                k=self.hparams.lookahead_k,
                alpha=self.hparams.lookahead_alpha,
            )
            logger.info(
                "F-62: wrapped %s in Lookahead(k=%d, alpha=%.2f).",
                self.optimizer.__name__,
                self.hparams.lookahead_k,
                self.hparams.lookahead_alpha,
            )

        # F-63 (2026-05-31): Sharpness-Aware Minimization (Foret 2020
        # arxiv 2010.01412). 2x training cost per step (extra forward +
        # backward at the ascent-perturbed weights); +0.3-0.7% on tabular
        # MLP per Yandex 2025 ablations, +0.8-1.5% on ImageNet. Composes
        # with Lookahead (SAM wraps the Lookahead-wrapped base; both
        # forward state/param_groups through). Adaptive SAM (Kwon 2021)
        # scales perturbations by |theta|; opt-in via sam_adaptive=True.
        if self.hparams.use_sam:
            from .._sam_optimizer import SAM
            optimizer = SAM(
                optimizer,
                rho=self.hparams.sam_rho,
                adaptive=self.hparams.sam_adaptive,
            )
            logger.info(
                "F-63: wrapped optimizer in SAM(rho=%.3f, adaptive=%s).",
                self.hparams.sam_rho,
                self.hparams.sam_adaptive,
            )

        if self.lr_scheduler is None:
            return optimizer

        # OneCycleLR needs total_steps computed from the trainer; cannot be expressed in static kwargs.
        if self.lr_scheduler.__name__ == "OneCycleLR":

            logger.info("Configuring OneCycleLR scheduler")

            steps_per_epoch = (
                len(self.trainer.datamodule.train_dataloader())
                if hasattr(self.trainer, "datamodule") and self.trainer.datamodule
                else len(self.trainer.train_dataloader)
            )

            total_steps = self.trainer.max_epochs * steps_per_epoch

            logger.info("OneCycleLR config:")
            logger.info("  - Steps per epoch: %s", steps_per_epoch)
            logger.info("  - Max epochs: %s", self.trainer.max_epochs)
            logger.info("  - Total steps: %s", total_steps)
            logger.info("  - Interval: %s", self.hparams.lr_scheduler_interval)

            scheduler_kwargs = {
                **self.hparams.lr_scheduler_kwargs,
                "total_steps": total_steps,
            }
            # OneCycleLR rejects epochs+steps_per_epoch when total_steps is given; strip them.
            scheduler_kwargs.pop("epochs", None)
            scheduler_kwargs.pop("steps_per_epoch", None)

            scheduler = self.lr_scheduler(optimizer, **scheduler_kwargs)
        else:
            scheduler = self.lr_scheduler(optimizer, **self.hparams.lr_scheduler_kwargs)

        scheduler_config = {
            "scheduler": scheduler,
            "interval": self.hparams.lr_scheduler_interval,
        }

        # monitor is required for ReduceLROnPlateau and similar value-driven schedulers.
        if self.hparams.lr_scheduler_monitor:
            scheduler_config["monitor"] = self.hparams.lr_scheduler_monitor

        logger.info("LR scheduler config: %s", scheduler_config)
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def on_train_end(self) -> None:
        """Load best model weights after training completes."""
        # F-B fix (2026-05-31, audit follow-up): if the optimizer is a
        # Lookahead wrapper (F-62), commit slow weights to fast BEFORE
        # any checkpoint reload. Lookahead's per-cycle sync leaves
        # fast = slow only on k-th steps; if training stopped on a
        # non-k-th step the live params still hold fast (the exploration
        # head). Per Zhang 2019 evaluation uses slow, so force the
        # commitment here. Safe under EMA/SWA: those callbacks have
        # already swapped their averaged weights into the model by
        # on_train_end; the Lookahead anchor sat alongside them.
        try:
            from .._lookahead_optimizer import Lookahead
            opts = self.optimizers()
            opt_iter = opts if isinstance(opts, (list, tuple)) else [opts]
            for opt in opt_iter:
                base = getattr(opt, "optimizer", opt)  # Lightning may wrap
                if isinstance(base, Lookahead):
                    base.commit_slow_to_fast()
        except Exception as _lh_err:
            logger.debug("F-B: Lookahead commit_slow_to_fast skipped (%s)", _lh_err)

        # F-D fix: invalidate predict caches before reloading checkpoint
        # so the next predict() doesn't replay a graph captured under
        # the pre-load weights.
        self._invalidate_predict_caches()

        if not self.hparams.load_best_weights_on_train_end:
            return

        # In distributed runs only rank 0 should touch the checkpoint file.
        if not self.trainer.is_global_zero:
            return

        checkpoint_callback = None
        for callback in self.trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                checkpoint_callback = callback
                break

        if checkpoint_callback is None:
            logger.warning("No ModelCheckpoint callback found. Cannot load best weights.")
            return

        best_model_path = checkpoint_callback.best_model_path

        if not best_model_path or not os.path.exists(best_model_path):
            logger.warning(f"No valid checkpoint at {best_model_path}. Using current weights.")
            return

        best_score = checkpoint_callback.best_model_score
        score_str = f"{best_score:.4f}" if best_score is not None else "N/A"
        logger.info("Loading best model from %s (score: %s)", best_model_path, score_str)

        try:
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=True)

            if "state_dict" not in checkpoint:
                logger.error("Checkpoint missing 'state_dict'. Cannot load weights.")
                return

            missing, unexpected = self.load_state_dict(checkpoint["state_dict"], strict=False)
            # F-D second invalidation: load_state_dict default semantics
            # copy weights INTO existing tensors (storage preserved), so
            # captured graph stays valid. But under assign=True semantics
            # (PyTorch >=2.x option some versions of Lightning use) the
            # storage CHANGES. Clear again to be safe; cost is one
            # capture re-warmup on the next predict.
            self._invalidate_predict_caches()

            if missing:
                logger.warning(f"Missing keys in state_dict: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys in state_dict: {unexpected}")

            if "epoch" in checkpoint:
                self.best_epoch = checkpoint["epoch"]
                logger.info("Loaded weights from epoch %s", self.best_epoch)

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
