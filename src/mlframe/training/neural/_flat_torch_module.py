"""MLPTorchModel carved out of ``mlframe.training.neural.flat``.

Re-imported at the parent module bottom so historical
``from mlframe.training.neural.flat import MLPTorchModel`` callers keep working.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint

from .base import MetricSpec, to_numpy_safe

logger = logging.getLogger("mlframe.training.neural.flat")


class MLPTorchModel(L.LightningModule):
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
    def __getstate__(self):  # type: ignore[no-untyped-def]
        state = self.__dict__.copy()
        state["_cuda_graph_predict_cache"] = {}
        state["_compiled_predict_fn"] = None
        state["_compile_predict_failed"] = False
        return state

    def __setstate__(self, state):  # type: ignore[no-untyped-def]
        self.__dict__.update(state)
        self._cuda_graph_predict_cache = {}
        self._compiled_predict_fn = None
        self._compile_predict_failed = False

    def _apply_torch_compile(self) -> None:
        """Apply torch.compile to the network if enabled.

        Compiled models cannot be pickled in PyTorch 2.8 ("cannot pickle 'ConfigModuleInstance' object"),
        which breaks checkpoint saving. See https://github.com/pytorch/pytorch/issues/126154.

        F-35 (2026-05-31) safety guards added per 2026-05-31 PyTorch
        optimization audit (Agent A torch.compile research):
          * LSTM/GRU/RNN networks: TorchDynamo INTENTIONALLY graph-breaks
            on these (pytorch/pytorch#167275, #140845). Compiled is SLOWER
            than eager due to host-device syncs around the cuDNN call.
            Detect + skip + WARN; users who set compile_network globally
            should NOT silently regress recurrent fits.
          * MLFRAME_TORCH_COMPILE_DEBUG=1 env var: routes graph_break +
            recompile events through torch._logging.set_logs so the next
            perf investigation has visibility. Off by default (logs would
            spam STDERR otherwise).
        """
        if not self.hparams.compile_network:
            return

        if torch.__version__ < "2.0":
            logger.warning("torch.compile requires PyTorch >= 2.0. Skipping compilation.")
            return

        # F-35 safety guard: LSTM/GRU/RNN are explicitly anti-compile.
        _recurrent_types = (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)
        try:
            _has_recurrent = any(
                isinstance(m, _recurrent_types) for m in self.network.modules()
            )
        except Exception:
            _has_recurrent = False
        if _has_recurrent:
            logger.warning(
                "torch.compile requested but network contains LSTM/GRU/RNN "
                "modules. TorchDynamo intentionally graph-breaks on these "
                "(pytorch/pytorch#167275, #140845); compiled is SLOWER than "
                "eager. Skipping compile for this network."
            )
            return

        # Opt-in debug logs for graph breaks + recompiles.
        import os as _os_dbg
        if _os_dbg.environ.get("MLFRAME_TORCH_COMPILE_DEBUG", "0") == "1":
            try:
                import torch._logging as _tlog
                _tlog.set_logs(graph_breaks=True, recompiles=True)
                logger.info(
                    "MLFRAME_TORCH_COMPILE_DEBUG=1: enabled torch._logging "
                    "graph_breaks + recompiles. Re-run with --no-cov + capture "
                    "STDERR to inspect."
                )
            except Exception as _dbg_err:
                logger.debug("torch._logging.set_logs failed: %s", _dbg_err)

        try:
            self.network = torch.compile(self.network, mode=self.hparams.compile_network)
            logger.info("Applied torch.compile with mode='%s'", self.hparams.compile_network)
        except Exception:
            logger.warning("Failed to apply torch.compile. Using uncompiled network.", exc_info=True)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(*args, **kwargs)

    def _unpack_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Unpack batch into features, labels, and optional sample_weight."""
        if isinstance(batch, dict):
            return batch["features"], batch["labels"], batch.get("sample_weight")
        elif isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                return batch[0], batch[1], batch[2]
            elif len(batch) == 2:
                return batch[0], batch[1], None
        raise TypeError(f"Unexpected batch format: {type(batch).__name__}")

    def _loss_unreduced(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Per-sample loss via self.loss_fn with reduction='none' when available.

        Most torch.nn loss modules expose a ``reduction`` attribute; toggling it
        round-trip preserves the loss type (BCE stays BCE, MSE stays MSE) so
        sample weighting doesn't silently swap loss semantics.
        """
        # Module loss (CrossEntropyLoss / BCEWithLogitsLoss / MSELoss / ...):
        # set reduction='none', call, restore.
        if hasattr(self.loss_fn, "reduction"):
            prev = self.loss_fn.reduction
            try:
                self.loss_fn.reduction = "none"
                return self.loss_fn(predictions, labels)
            finally:
                self.loss_fn.reduction = prev
        # Functional / lambda loss: try kwarg, fall back to CE/MSE shape-guess.
        try:
            return self.loss_fn(predictions, labels, reduction="none")
        except TypeError:
            if predictions.dim() == 2 and predictions.shape[1] > 1:
                return F.cross_entropy(predictions, labels, reduction="none")
            return F.mse_loss(predictions, labels, reduction="none")

    def _compute_weighted_loss(self, predictions: torch.Tensor, labels: torch.Tensor, sample_weight: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute loss with optional sample weighting.

        Args:
            predictions: Model predictions
            labels: Ground truth labels
            sample_weight: Optional per-sample weights

        Returns:
            Scalar loss tensor
        """
        # Defensive shape alignment: torch's MSELoss / L1Loss / SmoothL1Loss
        # emit a UserWarning ("Using a target size (torch.Size([N])) that is
        # different to the input size (torch.Size([N, 1]))") AND apply
        # incorrect broadcasting when predictions and labels disagree on a
        # singleton trailing dim. training_step + validation_step squeeze
        # raw_predictions in the common (N, 1) case, but other call paths
        # (test fixtures, ranker, custom loss_fn) can still enter here
        # mismatched. Squeeze the singleton dim on either side so the loss
        # function never sees a broadcasting collision.
        if predictions.dim() == 2 and predictions.shape[-1] == 1 and labels.dim() == 1:
            predictions = predictions.squeeze(-1)
        elif labels.dim() == 2 and labels.shape[-1] == 1 and predictions.dim() == 1:
            labels = labels.squeeze(-1)

        if sample_weight is None:
            return self.loss_fn(predictions, labels)

        # Honour self.loss_fn if it supports reduction='none' (every
        # torch.nn.*Loss does); the previous hard-coded CE/MSE branch silently
        # switched a BCE binary classifier (with sample_weight) to MSE,
        # producing a degenerate gradient.
        loss_unreduced = self._loss_unreduced(predictions, labels)

        # torch.dot fuses mul+sum into one kernel; ~1.7-2.2x faster than
        # (a*b).sum() across N=256-16384 on CPU. 1-D fast path covers the
        # common case (per-sample weights, scalar-per-sample loss); fall
        # back to broadcast-mul for any unexpected shape.
        weight_sum = sample_weight.sum()
        if loss_unreduced.dim() == 1 and sample_weight.dim() == 1 and loss_unreduced.shape == sample_weight.shape:
            raw = torch.dot(loss_unreduced, sample_weight)
        else:
            # Multilabel BCE / multiclass per-sample losses produce a 2-D
            # (B, K) loss tensor while sample_weight is 1-D (B,); torch
            # broadcasts (B,) as (1, B) which collides with the (B, K)
            # right-aligned shape (B vs K mismatch). Reshape 1-D weights to
            # (B, 1) so they broadcast across labels / classes uniformly.
            # Fuzz c0052 (multilabel + MLP + uniform weights) crashed here
            # before the unsqueeze: "tensor a (3) must match tensor b (30928)".
            if loss_unreduced.dim() == 2 and sample_weight.dim() == 1 and sample_weight.shape[0] == loss_unreduced.shape[0]:
                sample_weight_b = sample_weight.unsqueeze(1)
            else:
                sample_weight_b = sample_weight
            raw = (loss_unreduced * sample_weight_b).sum()
        # Safe divide avoids the prior ``if weight_sum > 0`` Python branch (a
        # forced GPU->CPU sync ~30 us per call) without losing the all-zero-
        # weight semantic: when weight_sum is 0, raw is also 0 (sum(loss * 0)),
        # so 0 / clamp(0, 1e-12) = 0 -- the same zero loss the legacy branch
        # returned. Bench at n=50k (c0117 batch shape, CUDA): 299 us -> 267 us
        # per call (~10%), bit-identical for both non-degenerate and all-zero-
        # weight cases.
        weighted_loss = raw / torch.clamp(weight_sum, min=1e-12)
        # F-10 (2026-05-30): emit a ONCE-per-LightningModule WARN on the
        # first all-zero-weight batch. Pre-fix the safe-divide path
        # silently returned 0 loss + 0 gradient with no log message --
        # the model trained on nothing and the operator saw a flat val
        # curve with no clue why. The check costs one GPU->CPU sync
        # (weight_sum.item()) per training_step UNTIL the warning fires,
        # after which the flag short-circuits everything. The sync is
        # only paid when sample_weight is not None (opt-in path); fits
        # without sample_weight pay nothing. The 2026-05-20 rationale
        # to drop the per-batch sync stands for fits where the operator
        # KNOWS sample_weight is well-behaved -- a once-per-fit sync
        # is the price of catching the silent-debug trap.
        if not getattr(self, "_warned_zero_weight_batch", False):
            if float(weight_sum.detach().item()) < 1e-12:
                logger.warning(
                    "All-zero sample_weight batch in MLP training_step "
                    "(batch_idx=%d): loss=0, gradient=0, model receives "
                    "no learning signal from this step. Suppressing "
                    "subsequent warnings for this LightningModule. If "
                    "the entire fit has zero-weight batches the model "
                    "will not learn -- check the sample_weight pipeline.",
                    getattr(self, "_current_batch_idx", -1),
                )
                self._warned_zero_weight_batch = True
        return weighted_loss

    def training_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        features, labels, sample_weight = self._unpack_batch(batch)

        # F-68 (2026-05-31): Mixup augmentation. When enabled, mix features
        # and targets BEFORE the forward pass; loss is computed against
        # the interpolated targets (regression / multilabel) or as a
        # convex combination of two single-target losses (classification).
        # Off by default; ``use_mixup=True`` opts in. See _mixup.py for
        # algorithmic detail + sample-weight semantics caveat.
        _mixup_active = (
            self.hparams.use_mixup
            and self.training
            and features.shape[0] >= 2
        )
        if _mixup_active:
            from ._mixup import mixup_batch
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
        if (
            raw_predictions.ndim == 2
            and raw_predictions.shape[1] == 1
            and labels.ndim == 1
        ):
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
            _abs_sums = [
                p.abs().sum().unsqueeze(0)
                for p in self.network.parameters()
                if id(p) not in self._l1_excluded_param_ids
            ]
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
        if (
            raw_predictions.ndim == 2
            and raw_predictions.shape[1] == 1
            and labels.ndim == 1
        ):
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

    def compute_metrics(self, predictions_and_labels: List[Tuple[torch.Tensor, torch.Tensor]], prefix: str = "val") -> None:
        """
        Compute and log all metrics given raw predictions and labels.
        Optimized: compute argmax, softmax, CPU/numpy only if needed, once each.

        Args:
            predictions_and_labels: List of (raw_predictions, labels) tuples from each batch (on CPU)
            prefix: Logging prefix ('train' or 'val')
        """
        raw_predictions, labels = zip(*predictions_and_labels)
        raw_predictions = torch.cat(raw_predictions)
        labels = torch.cat(labels)

        need_argmax = any(m.requires_argmax for m in self.metrics)
        need_softmax = any(m.requires_probs for m in self.metrics)
        need_cpu = any(m.requires_cpu for m in self.metrics)

        # argmax along the class dim only makes sense for multi-class K>1 logits (shape (N, K)).
        # Regression (dim==1) or multilabel BCE (each label independent) would silently emit
        # garbage from raw_predictions.argmax(dim=1); guard explicitly.
        _is_multiclass = (raw_predictions.dim() == 2 and raw_predictions.shape[1] > 1 and self.task_type != "multilabel")

        preds_dict = {}
        if need_argmax:
            if _is_multiclass:
                preds_dict["argmax"] = raw_predictions.argmax(dim=1)
            elif self.task_type == "multilabel":
                # Multilabel: per-label thresholded predictions at 0.5 (each output independent binary).
                preds_dict["argmax"] = (torch.sigmoid(raw_predictions) >= 0.5).long()
            elif self.task_type == "binary":
                # F-05 binary single-output: threshold sigmoid at 0.5 -> {0, 1}.
                preds_dict["argmax"] = (torch.sigmoid(raw_predictions).squeeze(-1) >= 0.5).long()
            else:
                # Regression: argmax has no meaning; skip metric below.
                preds_dict["argmax"] = None
        if need_softmax:
            if _is_multiclass:
                preds_dict["softmax"] = F.softmax(raw_predictions, dim=1)
            elif self.task_type == "multilabel":
                preds_dict["softmax"] = torch.sigmoid(raw_predictions)
            elif self.task_type == "binary":
                # F-05 binary single-output: convert (N, 1) sigmoid -> (N, 2)
                # column-stacked [P(y=0), P(y=1)] so the same probabilistic
                # metric (ICE, etc.) the multiclass path uses keeps working.
                p1 = torch.sigmoid(raw_predictions).reshape(-1)
                p0 = 1.0 - p1
                preds_dict["softmax"] = torch.stack([p0, p1], dim=1)
            else:
                preds_dict["softmax"] = raw_predictions

        labels_cpu = None
        if need_cpu:
            # cpu=False: tensors are already on CPU thanks to the per-step .cpu() in *_step.
            labels_cpu = to_numpy_safe(labels, cpu=False)

        # CPU-numpy memoisation keyed by tagged source (argmax / softmax / raw) NOT id(preds).
        # id() values can be reused after garbage collection across loop iterations and silently
        # return the wrong tensor's cached numpy view.
        cpu_cache: dict[str, np.ndarray] = {}

        for metric in self.metrics:
            if metric.requires_argmax:
                preds = preds_dict["argmax"]
                preds_tag = "argmax"
            elif metric.requires_probs:
                preds = preds_dict["softmax"]
                preds_tag = "softmax"
            else:
                preds = raw_predictions
                preds_tag = "raw"

            if preds is None:
                # argmax requested but logits aren't multi-class; skip silently to avoid garbage.
                continue

            if metric.requires_cpu:
                if preds_tag not in cpu_cache:
                    cpu_cache[preds_tag] = to_numpy_safe(preds, cpu=False)
                preds_np = cpu_cache[preds_tag]
                labels_np = labels_cpu
            else:
                preds_np = preds
                labels_np = labels

            try:
                value = metric.fcn(y_true=labels_np, y_score=preds_np)
                self.log(
                    f"{prefix}_{metric.name}",
                    value,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
            except Exception:
                logger.exception("Failed to compute metric %s_%s", prefix, metric.name)

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
            if _any_cuda and torch.cuda.is_available():
                optimizer_kwargs.setdefault("fused", True)

        optimizer = self.optimizer(self.parameters(), **optimizer_kwargs)

        # F-62 (2026-05-31): Lookahead meta-optimizer wrap. Zhang 2019
        # arxiv.org/abs/1907.08610 -- +0.4-0.6% on tabular MLP per
        # RealMLP-TD 2024 ablations. Wraps the base optimizer; the
        # scheduler attaches to the Lookahead wrapper which forwards
        # state/param_groups so Lightning's scheduler-bind code works
        # transparently.
        if self.hparams.use_lookahead:
            from ._lookahead_optimizer import Lookahead
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

    def _invalidate_predict_caches(self) -> None:
        """F-D fix (2026-05-31, audit follow-up): clear the F-40 CUDA-graph
        cache + F-39 torch.compile cache.

        The cache holds captured kernels that reference the parameter
        tensors' storage addresses at capture time. Most weight-update
        paths (Lookahead .copy_, EMA WeightAveraging.update_parameters,
        SWA .copy_, load_state_dict default) PRESERVE storage so the
        captured graph keeps producing correct values. But ANY path that
        REPLACES a nn.Parameter object (load_state_dict(assign=True),
        LoRA adapter swap, dynamic head replacement, user-explicit
        param mutation) leaves the captured graph pointing at stale
        storage -- replay produces predictions from PRE-swap weights
        with no exception. Same silent-correctness class as F-58.

        Call this AFTER any code path that could replace params
        (on_train_end checkpoint reload, user calls to set weights, etc.)
        Idempotent.
        """
        if hasattr(self, "_cuda_graph_predict_cache"):
            self._cuda_graph_predict_cache.clear()
        if hasattr(self, "_compiled_predict_fn"):
            self._compiled_predict_fn = None
            self._compile_predict_failed = False

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
            from ._lookahead_optimizer import Lookahead
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

    def _maybe_compile_predict_forward(self, x: torch.Tensor) -> torch.Tensor:
        """F-39 (2026-05-31): torch.compile(mode="reduce-overhead")
        fast path for the inference forward. Strictly more powerful than
        F-38's manual CUDA-graph path -- Inductor fuses elementwise
        chains (BN+act+Dropout) in addition to capturing kernels into
        CUDA graphs. Cost: 1-5s compile latency on first call.

        Gating:
          1. ``MLFRAME_TORCH_COMPILE_PREDICT=1`` env var (default off
             until validated on the target host's PyTorch + GPU combo)
          2. CUDA is available + the input tensor lives on CUDA
          3. No LSTM/GRU/RNN in the network (same anti-pattern as F-35
             compile guard + F-38 CUDA-graph gate)
          4. Prior compile attempt has not been cached as failed

        Returns None to signal "not applicable; caller should fall
        through to next path" (vs returning the eager output directly
        which would obscure the cache miss).
        """
        import os as _os
        if _os.environ.get("MLFRAME_TORCH_COMPILE_PREDICT", "0") != "1":
            return None
        if self._compile_predict_failed:
            return None
        if not torch.cuda.is_available():
            return None
        if not isinstance(x, torch.Tensor) or not x.is_cuda:
            return None
        _net = getattr(self.network, "_orig_mod", self.network)
        try:
            _recurrent = (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)
            if any(isinstance(m, _recurrent) for m in _net.modules()):
                return None
        except Exception:
            return None

        if self._compiled_predict_fn is None:
            try:
                # mode="reduce-overhead" enables CUDA graph trees +
                # Inductor fusion. dynamic=False is correct for the
                # cached-shape regime; tail batches that differ in
                # batch size trigger a recompile inside the cache (one-
                # off ~1-2s) which is amortised across the rest of the
                # predict pass.
                self._compiled_predict_fn = torch.compile(
                    self.network, mode="reduce-overhead", dynamic=False,
                )
                logger.info(
                    "F-39: torch.compile(mode='reduce-overhead') applied "
                    "to predict forward; first call will pay 1-5s compile "
                    "latency, subsequent calls run on CUDA graph trees + "
                    "Inductor-fused kernels."
                )
            except Exception as _comp_err:
                self._compile_predict_failed = True
                logger.warning(
                    "F-39: torch.compile(reduce-overhead) setup failed "
                    "(%s); falling back to eager predict + F-38 CUDA-graph "
                    "path (if env-gated).",
                    _comp_err,
                )
                return None
        try:
            return self._compiled_predict_fn(x)
        except Exception as _exec_err:
            self._compile_predict_failed = True
            self._compiled_predict_fn = None
            logger.warning(
                "F-39: compiled predict forward failed at execution "
                "(%s); permanently falling back to eager + F-38 path.",
                _exec_err,
            )
            return None

    def _maybe_cuda_graph_forward(self, x: torch.Tensor) -> torch.Tensor:
        """F-38 + F-40 (2026-05-31): CUDA-graph fast path for the
        inference forward via the LOW-LEVEL torch.cuda.CUDAGraph() API.

        Non-destructive: each captured graph is an independent CUDAGraph
        object with its own static input + output buffers. self.network
        and self.network.forward are NEVER mutated, so other shapes can
        still run through eager forward without crashing.

        Per-shape capture flow (cache miss path):
          1. Warmup: 3 eager forwards on a side stream (NVIDIA best
             practice -- lets the caching allocator settle, primes
             cuDNN's algo selection, validates the network is capturable
             for this shape).
          2. Static buffer allocation: ``static_in = empty_like(x)``,
             initial copy ``static_in.copy_(x)``.
          3. Capture: ``with torch.cuda.graph(g): static_out = self.network(static_in)``.
             Records every kernel into ``g``. static_out is the output
             tensor bound to the captured graph's output slot.
          4. Cache: store (g, static_in, static_out) keyed by
             (shape, dtype, device).
        Replay (cache hit):
          1. ``static_in.copy_(x, non_blocking=True)``.
          2. ``g.replay()`` -- single CPU-side launch + GPU runs the
             captured kernel sequence against the new input.
          3. Clone ``static_out`` (next replay overwrites it).

        Gating chain (any False -> eager fallback, zero overhead):
          1. ``MLFRAME_CUDA_GRAPH_PREDICT`` env var:
               "1" / "true" / "on" / "yes" -> ON (opt-in)
               unset / "0" / "false" / "off" -> OFF (default-off after the
               2026-06-01 cross-call determinism regression; F-40's
               default-on returned stale replay output when Lightning's
               predict loop reused the GPU allocator between successive
               ``_predict_raw`` calls)
          2. CUDA is available + the input tensor lives on CUDA
          3. No LSTM/GRU/RNN in the network (cuDNN control flow breaks
             capture; same anti-pattern as F-35 torch.compile guard)
          4. Successful capture (any failure caches the False sentinel
             so we don't retry forever)

        Tail-batch behaviour: the LAST batch of a predict pass is
        usually smaller than the trained batch size (drop_last=False on
        the predict dataloader). Its shape misses the cache and triggers
        a fresh capture (~3-5 ms one-off). The capture is NON-destructive
        (F-40 fix) so the previously-captured graph for the full-size
        batch still works on the NEXT predict pass.

        2026-05-31 F-38 attempt with make_graphed_callables was REVERTED
        because that API MUTATES self.network.forward (replaces it with
        the graphed_fn specialised to the first capture's shape). F-40
        uses the low-level CUDAGraph() API instead -- network stays
        untouched.
        """
        import os as _os
        # F-60 follow-up (2026-06-01): default-on caused predict() and
        # predict_proba() to return divergent values across successive
        # _predict_raw calls (observed max diff ~1.0 in [0, 1] sigmoid
        # output, 43% of test rows). Root cause: the captured graph's
        # static buffers can read stale GPU memory after Lightning's
        # predict loop releases its intermediates between calls -- the
        # replay returns the cached output rather than the recomputation
        # over the new input copied into _static_in. Until the
        # cross-call invalidation is solved, default OFF; users with a
        # validated host can opt back in via MLFRAME_CUDA_GRAPH_PREDICT=1.
        _env = _os.environ.get("MLFRAME_CUDA_GRAPH_PREDICT", "0").lower()
        if _env in ("0", "false", "off", "no", ""):
            return self(x)
        if not torch.cuda.is_available():
            return self(x)
        if not isinstance(x, torch.Tensor) or not x.is_cuda:
            return self(x)
        # Skip if the underlying network contains recurrent modules
        # (same anti-pattern as F-35 torch.compile guard).
        _net = getattr(self.network, "_orig_mod", self.network)
        try:
            _recurrent = (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)
            if any(isinstance(m, _recurrent) for m in _net.modules()):
                return self(x)
        except Exception:
            return self(x)

        _key = (tuple(x.shape), x.dtype, x.device)
        _cached = self._cuda_graph_predict_cache.get(_key)
        if _cached is False:
            # Previous capture attempt failed for this shape; permanent
            # eager fallback to avoid retry storms.
            return self(x)
        if _cached is not None:
            # Defensive: replay can fail if model state changed
            # (parameters swapped by an EMA callback between predict
            # calls, weight load via load_state_dict, etc.). Wrap in
            # try/except + evict + fall back to eager.
            try:
                _g, _static_in, _static_out = _cached
                _static_in.copy_(x, non_blocking=True)
                _g.replay()
                # Block until the captured kernel sequence finishes
                # writing _static_out before .clone() reads it. Without
                # this sync the replay was racing the host read on
                # subsequent _predict_raw calls, returning stale values.
                torch.cuda.synchronize()
                return _static_out.clone()
            except Exception as _replay_err:
                logger.warning(
                    "F-40: CUDA-graph replay failed for shape=%s (%s); "
                    "evicting cache entry + falling back to eager.",
                    tuple(x.shape), _replay_err,
                )
                self._cuda_graph_predict_cache.pop(_key, None)
                return self(x)

        # First time seeing this shape on this device + dtype. Try a
        # capture via the LOW-LEVEL CUDAGraph() API (non-destructive).
        try:
            # 3 warmup forwards on a side stream. wait_stream chains so
            # the side stream sees the caller's prior work; current
            # stream then waits on the side stream so subsequent ops
            # see the warmup's effects.
            _side_stream = torch.cuda.Stream()
            _side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(_side_stream):
                for _ in range(3):
                    _ = self.network(x.clone())
            torch.cuda.current_stream().wait_stream(_side_stream)
            torch.cuda.synchronize()

            # Static input + output buffers. Capture binds them into
            # the graph's input/output slots; replay just copies new x
            # into static_in and the captured kernel sequence writes
            # static_out.
            _static_in = torch.empty_like(x)
            _static_in.copy_(x)
            _g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(_g):
                _static_out = self.network(_static_in)

            self._cuda_graph_predict_cache[_key] = (_g, _static_in, _static_out)
            logger.info(
                "F-40: CUDA-graph captured for predict shape=%s dtype=%s.",
                tuple(x.shape), x.dtype,
            )
            # F-58 (2026-05-31): CRITICAL FIX.
            # After capture, ``_static_out`` is bound to the graph's output
            # slot but the kernels have only been RECORDED -- the buffer is
            # uninitialised memory (observed: zeros). Returning it directly
            # on the first call yields garbage predictions for the FIRST
            # batch, dropping aggregate R^2 from 0.998 to 0.659 on a 360-
            # row test split (n=64 per batch, 6 batches). All replays after
            # the first are correct, which is why this bug masquerades as
            # "first-batch random failure" and only surfaces in metrics.
            # Bench: vanilla PyTorch identical network/optim/data converges
            # to R^2=0.998, mlframe wrapper with this bug to 0.659; the
            # gap was exclusively the first-batch zeros.
            # Fix: do one replay AFTER capture so _static_out has the
            # actual computed values for this batch, AND the cache is
            # primed for subsequent same-shape calls. ~3 us extra on first
            # batch only (the rest of the predict pass already pays the
            # replay cost via the normal cache-hit path).
            _g.replay()
            return _static_out.clone()
        except Exception as _graph_err:
            self._cuda_graph_predict_cache[_key] = False
            logger.warning(
                "F-40: CUDA-graph capture failed for predict shape=%s "
                "(%s); falling back to eager forward for this shape.",
                tuple(x.shape), _graph_err,
            )
            return self(x)

    def predict_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Handle prediction for both (x, y) and x-only batches.

        Returns raw model output (logits for classification, values for regression).
        Softmax/argmax conversion is handled in the estimator's predict methods.
        """
        if self.training:
            logger.warning(f"Model was in training mode during predict_step at batch {batch_idx}. Switching to eval mode.")
            self.eval()

        # Accept both training/testing format (x, y) and prediction format (x only).
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        # F-35 (2026-05-31): torch.inference_mode() replaces torch.no_grad()
        # to be torch.compile-friendly. torch.no_grad() still graph-breaks
        # under TorchDynamo in some PyTorch 2.x versions (the "data-
        # dependent context manager" pattern); torch.inference_mode() is
        # the modern equivalent with cleaner graph capture semantics. The
        # observable behaviour is identical (no grad tracking) for
        # standard tensor ops; inference_mode additionally blocks
        # version-counter mutation which catches a class of user bugs
        # cheaply. Per torch.inference_mode docs.
        #
        # F-38 + F-39 (2026-05-31): two-tier accelerated predict.
        # Order:
        #   1. F-39 torch.compile(mode='reduce-overhead') [most powerful:
        #      CUDA graphs + Inductor fusion] — gated by
        #      MLFRAME_TORCH_COMPILE_PREDICT=1
        #   2. F-38 manual CUDA-graph capture via make_graphed_callables
        #      [graphs only, no fusion] — gated by
        #      MLFRAME_CUDA_GRAPH_PREDICT=1
        #   3. Eager fallback (always available)
        # Each gate returns None / falls through when not applicable,
        # so the default behaviour with no env vars set is byte-identical
        # to pre-fix eager forward.
        with torch.inference_mode():
            logits = self._maybe_compile_predict_forward(x)
            if logits is None:
                logits = self._maybe_cuda_graph_forward(x)

        # task_type='regression' (F-24) returns raw values regardless of
        # shape -- including (N, K) multi-target regression where the prior
        # ``logits.shape[1] > 1`` softmax branch would have silently
        # mangled the outputs. Check this FIRST so it short-circuits
        # before any classification-flavoured transform.
        if self.task_type == "regression":
            return logits
        # task_type='multilabel' returns per-label sigmoid (each output independent binary in [0, 1]);
        # task_type='binary' (F-05) returns sigmoid of 1-output logit -> P(y=1) shape (N, 1);
        # default multi-class K>1 path returns softmax rows that sum to 1.
        if logits.dim() == 2 and logits.shape[1] > 1:
            if self.task_type == "multilabel":
                return torch.sigmoid(logits)
            return torch.softmax(logits, dim=1)
        if self.task_type == "binary":
            # Binary 1-output sigmoid head: return P(y=1) in shape (N, 1).
            # The classifier wrapper stacks [1-p, p] for the (N, 2) contract.
            return torch.sigmoid(logits)
        else:
            # Regression (no task_type tag — legacy path): return raw values.
            return logits
