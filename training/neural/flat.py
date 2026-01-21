"""
MLP (Multi-Layer Perceptron) models for tabular/flat data.

This module provides:
- MLPTorchModel: PyTorch Lightning module for MLP training
- generate_mlp: Function to generate MLP architectures
- MLPNeuronsByLayerArchitecture: Enum for architecture patterns
"""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from enum import Enum, auto
from functools import partial
import numpy as np


# Local imports
from .base import MetricSpec, to_numpy_safe


# ----------------------------------------------------------------------------------------------------------------------------
# ENUMS
# ----------------------------------------------------------------------------------------------------------------------------


class MLPNeuronsByLayerArchitecture(Enum):
    Constant = auto()
    Declining = auto()
    Expanding = auto()
    ExpandingThenDeclining = auto()
    Autoencoder = auto()


# ----------------------------------------------------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------------------------------------------------


def get_valid_num_groups(num_channels, preferred_num_groups):
    for g in range(preferred_num_groups, 0, -1):
        if num_channels % g == 0:
            return g
    return 1  # Fallback to 1 (LayerNorm-like) if no divisor found


# ----------------------------------------------------------------------------------------------------------------------------
# Network generation
# ----------------------------------------------------------------------------------------------------------------------------


def generate_mlp(
    num_features: int,
    num_classes: int,
    nlayers: int = 1,
    first_layer_num_neurons: int = None,
    min_layer_neurons: int = 1,
    neurons_by_layer_arch: MLPNeuronsByLayerArchitecture = MLPNeuronsByLayerArchitecture.Constant,
    consec_layers_neurons_ratio: float = 1.1,
    activation_function: Callable = torch.nn.ReLU,
    weights_init_fcn: Callable = None,
    dropout_prob: float = 0.15,
    inputs_dropout_prob: float = 0.01,
    use_layernorm: bool = True,
    use_batchnorm: bool = False,
    use_layernorm_per_layer: bool = False,
    groupnorm_num_groups: int = 0,
    layer_norm_kwargs: dict = None,
    batch_norm_kwargs: dict = None,
    group_norm_kwargs: dict = None,
    verbose: int = 1,
):
    """Generates multilayer perceptron with specific architecture.
    If first_layer_num_neurons is not specified, uses num_features.
    Suitable in NAS and HPT/HPO procedures for generating ANN candidates.

    Args:
        num_features: Number of input features
        num_classes: Number of output classes (None/0 = feature extractor, 1 = regression, >1 = classification)
        nlayers: Number of hidden layers
        first_layer_num_neurons: Neurons in first layer (defaults to num_features)
        min_layer_neurons: Minimum neurons per layer
        neurons_by_layer_arch: Architecture pattern for neuron counts
        consec_layers_neurons_ratio: Ratio between consecutive layers
        activation_function: Activation function class (will be instantiated)
        weights_init_fcn: Weight initialization function
        dropout_prob: Dropout probability after each layer
        inputs_dropout_prob: Dropout probability for input features
        use_layernorm: Apply LayerNorm to inputs
        use_batchnorm: Apply BatchNorm after each layer
        use_layernorm_per_layer: Apply LayerNorm after each layer (in addition to input LayerNorm)
        groupnorm_num_groups: Number of groups for GroupNorm (0 = disabled)
        layer_norm_kwargs: Kwargs for LayerNorm
        batch_norm_kwargs: Kwargs for BatchNorm
        group_norm_kwargs: Kwargs for GroupNorm
        verbose: If 1, logs the network architecture (e.g., 100->50->25->1 [R, n=176, w=7.6k])
    """

    # ----------------------------------------------------------------------------------------------------------------------------
    # Auto inits and parameter setup
    # ----------------------------------------------------------------------------------------------------------------------------

    if layer_norm_kwargs is None:
        layer_norm_kwargs = dict(eps=1e-5)
    if batch_norm_kwargs is None:
        batch_norm_kwargs = dict(eps=1e-5, momentum=0.1)
    if group_norm_kwargs is None:
        group_norm_kwargs = dict(eps=1e-5)

    if not first_layer_num_neurons:
        first_layer_num_neurons = num_features

    # Don't modify min_layer_neurons directly - use effective_min_neurons instead
    effective_min_neurons = max(min_layer_neurons, num_classes) if num_classes and num_classes > 1 else min_layer_neurons

    # ----------------------------------------------------------------------------------------------------------------------------
    # Sanity checks
    # ----------------------------------------------------------------------------------------------------------------------------

    assert dropout_prob >= 0.0
    assert inputs_dropout_prob >= 0.0
    assert consec_layers_neurons_ratio >= 1.0
    assert nlayers >= 1 and isinstance(nlayers, int)
    assert min_layer_neurons >= 1 and isinstance(min_layer_neurons, int)
    assert num_classes is None or (num_classes >= 0 and isinstance(num_classes, int))
    assert first_layer_num_neurons >= min_layer_neurons and isinstance(first_layer_num_neurons, int)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Build network layers
    # ----------------------------------------------------------------------------------------------------------------------------

    layers = []
    layer_sizes = [num_features]  # Track layer sizes for verbose logging

    # Input normalization and dropout
    if inputs_dropout_prob > 0:
        layers.append(nn.Dropout(inputs_dropout_prob))
    if use_layernorm:
        layers.append(nn.LayerNorm(num_features, **layer_norm_kwargs))

    if groupnorm_num_groups > 0:
        num_groups_for_input = get_valid_num_groups(num_features, groupnorm_num_groups)
        if num_groups_for_input > 1:
            layers.append(nn.GroupNorm(num_groups=num_groups_for_input, num_channels=num_features, **group_norm_kwargs))

    # Cache mid_layer for architectures that need it
    mid_layer = nlayers // 2

    prev_layer_neurons = num_features
    cur_layer_neurons = first_layer_num_neurons
    cur_layer_virt_neurons = first_layer_num_neurons

    for layer in range(nlayers):

        # ----------------------------------------------------------------------------------------------------------------------------
        # Compute # of neurons in current layer using match statement (Python 3.10+)
        # ----------------------------------------------------------------------------------------------------------------------------

        if layer > 0:
            match neurons_by_layer_arch:
                case MLPNeuronsByLayerArchitecture.Constant:
                    cur_layer_virt_neurons = prev_layer_virt_neurons
                case MLPNeuronsByLayerArchitecture.Declining:
                    cur_layer_virt_neurons = prev_layer_virt_neurons / consec_layers_neurons_ratio
                case MLPNeuronsByLayerArchitecture.Expanding:
                    cur_layer_virt_neurons = prev_layer_virt_neurons * consec_layers_neurons_ratio
                case MLPNeuronsByLayerArchitecture.ExpandingThenDeclining:
                    if layer <= mid_layer:
                        cur_layer_virt_neurons = prev_layer_virt_neurons * consec_layers_neurons_ratio
                    else:
                        cur_layer_virt_neurons = prev_layer_virt_neurons / consec_layers_neurons_ratio
                case MLPNeuronsByLayerArchitecture.Autoencoder:
                    if layer <= mid_layer:
                        cur_layer_virt_neurons = prev_layer_virt_neurons / consec_layers_neurons_ratio
                    else:
                        cur_layer_virt_neurons = prev_layer_virt_neurons * consec_layers_neurons_ratio

            cur_layer_neurons = int(cur_layer_virt_neurons)

        # Check minimum neurons constraint
        if cur_layer_neurons < effective_min_neurons:
            if neurons_by_layer_arch == MLPNeuronsByLayerArchitecture.Autoencoder:
                # For autoencoder, allow smaller layers for symmetry, but enforce absolute minimum
                if cur_layer_neurons < 1:
                    cur_layer_neurons = 1
            elif layer > 0:
                # For other architectures, stop adding layers
                break
            else:
                # First layer must meet minimum
                cur_layer_neurons = effective_min_neurons

        # ----------------------------------------------------------------------------------------------------------------------------
        # Add linear layer with that many neurons
        # ----------------------------------------------------------------------------------------------------------------------------

        layers.append(nn.Linear(prev_layer_neurons, cur_layer_neurons))
        layer_sizes.append(cur_layer_neurons)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Add optional bells & whistles - batchnorm, layernorm, activation, dropout
        # ----------------------------------------------------------------------------------------------------------------------------

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(cur_layer_neurons, **batch_norm_kwargs))
        if use_layernorm_per_layer:
            layers.append(nn.LayerNorm(cur_layer_neurons, **layer_norm_kwargs))
        if activation_function:
            layers.append(activation_function())  # Instantiate activation function
        if dropout_prob > 0:
            layers.append(nn.Dropout(dropout_prob))

        prev_layer_neurons = cur_layer_neurons
        prev_layer_virt_neurons = cur_layer_virt_neurons

    # ----------------------------------------------------------------------------------------------------------------------------
    # Add final layer based on num_classes (Classification / Regression / Feature Extractor)
    # ----------------------------------------------------------------------------------------------------------------------------

    if num_classes is None or num_classes == 0:
        logger.warning("num_classes is None or 0 - creating feature extractor (no final layer)")
        model_type = "FE"
    elif num_classes == 1:
        layers.append(nn.Linear(prev_layer_neurons, 1))
        layer_sizes.append(1)
        model_type = "R"
    else:  # num_classes > 1
        layers.append(nn.Linear(prev_layer_neurons, num_classes))
        layer_sizes.append(num_classes)
        model_type = "C"

    model = nn.Sequential(*layers)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Log network architecture if verbose is enabled
    # ----------------------------------------------------------------------------------------------------------------------------

    if verbose == 1:
        # Calculate total neurons and weights
        total_neurons = sum(layer_sizes)
        total_weights = 0

        # Count parameters in all layers
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Linear layer: in_features * out_features + out_features (bias)
                total_weights += module.weight.numel()
                if module.bias is not None:
                    total_weights += module.bias.numel()
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                # Norm layers have weight and bias
                if hasattr(module, "weight") and module.weight is not None:
                    total_weights += module.weight.numel()
                if hasattr(module, "bias") and module.bias is not None:
                    total_weights += module.bias.numel()

        # Format numbers with 'k' suffix if >= 1000
        def format_num(n):
            if n >= 1000:
                return f"{n/1000:.1f}k"
            return str(n)

        architecture = "->".join(str(size) for size in layer_sizes)
        logger.info(f"Network architecture: {architecture} [{model_type}, n={format_num(total_neurons)}, w={format_num(total_weights)}]")

    # ----------------------------------------------------------------------------------------------------------------------------
    # Init weights explicitly if weights_init_fcn is set
    # ----------------------------------------------------------------------------------------------------------------------------

    if weights_init_fcn:

        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.BatchNorm1d)):
                if isinstance(weights_init_fcn, partial):
                    func_to_check = weights_init_fcn.func
                else:
                    func_to_check = weights_init_fcn

                # Initialize weights
                if hasattr(m, "weight") and m.weight is not None:
                    if func_to_check in (
                        torch.nn.init.xavier_normal_,
                        torch.nn.init.xavier_uniform_,
                        torch.nn.init.kaiming_normal_,
                        torch.nn.init.kaiming_uniform_,
                    ):
                        if m.weight.dim() >= 2:  # Only for Linear weights (2D)
                            weights_init_fcn(m.weight)
                        elif isinstance(m, nn.BatchNorm1d):  # BatchNorm weight (gamma, 1D)
                            torch.nn.init.normal_(m.weight, mean=1.0, std=0.02)  # Standard for BatchNorm
                    else:
                        weights_init_fcn(m.weight)

                # Initialize biases
                if hasattr(m, "bias") and m.bias is not None:
                    if func_to_check in (
                        torch.nn.init.xavier_normal_,
                        torch.nn.init.xavier_uniform_,
                        torch.nn.init.kaiming_normal_,
                        torch.nn.init.kaiming_uniform_,
                    ):
                        torch.nn.init.constant_(m.bias, 0.0)  # Standard for biases
                    else:
                        weights_init_fcn(m.bias)

        model.apply(init_weights)
        # Handle logging for partial functions
        init_name = weights_init_fcn.func.__name__ if isinstance(weights_init_fcn, partial) else weights_init_fcn.__name__
        logger.info(f"Applied {init_name} initialization to Linear weights; normal_/constant_ for BatchNorm weights/biases and Linear biases")

    model.example_input_array = torch.zeros(1, num_features)

    return model


# ----------------------------------------------------------------------------------------------------------------------------
# MLPTorchModel
# ----------------------------------------------------------------------------------------------------------------------------


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
            lr_scheduler: Learning rate scheduler class
            lr_scheduler_kwargs: Additional kwargs for scheduler
            compile_network: torch.compile mode (e.g., 'max-autotune', 'reduce-overhead')
            compute_trainset_metrics: Whether to compute metrics on training set
            lr_scheduler_interval: 'epoch' or 'step'
            lr_scheduler_monitor: Metric to monitor for scheduler (e.g., 'val_loss')
            load_best_weights_on_train_end: Load best checkpoint weights after training
        """
        super().__init__()

        # Validation
        if network is None:
            raise ValueError("network must be provided")
        if loss_fn is None:
            raise ValueError("loss_fn must be provided")
        if metrics is None:
            metrics = []
        if lr_scheduler_interval not in ["epoch", "step"]:
            raise ValueError(f"lr_scheduler_interval must be 'epoch' or 'step', got {lr_scheduler_interval}")

        # Set defaults
        optimizer = optimizer or torch.optim.AdamW
        optimizer_kwargs = optimizer_kwargs or {}
        lr_scheduler_kwargs = lr_scheduler_kwargs or {}

        # Save hyperparameters (excluding non-serializable objects)
        self.save_hyperparameters(ignore=["loss_fn", "metrics", "network", "optimizer", "lr_scheduler"])

        # Store components
        self.network = network
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.best_epoch = None

        # Initialize lists to store outputs during epoch
        self.training_step_outputs = []
        self.validation_step_outputs = []

        # Apply torch.compile if requested
        self._apply_torch_compile()

        # Set example input for ONNX export if available
        if hasattr(network, "example_input_array"):
            self.example_input_array = network.example_input_array
        else:
            logger.debug("Network lacks 'example_input_array'; ONNX export may require manual input")

    def _apply_torch_compile(self) -> None:
        """Apply torch.compile to the network if enabled.

        Compiled modles have problems with saving, at least in pytorch 2.8:
        cannot pickle 'ConfigModuleInstance' object
        https://github.com/pytorch/pytorch/issues/126154
        """
        if not self.hparams.compile_network:
            return

        if torch.__version__ < "2.0":
            logger.warning("torch.compile requires PyTorch >= 2.0. Skipping compilation.")
            return

        try:
            self.network = torch.compile(self.network, mode=self.hparams.compile_network)
            logger.info(f"Applied torch.compile with mode='{self.hparams.compile_network}'")
        except Exception as e:
            logger.warning(f"Failed to apply torch.compile: {e}. Using uncompiled network.")

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
        raise ValueError(f"Unexpected batch format: {type(batch)}")

    def _compute_weighted_loss(self, predictions: torch.Tensor, labels: torch.Tensor, sample_weight: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute loss with optional sample weighting.

        Args:
            predictions: Model predictions
            labels: Ground truth labels
            sample_weight: Optional per-sample weights

        Returns:
            Scalar loss tensor
        """
        if sample_weight is None:
            # No weighting - use original loss function
            return self.loss_fn(predictions, labels)

        # Compute per-sample losses using functional API with reduction='none'
        # Detect if this is classification (CrossEntropyLoss) or regression (MSELoss)
        if predictions.dim() == 2 and predictions.shape[1] > 1:
            # Classification: predictions are logits with shape (batch, num_classes)
            loss_unreduced = F.cross_entropy(predictions, labels, reduction="none")
        else:
            # Regression: predictions are values
            loss_unreduced = F.mse_loss(predictions, labels, reduction="none")

        # Apply sample weights and normalize by sum of weights
        weight_sum = sample_weight.sum()
        if weight_sum > 0:
            weighted_loss = (loss_unreduced * sample_weight).sum() / weight_sum
        else:
            # All weights are zero - return zero loss (no contribution from this batch)
            weighted_loss = torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)
        return weighted_loss

    def training_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        features, labels, sample_weight = self._unpack_batch(batch)
        raw_predictions = self(features)

        # Squeeze predictions for single-output regression to match label shape
        if raw_predictions.ndim == 2 and raw_predictions.shape[1] == 1:
            raw_predictions_for_loss = raw_predictions.squeeze(1)
        else:
            raw_predictions_for_loss = raw_predictions

        # Compute loss (with optional sample weighting)
        loss = self._compute_weighted_loss(raw_predictions_for_loss, labels, sample_weight)

        # Add L1 regularization if enabled
        if self.hparams.l1_alpha > 0:
            l1_norm = sum(p.abs().sum() for p in self.network.parameters())
            loss = loss + self.hparams.l1_alpha * l1_norm
            # Only log if trainer is attached (avoid warnings in unit tests)
            try:
                self.log("train_l1_norm", l1_norm, on_step=False, on_epoch=True)
            except RuntimeError:
                pass  # Not attached to trainer (unit tests)

        # Only log if trainer is attached (avoid warnings in unit tests)
        try:
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        except RuntimeError:
            pass  # Not attached to trainer (unit tests)

        # Store predictions for metric computation if needed
        result = {"loss": loss}
        if self.hparams.compute_trainset_metrics:
            # MEMORY OPTIMIZATION: Store outputs on CPU immediately to free GPU memory
            # This prevents GPU memory accumulation throughout the epoch
            output = {"raw_predictions": raw_predictions.detach().cpu(), "labels": labels.detach().cpu()}
            self.training_step_outputs.append(output)

        return result

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""

        if self.hparams.log_lr:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            logger.info(f"Epoch {self.current_epoch}, Step {self.global_step}: LR = {current_lr:.2e}")

        if not self.hparams.compute_trainset_metrics:
            return

        if not self.training_step_outputs:
            logger.warning("No training outputs collected for metric computation")
            return

        # Extract predictions and labels from collected outputs (already on CPU)
        preds_and_labels = [(out["raw_predictions"], out["labels"]) for out in self.training_step_outputs]

        # Compute metrics
        self.compute_metrics(preds_and_labels, prefix="train")

        # Clear outputs to free memory
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        features, labels, sample_weight = self._unpack_batch(batch)

        # No gradient computation needed
        raw_predictions = self(features)

        # Squeeze predictions for single-output regression to match label shape
        if raw_predictions.ndim == 2 and raw_predictions.shape[1] == 1:
            raw_predictions_for_loss = raw_predictions.squeeze(1)
        else:
            raw_predictions_for_loss = raw_predictions

        # Compute loss without L1 regularization for fair comparison (with optional sample weighting)
        loss = self._compute_weighted_loss(raw_predictions_for_loss, labels, sample_weight)

        # Only log if trainer is attached (avoid warnings in unit tests)
        try:
            self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        except RuntimeError:
            pass  # Not attached to trainer (unit tests)

        # MEMORY OPTIMIZATION: Store outputs on CPU immediately to free GPU memory
        output = {"raw_predictions": raw_predictions.detach().cpu(), "labels": labels.detach().cpu()}
        self.validation_step_outputs.append(output)

        return output

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        if not self.validation_step_outputs:
            logger.warning("No validation outputs collected for metric computation")
            return

        # Extract predictions and labels from collected outputs (already on CPU)
        preds_and_labels = [(out["raw_predictions"], out["labels"]) for out in self.validation_step_outputs]

        # Compute metrics
        self.compute_metrics(preds_and_labels, prefix="val")

        # Clear outputs to free memory
        self.validation_step_outputs.clear()

    def compute_metrics(self, predictions_and_labels: List[Tuple[torch.Tensor, torch.Tensor]], prefix: str = "val") -> None:
        """
        Compute and log all metrics given raw predictions and labels.
        Optimized: compute argmax, softmax, CPU/numpy only if needed, once each.

        Args:
            predictions_and_labels: List of (raw_predictions, labels) tuples from each batch (on CPU)
            prefix: Logging prefix ('train' or 'val')
        """
        # Concatenate all predictions and labels
        raw_predictions, labels = zip(*predictions_and_labels)
        raw_predictions = torch.cat(raw_predictions)
        labels = torch.cat(labels)

        # Determine which transformations are actually needed
        need_argmax = any(m.requires_argmax for m in self.metrics)
        need_softmax = any(m.requires_probs for m in self.metrics)
        need_cpu = any(m.requires_cpu for m in self.metrics)

        # Precompute transforms
        preds_dict = {}
        if need_argmax:
            preds_dict["argmax"] = raw_predictions.argmax(dim=1)
        if need_softmax:
            preds_dict["softmax"] = F.softmax(raw_predictions, dim=1)

        labels_cpu = None
        if need_cpu:
            # Already on CPU from storage optimization
            labels_cpu = to_numpy_safe(labels, cpu=False)

        # Compute metrics
        for metric in self.metrics:
            # Select correct prediction type
            if metric.requires_argmax:
                preds = preds_dict["argmax"]
            elif metric.requires_probs:
                preds = preds_dict["softmax"]
            else:
                preds = raw_predictions

            # Convert to CPU/numpy if needed
            if metric.requires_cpu:
                key = f"cpu_{id(preds)}"
                if key not in preds_dict:
                    # Already on CPU from storage optimization
                    preds_dict[key] = to_numpy_safe(preds, cpu=False)
                preds_np = preds_dict[key]
                labels_np = labels_cpu
            else:
                preds_np = preds
                labels_np = labels

            # Compute and log
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
            except Exception as e:
                logger.error(f"Failed to compute metric {prefix}_{metric.name}: {e}")

    def configure_optimizers(self):
        """Configure optimizer and optional learning rate scheduler."""
        # Create optimizer
        optimizer_kwargs = {"lr": self.hparams.learning_rate, **self.hparams.optimizer_kwargs}
        optimizer = self.optimizer(self.parameters(), **optimizer_kwargs)

        # Return optimizer only if no scheduler
        if self.lr_scheduler is None:
            return optimizer

        # Special handling for OneCycleLR
        if self.lr_scheduler.__name__ == "OneCycleLR":

            logger.info("Configuring OneCycleLR scheduler")

            # Calculate total steps
            steps_per_epoch = (
                len(self.trainer.datamodule.train_dataloader())
                if hasattr(self.trainer, "datamodule") and self.trainer.datamodule
                else len(self.trainer.train_dataloader)
            )

            total_steps = self.trainer.max_epochs * steps_per_epoch

            logger.info(f"OneCycleLR config:")
            logger.info(f"  - Steps per epoch: {steps_per_epoch}")
            logger.info(f"  - Max epochs: {self.trainer.max_epochs}")
            logger.info(f"  - Total steps: {total_steps}")
            logger.info(f"  - Interval: {self.hparams.lr_scheduler_interval}")

            # Update kwargs with calculated values
            scheduler_kwargs = {
                **self.hparams.lr_scheduler_kwargs,
                "total_steps": total_steps,
            }

            # Remove epochs/steps_per_epoch if present (use total_steps instead)
            scheduler_kwargs.pop("epochs", None)
            scheduler_kwargs.pop("steps_per_epoch", None)

            scheduler = self.lr_scheduler(optimizer, **scheduler_kwargs)
        else:
            # Normal scheduler creation
            scheduler = self.lr_scheduler(optimizer, **self.hparams.lr_scheduler_kwargs)

        # Configure scheduler settings
        scheduler_config = {
            "scheduler": scheduler,
            "interval": self.hparams.lr_scheduler_interval,
        }

        # Add monitor if specified (required for ReduceLROnPlateau)
        if self.hparams.lr_scheduler_monitor:
            scheduler_config["monitor"] = self.hparams.lr_scheduler_monitor

        logger.info(f"LR scheduler config: {scheduler_config}")
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def on_train_end(self) -> None:
        """Load best model weights after training completes."""
        if not self.hparams.load_best_weights_on_train_end:
            return

        # Only rank 0 handles checkpoint loading in distributed settings
        if not self.trainer.is_global_zero:
            return

        # Find ModelCheckpoint callback
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

        # Log checkpoint info
        best_score = checkpoint_callback.best_model_score
        score_str = f"{best_score:.4f}" if best_score is not None else "N/A"
        logger.info(f"Loading best model from {best_model_path} (score: {score_str})")

        try:
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)

            if "state_dict" not in checkpoint:
                logger.error("Checkpoint missing 'state_dict'. Cannot load weights.")
                return

            # Load state dict
            missing, unexpected = self.load_state_dict(checkpoint["state_dict"], strict=False)

            if missing:
                logger.warning(f"Missing keys in state_dict: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys in state_dict: {unexpected}")

            # Store best epoch info
            if "epoch" in checkpoint:
                self.best_epoch = checkpoint["epoch"]
                logger.info(f"Loaded weights from epoch {self.best_epoch}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)

    def predict_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Handle prediction for both (x, y) and x-only batches.

        Returns raw model output (logits for classification, values for regression).
        Softmax/argmax conversion is handled in the estimator's predict methods.
        """
        # Ensure model is in eval mode
        if self.training:
            logger.warning(f"Model was in training mode during predict_step at batch {batch_idx}. Switching to eval mode.")
            self.eval()

        # Handle both training/testing format (x, y) and prediction format (x only)
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        # Forward pass - return raw logits/values
        with torch.no_grad():
            logits = self(x)

        # For classification, apply softmax to get probabilities
        # (MLPTorchModel doesn't have is_classification attribute, so check network output)
        if logits.dim() == 2 and logits.shape[1] > 1:
            # Multi-class classification - return probabilities
            return torch.softmax(logits, dim=1)
        else:
            # Regression or binary classification - return raw values
            return logits
