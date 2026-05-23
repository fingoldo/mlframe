"""
MLP (Multi-Layer Perceptron) models for tabular/flat data.

This module provides:
- MLPTorchModel: PyTorch Lightning module for MLP training
- generate_mlp: Function to generate MLP architectures
- MLPNeuronsByLayerArchitecture: Enum for architecture patterns
"""

from __future__ import annotations


import logging
import os
from enum import Enum, auto
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint

from .base import MetricSpec, to_numpy_safe

logger = logging.getLogger(__name__)


class MLPNeuronsByLayerArchitecture(Enum):
    Constant = auto()
    Declining = auto()
    Expanding = auto()
    ExpandingThenDeclining = auto()
    Autoencoder = auto()


def get_valid_num_groups(num_channels, preferred_num_groups):
    for g in range(preferred_num_groups, 0, -1):
        if num_channels % g == 0:
            return g
    return 1  # Fallback to 1 (LayerNorm-like) if no divisor found


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
        activation_function: Activation function class (will be instantiated).
            ``torch.nn.Identity`` or ``None`` with ``nlayers>=2`` is the
            "linear MLP" footgun -- collapses to a single affine map with
            3x redundant parameterisation, bad optimisation, and known
            catastrophic OOD extrapolation under covariate shift (prod
            TVT 2026-05-22: R^2=-326). Pick ``nlayers=1`` with Identity
            for honest linear, or pick a real nonlinearity for a
            nonlinear MLP. A WARN fires if the footgun config is detected.
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

    if layer_norm_kwargs is None:
        layer_norm_kwargs = dict(eps=1e-5)
    if batch_norm_kwargs is None:
        batch_norm_kwargs = dict(eps=1e-5, momentum=0.1)
    if group_norm_kwargs is None:
        group_norm_kwargs = dict(eps=1e-5)

    if not first_layer_num_neurons:
        first_layer_num_neurons = num_features

    # Don't modify min_layer_neurons directly; use effective_min_neurons instead.
    effective_min_neurons = max(min_layer_neurons, num_classes) if num_classes and num_classes > 1 else min_layer_neurons

    # Validate API-boundary args explicitly so failures survive `python -O`
    # (asserts are stripped) and produce informative ValueError messages.
    if dropout_prob < 0.0:
        raise ValueError(f"dropout_prob must be >= 0.0, got {dropout_prob!r}")
    if inputs_dropout_prob < 0.0:
        raise ValueError(f"inputs_dropout_prob must be >= 0.0, got {inputs_dropout_prob!r}")
    if consec_layers_neurons_ratio < 1.0:
        raise ValueError(
            f"consec_layers_neurons_ratio must be >= 1.0, got {consec_layers_neurons_ratio!r}"
        )
    if not isinstance(nlayers, int) or isinstance(nlayers, bool):
        raise TypeError(f"nlayers must be an int, got {type(nlayers).__name__}")
    if nlayers < 1:
        raise ValueError(f"nlayers must be >= 1, got {nlayers!r}")
    if not isinstance(min_layer_neurons, int) or isinstance(min_layer_neurons, bool):
        raise TypeError(f"min_layer_neurons must be an int, got {type(min_layer_neurons).__name__}")
    if min_layer_neurons < 1:
        raise ValueError(f"min_layer_neurons must be >= 1, got {min_layer_neurons!r}")
    if num_classes is not None:
        if not isinstance(num_classes, int) or isinstance(num_classes, bool):
            raise TypeError(f"num_classes must be None or an int, got {type(num_classes).__name__}")
        if num_classes < 0:
            raise ValueError(f"num_classes must be >= 0, got {num_classes!r}")
    if not isinstance(first_layer_num_neurons, int) or isinstance(first_layer_num_neurons, bool):
        raise TypeError(f"first_layer_num_neurons must be an int, got {type(first_layer_num_neurons).__name__}")
    if first_layer_num_neurons < min_layer_neurons:
        raise ValueError(
            f"first_layer_num_neurons must be >= min_layer_neurons "
            f"({min_layer_neurons}), got {first_layer_num_neurons!r}"
        )

    # Identity-MLP footgun guard. ``nn.Identity`` (or ``None``) on a
    # multi-layer net composes to a single affine map but with 3x
    # redundantly-parameterised matrices: bad optimisation landscape,
    # weight_decay applied per-matrix instead of per-effective-coef,
    # and CATASTROPHIC OOD extrapolation under covariate shift
    # (production TVT 2026-05-22: 25->32->16->1 Identity-MLP went to
    # ~-17 sigma on the test split, R^2=-326 while Ridge nailed R^2=1.00
    # on the same data). For a truly linear regressor, ``nlayers=1``
    # gives an honest single Linear -> Identity which is well-conditioned
    # AND has the same expressivity. Multi-layer Identity is always a
    # mistake; warn loudly so the operator picks one or the other.
    _is_identity_activation = (
        activation_function is None
        or activation_function is nn.Identity
    )
    if _is_identity_activation and nlayers >= 2 and num_classes != 0:
        logger.warning(
            "generate_mlp: activation_function=%s with nlayers=%d on a %s "
            "head will COLLAPSE to a single affine map at inference. "
            "The 3+ redundantly-parameterised matrices DO NOT add "
            "expressivity (any composition of linear maps is linear), "
            "but they DO degrade optimisation and catastrophically "
            "amplify OOD-extrapolation on unseen-groups test splits "
            "(prod TVT 2026-05-22: Identity-MLP R^2=-326 vs Ridge R^2=1.00 "
            "on identical data). Pick one: set nlayers=1 for an honest "
            "linear model, OR pick a real nonlinearity (nn.ReLU, nn.GELU, "
            "nn.LeakyReLU) for an actual nonlinear function.",
            "Identity/None" if activation_function is None
            else activation_function.__name__,
            nlayers,
            "regression" if num_classes == 1 else "classification",
        )

    layers = []
    layer_sizes = [num_features]  # tracked for verbose logging

    if inputs_dropout_prob > 0:
        layers.append(nn.Dropout(inputs_dropout_prob))
    if use_layernorm:
        layers.append(nn.LayerNorm(num_features, **layer_norm_kwargs))

    if groupnorm_num_groups > 0:
        num_groups_for_input = get_valid_num_groups(num_features, groupnorm_num_groups)
        if num_groups_for_input > 1:
            layers.append(nn.GroupNorm(num_groups=num_groups_for_input, num_channels=num_features, **group_norm_kwargs))

    mid_layer = nlayers // 2

    prev_layer_neurons = num_features
    cur_layer_neurons = first_layer_num_neurons
    cur_layer_virt_neurons = first_layer_num_neurons
    prev_layer_virt_neurons = first_layer_num_neurons  # carries over into the if/elif chain on iterations >= 1

    for layer in range(nlayers):

        if layer > 0:
            if neurons_by_layer_arch == MLPNeuronsByLayerArchitecture.Constant:
                cur_layer_virt_neurons = prev_layer_virt_neurons
            elif neurons_by_layer_arch == MLPNeuronsByLayerArchitecture.Declining:
                cur_layer_virt_neurons = prev_layer_virt_neurons / consec_layers_neurons_ratio
            elif neurons_by_layer_arch == MLPNeuronsByLayerArchitecture.Expanding:
                cur_layer_virt_neurons = prev_layer_virt_neurons * consec_layers_neurons_ratio
            elif neurons_by_layer_arch == MLPNeuronsByLayerArchitecture.ExpandingThenDeclining:
                if layer <= mid_layer:
                    cur_layer_virt_neurons = prev_layer_virt_neurons * consec_layers_neurons_ratio
                else:
                    cur_layer_virt_neurons = prev_layer_virt_neurons / consec_layers_neurons_ratio
            elif neurons_by_layer_arch == MLPNeuronsByLayerArchitecture.Autoencoder:
                if layer <= mid_layer:
                    cur_layer_virt_neurons = prev_layer_virt_neurons / consec_layers_neurons_ratio
                else:
                    cur_layer_virt_neurons = prev_layer_virt_neurons * consec_layers_neurons_ratio

            cur_layer_neurons = int(cur_layer_virt_neurons)

        if cur_layer_neurons < effective_min_neurons:
            if neurons_by_layer_arch == MLPNeuronsByLayerArchitecture.Autoencoder:
                # Autoencoder: allow smaller layers for symmetry, but enforce absolute minimum of 1.
                if cur_layer_neurons < 1:
                    cur_layer_neurons = 1
            elif layer > 0:
                # Other architectures: stop adding layers once below minimum.
                break
            else:
                cur_layer_neurons = effective_min_neurons

        layers.append(nn.Linear(prev_layer_neurons, cur_layer_neurons))
        layer_sizes.append(cur_layer_neurons)

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(cur_layer_neurons, **batch_norm_kwargs))
        if use_layernorm_per_layer:
            layers.append(nn.LayerNorm(cur_layer_neurons, **layer_norm_kwargs))
        if activation_function:
            layers.append(activation_function())
        if dropout_prob > 0:
            layers.append(nn.Dropout(dropout_prob))

        prev_layer_neurons = cur_layer_neurons
        prev_layer_virt_neurons = cur_layer_virt_neurons

    # Final layer: num_classes None/0 = feature extractor, 1 = regression, >1 = classification.
    if num_classes is None or num_classes == 0:
        logger.warning("num_classes is None or 0; creating feature extractor (no final layer)")
        model_type = "FE"
    elif num_classes == 1:
        layers.append(nn.Linear(prev_layer_neurons, 1))
        layer_sizes.append(1)
        model_type = "R"
    else:
        layers.append(nn.Linear(prev_layer_neurons, num_classes))
        layer_sizes.append(num_classes)
        model_type = "C"

    model = nn.Sequential(*layers)

    if verbose == 1:
        total_neurons = sum(layer_sizes)
        total_weights = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total_weights += module.weight.numel()
                if module.bias is not None:
                    total_weights += module.bias.numel()
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                if hasattr(module, "weight") and module.weight is not None:
                    total_weights += module.weight.numel()
                if hasattr(module, "bias") and module.bias is not None:
                    total_weights += module.bias.numel()

        def format_num(n):
            if n >= 1000:
                return f"{n/1000:.1f}k"
            return str(n)

        # Include regularisation, activation and init in the log; the bare layer chain hides choices that
        # silently sabotage training (e.g. collapsed predictions when dropout/BN/init misconfigured).
        arch_name = getattr(neurons_by_layer_arch, "name", str(neurons_by_layer_arch))
        if nlayers > 1:
            arch_descr = f"{arch_name}(r={consec_layers_neurons_ratio:g})"
        else:
            arch_descr = arch_name

        if activation_function is None:
            act_descr = "identity"
        else:
            act_descr = getattr(activation_function, "__name__", type(activation_function).__name__)

        norm_parts = []
        if use_batchnorm:
            norm_parts.append("BN")
        if use_layernorm:
            norm_parts.append("LN_in")
        if use_layernorm_per_layer:
            norm_parts.append("LN_per_layer")
        if groupnorm_num_groups > 0:
            norm_parts.append(f"GN({groupnorm_num_groups})")
        norm_descr = "+".join(norm_parts) if norm_parts else "none"

        if weights_init_fcn is None:
            init_descr = "default"
        else:
            # functools.partial wraps the real callable in .func; bare functions / lambdas expose __name__ directly.
            _wf = getattr(weights_init_fcn, "func", weights_init_fcn)
            init_descr = getattr(_wf, "__name__", type(_wf).__name__)

        architecture = "->".join(str(size) for size in layer_sizes)
        logger.info(
            "Network architecture: %s [%s, n=%s, w=%s] arch=%s act=%s "
            "drop=in:%g/hid:%g norm=%s init=%s",
            architecture, model_type,
            format_num(total_neurons), format_num(total_weights),
            arch_descr, act_descr,
            inputs_dropout_prob, dropout_prob,
            norm_descr, init_descr,
        )

    if weights_init_fcn:

        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.BatchNorm1d)):
                if isinstance(weights_init_fcn, partial):
                    func_to_check = weights_init_fcn.func
                else:
                    func_to_check = weights_init_fcn

                # Xavier/Kaiming inits assume 2D fan-in/fan-out tensors. Applying them to BN's 1D gamma
                # raises ValueError, so we fall back to N(1.0, 0.02) for BN gamma in those cases.
                if hasattr(m, "weight") and m.weight is not None:
                    if func_to_check in (
                        torch.nn.init.xavier_normal_,
                        torch.nn.init.xavier_uniform_,
                        torch.nn.init.kaiming_normal_,
                        torch.nn.init.kaiming_uniform_,
                    ):
                        if m.weight.dim() >= 2:
                            weights_init_fcn(m.weight)
                        elif isinstance(m, nn.BatchNorm1d):
                            torch.nn.init.normal_(m.weight, mean=1.0, std=0.02)
                    else:
                        weights_init_fcn(m.weight)

                if hasattr(m, "bias") and m.bias is not None:
                    if func_to_check in (
                        torch.nn.init.xavier_normal_,
                        torch.nn.init.xavier_uniform_,
                        torch.nn.init.kaiming_normal_,
                        torch.nn.init.kaiming_uniform_,
                    ):
                        torch.nn.init.constant_(m.bias, 0.0)
                    else:
                        weights_init_fcn(m.bias)

        model.apply(init_weights)
        init_name = weights_init_fcn.func.__name__ if isinstance(weights_init_fcn, partial) else weights_init_fcn.__name__
        logger.info("Applied %s initialization to Linear weights; normal_/constant_ for BatchNorm weights/biases and Linear biases", init_name)

    # Degenerate-init probe (audit Agent A round-2 P1, landed 2026-05-23).
    # The Identity-activation guard above catches the "stack of Linear
    # -> Identity" footgun BEFORE the network is built. This probe runs
    # AFTER ``weights_init_fcn`` has been applied so it catches the
    # COMPLEMENTARY init-side pathologies that produce dead layers:
    #   * ``weights_init_fcn=torch.nn.init.zeros_`` (all weights zero,
    #     layer outputs zero, no gradient signal)
    #   * accidental ``constant_`` init (all weights identical)
    #   * any callable that produces zero-variance weight matrices
    #
    # bench-attempt-rejected (2026-05-23, c0039 / iter256): full
    # ``torch.linalg.matrix_rank`` (SVD-based) cost 785ms per Linear
    # layer = 7.85s for a 10-layer suite (5.2pct of total wall on c0039).
    # std-based check below is O(n*m) vs SVD's O(n*m^2), runs in
    # microseconds, and catches every common-case pathology this probe
    # was added to detect (zeros_/constant_/scalar-init). The rare
    # "non-zero-std but rank-deficient by construction" case (e.g. a
    # custom init that copies one row across the matrix) is a
    # model-design bug that should be caught at design time, not at
    # every fit. Bench: profiling/bench_mlp_rank_probe_std_vs_svd.py.
    try:
        _worst_std: float = float("inf")
        _worst_layer_name: str = ""
        for _name, _module in model.named_modules():
            if isinstance(_module, nn.Linear):
                with torch.no_grad():
                    _W = _module.weight.detach()
                    # Population std (unbiased=False) avoids n>1 special
                    # case for 1-element weights (1x1 Linear).
                    _std = float(torch.std(_W, unbiased=False).item())
                    if _std < _worst_std:
                        _worst_std = _std
                        _worst_layer_name = _name or f"Linear({_W.shape[1]}->{_W.shape[0]})"
        # Threshold: 1e-8 catches zeros_ (std=0) and constant_ (std=0)
        # without false-positives on legitimate kaiming/xavier inits whose
        # std is typically 0.01-0.2 depending on fan_in / fan_out.
        if _worst_std < 1e-8:
            logger.warning(
                "generate_mlp: degenerate Linear layer detected -- "
                "weakest layer '%s' has weight std %.2e (~zero). "
                "Common causes: weights_init_fcn=zeros_ / constant_, or "
                "pathological init. The model will not learn useful "
                "representations on this layer; pick a non-degenerate "
                "init (kaiming_normal_ / xavier_uniform_).",
                _worst_layer_name, _worst_std,
            )
    except Exception as _rank_err:
        logger.debug(
            "generate_mlp: degenerate-init probe failed (non-fatal): %s",
            _rank_err,
        )

    model.example_input_array = torch.zeros(1, num_features)

    return model


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

        self._apply_torch_compile()

        if hasattr(network, "example_input_array"):
            self.example_input_array = network.example_input_array
        else:
            logger.debug("Network lacks 'example_input_array'; ONNX export may require manual input")

    def _apply_torch_compile(self) -> None:
        """Apply torch.compile to the network if enabled.

        Compiled models cannot be pickled in PyTorch 2.8 ("cannot pickle 'ConfigModuleInstance' object"),
        which breaks checkpoint saving. See https://github.com/pytorch/pytorch/issues/126154.
        """
        if not self.hparams.compile_network:
            return

        if torch.__version__ < "2.0":
            logger.warning("torch.compile requires PyTorch >= 2.0. Skipping compilation.")
            return

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
        # weight cases. The once-per-process all-zero-weight WARN is dropped:
        # zero gradient surfaces downstream as flat val loss, and per-batch
        # sync to log a single warning was a poor trade.
        weighted_loss = raw / torch.clamp(weight_sum, min=1e-12)
        return weighted_loss

    def training_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        features, labels, sample_weight = self._unpack_batch(batch)
        raw_predictions = self(features)

        # Squeeze single-output regression to match label shape.
        if raw_predictions.ndim == 2 and raw_predictions.shape[1] == 1:
            raw_predictions_for_loss = raw_predictions.squeeze(1)
        else:
            raw_predictions_for_loss = raw_predictions

        loss = self._compute_weighted_loss(raw_predictions_for_loss, labels, sample_weight)

        if self.hparams.l1_alpha > 0:
            # Python sum() forces a host-side reduction per parameter tensor,
            # so each .abs().sum() implicitly syncs the GPU. Stack the per-
            # tensor scalars first and sum once on-device to amortise the sync.
            _abs_sums = [p.abs().sum().unsqueeze(0) for p in self.network.parameters()]
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

        if raw_predictions.ndim == 2 and raw_predictions.shape[1] == 1:
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
            else:
                # Regression / binary single-output: argmax has no meaning; skip metric below.
                preds_dict["argmax"] = None
        if need_softmax:
            if _is_multiclass:
                preds_dict["softmax"] = F.softmax(raw_predictions, dim=1)
            elif self.task_type == "multilabel":
                preds_dict["softmax"] = torch.sigmoid(raw_predictions)
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
        optimizer = self.optimizer(self.parameters(), **optimizer_kwargs)

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

            if missing:
                logger.warning(f"Missing keys in state_dict: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys in state_dict: {unexpected}")

            if "epoch" in checkpoint:
                self.best_epoch = checkpoint["epoch"]
                logger.info("Loaded weights from epoch %s", self.best_epoch)

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)

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

        with torch.no_grad():
            logits = self(x)

        # task_type='multilabel' returns per-label sigmoid (each output independent binary in [0, 1]);
        # default multi-class K>1 path returns softmax rows that sum to 1.
        if logits.dim() == 2 and logits.shape[1] > 1:
            if self.task_type == "multilabel":
                return torch.sigmoid(logits)
            return torch.softmax(logits, dim=1)
        else:
            # Regression or binary classification: return raw values.
            return logits
