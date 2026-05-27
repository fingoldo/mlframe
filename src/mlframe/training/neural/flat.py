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


class Snake(nn.Module):
    """Snake activation: ``x + (1/alpha) * sin^2(alpha * x)``.

    Liu et al. 2020 "Neural Networks Fail to Learn Periodic Functions
    and How to Fix It" (NeurIPS). Useful when the target depends on
    cyclic / quasi-periodic inputs (azimuth, dogleg severity, formation
    crossing index) -- ReLU / Tanh / GELU all attenuate periodic
    signals because they're locally monotonic, Snake preserves them.

    ``alpha`` controls frequency. The default 1.0 reproduces the
    paper's "snake" curve; opt-in learnable per-channel alpha via
    ``alpha_learnable=True``.

    Forward shape: identity-preserving (input shape == output shape).
    """

    def __init__(self, alpha: float = 1.0, alpha_learnable: bool = False) -> None:
        super().__init__()
        if alpha_learnable:
            self.alpha = nn.Parameter(torch.tensor(float(alpha), dtype=torch.float32))
        else:
            self.register_buffer(
                "alpha", torch.tensor(float(alpha), dtype=torch.float32),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x + (1/alpha) * sin^2(alpha * x) = x + (1 - cos(2*alpha*x)) / (2*alpha)
        # Latter form is numerically stabler for small alpha.
        a = self.alpha
        return x + (1.0 - torch.cos(2.0 * a * x)) / (2.0 * a + 1e-12)

    def extra_repr(self) -> str:
        return f"alpha={float(self.alpha):.4g}"


class _BoundedTanhOutput(nn.Module):
    """Bounded-range output head: ``tanh(x) * scale + center``.

    Wraps the last ``nn.Linear`` of a regression MLP to HARD-CAP the
    output to ``[center - scale, center + scale]``. Composes cleanly with
    ``_TTRWithEvalSetScaling``: when y is z-scored by the TTR transformer
    and ``scale``/``center`` are computed on the SCALED y the MLP sees at
    fit-time, the tanh window is in scaled space and the TTR's
    ``inverse_transform`` unwinds it back to raw-y space correctly.

    Why this fix is independent of the defensive TTR predict clip
    (which lives at ``_TTRWithEvalSetScaling.predict``): the TTR clip
    BOUNDS the damage from runaway predictions; this output activation
    PREVENTS the MLP's affine-composition from emitting them in the
    first place. The MLP gradient sees the bound during training and
    learns parameters that keep activations inside the window; the TTR
    clip only catches what slips through at inference.

    ``scale`` and ``center`` are registered as non-trainable BUFFERS so
    they (a) move to the right device with ``.to(device)`` calls,
    (b) save/load with state_dict, and (c) do not get updated by the
    optimizer (they are fixed properties of the train target).
    """

    def __init__(self, scale: float, center: float) -> None:
        super().__init__()
        self.register_buffer(
            "scale", torch.tensor(float(scale), dtype=torch.float32),
        )
        self.register_buffer(
            "center", torch.tensor(float(center), dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x) * self.scale + self.center

    def extra_repr(self) -> str:
        return f"scale={float(self.scale):.4g}, center={float(self.center):.4g}"


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
    output_activation: str = "linear",
    output_activation_scale: Optional[float] = None,
    output_activation_center: Optional[float] = None,
    spectral_norm: bool = False,
    spectral_norm_n_power_iterations: int = 1,
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
            catastrophic OOD extrapolation under covariate shift (observed
            in prod: R^2=-326). Pick ``nlayers=1`` with Identity
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
    # (observed in prod: 25->32->16->1 Identity-MLP went to
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
            "(observed in prod: Identity-MLP R^2=-326 vs Ridge R^2=1.00 "
            "on identical data). Pick one: set nlayers=1 for an honest "
            "linear model, OR pick a real nonlinearity (nn.ReLU, nn.GELU, "
            "nn.LeakyReLU) for an actual nonlinear function.",
            "Identity/None" if activation_function is None
            else activation_function.__name__,
            nlayers,
            "regression" if num_classes == 1 else "classification",
        )

    # ``spectral_norm`` wrap helper. Bounds the spectral norm
    # (largest singular value) of each Linear's weight matrix to <= 1
    # via power iteration. Composes with weight init: SN is applied
    # AFTER initialisation, so we still get the user's init at t=0
    # but never see a weight matrix whose Lipschitz constant exceeds 1
    # at any subsequent step. The downstream effect: every Linear
    # contracts the input by at most ||x||, the activations (Tanh /
    # GELU / Mish / Snake) are themselves Lipschitz, and the whole
    # network is therefore globally Lipschitz with a known bound.
    # OOD inputs cannot produce output magnitudes more than ~depth
    # times their input norm -- the catastrophic-extrapolation modes
    # (R^2=-326, R^2=-30) that motivated the bounded-output and
    # envelope-clip defences become geometrically impossible.
    def _maybe_sn(module: nn.Module) -> nn.Module:
        if not spectral_norm:
            return module
        return nn.utils.spectral_norm(
            module, n_power_iterations=spectral_norm_n_power_iterations,
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

        layers.append(_maybe_sn(nn.Linear(prev_layer_neurons, cur_layer_neurons)))
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
        layers.append(_maybe_sn(nn.Linear(prev_layer_neurons, 1)))
        layer_sizes.append(1)
        model_type = "R"
    else:
        layers.append(_maybe_sn(nn.Linear(prev_layer_neurons, num_classes)))
        layer_sizes.append(num_classes)
        model_type = "C"

    # Bounded output head (Fix 1, 2026-05-26). Appended ONLY for regression
    # (``num_classes == 1``) and only when caller passes a non-default
    # ``output_activation``. ``"linear"`` is the historical no-op default.
    # ``"tanh_train_range"`` requires ``output_activation_scale`` +
    # ``output_activation_center`` (computed by the estimator from y_train).
    # Caps MLP output to ``[center - scale, center + scale]`` -> kills the
    # affine-composition extrapolation that motivated the TTR predict-clip.
    if output_activation != "linear" and num_classes == 1:
        if output_activation == "tanh_train_range":
            if output_activation_scale is None or output_activation_center is None:
                raise ValueError(
                    "output_activation='tanh_train_range' requires both "
                    "output_activation_scale and output_activation_center "
                    "to be non-None floats (computed by the caller from "
                    "y_train range + std)."
                )
            layers.append(_BoundedTanhOutput(
                scale=output_activation_scale,
                center=output_activation_center,
            ))
        else:
            raise ValueError(
                f"Unknown output_activation={output_activation!r}; expected "
                f"one of: 'linear', 'tanh_train_range'."
            )

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
        _sn_descr = " SN" if spectral_norm else ""
        logger.info(
            "Network architecture: %s [%s, n=%s, w=%s] arch=%s act=%s "
            "drop=in:%g/hid:%g norm=%s init=%s%s",
            architecture, model_type,
            format_num(total_neurons), format_num(total_weights),
            arch_descr, act_descr,
            inputs_dropout_prob, dropout_prob,
            norm_descr, init_descr,
            _sn_descr,
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


# MLPTorchModel carved to ``_flat_torch_module``; re-exported below.
from ._flat_torch_module import MLPTorchModel  # noqa: F401, E402

