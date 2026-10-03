"""Helpers carved out of ``flat`` to keep that module under its size budget."""

from __future__ import annotations


import logging
from functools import partial

import torch
import torch.nn as nn

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)
from ._flat_layers import (
    MLPNeuronsByLayerArchitecture,
    _BoundedTanhOutput,
    _ResidualLinearBlock,
    get_valid_num_groups,
)


def _generate_mlp_num_classes_none(num_classes):
    """Block of generate_mlp starting at ``if num_classes is not None:``."""
    if num_classes is not None:
        if not isinstance(num_classes, int) or isinstance(num_classes, bool):
            raise TypeError(f"num_classes must be None or an int, got {type(num_classes).__name__}")
        if num_classes < 0:
            raise ValueError(f"num_classes must be >= 0, got {num_classes!r}")


def _generate_mlp_accuracy_classification_trunk_downstream(numerical_embedding, numerical_embedding_kwargs, num_features, layers, layer_sizes):
    """Block of generate_mlp starting at ``if numerical_embedding is not None:``."""
    if numerical_embedding is not None:
        from mlframe.training.neural._numerical_embeddings import PeriodicLinearEmbedding
        _ne_kwargs = dict(numerical_embedding_kwargs or {})
        if numerical_embedding == "plr":
            _emb = PeriodicLinearEmbedding(in_features=num_features, **_ne_kwargs)
        else:
            raise ValueError(f"Unknown numerical_embedding={numerical_embedding!r}; " "supported: 'plr' (Periodic-Linear-ReLU).")
        layers.append(_emb)
        layer_sizes.append(_emb.out_features)
        # Override num_features for everything that follows so the
        # input-side Dropout / LayerNorm / GroupNorm and the first
        # hidden Linear see the EMBEDDED dimension, not the raw.
        num_features = _emb.out_features
    return num_features


def _generate_mlp_groupnorm_num_groups(groupnorm_num_groups, num_features, layers, group_norm_kwargs):
    """Block of generate_mlp starting at ``if groupnorm_num_groups > 0:``."""
    if groupnorm_num_groups > 0:
        num_groups_for_input = get_valid_num_groups(num_features, groupnorm_num_groups)
        if num_groups_for_input > 1:
            layers.append(nn.GroupNorm(num_groups=num_groups_for_input, num_channels=num_features, **group_norm_kwargs))


def _generate_mlp_layer(layer, neurons_by_layer_arch, prev_layer_virt_neurons, consec_layers_neurons_ratio, mid_layer, cur_layer_neurons, cur_layer_virt_neurons):
    """Block of generate_mlp starting at ``if layer > 0:``."""
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
    return cur_layer_neurons, cur_layer_virt_neurons


def _generate_mlp_use_residual(use_residual, spectral_norm, prev_layer_neurons, cur_layer_neurons, activation_function, dropout_prob, use_batchnorm, use_layernorm_per_layer, batch_norm_kwargs, layer_norm_kwargs, spectral_norm_n_power_iterations, layers, layer_sizes, _maybe_sn):
    """Block of generate_mlp starting at ``if use_residual:``."""
    if use_residual:
        # C6 (F-31): residual block bundles the Linear+norm+act+dropout
        # AND the skip connection (identity if dims match, bias-free
        # projection otherwise). Spectral norm is not threaded through
        # the residual path (the skip projection would also need it for
        # a true Lipschitz bound) — fall back to plain Linear with a
        # WARN if the user combined both.
        if spectral_norm:
            log_throttle(
                logger,
                "flat_mlp_residual_spectral_norm_approximate",
                logging.WARNING,
                "use_residual=True + spectral_norm=True: spectral norm "
                "is applied to the BODY Linear only, NOT to the skip "
                "projection; the global Lipschitz bound is therefore "
                "approximate. For an exact bound use one or the other.",
            )
        _block = _ResidualLinearBlock(
            in_dim=prev_layer_neurons,
            out_dim=cur_layer_neurons,
            activation_cls=activation_function,
            dropout_prob=dropout_prob,
            use_batchnorm=use_batchnorm,
            use_layernorm_per_layer=use_layernorm_per_layer,
            batch_norm_kwargs=batch_norm_kwargs,
            layer_norm_kwargs=layer_norm_kwargs,
        )
        if spectral_norm:
            _block.linear = nn.utils.spectral_norm(
                _block.linear, n_power_iterations=spectral_norm_n_power_iterations,
            )
        layers.append(_block)
        layer_sizes.append(cur_layer_neurons)
    else:
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


def _generate_mlp_final_layer_num_classes(num_classes, layers, _maybe_sn, prev_layer_neurons, layer_sizes):
    """Block of generate_mlp starting at ``if num_classes is None or num_classes == 0:``."""
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
    return model_type


def _generate_mlp_affine_composition_extrapolation_motivated(output_activation, num_classes, output_activation_scale, output_activation_center, layers):
    """Block of generate_mlp starting at ``if output_activation != "linear" and num_classes == 1:``."""
    if output_activation != "linear" and num_classes == 1:
        if output_activation == "tanh_train_range":
            if output_activation_scale is None or output_activation_center is None:
                # Hard raise: the caller-side contract is "auto-derive
                # scale/center BEFORE calling generate_mlp". The auto-fill
                # in ``neural.base._fit_inner_network`` (line ~627) is
                # OR-based on either being None and fills missing fields
                # in place; a direct caller that bypasses that path is
                # a programmer error worth surfacing immediately. Test
                # ``test_tanh_train_range_requires_scale_and_center`` pins
                # this contract; an earlier (2026-06-01) soft fallback
                # silently demoted misconfigured calls and broke that
                # test. The actual orchestration bug it was meant to mask
                # turned out to live on the S: recovered tree only.
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
            raise ValueError(f"Unknown output_activation={output_activation!r}; expected " f"one of: 'linear', 'tanh_train_range'.")


def _generate_mlp_verbose(verbose, layer_sizes, model, neurons_by_layer_arch, nlayers, consec_layers_neurons_ratio, activation_function, use_batchnorm, use_layernorm, use_layernorm_per_layer, groupnorm_num_groups, weights_init_fcn, spectral_norm, model_type, inputs_dropout_prob, dropout_prob):
    """Block of generate_mlp starting at ``if verbose == 1:``."""
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
            """Formats a neuron/weight count as e.g. ``7.6k`` above 1000, else the plain integer string."""
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


def _generate_mlp_weights_init_fcn(weights_init_fcn, model):
    """Block of generate_mlp starting at ``if weights_init_fcn:``."""
    if weights_init_fcn:

        def init_weights(m):
            """Applies ``weights_init_fcn`` to Linear/BatchNorm1d weights and biases, falling back to N(1.0, 0.02) for BatchNorm gamma when the init fn is Xavier/Kaiming (which require >=2D tensors and would raise on BN's 1D gamma)."""
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


def _generate_mlp_every_fit_bench_profiling(model):
    """Block of generate_mlp starting at ``try:``."""
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
