"""Translate sklearn MLP overrides onto the mlframe MLP and recurrent-config kwargs.

``feature_drift_report`` re-exports each translator, so its public surface is unchanged.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("mlframe.training.feature_drift_report")


def _translate_alpha(sklearn_overrides, existing_mlp_kwargs, out, untranslated) -> None:
    """Route the sklearn ``alpha`` (L2) override into the optimizer's ``weight_decay``, keeping a caller-pinned optimizer."""
    if "alpha" in sklearn_overrides:
        try:
            import torch  # local import: torch is an MLP-only optional dep.
            out.setdefault("model_params", {})
            # Preserve a caller-supplied optimizer (e.g. MuonAdamWHybrid).
            # Audit-pass-10 #1 surfaced this as a real bug: hardcoding
            # torch.optim.AdamW here silently overwrote the user's Muon
            # choice whenever ``alpha`` was in sklearn_overrides AND the
            # downstream deep-merge at
            # ``training/core/_phase_train_one_target_model_setup.py``
            # had ``{**existing, **override}`` direction. We now only
            # set the optimizer when the caller did not pin one.
            _existing_optimizer = None
            if existing_mlp_kwargs:
                _existing_model_params = existing_mlp_kwargs.get("model_params") or {}
                _existing_optimizer = _existing_model_params.get("optimizer")
            if _existing_optimizer is None:
                out["model_params"]["optimizer"] = torch.optim.AdamW
            out["model_params"].setdefault("optimizer_kwargs", {})
            out["model_params"]["optimizer_kwargs"]["weight_decay"] = float(sklearn_overrides["alpha"])
        except Exception as e:
            logger.debug("could not translate sklearn 'alpha' override to weight_decay: %s", e)
            untranslated.append("alpha")


def _translate_hidden_layer_sizes(sklearn_overrides, out, untranslated) -> None:
    """Map the sklearn ``hidden_layer_sizes`` topology onto the network depth, first and minimum layer widths, and width ratio."""
    if "hidden_layer_sizes" in sklearn_overrides:
        hls = sklearn_overrides["hidden_layer_sizes"]
        if isinstance(hls, (tuple, list)) and len(hls) >= 1:
            first = int(hls[0])
            minl = int(hls[-1])
            consec = (first / minl) if minl > 0 else 1.0
            out.setdefault("network_params", {})
            out["network_params"]["nlayers"] = len(hls)
            out["network_params"]["first_layer_num_neurons"] = first
            out["network_params"]["min_layer_neurons"] = minl
            out["network_params"]["consec_layers_neurons_ratio"] = float(consec)
        else:
            untranslated.append("hidden_layer_sizes")


def _translate_activation(sklearn_overrides, out, untranslated) -> None:
    """Map the sklearn ``activation`` name onto the matching torch activation; ``identity`` is a single linear layer with no dropout."""
    if "activation" in sklearn_overrides:
        try:
            import torch
            _act = str(sklearn_overrides["activation"]).lower()
            out.setdefault("network_params", {})
            if _act == "relu":
                out["network_params"]["activation_function"] = torch.nn.ReLU
            elif _act == "tanh":
                out["network_params"]["activation_function"] = torch.nn.Tanh
            elif _act in ("leaky_relu", "leakyrelu"):
                out["network_params"]["activation_function"] = torch.nn.LeakyReLU
            elif _act == "identity":
                # Identity activation = single linear transform on the
                # input. ``generate_mlp`` requires ``nlayers >= 1``;
                # force ``nlayers=1`` so the network is an honest
                # ``Linear(in, out) -> Identity`` instead of a multi-
                # layer stack that mathematically collapses but
                # is redundantly parameterised, optimises poorly,
                # and catastrophically OOD-extrapolates under
                # covariate shift (observed in prod: 25->32->16->1
                # Identity stack went to ~-17 sigma on test split,
                # R^2=-326 while Ridge R^2=1.00 on the same data).
                out["network_params"]["activation_function"] = torch.nn.Identity
                out["network_params"]["nlayers"] = 1
                # Aux dropout sources also break linearity if non-zero --
                # zero them so the network really IS linear end-to-end.
                out["network_params"]["dropout_prob"] = 0.0
                out["network_params"]["inputs_dropout_prob"] = 0.0
            else:
                untranslated.append(f"activation={_act}")
        except Exception as e:
            logger.debug("could not translate sklearn activation override: %s", e)
            untranslated.append("activation")


def translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs(
    sklearn_overrides: dict[str, Any],
    existing_mlp_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate a sklearn-MLPRegressor-shape override dict (produced by the
    robustness bench) into the nested ``mlp_kwargs`` shape consumed by the
    mlframe MLP builder in ``training/helpers.py:get_training_configs``.

    Keys recognised on input (sklearn naming):
      - ``alpha`` (float): L2 weight decay -> mlframe routes via the
        caller-resolved optimizer's ``weight_decay`` kwarg (AdamW /
        MuonAdamWHybrid / any other ``torch.optim`` that accepts
        ``weight_decay``). When ``existing_mlp_kwargs`` already sets
        ``model_params["optimizer"]`` we DO NOT overwrite it -- the
        caller's choice wins, only ``optimizer_kwargs["weight_decay"]``
        is injected. When no optimizer is preset, we default to
        ``torch.optim.AdamW`` (the LightningModule's own default at
        ``training/neural/_flat_torch_module.py:86``).
      - ``hidden_layer_sizes`` (tuple[int, ...]): output topology ->
        ``network_params`` ``nlayers`` + neuron sizes.
      - ``activation`` (str): {"relu", "tanh", "identity", "leaky_relu"}.
        ``identity`` collapses to ``nlayers=0`` (a literal linear head).

    Anything else is passed through verbatim under
    ``mlp_kwargs["model_params"]`` so the caller can extend the bench
    without changing this translator.

    The translation is informed but not exhaustive: the bench validates
    the MECHANISM (regularization + capacity + activation), not specific
    field names; the torch-backed mlframe MLP is a different
    implementation. If a bench winner uses a key we cannot translate, we
    skip it and surface the skipped keys in the returned dict's
    ``__untranslated__`` entry for caller-side logging.
    """
    if not sklearn_overrides:
        return {}

    out: dict[str, Any] = {}
    untranslated: list[str] = []

    _translate_alpha(sklearn_overrides, existing_mlp_kwargs, out, untranslated)
    _translate_hidden_layer_sizes(sklearn_overrides, out, untranslated)
    _translate_activation(sklearn_overrides, out, untranslated)

    for k, v in sklearn_overrides.items():
        if k in {"alpha", "hidden_layer_sizes", "activation"}:
            continue
        out.setdefault("model_params", {})[k] = v

    if untranslated:
        out["__untranslated__"] = untranslated
    return out


def translate_sklearn_mlp_overrides_to_recurrent_config_kwargs(
    sklearn_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Translate the same sklearn-MLP-shape override into the
    ``RecurrentConfig`` field shape used by mlframe's LSTM / GRU / RNN /
    Transformer wrappers.

    Mapping (BEST-EFFORT, not empirically grounded for recurrent yet):
      - ``alpha`` -> ``weight_decay`` (RecurrentConfig has it directly).
      - ``hidden_layer_sizes`` -> ``mlp_hidden_sizes`` (the post-recurrent
        MLP head; the recurrent layers themselves are uncontrolled here).
      - ``activation`` -> no direct mapping. ``identity`` cannot collapse
        the recurrent cell to linear, so this key is recorded as
        untranslated rather than silently dropped.

    This translator is NOT used at runtime until
    ``ROBUST_RECURRENT_OVERRIDES_UNDER_DRIFT`` is populated (currently
    empty). The empirical grounding for recurrent will require a
    sequence-DGP sweep -- the MLP sweep does not transfer because the
    recurrent failure mode (hidden-state extrapolation) is mechanically
    different from MLP feature-range extrapolation.
    """
    if not sklearn_overrides:
        return {}
    out: dict[str, Any] = {}
    untranslated: list[str] = []
    if "alpha" in sklearn_overrides:
        out["weight_decay"] = float(sklearn_overrides["alpha"])
    if "hidden_layer_sizes" in sklearn_overrides:
        hls = sklearn_overrides["hidden_layer_sizes"]
        if isinstance(hls, (tuple, list)) and len(hls) >= 1:
            out["mlp_hidden_sizes"] = tuple(int(h) for h in hls)
        else:
            untranslated.append("hidden_layer_sizes")
    if "activation" in sklearn_overrides:
        # Recurrent cells (LSTM/GRU/RNN) have hard-coded gate activations;
        # transformer uses softmax+MLP. None of them have a single
        # "activation" knob the way sklearn MLPRegressor does. Skip rather
        # than silently force an incompatible value.
        untranslated.append(f"activation={sklearn_overrides['activation']} (no direct recurrent equivalent)")
    if untranslated:
        out["__untranslated__"] = untranslated
    return out
