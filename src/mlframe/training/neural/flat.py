"""
MLP (Multi-Layer Perceptron) models for tabular/flat data.

This module provides:
- MLPTorchModel: PyTorch Lightning module for MLP training
- generate_mlp: Function to generate MLP architectures
- MLPNeuronsByLayerArchitecture: Enum for architecture patterns
"""

from __future__ import annotations


import logging
from typing import Callable, Optional, cast, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


from ._flat_layers import (  # noqa: F401  -- carved helpers
    MLPNeuronsByLayerArchitecture,
    get_valid_num_groups,
    _BoundedTanhOutput,
    _ResidualLinearBlock,
)
from ._flat_generate_helpers import (
    _generate_mlp_num_classes_none,
    _generate_mlp_accuracy_classification_trunk_downstream,
    _generate_mlp_groupnorm_num_groups,
    _generate_mlp_layer,
    _generate_mlp_use_residual,
    _generate_mlp_final_layer_num_classes,
    _generate_mlp_affine_composition_extrapolation_motivated,
    _generate_mlp_verbose,
    _generate_mlp_weights_init_fcn,
    _generate_mlp_every_fit_bench_profiling,
)

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
        """Applies the Snake activation elementwise (shape-preserving)."""
        # x + (1/alpha) * sin^2(alpha * x) = x + (1 - cos(2*alpha*x)) / (2*alpha)
        # Latter form is numerically stabler for small alpha.
        a = self.alpha
        return cast(torch.Tensor, x + (1.0 - torch.cos(2.0 * a * x)) / (2.0 * a + 1e-12))

    def extra_repr(self) -> str:
        """Reports the current ``alpha`` value for ``print(module)`` / ``repr``."""
        return f"alpha={float(self.alpha):.4g}"


def generate_mlp(
    num_features: int,
    num_classes: int,
    nlayers: int = 1,
    first_layer_num_neurons: Optional[int] = None,
    min_layer_neurons: int = 1,
    neurons_by_layer_arch: MLPNeuronsByLayerArchitecture = MLPNeuronsByLayerArchitecture.Constant,
    consec_layers_neurons_ratio: float = 1.1,
    activation_function: Optional[Callable] = torch.nn.ReLU,
    weights_init_fcn: Optional[Callable] = None,
    dropout_prob: float = 0.15,
    inputs_dropout_prob: float = 0.01,
    use_layernorm: bool = False,
    use_batchnorm: bool = False,
    use_layernorm_per_layer: bool = False,
    use_residual: bool = False,
    numerical_embedding: Optional[str] = None,
    numerical_embedding_kwargs: Optional[dict] = None,
    categorical_cardinalities: Optional[list] = None,
    categorical_embed_dim: Optional[int] = None,
    groupnorm_num_groups: int = 0,
    layer_norm_kwargs: Optional[dict] = None,
    batch_norm_kwargs: Optional[dict] = None,
    group_norm_kwargs: Optional[dict] = None,
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
        use_layernorm: Apply ``nn.LayerNorm(num_features)`` to inputs.
            Default ``False`` (F-03, 2026-05-30).

            When to LEAVE False (the default, recommended for tabular):
            inputs are heterogeneous-scale columns (raw numerics in
            different units). LayerNorm-per-row z-scores ACROSS the
            row's features, mixing them into one sequence and
            discarding per-feature scale. Measured (n=1200, d=3,
            scales 1.0 / 1000 / 10): test R^2 +0.083 (LN on) vs +0.989
            (use_batchnorm=True) vs +0.998 (StandardScaler upstream).
            Single-feature regression with LN on collapses to
            predict-the-mean (R^2 ~0) because the per-row variance is
            always 0 -> the normalised input is the zero tensor.

            When to FLIP True: inputs are homogeneous-scale columns
            that legitimately share a unit -- post-StandardScaler
            features, an embedding output (e.g. Transformer / CNN
            penultimate activations), an RNN hidden-state slice. In
            those regimes the per-row z-score IS roughly the per-
            feature z-score and LN-on costs LESS (measured: gap of
            ~0.18 R^2 on homogeneous N(0,1) inputs vs ~0.9 R^2 on
            heterogeneous-scale inputs -- order of magnitude smaller).
            This is the regime where LayerNorm was originally proposed
            (Ba et al. 2016) and shines on deeper / nonlinear /
            high-variance-activation networks; it does NOT generalise
            to shallow tabular MLPs with linear-ish targets, where it
            adds a small drag even when inputs are already normalised.

            The mlframe suite has been overriding this to False at
            trainer.py:697 since 2026-05-21; the default flip here
            propagates the fix to direct ``generate_mlp`` callers
            that don't go through the suite. For per-feature input
            normalisation use ``use_batchnorm=True`` (BN on hidden
            activations, which transitively normalises the raw-input
            feature mix the first Linear sees) or
            ``sklearn.preprocessing.StandardScaler`` upstream of fit().
        use_batchnorm: Apply BatchNorm after each layer
        use_layernorm_per_layer: Apply LayerNorm after each layer (in addition to input LayerNorm)
        use_residual: Use the ResNet-tabular block (Linear -> norm -> activation -> dropout, plus
            an identity/projected skip connection) instead of the plain feedforward block.
        numerical_embedding: Optional embedding applied to numeric inputs before the MLP body.
            ``None`` (default) = no embedding. ``"plr"`` = Periodic-Linear-ReLU embedding.
        numerical_embedding_kwargs: Kwargs forwarded to the ``numerical_embedding`` constructor.
        categorical_cardinalities: Per-categorical category counts (excluding the reserved unknown row). When set, a learnable
            ``CategoricalEmbedding`` is prepended; the first ``len(categorical_cardinalities)`` input columns are treated as integer cat
            codes (leading columns, set by the estimator's fit-boundary factorizer) and the rest as numeric passthrough. ``None`` (default)
            = no categorical embedding (hook is a no-op).
        categorical_embed_dim: Fixed per-cat embedding width; ``None`` (default) uses the fastai heuristic ``min(50, round(1.6*card**0.56))``.
        groupnorm_num_groups: Number of groups for GroupNorm (0 = disabled)
        layer_norm_kwargs: Kwargs for LayerNorm
        batch_norm_kwargs: Kwargs for BatchNorm
        group_norm_kwargs: Kwargs for GroupNorm
        output_activation: Final-layer activation for regression (``num_classes == 1``). ``"linear"``
            (default) is a no-op. ``"tanh_train_range"`` squashes output into a train-range-derived
            band and requires both ``output_activation_scale``/``output_activation_center``.
        output_activation_scale: Scale term for ``output_activation="tanh_train_range"`` (typically
            derived by the estimator from ``y_train``); required when that mode is selected.
        output_activation_center: Center term for ``output_activation="tanh_train_range"``
            (typically derived by the estimator from ``y_train``); required when that mode is selected.
        spectral_norm: Wrap each layer's Linear with ``nn.utils.spectral_norm`` (Lipschitz-bounds
            the weight matrix); combined with ``use_residual=True`` this also spectral-norms the
            residual block's inner Linear.
        spectral_norm_n_power_iterations: Power-iteration count for ``spectral_norm``'s singular
            value estimate; only used when ``spectral_norm=True``.
        verbose: If 1, logs the network architecture (e.g., 100->50->25->1 [R, n=176, w=7.6k])
    """

    layer: Any = None
    if layer_norm_kwargs is None:
        layer_norm_kwargs = dict(eps=1e-5)
    if batch_norm_kwargs is None:
        batch_norm_kwargs = dict(eps=1e-5, momentum=0.1)
    if group_norm_kwargs is None:
        group_norm_kwargs = dict(eps=1e-5)

    if first_layer_num_neurons is None or first_layer_num_neurons <= 0:
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
        raise ValueError(f"consec_layers_neurons_ratio must be >= 1.0, got {consec_layers_neurons_ratio!r}")
    if not isinstance(nlayers, int) or isinstance(nlayers, bool):
        raise TypeError(f"nlayers must be an int, got {type(nlayers).__name__}")
    if nlayers < 1:
        raise ValueError(f"nlayers must be >= 1, got {nlayers!r}")
    if not isinstance(min_layer_neurons, int) or isinstance(min_layer_neurons, bool):
        raise TypeError(f"min_layer_neurons must be an int, got {type(min_layer_neurons).__name__}")
    if min_layer_neurons < 1:
        raise ValueError(f"min_layer_neurons must be >= 1, got {min_layer_neurons!r}")
    _generate_mlp_num_classes_none(num_classes)
    if not isinstance(first_layer_num_neurons, int) or isinstance(first_layer_num_neurons, bool):
        raise TypeError(f"first_layer_num_neurons must be an int, got {type(first_layer_num_neurons).__name__}")
    if first_layer_num_neurons < min_layer_neurons:
        raise ValueError(f"first_layer_num_neurons must be >= min_layer_neurons " f"({min_layer_neurons}), got {first_layer_num_neurons!r}")

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
    _is_identity_activation = activation_function is None or activation_function is nn.Identity
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
            "Identity/None" if activation_function is None else activation_function.__name__,
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
        """Wraps ``module`` with ``nn.utils.spectral_norm`` when ``spectral_norm`` is enabled, else returns it unchanged."""
        if not spectral_norm:
            return module
        return nn.utils.spectral_norm(
            module, n_power_iterations=spectral_norm_n_power_iterations,
        )

    layers: list[nn.Module] = []
    layer_sizes = [num_features]  # tracked for verbose logging

    # Learnable categorical entity embeddings. When ``categorical_cardinalities`` is set, the FIRST ``k`` input columns are integer category
    # codes (the estimator factorizes raw cats into codes at the fit boundary and reorders them leading); the rest are numeric. A
    # ``CategoricalEmbedding`` maps each cat code through its own ``nn.Embedding`` (trained end-to-end) and passes the numeric block through.
    # Per Guo & Berkhahn 2016 this recovers non-monotone category->target structure a single target-encoded scalar column cannot. Runs BEFORE
    # the numerical embedding so the post-cat-embedding width is what the PLR / trunk sees; cats being the leading columns is the layout
    # contract with the estimator boundary.
    if categorical_cardinalities:
        from ._categorical_embeddings import CategoricalEmbedding
        _k_cat = len(categorical_cardinalities)
        _cat_emb = CategoricalEmbedding(
            cardinalities=list(categorical_cardinalities),
            embed_dim=categorical_embed_dim,
        )
        _cat_emb.set_num_numeric(max(0, num_features - _k_cat))
        layers.append(_cat_emb)
        layer_sizes.append(_cat_emb.out_features)
        # Override num_features so the input-side Dropout / LayerNorm / GroupNorm and the first hidden Linear (and any numerical embedding
        # below) size to the post-embedding width, not the raw code+numeric width.
        num_features = _cat_emb.out_features

    # Numerical-feature embeddings: the PLR (Periodic-Linear-ReLU) embedding maps each scalar feature to a high-dim representation via sin/cos
    # at K learnable frequencies + a per-feature Linear projection (RealMLP-TD, Holzmuller et al. NeurIPS 2024: +20.6% R^2 regression / +2.3%
    # accuracy classification). The trunk downstream then operates on the embedded representation as if it were the raw feature vector.
    num_features = _generate_mlp_accuracy_classification_trunk_downstream(numerical_embedding, numerical_embedding_kwargs, num_features, layers, layer_sizes)

    if inputs_dropout_prob > 0:
        layers.append(nn.Dropout(inputs_dropout_prob))
    if use_layernorm:
        layers.append(nn.LayerNorm(num_features, **layer_norm_kwargs))

    _generate_mlp_groupnorm_num_groups(groupnorm_num_groups, num_features, layers, group_norm_kwargs)

    mid_layer = nlayers // 2

    prev_layer_neurons = num_features
    cur_layer_neurons: int = first_layer_num_neurons
    cur_layer_virt_neurons: float = first_layer_num_neurons
    prev_layer_virt_neurons: float = first_layer_num_neurons  # carries over into the if/elif chain on iterations >= 1

    for layer in range(nlayers):

        cur_layer_neurons, cur_layer_virt_neurons = _generate_mlp_layer(layer, neurons_by_layer_arch, prev_layer_virt_neurons, consec_layers_neurons_ratio, mid_layer, cur_layer_neurons, cur_layer_virt_neurons)

        if cur_layer_neurons < effective_min_neurons:
            if neurons_by_layer_arch == MLPNeuronsByLayerArchitecture.Autoencoder:
                # Autoencoder: allow smaller layers for symmetry, but enforce absolute minimum of 1.
                if cur_layer_neurons < 1:
                    cur_layer_neurons = 1
            elif layer > 0:
                # Other architectures: stop adding layers once below minimum.
                break
            else:
                cur_layer_neurons = int(effective_min_neurons)

        _generate_mlp_use_residual(use_residual, spectral_norm, prev_layer_neurons, cur_layer_neurons, activation_function, dropout_prob, use_batchnorm, use_layernorm_per_layer, batch_norm_kwargs, layer_norm_kwargs, spectral_norm_n_power_iterations, layers, layer_sizes, _maybe_sn)

        prev_layer_neurons = cur_layer_neurons
        prev_layer_virt_neurons = cur_layer_virt_neurons

    # Final layer: num_classes None/0 = feature extractor, 1 = regression, >1 = classification.
    model_type = _generate_mlp_final_layer_num_classes(num_classes, layers, _maybe_sn, prev_layer_neurons, layer_sizes)

    # Bounded output head (Fix 1, 2026-05-26). Appended ONLY for regression
    # (``num_classes == 1``) and only when caller passes a non-default
    # ``output_activation``. ``"linear"`` is the historical no-op default.
    # ``"tanh_train_range"`` requires ``output_activation_scale`` +
    # ``output_activation_center`` (computed by the estimator from y_train).
    # Caps MLP output to ``[center - scale, center + scale]`` -> kills the
    # affine-composition extrapolation that motivated the TTR predict-clip.
    _generate_mlp_affine_composition_extrapolation_motivated(output_activation, num_classes, output_activation_scale, output_activation_center, layers)

    model = nn.Sequential(*layers)

    _generate_mlp_verbose(verbose, layer_sizes, model, neurons_by_layer_arch, nlayers, consec_layers_neurons_ratio, activation_function, use_batchnorm, use_layernorm, use_layernorm_per_layer, groupnorm_num_groups, weights_init_fcn, spectral_norm, model_type, inputs_dropout_prob, dropout_prob)

    _generate_mlp_weights_init_fcn(weights_init_fcn, model)

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
    _generate_mlp_every_fit_bench_profiling(model)

    # example_input_array MUST reflect the USER-facing input shape
    # (raw features), NOT the embedded shape — the model accepts raw
    # X via the embedding layer at position 0. layer_sizes[0] preserves
    # the original raw num_features even when numerical_embedding overrode
    # num_features for downstream construction.
    model.example_input_array = torch.zeros(1, layer_sizes[0])

    return model


# MLPTorchModel carved to ``_flat_torch_module``; re-exported below.
from ._flat_torch_module import MLPTorchModel  # noqa: F401
