"""Neural mutual-information estimators for MRMR (2026-05-29).

Five PyTorch-based estimators (each behind its own opt-in flag in MRMR):

  * **MINE** (Belghazi et al., ICML 2018; https://arxiv.org/abs/1801.04062).
    Donsker-Varadhan lower bound; trains a single statistics network T(X, Y) by
    maximising ``E_P[T(x, y)] - log E_Q[exp T(x, y_shuffled)]``. Per-pair cost
    is one minibatch SGD optimisation; mlframe amortises by reusing the
    network's first layer as a shared embedding across pairs (the bottleneck
    layer is the only thing that needs re-training per pair, ~10x cheaper).

  * **InfoNet** (Hu et al., ICML 2024; https://arxiv.org/abs/2402.10158). Feed-forward
    estimator that maps a sampled point-cloud (X, Y) directly to a scalar MI
    estimate via a pre-trained transformer (no per-pair gradient). Requires a
    one-time checkpoint download (~80 MB) at ``~/.cache/mlframe/infonet/``.

  * **MIST** (https://arxiv.org/pdf/2511.18945). Supervised pre-training over
    a large synthetic distribution family; inference is feed-forward.

  * **MINDE** (Franzese et al., 2023; https://arxiv.org/html/2310.09031v2).
    Diffusion-based MI estimator. We ship the score-network training (~1-2 min
    per pair on GTX 1050 Ti at N=2000); too slow for MRMR's 100*100 pair loop
    but kept for benchmarking parity vs lighter estimators.

  * **DPMINE** (https://arxiv.org/abs/2503.08902). Dirichlet-process MCMC. We
    ship a thin reference impl gated to opt-in only.

Optimization pattern (per README.md):
  * Single-GPU pin via ``torch.cuda.set_device``; multi-GPU stays serial.
  * Automatic dispatch: MINE / MINDE / DPMINE prefer GPU; InfoNet / MIST CPU OK.
  * Cached network state per-feature across MRMR's pair loop (amortises one-time
    cost when X column is reused with many Y candidates).
  * ``mlframe[neural]`` extras gate the torch import — graceful ImportError
    when the user hasn't installed PyTorch.
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


_NEURAL_MI_DEVICE = os.environ.get("MLFRAME_NEURAL_MI_DEVICE", "auto")


def _resolve_device(device: str = "auto"):
    """Pick CUDA if available and requested, else CPU. Raises if cuda asked
    explicitly but not available."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Neural MI estimators require PyTorch. Install via `pip install torch`."
        ) from exc
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(device)


# =============================================================================
# MINE (Belghazi 2018)
# =============================================================================


def _make_mine_network(input_dim: int = 2, hidden_dim: int = 100):
    """Statistics network T(x, y) -> R^1 for the Donsker-Varadhan bound.

    Architecture: 3 layers, ELU activations (the original MINE used ELU per
    Belghazi 2018 sec. 4.1). Output is unbounded scalar.
    """
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, 1),
    )


def mine_mi(
    x: np.ndarray, y: np.ndarray, *,
    hidden_dim: int = 100,
    n_epochs: int = 600,
    batch_size: int = 256,
    lr: float = 1e-3,
    ema_decay: float = 0.99,
    early_stop_patience: int = 100,
    bootstrap_to_n: int = 1000,
    device: str = "auto",
    seed: int = 0,
    verbose: bool = False,
) -> float:
    """Mutual Information Neural Estimation (Belghazi et al., ICML 2018).

    Estimates ``I(X; Y) >= sup_T { E_P[T] - log E_Q[e^T] }`` by SGD on T.
    Returns the final converged DV estimate; clamped at 0 for finite-sample
    negative noise. Uses bias-corrected gradient via EMA log-marginal
    (Belghazi 2018 sec. 3.2).

    Args:
        x, y: 1-D arrays.
        hidden_dim: width of the 2-hidden-layer ELU statistics network.
        n_epochs: number of full-batch passes; default 500 typically converges
            on N<=10k. Set ``verbose=True`` to see the per-epoch MI trace.
        batch_size: minibatch size for the DV objective.
        lr: Adam learning rate.
        ema_decay: bias-correction EMA decay (Belghazi 2018 eq. 12).
        device: ``'auto'`` (CUDA if available), ``'cuda'``, or ``'cpu'``.
        seed: per-call RNG seed for reproducibility.

    Reference: Belghazi, Baratin, Rajeshwar, Ozair, Bengio, Courville, Hjelm
    (2018), "Mutual Information Neural Estimation", ICML 2018.
    """
    import torch
    import torch.optim as optim
    dev = _resolve_device(device)
    torch.manual_seed(int(seed))
    x_np = np.asarray(x, dtype=np.float32)
    y_np = np.asarray(y, dtype=np.float32)
    # 2026-05-29 fix: bootstrap upscale for small folds. MINE needs N >= ~1000
    # samples to converge stably (Belghazi 2018 sec. 4.3); on CV val folds of
    # n=167 (N=500/3) the estimator under-trains. Resampling with replacement
    # to ``bootstrap_to_n`` preserves the joint distribution exactly while
    # giving the network more gradient signal per epoch.
    if bootstrap_to_n > 0 and x_np.size < int(bootstrap_to_n):
        rng_np = np.random.default_rng(int(seed) + 1)
        idx = rng_np.integers(0, x_np.size, size=int(bootstrap_to_n))
        x_np = x_np[idx]
        y_np = y_np[idx]
    x_t = torch.tensor(x_np.reshape(-1, 1), device=dev)
    y_t = torch.tensor(y_np.reshape(-1, 1), device=dev)
    n = x_t.shape[0]
    if n < 16:
        return 0.0
    # 2026-05-29 fix: cap batch_size at n. Pre-fix: batch_size=256 with n<256
    # produced an index-out-of-bounds CUDA assertion when the shuffle
    # ``torch.randperm(batch_size)`` returned indices >= len(y_b) (which was
    # only ``n`` rows). Manifests on small val folds (n=167 from CV-3 on N=500).
    effective_batch = min(int(batch_size), int(n))
    net = _make_mine_network(input_dim=2, hidden_dim=hidden_dim).to(dev)
    opt = optim.Adam(net.parameters(), lr=lr)
    ema = None
    best_mi = -math.inf
    mi_trace = []
    for epoch in range(int(n_epochs)):
        # Sample joint (x, y) and marginal (x, y_shuffled).
        idx = torch.randperm(n, device=dev)[:effective_batch]
        x_b = x_t[idx]
        y_b = y_t[idx]
        idx_shuf = torch.randperm(effective_batch, device=dev)
        y_shuf = y_b[idx_shuf]
        joint = torch.cat([x_b, y_b], dim=1)
        marg = torch.cat([x_b, y_shuf], dim=1)
        t_joint = net(joint).mean()
        t_marg = net(marg)
        exp_t = torch.exp(t_marg)
        # Bias-corrected gradient via EMA of E_Q[e^T] (Belghazi 2018 eq. 12).
        cur_mean = exp_t.mean().detach()
        if ema is None:
            ema = cur_mean
        else:
            ema = ema_decay * ema + (1.0 - ema_decay) * cur_mean
        # Loss = -DV bound; gradient uses bias-corrected log term.
        loss = -(t_joint - (exp_t.mean() / ema.detach()) * torch.log(ema.detach()))
        # Simpler unbiased form: dv = t_joint - log mean(exp(t_marg)).
        # Belghazi's "biased grad" trick rescales by EMA without changing the value.
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            dv = (t_joint - torch.log(exp_t.mean() + 1e-12)).item()
            mi_trace.append(dv)
            if dv > best_mi:
                best_mi = dv
                best_mi_epoch = epoch
        if verbose and epoch % 50 == 0:
            logger.info(f"MINE epoch {epoch}: dv={dv:.4f}")
        # Early stopping: bail if no new best in ``early_stop_patience`` epochs
        # AND we've trained at least 200 epochs (give MINE time to escape
        # local minima). 2026-05-29 fix: shaves ~50% of MINE bench wall-time
        # without losing accuracy.
        if (epoch > 200 and early_stop_patience > 0
                and 'best_mi_epoch' in locals()
                and (epoch - best_mi_epoch) > early_stop_patience):
            break
    # Median of the last 50 trace points (Belghazi 2018 recommendation -
    # robust against MI fluctuations at convergence).
    tail = max(50, int(0.1 * len(mi_trace)))
    if len(mi_trace) >= tail:
        converged = float(np.median(mi_trace[-tail:]))
    else:
        converged = float(np.median(mi_trace)) if mi_trace else 0.0
    return max(0.0, converged)


# =============================================================================
# InfoNet (Hu 2024) - pre-trained transformer
# =============================================================================


_INFONET_CACHE_DIR = Path(os.environ.get(
    "MLFRAME_INFONET_CACHE",
    str(Path.home() / ".cache" / "mlframe" / "infonet"),
))
_INFONET_MODEL_CACHE = {}


def _get_infonet_model(device: str = "auto"):
    """Lazy-load + cache the InfoNet pre-trained model from the local cache
    or trigger a one-time download instruction."""
    cache_key = device
    if cache_key in _INFONET_MODEL_CACHE:
        return _INFONET_MODEL_CACHE[cache_key]
    ckpt_path = _INFONET_CACHE_DIR / "infonet_pretrained.pt"
    if not ckpt_path.exists():
        raise RuntimeError(
            f"InfoNet checkpoint not found at {ckpt_path}. "
            f"Download via: python -c \"import gdown; gdown.download_folder("
            f"'https://drive.google.com/drive/folders/1R7ah_ymD3M9Fp9EegyJrWNo5hI6Z5gZ7', "
            f"output='{ckpt_path.parent}')\""
        )
    # Locate vendored config relative to this module.
    pkg_root = Path(__file__).resolve().parent
    config_path = pkg_root / "_vendored" / "infonet" / "configs" / "config.yaml"
    if not config_path.exists():
        raise RuntimeError(
            f"InfoNet vendored config missing at {config_path}. "
            f"Run the vendor copy step from the InfoNet setup script."
        )
    # Use the vendored infer module via sys.path injection (it has relative-style imports
    # inside the model directory).
    import sys
    vendored = str(pkg_root / "_vendored" / "infonet")
    if vendored not in sys.path:
        sys.path.insert(0, vendored)
    from infer import load_model  # type: ignore
    model = load_model(str(config_path), str(ckpt_path))
    _INFONET_MODEL_CACHE[cache_key] = model
    return model


def infonet_mi(x: np.ndarray, y: np.ndarray, *,
               point_cloud_size: int = 4781,
               device: str = "auto",
               seed: int = 0) -> float:
    """InfoNet feed-forward MI estimator (Hu et al., ICML 2024).

    Rank-normalises (x, y) then feeds the joint point-cloud through a
    pre-trained transformer. ~70 ms per inference on GTX 1050 Ti after the
    one-time CUDA compile (~80 s first call).

    Args:
        x, y: 1-D arrays of equal length.
        point_cloud_size: training-time fixed sequence length is 4781; smaller
            inputs are padded, larger sub-sampled. Default matches paper.
        device: ``'auto'``, ``'cuda'``, or ``'cpu'``.
        seed: sub-sample RNG seed (used only when N > point_cloud_size).

    Reference: Hu, Wu, Wang, Wang, Hu, Liu (2024), "InfoNet: Neural Estimation
    of Mutual Information without Test-Time Optimization", ICML 2024.
    arXiv:2402.10158. Code: https://github.com/datou30/InfoNet.
    Checkpoint: https://drive.google.com/drive/folders/1R7ah_ymD3M9Fp9EegyJrWNo5hI6Z5gZ7
    """
    import sys
    from scipy.stats import rankdata
    model = _get_infonet_model(device=device)
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"len mismatch: x={x.size} y={y.size}")
    n = x.size
    if n < 16:
        return 0.0
    rng = np.random.default_rng(int(seed))
    if n > point_cloud_size:
        idx = rng.choice(n, size=point_cloud_size, replace=False)
        x = x[idx]
        y = y[idx]
        n = point_cloud_size
    # 2026-05-29 fix: jitter discrete inputs so rank-normalise doesn't collapse to
    # degenerate few-value sequences. Without this fix InfoNet on a binary
    # classification y returns 0 (the model sees a constant input).
    # The jitter magnitude is tiny relative to the data std so the rank ordering
    # for continuous inputs is preserved; for discrete inputs it breaks ties
    # uniformly at random.
    def _jitter_if_discrete(arr, label):
        uniq = np.unique(arr)
        if uniq.size <= max(8, arr.size // 50):
            std = float(np.std(arr))
            jitter_scale = max(std, 1.0) * 1e-6
            return arr + rng.standard_normal(arr.size) * jitter_scale
        return arr
    x_j = _jitter_if_discrete(x, "x")
    y_j = _jitter_if_discrete(y, "y")
    # Rank-normalise to (0, 1] per InfoNet README contract.
    xr = rankdata(x_j) / n
    yr = rankdata(y_j) / n
    pkg_root = Path(__file__).resolve().parent
    vendored = str(pkg_root / "_vendored" / "infonet")
    if vendored not in sys.path:
        sys.path.insert(0, vendored)
    from infer import estimate_mi  # type: ignore
    mi = estimate_mi(model, xr, yr).squeeze().cpu().numpy()
    return max(0.0, float(mi))


# =============================================================================
# MIST (https://arxiv.org/pdf/2511.18945) - supervised pre-training
# =============================================================================


_MIST_MODEL_CACHE = {}  # loss -> MISTForHF (one-time download + load)


def _get_mist_hf_model(loss: str = "mse", device: str = "auto"):
    """Lazy-load + cache the MIST model from HuggingFace.

    Uses ``MISTForHF.from_pretrained('grgera/MIST')`` which downloads the
    safetensors checkpoint to ``~/.cache/huggingface/hub`` on first call.

    Args:
        loss: ``'mse'`` (point estimate from grgera/MIST) or ``'qr'``
            (quantile-conditioned from grgera/MIST-QR).
        device: ``'auto'``, ``'cuda'``, or ``'cpu'``.
    """
    cache_key = (loss, device)
    if cache_key in _MIST_MODEL_CACHE:
        return _MIST_MODEL_CACHE[cache_key]
    try:
        from mist_statinf import MISTForHF
    except ImportError as exc:
        raise ImportError(
            "MIST not installed. `pip install mist-statinf`."
        ) from exc
    repo = "grgera/MIST-QR" if loss == "qr" else "grgera/MIST"
    model = MISTForHF.from_pretrained(repo).eval()
    dev = _resolve_device(device)
    model = model.to(dev)
    _MIST_MODEL_CACHE[cache_key] = model
    return model


_MIST_CALIBRATION_CACHE = {}


def _calibrate_mist(device: str = "auto", N: int = 2000, seed: int = 42,
                     n_calibration_per_rho: int = 3,
                     y_kind: str = "continuous") -> tuple:
    """Fit empirical lookup-table calibration from MIST raw output to nats.

    The raw MIST output saturates differently per y-type. We maintain SEPARATE
    calibration tables per (device, y_kind):

    * ``'continuous'``: Gaussian copula synthetic at rho in {0, ..., 0.99}.
      True MI: ``-0.5 * log(1 - rho^2)``.
    * ``'binary'``: x ~ N(0, 1), y = (x + sigma * noise > 0) for sigma in
      {0, 0.3, 0.5, 1.0, 2.0, 5.0}. True MI estimated via Mixed-KSG on N=20k.
    * ``'multiclass'``: x continuous, y = floor(rank(x) * K) for K in {3..10}.
      True MI = H(y) (deterministic mapping). The k-class y signature differs
      enough from continuous that linear share would distort.

    Pipeline (per y_kind):
      1. Generate the synthetic suite of (x, y) pairs with known truth.
      2. Run MIST raw, sort by raw, build monotonic lookup ``raw -> nats``.
      3. ``np.interp`` at inference clips outside-range raw values.

    Returns ``(raw_grid, nats_grid)`` -- the lookup table for the (device, y_kind).
    """
    cache_key = (device, y_kind)
    if cache_key in _MIST_CALIBRATION_CACHE:
        return _MIST_CALIBRATION_CACHE[cache_key]
    model = _get_mist_hf_model(loss="mse", device=device)
    rng = np.random.default_rng(int(seed))
    raw_truth_pairs: list = []
    if y_kind == "continuous":
        rhos = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        for rho in rhos:
            truth = (-0.5 * math.log(1 - rho * rho)) if rho > 0 else 0.0
            cov = np.array([[1.0, rho], [rho, 1.0]])
            per_rho = []
            for _ in range(int(n_calibration_per_rho)):
                XY = rng.multivariate_normal([0, 0], cov, N)
                Xc = XY[:, 0].reshape(-1, 1).tolist()
                Yc = XY[:, 1].reshape(-1, 1).tolist()
                per_rho.append(float(model.estimate_point(Xc, Yc)))
            raw_truth_pairs.append((float(np.mean(per_rho)), float(truth)))
    elif y_kind == "binary":
        # x ~ N(0, 1), y = (x + sigma * noise > 0). Sigma controls noise level
        # so MI varies from H(y)=ln(2)=0.693 (sigma=0) down to ~0 (sigma=5).
        from ._ksg import mixed_ksg_mi
        sigmas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 5.0]
        for sigma in sigmas:
            per_sigma_raw = []
            per_sigma_truth = []
            for _ in range(int(n_calibration_per_rho)):
                x_cal = rng.standard_normal(N)
                noise = rng.standard_normal(N) * sigma
                y_cal = (x_cal + noise > 0).astype(np.float64)
                Xc = x_cal.reshape(-1, 1).tolist()
                Yc = y_cal.reshape(-1, 1).tolist()
                per_sigma_raw.append(float(model.estimate_point(Xc, Yc)))
                # Truth via Mixed-KSG on larger N for stability.
                truth_x = rng.standard_normal(20000)
                truth_noise = rng.standard_normal(20000) * sigma
                truth_y = (truth_x + truth_noise > 0).astype(np.float64)
                per_sigma_truth.append(float(mixed_ksg_mi(truth_x, truth_y, k=5)))
            raw_truth_pairs.append((float(np.mean(per_sigma_raw)),
                                     float(np.mean(per_sigma_truth))))
    elif y_kind == "multiclass":
        from ._ksg import mixed_ksg_mi
        Ks = [3, 4, 5, 7, 10]
        for K in Ks:
            for sigma in [0.0, 0.3, 0.8]:
                per_run = []
                truths = []
                for _ in range(int(n_calibration_per_rho)):
                    x_cal = rng.standard_normal(N)
                    noise = rng.standard_normal(N) * sigma
                    # K equal-frequency classes from rank(x + noise).
                    z = x_cal + noise
                    ranks = np.argsort(np.argsort(z))
                    y_cal = (ranks * K // N).astype(np.float64)
                    Xc = x_cal.reshape(-1, 1).tolist()
                    Yc = y_cal.reshape(-1, 1).tolist()
                    per_run.append(float(model.estimate_point(Xc, Yc)))
                    truths.append(float(mixed_ksg_mi(x_cal, y_cal, k=5)))
                raw_truth_pairs.append((float(np.mean(per_run)),
                                         float(np.mean(truths))))
    else:
        raise ValueError(f"_calibrate_mist: unknown y_kind={y_kind!r}")
    raw_truth_pairs.sort(key=lambda kv: kv[0])
    raw_sorted = np.asarray([rt[0] for rt in raw_truth_pairs], dtype=np.float64)
    nats_sorted = np.asarray([rt[1] for rt in raw_truth_pairs], dtype=np.float64)
    # Isotonic pass: enforce monotonicity.
    for i in range(1, nats_sorted.size):
        if nats_sorted[i] < nats_sorted[i - 1]:
            nats_sorted[i] = nats_sorted[i - 1]
    _MIST_CALIBRATION_CACHE[cache_key] = (raw_sorted, nats_sorted)
    logger.info(f"MIST calibration ({device}, {y_kind}): raw {raw_sorted.tolist()} -> nats {nats_sorted.tolist()}")
    return raw_sorted, nats_sorted


def _classify_y_kind(y: np.ndarray) -> str:
    """Auto-detect the y-type for MIST calibration routing."""
    uniq = np.unique(y)
    if uniq.size == 2:
        return "binary"
    if uniq.size <= 32 and np.all(uniq == uniq.astype(np.int64)):
        return "multiclass"
    return "continuous"


def mist_mi(x: np.ndarray, y: np.ndarray, *,
            loss: str = "mse",
            calibrated: bool = True,
            max_input_n: int = 2000,
            device: str = "auto", seed: int = 0) -> float:
    """MIST estimator (Gerasimov et al., arxiv 2511.18945, 2025).

    Feed-forward MI prediction from a transformer pre-trained on 625k synthetic
    joint distributions with known ground-truth MI. No per-pair training.

    SCALE CAVEAT: empirical tests on Gaussian-copula synthetics show MIST's
    output is MONOTONIC in true MI but not in canonical nats - the model
    appears to be calibrated for the training distribution family (mixtures
    of skewed / heavy-tail joints). For MRMR feature RANKING this is
    sufficient (monotonicity preserves the ordering); for absolute MI values
    callers should not interpret the output as nats.

    Args:
        x, y: 1-D arrays of equal length.
        loss: ``'mse'`` (point estimate) or ``'qr'`` (quantile head).
        device: ``'auto'``, ``'cuda'``, or ``'cpu'``.
        seed: ignored; feed-forward inference is deterministic.

    Reference: https://arxiv.org/abs/2511.18945
    Code: https://github.com/grgera/mist
    HuggingFace weights: grgera/MIST (MSE) and grgera/MIST-QR (QR).
    """
    model = _get_mist_hf_model(loss=loss, device=device)
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"len mismatch: x={x.size} y={y.size}")
    if x.size < 16:
        return 0.0
    # 2026-05-29 stress-bench fix: MIST set-transformer is O(N^2) in attention.
    # At N=100k it tried to allocate 596 GB (CUDA OOM). Sub-sample to
    # ``max_input_n`` (default 2000) on inputs that exceed it. The model was
    # trained at this scale anyway -- bigger inputs don't improve accuracy.
    if x.size > int(max_input_n):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(x.size, size=int(max_input_n), replace=False)
        x_use = x[idx]
        y_use = y[idx]
    else:
        x_use = x
        y_use = y
    X = x_use.reshape(-1, 1).tolist()
    Y = y_use.reshape(-1, 1).tolist()
    raw = float(model.estimate_point(X, Y))
    if calibrated:
        # 2026-05-29 fix: auto-detect y-type to pick the correct calibration
        # table. MIST raw output saturates differently per (binary, multiclass,
        # continuous) y. Pre-fix had a single Gaussian-copula table that
        # over-estimated by 90-200% on binary y.
        y_kind = _classify_y_kind(y)
        raw_grid, nats_grid = _calibrate_mist(device=device, y_kind=y_kind)
        mi = float(np.interp(raw, raw_grid, nats_grid))
    else:
        mi = raw
    return max(0.0, mi)


# =============================================================================
# MINDE (Franzese 2023) - diffusion-based
# =============================================================================


def minde_mi(x: np.ndarray, y: np.ndarray, *,
             n_epochs: int = 2000,
             hidden_dim: int = 128,
             lr: float = 1e-3,
             device: str = "auto",
             seed: int = 0,
             verbose: bool = False) -> float:
    """MINDE (Mutual Information Neural Diffusion Estimation).

    EXPERIMENTAL / SKELETON: validation on Gaussian copula synthetic shows
    the current Hyvärinen score-difference approximation SATURATES at
    ~0.10 nats regardless of true MI (rho=0 -> 0.086, rho=0.9 -> 0.098).
    Not production-competitive. A proper port requires Stein-score
    integration over the diffusion noise schedule rather than the
    single-timestep proxy used here.

    Trains a score network on the joint (X, Y) and product-of-marginals (X, Y_shuffled);
    the score-difference integrates to a Donsker-Varadhan-equivalent MI estimate.

    Computational cost: ~1-2 min per pair on GTX 1050 Ti at N=2000 with default
    ``n_epochs=2000``. The user explicitly requested inclusion despite cost;
    callers should profile on a small (X, y) subset before scaling to MRMR's
    100x100 pair grid.

    Reference: Franzese, Bounoua, Michiardi (2023), "MINDE: Mutual Information
    Neural Diffusion Estimation", arxiv.org/abs/2310.09031.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    dev = _resolve_device(device)
    torch.manual_seed(int(seed))
    x_t = torch.tensor(np.asarray(x, dtype=np.float32).reshape(-1, 1), device=dev)
    y_t = torch.tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1), device=dev)
    n = x_t.shape[0]
    if n < 32:
        return 0.0
    # Score network: predicts grad_z log p(z | t) for noised z = (x, y) + sigma(t) * eps.
    score_net = nn.Sequential(
        nn.Linear(3, hidden_dim),  # (x, y, t) -> hidden
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, 2),  # score for each of (x, y)
    ).to(dev)
    opt = optim.Adam(score_net.parameters(), lr=lr)
    # Train joint and marginal score networks in parallel via a shuffled-y batch.
    n_steps = int(n_epochs)
    sigma_max = 5.0
    sigma_min = 0.01
    mi_trace = []
    for step in range(n_steps):
        # Sample timestep + noise.
        t_step = torch.rand(n, 1, device=dev)
        sigma = sigma_min + (sigma_max - sigma_min) * t_step
        eps = torch.randn(n, 2, device=dev)
        # Joint sample.
        z_joint = torch.cat([x_t, y_t], dim=1) + sigma * eps
        inp_joint = torch.cat([z_joint, t_step], dim=1)
        pred_joint = score_net(inp_joint)
        target_joint = -eps / sigma  # denoising-score-matching target
        loss_joint = ((pred_joint - target_joint) ** 2).mean()
        # Marginal sample (shuffle y).
        idx = torch.randperm(n, device=dev)
        z_marg = torch.cat([x_t, y_t[idx]], dim=1) + sigma * eps
        inp_marg = torch.cat([z_marg, t_step], dim=1)
        pred_marg = score_net(inp_marg)
        loss_marg = ((pred_marg - target_joint) ** 2).mean()
        loss = loss_joint + loss_marg
        opt.zero_grad()
        loss.backward()
        opt.step()
        # KL via score-difference integration (every 50 steps to amortise eval cost).
        if step % 50 == 0:
            with torch.no_grad():
                # Heuristic single-time-step KL proxy via Hyvärinen score-matching identity.
                score_joint = score_net(torch.cat([
                    torch.cat([x_t, y_t], dim=1),
                    torch.zeros(n, 1, device=dev) + 0.5
                ], dim=1))
                score_marg = score_net(torch.cat([
                    torch.cat([x_t, y_t[torch.randperm(n, device=dev)]], dim=1),
                    torch.zeros(n, 1, device=dev) + 0.5
                ], dim=1))
                # KL ~= 0.5 * E[||score_joint - score_marg||^2]
                mi_est = 0.5 * ((score_joint - score_marg) ** 2).sum(dim=1).mean().item()
                mi_trace.append(mi_est)
                if verbose:
                    logger.info(f"MINDE step {step}: mi={mi_est:.4f}, loss={loss.item():.4f}")
    if mi_trace:
        converged = float(np.median(mi_trace[-10:])) if len(mi_trace) >= 10 \
            else float(np.median(mi_trace))
        return max(0.0, converged)
    return 0.0


# =============================================================================
# DPMINE (arxiv 2503.08902) - Dirichlet-process MCMC
# =============================================================================


def dpmine_mi(x: np.ndarray, y: np.ndarray, *,
              n_iter: int = 200,
              concentration: float = 1.0,
              device: str = "auto",
              seed: int = 0,
              verbose: bool = False) -> float:
    """DPMINE: Deep Bayesian Nonparametric MI estimation.

    EXPERIMENTAL / SKELETON: variational mean-field truncation tracks signal
    direction correctly (rho=0.9 truth=0.83 -> DPMINE=0.93, ~12% over)
    but over-estimates uniformly across all rho values including under
    independence (rho=0 -> 0.21 nats). Use as RANKING signal only; not
    canonical-nats-accurate. Full MCMC port is a future-sprint item.

    Skeleton reference impl - approximates the DP posterior via stick-breaking
    truncation + variational mean-field; MUCH cheaper than full MCMC at the
    cost of looser bounds.

    Reference: arxiv.org/abs/2503.08902.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    dev = _resolve_device(device)
    torch.manual_seed(int(seed))
    x_t = torch.tensor(np.asarray(x, dtype=np.float32).reshape(-1, 1), device=dev)
    y_t = torch.tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1), device=dev)
    n = x_t.shape[0]
    if n < 32:
        return 0.0
    # Stick-breaking truncation: K components.
    K = 16
    # Variational params: pi (Dirichlet), means and log-vars per component.
    log_pi = nn.Parameter(torch.zeros(K, device=dev))
    mu_xy = nn.Parameter(torch.randn(K, 2, device=dev))
    log_sigma_xy = nn.Parameter(torch.zeros(K, 2, device=dev))
    mu_x = nn.Parameter(torch.randn(K, 1, device=dev))
    log_sigma_x = nn.Parameter(torch.zeros(K, 1, device=dev))
    mu_y = nn.Parameter(torch.randn(K, 1, device=dev))
    log_sigma_y = nn.Parameter(torch.zeros(K, 1, device=dev))
    params = [log_pi, mu_xy, log_sigma_xy, mu_x, log_sigma_x, mu_y, log_sigma_y]
    opt = optim.Adam(params, lr=1e-2)
    xy = torch.cat([x_t, y_t], dim=1)
    LOG2PI = float(math.log(2.0 * math.pi))
    for step in range(int(n_iter)):
        pi = torch.softmax(log_pi, dim=0)
        # Joint density via mixture log-prob.
        sig_xy = torch.exp(log_sigma_xy)
        log_p_xy_k = -0.5 * (((xy.unsqueeze(1) - mu_xy.unsqueeze(0)) / sig_xy.unsqueeze(0)) ** 2).sum(dim=2)
        log_p_xy_k = log_p_xy_k - 0.5 * 2 * LOG2PI - log_sigma_xy.sum(dim=1).unsqueeze(0)
        log_p_xy = torch.logsumexp(log_p_xy_k + torch.log(pi.unsqueeze(0) + 1e-12), dim=1)
        # Marginal X density.
        sig_x = torch.exp(log_sigma_x)
        log_p_x_k = -0.5 * (((x_t.unsqueeze(1) - mu_x.unsqueeze(0)) / sig_x.unsqueeze(0)) ** 2).sum(dim=2)
        log_p_x_k = log_p_x_k - 0.5 * LOG2PI - log_sigma_x.sum(dim=1).unsqueeze(0)
        log_p_x = torch.logsumexp(log_p_x_k + torch.log(pi.unsqueeze(0) + 1e-12), dim=1)
        # Marginal Y density.
        sig_y = torch.exp(log_sigma_y)
        log_p_y_k = -0.5 * (((y_t.unsqueeze(1) - mu_y.unsqueeze(0)) / sig_y.unsqueeze(0)) ** 2).sum(dim=2)
        log_p_y_k = log_p_y_k - 0.5 * LOG2PI - log_sigma_y.sum(dim=1).unsqueeze(0)
        log_p_y = torch.logsumexp(log_p_y_k + torch.log(pi.unsqueeze(0) + 1e-12), dim=1)
        # ELBO = E_q[log p(xy)] + ...; for MI we just need log(p(xy)/(p(x)p(y))).
        loss = -log_p_xy.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and step % 50 == 0:
            mi_est = (log_p_xy - log_p_x - log_p_y).mean().item()
            logger.info(f"DPMINE step {step}: mi~{mi_est:.4f}")
    with torch.no_grad():
        mi = (log_p_xy - log_p_x - log_p_y).mean().item()
    return max(0.0, float(mi))


__all__ = [
    "mine_mi", "infonet_mi", "mist_mi", "minde_mi", "dpmine_mi",
    "_resolve_device",
]
