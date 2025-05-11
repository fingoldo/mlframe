# ----------------------------------------------------------------------------------------------------------------------------
# Numba
# ----------------------------------------------------------------------------------------------------------------------------

import numpy as np
from numba import njit, prange

USE_FASTMATH: bool = True

# ----------------------------------------------------------------------------------------------------------------------------
# GROK
# ----------------------------------------------------------------------------------------------------------------------------


@njit(fastmath=USE_FASTMATH)
def grok_compute_joint_hist(a: np.ndarray, b: np.ndarray, n_bins: int, dtype: object = np.int64):
    hist = np.zeros((n_bins, n_bins), dtype=dtype)
    for i in range(len(a)):
        hist[a[i], b[i]] += 1
    return hist


@njit(fastmath=USE_FASTMATH)
def grok_mutual_information_old(a: np.ndarray, b: np.ndarray, n_bins: int = 15, hist_dtype: object = np.int64):
    joint_hist = grok_compute_joint_hist(a=a, b=b, n_bins=n_bins, dtype=hist_dtype)
    a_hist = np.sum(joint_hist, axis=1)
    b_hist = np.sum(joint_hist, axis=0)
    n_samples = len(a)
    mi = 0.0
    for x in range(n_bins):
        for y in range(n_bins):
            if joint_hist[x, y] > 0:
                p_joint = joint_hist[x, y] / n_samples
                p_a = a_hist[x] / n_samples
                p_b = b_hist[y] / n_samples
                mi += p_joint * np.log(p_joint / (p_a * p_b))
    return mi


@njit(fastmath=USE_FASTMATH)
def grok_mutual_information(a: np.ndarray, b: np.ndarray, inv_n_samples: float, log_n_samples: float, n_bins: int = 15, hist_dtype: object = np.int64):
    joint_hist = grok_compute_joint_hist(a=a, b=b, n_bins=n_bins, dtype=hist_dtype)
    a_hist = np.sum(joint_hist, axis=1)
    b_hist = np.sum(joint_hist, axis=0)
    mi = 0.0
    for x in range(n_bins):
        for y in range(n_bins):
            if joint_hist[x, y] > 0:
                joint_count = joint_hist[x, y]
                p_joint = joint_count * inv_n_samples
                log_term = np.log(joint_count) - np.log(a_hist[x]) - np.log(b_hist[y]) + log_n_samples
                mi += p_joint * log_term
    return mi


@njit(parallel=True)
def grok_compute_mutual_information(
    data: np.ndarray, target_indices: np.ndarray | list[int], n_bins: int = 15, hist_dtype=np.int64, out_dtype=np.float64
) -> np.ndarray:
    """
    MI of every specified target column against every column in `data`.

    Parameters
    ----------
    data            : int8 ndarray, shape (n_samples, n_cols), already binned 0-14
    target_indices  : iterable of int column indices
    n_bins          : number of discrete bins (default 15)

    Returns
    -------
    mi_matrix       : float64 ndarray, shape (n_targets, n_cols)
                      Row k = MI(target_indices[k], all columns)
    """

    n_samples, n_columns = data.shape
    K = len(target_indices)
    mi_results = np.zeros((K, n_columns), dtype=out_dtype)

    inv_n_samples = 1.0 / n_samples
    log_n_samples = np.log(n_samples)

    for t in range(K):
        target = target_indices[t]
        target_col = data[:, target]
        for j in prange(n_columns):
            if j != target:
                mi_results[t, j] = grok_mutual_information(
                    target_col, data[:, j], n_bins=n_bins, inv_n_samples=inv_n_samples, log_n_samples=log_n_samples, hist_dtype=hist_dtype
                )
            else:
                mi_results[t, j] = np.nan
    return mi_results


# ----------------------------------------------------------------------------------------------------------------------------
# ChatGPT
# ----------------------------------------------------------------------------------------------------------------------------


# Single-pair MI (15 discrete bins, natural-log base)
@njit(fastmath=USE_FASTMATH)
def _chatgpt_mi_pair(x: np.ndarray, y: np.ndarray, n_bins: int = 15, hist_dtype=np.int64) -> float:
    """Mutual information between two 1-D int8 vectors already binned to 0..n_bins-1."""

    # 1) joint counts

    joint = np.zeros((n_bins, n_bins), dtype=hist_dtype)
    for k in range(x.size):
        joint[x[k], y[k]] += 1

    # 2) marginals

    row = np.zeros(n_bins, dtype=hist_dtype)  # P(x)
    col = np.zeros(n_bins, dtype=hist_dtype)  # P(y)
    for i in range(n_bins):
        for j in range(n_bins):
            c = joint[i, j]
            row[i] += c
            col[j] += c

    # 3) MI

    N = x.size
    mi = 0.0
    for i in range(n_bins):
        if row[i] == 0:
            continue
        p_i = row[i] / N
        for j in range(n_bins):
            c = joint[i, j]
            if c == 0 or col[j] == 0:
                continue
            p_ij = c / N
            p_j = col[j] / N
            mi += p_ij * np.log(p_ij / (p_i * p_j))
    return mi


# All features vs. one target (parallel over the wide axis)
@njit(parallel=True, fastmath=USE_FASTMATH)
def _chatgpt_mi_one_target(
    data: np.ndarray, target_idx: int, n_bins: int = 15, hist_dtype=np.int64, out_dtype=np.float64
) -> np.ndarray:  # shape (n_samples, n_cols), int8
    """Vector of MI(target, every feature)."""
    n_rows, n_cols = data.shape
    y = data[:, target_idx]
    out = np.empty(n_cols, dtype=out_dtype)

    # Parallel loop across *features* – this is the expensive axis.
    for c in prange(n_cols):
        out[c] = _chatgpt_mi_pair(data[:, c], y, n_bins, hist_dtype=hist_dtype)

    return out


# Public API: many targets vs. all features
def chatgpt_compute_mutual_information(
    data: np.ndarray, target_indices: np.ndarray | list[int], n_bins: int = 15, hist_dtype=np.int64, out_dtype=np.float64
) -> np.ndarray:
    """
    MI of every specified target column against every column in `data`.

    Parameters
    ----------
    data            : int8 ndarray, shape (n_samples, n_cols), already binned 0-14
    target_indices  : iterable of int column indices
    n_bins          : number of discrete bins (default 15)

    Returns
    -------
    mi_matrix       : float64 ndarray, shape (n_targets, n_cols)
                      Row k = MI(target_indices[k], all columns)
    """
    # Safety – make sure the array is C-contiguous int8 for maximum speed.
    if data.dtype != np.int8 or not data.flags.c_contiguous:
        data = np.ascontiguousarray(data, dtype=np.int8)

    targets = np.asarray(target_indices, dtype=np.int64)
    out = np.empty((targets.size, data.shape[1]), dtype=out_dtype)

    # Few targets, many features ⇒ parallel inside _mi_one_target
    for k, t in enumerate(targets):
        out[k, :] = _chatgpt_mi_one_target(data=data, target_idx=int(t), n_bins=n_bins, hist_dtype=hist_dtype, out_dtype=out_dtype)

    return out


# ----------------------------------------------------------------------------------------------------------------------------
# DeepSeek
# ----------------------------------------------------------------------------------------------------------------------------


@njit(parallel=True, fastmath=USE_FASTMATH)
def deepseek_compute_mutual_information(
    data: np.ndarray, target_indices: np.ndarray | list[int], n_bins: int = 15, hist_dtype=np.int64, out_dtype=np.float64
) -> np.ndarray:
    """
    MI of every specified target column against every column in `data`.

    Parameters
    ----------
    data            : int8 ndarray, shape (n_samples, n_cols), already binned 0-14
    target_indices  : iterable of int column indices
    n_bins          : number of discrete bins (default 15)

    Returns
    -------
    mi_matrix       : float64 ndarray, shape (n_targets, n_cols)
                      Row k = MI(target_indices[k], all columns)
    """

    n_samples, n_columns = data.shape
    n_targets = len(target_indices)

    # Precompute marginals and sum_N_log_N for each column
    marginals = np.zeros((n_columns, n_bins), dtype=hist_dtype)
    sum_N_log_N = np.zeros(n_columns, dtype=out_dtype)

    for col in prange(n_columns):
        counts = np.zeros(n_bins, dtype=hist_dtype)
        for i in range(n_samples):
            val = data[i, col]
            counts[val] += 1
        marginals[col] = counts
        s = 0.0
        for b in range(n_bins):
            c = counts[b]
            if c > 0:
                s += c * np.log(c)
        sum_N_log_N[col] = s

    N = n_samples
    N_log_N = N * np.log(N) if N > 0 else 0.0

    mi_results = np.zeros((n_targets, n_columns), dtype=out_dtype)
    n_total_pairs = n_targets * n_columns

    for pair_idx in prange(n_total_pairs):
        t_idx = pair_idx // n_columns
        feature_col = pair_idx % n_columns
        target_col = target_indices[t_idx]

        joint = np.zeros((n_bins, n_bins), dtype=hist_dtype)
        for i in range(n_samples):
            y_val = data[i, target_col]
            x_val = data[i, feature_col]
            joint[y_val, x_val] += 1

        sum_Nxy_log_Nxy = 0.0
        for y_bin in range(n_bins):
            for x_bin in range(n_bins):
                n = joint[y_bin, x_bin]
                if n > 0:
                    sum_Nxy_log_Nxy += n * np.log(n)

        sum_Ny_log_Ny = sum_N_log_N[target_col]
        sum_Nx_log_Nx = sum_N_log_N[feature_col]

        mi = (sum_Nxy_log_Nxy - sum_Ny_log_Ny - sum_Nx_log_Nx + N_log_N) / N
        mi_results[t_idx, feature_col] = mi

    return mi_results
