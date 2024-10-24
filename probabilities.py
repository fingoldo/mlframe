import numpy as np
from numba import njit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, brier_score_loss


@njit()
def generate_probs_from_outcomes(
    outcomes: np.ndarray, chunk_size: int = 20, scale: float = 0.1, nbins: int = 10, bins_std: float = 0.1, flip_percent: float = 0.6
) -> np.ndarray:
    """Can we generate hypothetical ground truth probs knowing the outcomes in advance?
    Our model probs will (hopefully) be calibrated. So, we need synthetic probs to be calibrated, too. With some degree of fitness.
    We also need to cover broad range of probs.
    So, how to achieve this?

    0)  if flip_percent is specified, for a random portion of data zeroes and ones are flipped. this will lower ROC AUC.
    1) we can work with small random chunks/subsets of data
    2) for every chunk, its real freq is computed.
    3) for every observation, 'exact' prob is drawn from some distribution (uniform or, say, gaussian) with center in real freq.
    then, if bins_std is specified, constant bin noise is applied to all observations of the chunk.

    final result is clipped to [0,1]
    """
    n = len(outcomes)
    indices = np.arange(n)
    np.random.shuffle(indices)

    probs = np.empty(n, dtype=np.float32)
    bin_offsets = (np.random.random(size=nbins) - 0.5) * bins_std

    if flip_percent:
        # flip some bits to worsen our so far perfect predictive power
        flip_size = int(n * flip_percent)
        if flip_size:
            outcomes = outcomes.copy()
            flip_indices = np.random.choice(indices, size=flip_size)
            outcomes[flip_indices] = 1 - outcomes[flip_indices]

    l = 0  # left border
    for idx in range(n // chunk_size):  # traverse randomly selected chunks/subsets of original data
        r = (idx + 1) * chunk_size  # right border
        freq = outcomes[l:r].mean()  # find real event occuring frequency in current chunk of observation

        # add pregenerated offset for particular bin
        bin_idx = int(freq * nbins)
        freq = freq + bin_offsets[bin_idx]

        # add small symmetric random noise. it must be higher when freq approaches [0;1] borders.
        probs[l:r] = freq + (np.random.random(size=chunk_size) - 0.5) * scale * np.abs(freq - 0.5)

        l = r

    return np.clip(probs, 0.0, 1.0)


from scipy.special import logit, expit


def generate_similar_probs_logit_space(predicted_probs, true_outcomes, noise_scale=0.05):
    """
    Perturb probabilities by applying noise in logit space (log-odds),
    then converting back to probability space.

    Args:
        predicted_probs (np.ndarray): Original predicted probabilities (0-1).
        true_outcomes (np.ndarray): True binary outcomes (0 or 1).
        noise_scale (float): Scale of Gaussian noise applied in logit space.

    Returns:
        np.ndarray: A new set of perturbed probabilities.
    """
    # Convert probabilities to logit (log-odds) space
    logit_probs = logit(np.clip(predicted_probs, 1e-6, 1 - 1e-6))  # Avoid log(0) issues

    # Add Gaussian noise in logit space
    perturbed_logit = logit_probs + np.random.normal(0, noise_scale, size=logit_probs.shape)

    # Convert back to probability space using the sigmoid function
    similar_probs = expit(perturbed_logit)

    return similar_probs


def generate_similar_probs_random_walk(predicted_probs, true_outcomes, step_size=0.05, n_steps=1):
    """
    Generates perturbed probabilities using a small random walk.

    Args:
        predicted_probs (np.ndarray): Original predicted probabilities (0-1).
        true_outcomes (np.ndarray): True binary outcomes (0 or 1).
        step_size (float): Size of each random walk step.
        n_steps (int): Number of random walk steps.

    Returns:
        np.ndarray: A new set of perturbed probabilities.
    """
    similar_probs = predicted_probs.copy()

    for _ in range(n_steps):
        # Add or subtract random small values for the random walk
        random_step = np.random.uniform(-step_size, step_size, size=predicted_probs.shape)
        similar_probs += random_step

        # Ensure probabilities stay within [0, 1]
        similar_probs = np.clip(similar_probs, 0, 1)

    return similar_probs


def generate_similar_probs(predicted_probs, true_outcomes, noise_scale=0.05, n_iterations=100):
    """
    Generates a random ndarray of similar probabilities that has approximately the same
    Brier Score and ROC AUC as the input predicted_probs.

    Args:
        predicted_probs (np.ndarray): Original predicted probabilities (0-1).
        true_outcomes (np.ndarray): True binary outcomes (0 or 1).
        noise_scale (float): Standard deviation of the noise to add.
        n_iterations (int): Number of iterations for fine-tuning the noise scale.

    Returns:
        np.ndarray: A similar_probs array with approximately the same Brier Score and ROC AUC.
    """
    original_brier_score = brier_score_loss(true_outcomes, predicted_probs)
    original_auc = roc_auc_score(true_outcomes, predicted_probs)

    similar_probs = None

    for _ in range(n_iterations):
        # Add Gaussian noise and clip the values to keep them in [0, 1]
        noisy_probs = predicted_probs + np.random.normal(loc=0.0, scale=noise_scale, size=predicted_probs.shape)
        noisy_probs = np.clip(noisy_probs, 0, 1)

        # Calculate new Brier Score and AUC
        new_brier_score = brier_score_loss(true_outcomes, noisy_probs)
        new_auc = roc_auc_score(true_outcomes, noisy_probs)

        # Check if the new metrics are close to the original
        if np.isclose(new_brier_score, original_brier_score, rtol=0.01) and np.isclose(new_auc, original_auc, rtol=0.01):
            similar_probs = noisy_probs
            break

    # If the loop doesn't converge, return the best found so far
    return similar_probs if similar_probs is not None else noisy_probs


def generate_similar_probs_by_ranking(predicted_probs, true_outcomes, n_bins: int = 10, noise_scale: float = 0.001):
    """
    Generates a new set of probabilities by shuffling within ranked bins,
    preserving the ranking and maintaining Brier Score and ROC AUC.

    Args:
        predicted_probs (np.ndarray): Original predicted probabilities (0-1).
        true_outcomes (np.ndarray): True binary outcomes (0 or 1).
        n_bins (int): Number of bins to group the probabilities into (e.g., quantiles).
        noise_scale (float): Amount of random noise to add after shuffling to introduce variation.

    Returns:
        np.ndarray: A new set of similar probabilities.
    """
    # Get ranks of predicted probabilities
    ranks = rankdata(predicted_probs, method="ordinal")
    n = len(predicted_probs)

    # Create bins based on the ranks (grouping into approximately equal-sized bins)
    bins = np.floor_divide(ranks * n_bins, n)

    similar_probs = predicted_probs.copy()

    # Shuffle within each bin
    for bin_value in np.unique(bins):
        bin_indices = np.where(bins == bin_value)[0]
        # Extract the probabilities in this bin
        bin_probs = similar_probs[bin_indices]
        # Shuffle them
        np.random.shuffle(bin_probs)
        # Assign the shuffled values back to their positions
        similar_probs[bin_indices] = bin_probs

    if noise_scale:
        # Add small noise to ensure variation
        similar_probs += np.random.normal(0, noise_scale, size=predicted_probs.shape)

        # Ensure probabilities are still between 0 and 1
        similar_probs = np.clip(similar_probs, 0, 1)

    return similar_probs
