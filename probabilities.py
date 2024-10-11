import numpy as np
from numba import njit


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
