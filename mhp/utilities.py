from math import log, ceil, floor
import numpy as np
from tick.hawkes.simulation import SimuHawkesExpKernels

# ---------------------------------------------------------------------------------------------------------
def find_exp_percentile(decay, percentile):
    """Compute the n-th percentile of exponential kernel

    Arguments:
        decay {float} -- how fast function decays
        percentile {int} -- 

    Returns:
        float -- value x such that percentile / 100 % of mass is smaller than x
    """
    return (-log(1 - percentile / 100)) / decay

# ---------------------------------------------------------------------------------------------------------
def decay_for_lag(p, delta):
    """Compute required decay of exp such that 95th percentile falls in the middle of the p-th bin after event

    Arguments:
        p {int} -- lag of INAR, number of subsequent bins affected by 95% of mass. For 1, 95th percentile in middle of next bin, i.e 3/2 * delta. For same bin, p = 0
        delta {int} -- discretization parameter

    Returns:
        float -- aforesaid 95th percentile
    """
    return find_exp_percentile(p * delta + delta / 2, 95)

# ---------------------------------------------------------------------------------------------------------
def lag_from_decay(decay, delta):
    """Compute how many bins further is the 95th percentile (from the point the event took place)

    Arguments:
        decay {int} -- decay param of exponential
        delta {int} -- discretization parameter

    Returns:
        int -- bin distance
    """
    return int(find_exp_percentile(decay, 95) / delta)

# ---------------------------------------------------------------------------------------------------------
def spectral_radius(adjacency):
    """Return spectral radius of adjacency matrix

    Arguments:
        adjacency {np.ndarray} -- adjacency matrix of Hawkes process

    Returns:
        float -- spectral_radius
    """
    sim = SimuHawkesExpKernels(adjacency, 0)
    return sim.spectral_radius()

# ---------------------------------------------------------------------------------------------------------
def sparse_matrix(dim, radius=0.9):
    """Generate intensity matrix with fixed spectral radius

    Arguments:
        dim {int} -- number of process

    Returns:
        np.ndarray -- array of intensities
    """
    rate = 1 if dim > 3 else 2
    matrix = np.zeros((dim, dim))
    # Makes sure that spectral radius is not zero
    while spectral_radius(matrix) == 0:
        matrix = np.random.binomial(n=1, p=rate * np.log(dim)/dim,
                                    size=(dim, dim)) * np.random.uniform(0.5, 1.0, size=(dim, dim))
        if spectral_radius(matrix) != 0:
            simHawks = SimuHawkesExpKernels(matrix, decays=1)
            simHawks.adjust_spectral_radius(radius)

    return simHawks.adjacency

# ---------------------------------------------------------------------------------------------------------
def mean_intensity(adjacency, decay, baseline):
    """Return mean intensity of adjacency matrix

    Arguments:
        adjacency {np.ndarray} -- intensity matrix
        decay {float} -- kernel's decay
        baseline {np.ndarray} -- baseline intensities

    Returns:
        np.ndarray -- mean intensities
    """
    hawkes = SimuHawkesExpKernels(
        adjacency, decays=float(decay), baseline=baseline)
    return hawkes.mean_intensity()

# ---------------------------------------------------------------------------------------------------------
def base_intensity(dim):
    """Generate random baseline intensity vector

    Arguments:
        dim {int} -- number of processes

    Returns:
        np.ndarray --
    """
    return np.random.uniform(0, 0.1, dim)

# ---------------------------------------------------------------------------------------------------------
def exp_kernel(decay, intensity, x):
    """Compute exp kernel at x

    Arguments:
        decay {float} -- beta parameter, decay
        intensity {float} -- alpha parameter, jump intensity
        x  -- {Number | np.ndarray}

    Returns:
         value of exponential at x
    """
    return intensity * decay * np.exp(-decay * x)

# ---------------------------------------------------------------------------------------------------------
def simulate_mhp(adjacency, decay, baseline, events, delta, seed=None):
    """Simulate MHP with provided parameters

    Args:
        adjacency (np.ndarray): 2D array containing the inter-process influence coefficients
        decay (float): Positive decay parameter of the exponential
        baseline (np.ndarray): 1D array of baseline intensities
        events (int): Number of events to simulate
        delta (float): Width of aggregation intervals
        seed (int, optional): Original value of random values generator. Defaults to None.

    Returns:
        np.ndarray: Aggregated Hawkes process.
    """

    hawkes = SimuHawkesExpKernels(adjacency, float(
        decay), baseline, max_jumps=events, verbose=False, seed=seed)

    hawkes.simulate()
    return discretize(hawkes.timestamps, delta)

# ---------------------------------------------------------------------------------------------------------
def discretize(timestamps, delta):
    """Aggregates event arrival times into bin counts

    Args:
        timestamps (list[np.ndarray]): List of per-process array of arrival times
        delta (float): Quantization parameter

    Returns:
        np.ndarray: Aggregated process array (each row represent a dimension/node)
    """
    time_max = max([0 if len(p) == 0 else np.max(p)  for p in timestamps])
    # Account for estimation lag, otherwise failing at estimation
    n_process = len(timestamps)
    bins = np.zeros((n_process, ceil(time_max / delta)))

    for line, process in zip(bins, timestamps):
        for event in process:
            line[int(event / delta)] += 1

    return bins

# ---------------------------------------------------------------------------------------------------------
def true_positive(adj_test, adj_true, threshold=0.05):
    assert adj_test.shape == adj_true.shape, \
        "Parameters should have same shape"
    return np.sum((adj_test >= threshold) * (adj_true > 0))

# ---------------------------------------------------------------------------------------------------------
def false_positive(adj_test, adj_true, threshold=0.05):
    assert adj_test.shape == adj_true.shape, \
        "Parameters should have same shape"
    return np.sum((adj_test >= threshold) * (adj_true == 0))

# ---------------------------------------------------------------------------------------------------------
def false_negative(adj_test, adj_true, threshold=0.05):
    assert adj_test.shape == adj_true.shape, \
        "Parameters should have same shape"
    return np.sum((adj_test < threshold) * (adj_true > 0))

# ---------------------------------------------------------------------------------------------------------
def true_negative(adj_test, adj_true, threshold=0.05):
    assert adj_test.shape == adj_true.shape, \
        "Parameters should have same shape"
    return np.sum((adj_test < threshold) * (adj_true == 0))

# ---------------------------------------------------------------------------------------------------------
def recall(adj_test, adj_true, threshold=0.05):
    assert adj_test.shape == adj_true.shape, \
        "Parameters should have same shape"
    return true_positive(adj_test, adj_true, threshold) / np.sum(adj_true > 0)

# ---------------------------------------------------------------------------------------------------------
def precision(adj_test, adj_true, threshold=0.05):
    assert adj_test.shape == adj_true.shape, \
        "Parameters should have same shape"
    return true_positive(adj_test, adj_true, threshold) / np.sum(adj_test > threshold)

# ---------------------------------------------------------------------------------------------------------
def fscore(adj_test, adj_true, threshold=0.05, beta=1.0):
    rec_val = recall(adj_test, adj_true, threshold)
    prec_val = precision(adj_test, adj_true, threshold)
    return (1 + beta ** 2) * prec_val * rec_val / (beta ** 2 * prec_val + rec_val)

# ---------------------------------------------------------------------------------------------------------
def f1_score(adj, estimates, threshold):
    est = np.max(estimates, axis=2) >= threshold
    return fscore(est, adj, threshold)
