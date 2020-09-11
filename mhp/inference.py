import numpy as np

def __Z(bins, p):
    """Produces Z matrix of estimator from dicrete bins

    Arguments:
        bins {np.ndarray} -- array of bins, per process event counts, 
        p {int} -- lag of INAR process

    Returns:
        ray -- Z matrix of estimator
    """
    bins = np.array(bins)
    d = bins.shape[0]
    assert d > 0, "Need non-empty bins list to work"

    n = bins.shape[1]
    if p >= n:
        return None

    n_cols = n - p
    res = np.zeros((d * p + 1, n_cols))

    for i, shift_i in enumerate(range(p - 1, -1, -1)):
        for k, bin_k in enumerate(bins):
            res[i * d + k] = bin_k[shift_i: shift_i + n_cols]

    res[-1] = np.ones(n_cols)
    return res

# ---------------------------------------------------------------------------------------------------------
def __Y(bins, p):
    """Produce Y matrix of estimator

    Arguments:
        bins {np.ndparray} -- array of bins, per process event counts
        p {int} -- lag of INAR process

    Returns:
        np.ndarray -- Y matrix of estimator
    """
    return bins[:, p:]

# ---------------------------------------------------------------------------------------------------------
def __theta_CLS(bins, p):
    """Compute conditional least square estimator of INAR process
    
    Arguments:
        bins {np.ndarray} -- array of bins, per process event counts
        p {int} -- lag of INAR process
    
    Returns:
        np.ndarray -- d x (dp + 1) estimator
    """
    y_matrix = __Y(bins, p)
    z_matrix = __Z(bins, p)
    if z_matrix is None:
        return None
    
    z_matrix_t = z_matrix.T
    try:
        zzti = np.linalg.inv(np.dot(z_matrix, z_matrix_t))
    except np.linalg.LinAlgError:
        return None

    return y_matrix.dot(z_matrix_t).dot(zzti)

# ---------------------------------------------------------------------------------------------------------
class InarInference:
    """Create an inference object. Sets the aggregation interval width to delta and considers that after *lag* bins, the influence of events on intensity is neligible.
    """

    def __init__(self, lag, delta):
        """Initialize inference object

        Args:
            lag (int): Number of adjacent intervals/bins influenced by an event
            delta (float): Aggregation width
        """
        assert int(lag) == lag, "Require integer valued lag"
        assert lag >= 0, "Require non-negative lag"
        assert delta > 0, "Require positive delta"

        self.__lag = lag
        self.__delta = delta

    # ---------------------------------------------------------------------------------------------------------
    
    def fit(self, bins):
        theta = __theta_CLS(bins, self.__lag)
        # In case estimation fails, carry error further
        if theta is None:
            return (None, None)
        # divide by delta to go from inar to hawkes
        theta /= self.__delta
        dim = theta.shape[0]
        # matrix of estimates of the form (from, to, at)
        # where from/to relates to origin/dest. of influence
        # and at relates to sample.
        res = np.zeros((dim, dim, self.__lag))
        for col_start in range(dim):
            indexes = np.ix_(range(dim), [dim * k + col_start for k in range(self.__lag)])
            res[:, col_start, :] = theta[indexes]

        eta = theta[:, -1]

        return res, eta

    # ---------------------------------------------------------------------------------------------------------

class WithPhaseInference:
    # TODO: Add PyTorch implementation of inference when there is a phase shift between dimensions
    pass