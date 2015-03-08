
"""
Module which groups all the information theory based measures of distances and
similarity of this package.

TODO
----
Discretization for transformation ts module
"""

from sklearn.metrics import mutual_info_score
import numpy as np
from pyCausality.TimeSeries.Mesaures.measures import entropy


def mutualInformation(X, bins):
    """Computation of the mutual information between each pair of variables in
    the system.
    """

    # Initialization of the values
    n = X.shape[1]
    MI = np.zeros((n, n))

    # Loop over the possible combinations of pairs
    for i in range(X.shape):
        for j in range(i, X.shape):
            aux = mutualInformation_1to1(X[:, i], X[:, j], bins)
            # Assignation
            MI[i, j] = aux
            MI[j, i] = aux

    return MI


def mutualInformation_1to1(x, y, bins):
    """Computation of the mutual information between two time-series.

    Parameters
    ----------
    x: array_like, shape (N,)
        time series to compute difference.
    y: array_like, shape (N,)
        time series to compute difference.
    bins: int or tuple of ints or tuple of arrays
        the binning information.

    Returns
    -------
    mi: float
        the measure of mutual information between the two time series.

    """

    ## 1. Discretization
    if bins is None:
        # Compute contingency matrix
        pass
    else:
        c_xy = np.histogram2d(x, y, bins)[0]

    ## 2. Compute mutual information from contingency matrix
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def conditional_entropy(x, y):
    """
    TODO
    ----
    Compute correctly
    Check with the real matrices
    """
    # Discretized signals
    ## xy and number of regimes
    p_xy = prob_xy(x, y)
    p_x = prob_x(x)
    p_y = prob_x(y)

    # Conditional probability
    p_x_y = np.divide(p_xy, p_y)
    # Sum over possible combination of regimes
    H_x_y = np.sum(p_xy, np.log(p_x_y))

    return H_x_y


def information_GCI_ind(X, bins):
    ''' Baseline method to compute scores based on Information Geometry Causal
    Inference, it boils down to computing the difference in entropy between
    pairs of variables:    scores(i, j) = H(j) - H(i)

    Parameters
    ----------
    X: array_like, shape(N,)
        the time-series of the system.
    bins: array_like, shape (Nintervals+1,)
        information of binning or discretizing.

    Returns
    -------
    scores: array_like, shape (Nelements, Nelements)
        the matrix of the system.

    Reference
    ---------
    .. [1] P. Daniuis, D. Janzing, J. Mooij, J. Zscheischler, B. Steudel,
    K. Zhang, B. Schalkopf: Inferring deterministic causal relations.
    Proceedings of the 26th Annual Conference on Uncertainty in Artificial
    Intelligence (UAI-2010).
    http://event.cwi.nl/uai2010/papers/UAI2010_0121.pdf

    '''
    ## DOUBT: Only for discrete signals?
    # Compute the entropy
    if bins is None:
        H = entropy(X)

    ## Compute the scores as entropy differences (vectorized :-))
    n = H.shape[0]
    scores = np.zeros(shape=(n, n))

    ## Loop over all the possible pairs of elements
    for i in range(n):
        for j in range(n):
            scores[i, j] = H[j] - H[i]

    return scores


# import time
# t0 = time.time()
# mi1 = mutualInformation_1to1(X1, X2, 20)
# t1 = time.time()
# mi2 = calc_MI(X1, X2, 20)
# t2 = time.time()
# mi3 = mutualInformation_1to1(X1, X2, 20)
# print t1-t0
# print t2-t1
# print time.time()-t2
