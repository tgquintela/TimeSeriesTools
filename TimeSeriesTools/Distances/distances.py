

"""
Distances or similarities between time series are functions that computes
a distance measure or similarity measure pairwise and returns a matrix of
distances between each timeserie.
"""


import numpy as np
#import math

from pyCausality.TimeSeries.measures import entropy


def genral_comparison(X, method, **kwargs):
    """Function which acts as a switcher and wraps all the comparison functions
    available in this package.

    Parameters
    ----------
    X: array_like, shape(N, M)
        a collection of M time-series.
    method: str, optional
        measure we want to use for make the comparison.
    kwargs: dict
        parameters for the functions selected by the method parameter.

    Returns
    -------
    comparisons: array_like, shape (M, M)
        returns the measure of all the possible time series versus all the
        others.

    """

    if method == '':
        pass

    return comparisons


def comparison_1v1(x, y, method, **kwargs):
    """Function which acts as a switcher and wraps all the comparison functions
    available in this package.

    Parameters
    ----------
    x: array_like, shape(N,)
        time-serie with N times.
    y: array_like, shape(N,)
        time-serie with N times.
    method: str, optional
        measure we want to use for make the comparison.
    kwargs: dict
        parameters for the functions selected by the method parameter.

    Returns
    -------
    comparisons: float
        returns the measure of comparison between the two time-series given
        using the measure specified in the inputs.

    """

    return comparison


###############################################################################
############################### SIMILARITIES ##################################
###############################################################################


################################ CORRELATION ##################################
# See more in
# http://scikit-learn.org/stable/modules/covariance.html
def correlation_without_lag(dynamics):
    '''Correlation measure without lag.
    '''

    # Quicker?
    #M = np.corrcoef(X.T)

    return dynamics.corr('pearson')


def lagged_PearsonCorrelation(X, timelag=0):
    '''Computes the Pearson Correlation pairwise in between the lagged
    time-series of all the elements of the system.
    '''

    # initialization
    n = X.shape[1]
    m = 1 if type(timelag) in [int, float] else len(timelag)
    if m == 1:
        timelag = [timelag]
    M = np.zeros((n, n, m))

    # loop over the required timelags
    for lag in range(m):
        tlag = timelag[lag]
        # loop over the possible combinations of elements
        for i in range(n):
            for j in range(n):
                aux = np.corrcoef(X[tlag:, i], X[:X.shape[0]-tlag, j])
                # assignation
                M[i, j, lag] = aux[1, 0]

    return M


#def myxcorr_1to1(x1, y1, timeLag=0):
#    # Not mine
#    # Not used
#    # TODO: Test this
#    crossCorr = []
#    x = ((np.array(x1))[np.newaxis])
#    y = ((np.array(y1))[np.newaxis])
#    n = x.shape[1]
#    for i in range(-timeLag-1, 0, 1):
#        j = -i
#        suma = np.dot(y[:, j-1:n], x[:, 0:n-j+1])
#        crossCorr.append(suma)
#    for i in range(0, timeLag, 1):
#        suma = np.dot(x[:, i:n], y[:, 0:n-i])
#        crossCorr.append(suma)
#    crossCorr_array = np.asarray(crossCorr)
#    x_sum = np.dot(x, x)
#    final_result = crossCorr_array/math.sqrt(x_sum)
#    y_sum = np.dot(y, y)
#    final_result_1 = final_result/math.sqrt(y_sum)
#    return final_result_1


############################# INFORMATION-BASED ###############################
from sklearn.metrics import mutual_info_score


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
    """

    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


### JOINT ENTROPY
#def calc_MI(X,Y,bins):
##http://stackoverflow.com/questions/20491028/optimal-way-
##for-calculating-columnwise-mutual-information-using-numpy
#   c_XY = np.histogram2d(X,Y,bins)[0]
#   c_X = np.histogram(X,bins)[0]
#   c_Y = np.histogram(Y,bins)[0]
#   H_X = shan_entropy(c_X)
#   H_Y = shan_entropy(c_Y)
#   H_XY = shan_entropy(c_XY)
#   MI = H_X + H_Y - H_XY
#   return MI


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


def computeIGCI_ind(F):
    ''' Baseline method to compute scores based on Information Geometry Causal
    Inference,
    Inspired by:
    [1]  P. Daniuis, D. Janzing, J. Mooij, J. Zscheischler, B. Steudel,
         K. Zhang, B. Schalkopf:  Inferring deterministic causal relations.
         Proceedings of the 26th Annual Conference on Uncertainty in Artificial
         Intelligence (UAI-2010).
         http://event.cwi.nl/uai2010/papers/UAI2010_0121.pdf
    It boils down to computing the difference in entropy between pairs of
    variables:
        scores(i, j) = H(j) - H(i)
    '''
    ## DOUBT: Only for discrete signals?
    #D = discretizeFluorescenceSignal(F)
    # Compute the entropy
    H = entropy(F)

    ## Compute the scores as entropy differences (vectorized :-))
    n = H.shape[0]
    scores = np.zeros(shape=(n, n))

    #scores = numpy.vstack([scores, H.T[0]])
    for i in range(n):
        for j in range(n):
            scores[i, j] = H[j] - H[i]

    return scores


def dtw(ts1, ts2, options):
    """The implementation of the dynamic time warping measure between two ts.
    """

    pass
