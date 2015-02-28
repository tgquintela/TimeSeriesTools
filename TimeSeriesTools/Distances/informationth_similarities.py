
"""
Module which groups all the information theory based measures of distances and
similarity of this package.
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
    """

    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


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

    #Loop over all the possible pairs of elements
    for i in range(n):
        for j in range(n):
            scores[i, j] = H[j] - H[i]

    return scores


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
