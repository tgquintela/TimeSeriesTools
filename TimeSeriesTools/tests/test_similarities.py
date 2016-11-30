
"""
Test similarities
-----------------
Collection of tests for the similarities module.

"""

import numpy as np
from ..Similarities.aux_functions import KL_divergence, average_prob,\
    Jensen_Shannon_divergence
from ..Similarities.dtw import dtw
from ..Similarities.magnitude_similarities import general_dtw
from ..Similarities.correlation_similarities import lagged_PearsonCorrelation
from ..Similarities.informationth_similarities import mutualInformation,\
    mutualInformation_1to1, conditional_entropy, information_GCI_ind
from ..Similarities.similarities import general_lag_distance,\
    general_distance_M, general_comparison


def test():
    ## Artificial data
    ##################
    p, q = np.random.random(10), np.random.random(10)
    p /= np.sum(p)
    q /= np.sum(q)
    probs, qrobs = np.random.random((10, 10)), np.random.random((10, 10))
    probs /= np.sum(probs)
    qrobs /= np.sum(qrobs)

    ## Auxiliar functions
    ##############
    KL_divergence(p, q)
    Jensen_Shannon_divergence(p, q)
    average_prob(p, q)
    average_prob(probs, qrobs)

    ## Magnitude similarity (dist)
    ###############################
    x, y = np.random.randn(200).cumsum(), np.random.randn(200).cumsum()
    dtw(x, y, dist=None)
    general_dtw(x, y, 'rpy2')

    ## Magnitude similarity (dist)
    ###############################
    X = np.random.randn(1000, 4).cumsum(0)
    lagged_PearsonCorrelation(X, timelag=0)
    lagged_PearsonCorrelation(X, timelag=2)

    X_disc = np.random.randint(0, 10, (1000, 2))
    mutualInformation_1to1(X_disc[:, 0], X_disc[:, 1], bins=10)
    mutualInformation(X_disc, bins=10)
    conditional_entropy(X_disc[:, 0], X_disc[:, 1])
    information_GCI_ind(X, bins=None)

    method_f = lambda x, y: np.max(x-y)
    tlags = [0, 2, 3]
    general_lag_distance(X, method_f, tlags, simmetrical=False, kwargs={})
    general_lag_distance(X, method_f, tlags, simmetrical=True, kwargs={})

    pars_lag = {'method_f': method_f, 'tlags': tlags, 'simmetrical': False}
    general_comparison(X, 'lag_based', pars_lag)

    general_distance_M(X, method_f, simmetrical=True, kwargs={})
    general_distance_M(X, method_f, simmetrical=False, kwargs={})
    pars_dist = {'method_f': method_f, 'simmetrical': False}
    general_comparison(X, 'static_based', pars_dist)
