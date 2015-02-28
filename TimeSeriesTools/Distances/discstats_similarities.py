
"""
Module which groups all the distance and similarity measures based on the
statistics of discrete events.
From statistics of events we can compute the tensor of coincidences. The
collapsing of this tensor into a matrix is the task of the functions of this
module.
"""

import numpy as np


def aux(C, method, weigths):
    """
    """

    if type(method).__name__ == 'function':
        comparisons = method(C, weigths)
    elif method == 'linear_comb':
        comparisons = linear_temp_combination(C, weigths)
    elif method == 'log_comb':
        comparisons = log_temp_combination(C, weigths)

    return comparisons


def linear_temp_combination(C, weigths):
    """
    """
    # Check dimensionality of weights
    assert C.shape[2] == weigths.shape[0]
    #
    comparisons = np.sum(np.multiply(C, weigths), axis=2)
    return comparisons


def log_temp_combination(C, weigths):
    """
    """
    # Check dimensionality of weights
    assert C.shape[2] == weigths.shape[0]
    #
    comparisons = np.sum(np.multiply(-np.log(C), weigths), axis=2)
    return comparisons
