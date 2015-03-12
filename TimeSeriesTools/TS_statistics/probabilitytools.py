
"""
This module contains functions related with probability and complements the
usual numpy or scipy tools.
"""

import numpy as np


def compute_conditional_probs(probs, marginal_vars):
    """Function which computes the conditional probability from the joint
    probability. We have to inform about the dependant variables.

    Parameters
    ----------
    probs: array_like
        multidimensional array which all the possible combinations of states of
        all the variables and the probability of being in each of thie
        combinations.
    marginal_vars: list or array_like of int
        the index of which variables we want to compute the marginal variables.

    Returns
    -------
    p_y_x: array_like
        the conditional probability.
    """

    ## Preparing needed variables
    n_vars = len(probs.shape)
    dependants = [i for i in range(n_vars) if i not in marginal_vars]
    dependants = np.sort(dependants)[::-1]
    marginal_vars = np.sort(marginal_vars)[::-1]
    n_np = dependants.shape[0]

    ## Computing dependendants
    p_x = compute_marginal_probs(probs, marginal_vars)

    ## Compute conditioned prob
    # Compute swap
    swp = np.array([[dependants[i], -i-1] for i in range(n_np)])
    # Swap axis
    for i in range(swp.shape[0]):
        probs = np.swapaxes(probs, swp[i, 0], swp[i, 1])
    # Division
    p_y_x = np.divide(probs, p_x)

    # Reswap axis
    for i in range(swp.shape[0]):
        p_y_x = np.swapaxes(p_y_x, swp[i, 1], swp[i, 0])
    for i in range(swp.shape[0]):
        probs = np.swapaxes(probs, swp[i, 1], swp[i, 0])

    return p_y_x


def compute_marginal_probs(probs, marginal_vars):
    """Function which computes marginal probabilities given the variables we
    want to marginalize.

    Parameters
    ----------
    probs: array_like
        the joint probability distribution.
    marginal_vars: list or array of int
        the indexes of the variables to marginalize.

    Returns
    -------
    p_x: array_like
        the marginal probability distribution.

    """

    ## Formatting inputs
    # Formatting marginal variables
    marginal_vars = np.sort(marginal_vars)[::-1]

    ## Marginalizing
    p_x = probs[:]
    for d in marginal_vars:
        nstates = p_x.shape[d]
        p_x = np.tensordot(np.ones(nstates), p_x, axes=np.array([0, d]))

    return p_x
