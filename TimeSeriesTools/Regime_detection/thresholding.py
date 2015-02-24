

""" TODO:
These functions only are examples of the work to detection
"""

import numpy as np


def discretize_with_thresholds(array, thres, values=[]):
    '''This function discretize the array to the values given.

    Parameters
    ----------
    array : array_like, shape (N,)
        the values of the signal.
    thres : float, list of floats, array_like
        thresholds values to discretize the array. Threholds can be of 2 types:
            * absolute thresholds: one or more values (float or list of floats)
                                   they are static thresholds for all the ts.
            * moving thresholds: array or matrix of values which represents the
                                 thresholds in each point of time.
    values: list
        the values to discretize the array.

    Returns
    -------
    aux : array_like, shape (N,)
        a array with a discretized values.

    '''

    ## 1. Preparing thresholds and values
    if type(thres) == float:
        thres = np.array([thres], ndmin=2).T
    elif type(thres) == list:
        thres = np.array(thres, ndmin=2).T
    elif type(thres) in [np.array, np.ndarray]:
        thres = np.array(thres, ndmin=2)
        if thres.shape[0] == array.shape[0]:
            pass
        elif thres.shape[1] == array.shape[0]:
            thres = thres.T
        else:
            raise ValueError("Not correct shape for the thresholds.")

    if values == []:
        values = range(thres.shape[0]+1)
    else:
        assert thres.shape[0] == len(values)-1
    n = array.shape[0]
    mini = np.array([0]*n)*(array.max()-array.min()) + array.min()*np.array([1]*n)
    maxi = np.array([1]*n)*array.max()
    thres = np.hstack([mini, thres.T, maxi]).T

    ## 2. Fill the new vector discretized signal to the given values
    aux = np.zeros(array.shape)
    for i in range(len(values)):
        indices = np.logical_and(array >= thres[i,:], array <= thres[i+1,:])
        indices = np.nonzero(indices)[0].astype(int)
        aux[indices] = values[i]

    return aux


