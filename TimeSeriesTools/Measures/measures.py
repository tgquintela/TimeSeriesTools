
"""
This module is composed by a group of functions that computes some measures
in the time-series individually.

TODO
----
entropy rate (paper transfer entropy)

"""

import numpy as np
from hurst_measures import hurst


def measure_ts(X, method, **kwargs):
    """Function which acts as a switcher and wraps all the possible functions
    related with the measure of a property in a time-series.

    Parameters
    ----------
    X: array_like, shape(N, M)
        signals of the elements of the system. They are recorded in N times M
        elements of the system.
    method: str, optional
        possible methods to be used in order to measure some paramter of the
        time series given.
    kwargs: dict
        extra variables to be used in the selected method. It is required that
        the keys match with the correct parameters of the method selected. If
        this is not the case will raise an error.

    Returns
    -------
    measure: array_like, shape(M, p)
        is the resultant measure of each time series of the system. The
        selected measure can be a multidimensional measure and returns p values
        for each time series.

    """

    # Switcher
    if method == 'entropy':
        measure = entropy(X, **kwargs)
    elif method == 'hurst':
        measure = hurst(X, **kwargs)
    elif method == 'pfd':
        measure = pfd(X, **kwargs)
    elif method not in ['entropy', 'hurst', 'pfd']:
        pass

    return measure


###############################################################################
###################### Hjorth mobility ########################################
def hjorth(X):
    """ Compute Hjorth mobility and complexity of a time series.

    Notes
    -----
    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.

    Parameters
    ----------
    X : array_like, shape(N,)
        a 1-D real time series.

    Returns
    -------
    HM : float
        Hjorth mobility
    Comp : float
        Hjorth complexity

    References
    ----------
    .. [1] B. Hjorth, "EEG analysis based on time domain properties,"
       Electroencephalography and Clinical Neurophysiology , vol. 29,
       pp. 306-310, 1970.
    """

    # Compute the first order difference
    D = np.diff(X)
    # pad the first difference
    D = np.hstack([X[0], D])

    #
    n = X.shape[0]
    M2 = np.float(np.sum(D ** 2))/n
    TP = np.sum(X ** 2)
    M4 = np.sum((D[1:] - D[:D.shape[0]-1])**2)/n

    # Hjorth Mobility and Complexity
    HM = np.sqrt(M2 / TP)
    Comp = np.sqrt(np.float(M4) * TP / M2 / M2)

    return HM, Comp
