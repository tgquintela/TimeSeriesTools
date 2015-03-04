

"""
Distances or similarities between time series are functions that computes a
distance measure or similarity measure pairwise and returns a matrix of
distances between each timeserie.
"""

import numpy as np
#import math


def general_comparison(X, method, **kwargs):
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

    if type(method).__name__ == 'function':
        comparisons = method(X, **kwargs)
    elif method == 'lag_based':
        comparisons = general_lag_distance(X, **kwargs)

    return comparisons


def general_lag_distance(X, method_f, tlags, simmetrical=False, kwargs={}):
    """Build a 3d matrix of distance using the method_f given.

    Parameters
    ----------
    X: array_like, shape(Ntimes, Nelements)
        the signals of the system.
    maxt: integer or list or array_like
        max lag to be considered.
    method_f: function
        the function 1v1 to compute the distance or similarity desired.
    simmetrical: boolean
        the possibility to safe computational power only computing one
        directional pairs.
    kwargs: dict
        the parameters of the selected method. The parameters needed by
        method_f.

    Returns
    -------
    M: array_like, shape (Nelements, Nelements, nlags)
        the matrix of possible distances for each choosen lag time.

    """

    ## 0. Format inputs and needed variables
    # Format lags
    lags = np.array([tlags]) if type(tlags) in [int, list] else tlags
    lags = lags.reshape(-1)
    # Elements
    n = X.shape[1]
    tl = lags.shape[0]

    ## Build the tensor
    M = np.zeros((n, n, tl))
    # Loop over lags
    for l in range(lags.shape[0]):
        tlag = lags[l]
        # Loop over the possible combinations of elements
        for i in range(n):
            # Loop over the necessary considering simmetrical or not
            if simmetrical:
                pairedelements = range(i, n)
            else:
                pairedelements = range(n)
            # Loop over the pair elements
            for j in pairedelements:
                M[i, j, l] = method_f(X[tlag:, i], X[:X.shape[0]-tlag, j],
                                      **kwargs)

    return M


def general_distance_M(X, method_f, simmetrical, kwargs):
    """This function is an applicator of a method given by the function
    method_f.

    Parameters
    ----------
    X: array_like, shape(Ntimes, Nelements)
        the signals of the system.
    maxt: integer or list or array_like
        max lag to be considered.
    method_f: function
        the function 1v1 to compute the distance or similarity desired.
    simmetrical: boolean
        the possibility to safe computational power only computing one
        directional pairs.
    kwargs: dict
        the parameters of the selected method. The parameters needed by
        method_f.

    """

    # Initialization
    n = X.shape[1]
    M = np.zeros((n, n))
    # Loop over the possible combinations of elements
    for i in range(n):
        # Loop over the necessary considering simmetrical or not
        if simmetrical:
            pairedelements = range(i, n)
        else:
            pairedelements = range(n)
        # Loop over the pair elements
        for j in pairedelements:
            M[i, j] = method_f(X[:, i], X[:, j], **kwargs)

    return M


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

    return comparisons


def comparison_f_1v1(x, y, method, **kwargs):
    """Function which acts as a switcher and wraps all the comparison functions
    available in this package and returns a instantiable function.

    Parameters
    ----------
    method: str, optional
        measure we want to use for make the comparison.
    kwargs: dict
        parameters for the functions selected by the method parameter.

    Returns
    -------
    comparator: function
        the instantiable function which could be called to compare two time
        series.
    """

    if type(method).__name__ == 'function':
        comparator = lambda x, y: method(x, y, **kwargs)
    elif method == '':
        pass

    return comparator

