
"""
This module contains the functions which computes interesting statistics. They
can be applied to continious time-series for obtaining important statistics for
the study of causality or prediction.
"""

from itertools import product
import numpy as np


from TimeSeriesTools.Transformation.value_discretization import \
    threshold_binning_builder, discretize_with_thresholds


def prob_markov_proc_ind(X_info, bin_info):
    """
    Parameters
    ----------
    X_info: tuple or list of arrays
        arrays between which we compute probabilities.
    bin_info: tuple or list of ints or arrays
        info about discretization

    """
    return probs


def prob_xy(X, bins=0, maxl=0):
    """Wrapper for prob_xy_ind. It computes the probability for all a matrix of
    dynamics.

    Parameters
    ----------
    X: array_like, shape (N, Melements)
        the signals of the system.
    bins: int
        number of bins to study the statistics.

    Returns
    -------
    probs: array_like, shape(Nelements, Nelements, maxl, n_bins, n_bins)
        the probability for each pair of elements, timelag and range of values.

    """

    # 0. Formating inputs and preparing needed variables

    ## 1. Discretization
    discretizor = threshold_binning_builder(X, bins)
    X = discretize_with_thresholds(X, discretizor)

    ## 2. Needed variables
    values = np.unique(X)
    n_bins = values.shape[0]
    n = X.shape[1]
    nt = X.shape[0]
    pairs = product(range(n), range(n))

    ## 3. Compute probability
    probs = np.zeros((n, n, maxl+1, n_bins, n_bins))
    for p in pairs:
        for tlag in range(maxl+1):
            p0, p1 = p[0], p[1]
            probs[p0, p1, tlag, :, :], _, _ = prob_xy_ind(X[tlag:, p0],
                                                          X[:nt-tlag, p1],
                                                          0, tlag)

    return probs, discretizor, values


def prob_xy_ind(x, y, bins=0, timelag=0, normalize=True):
    """Probability of the signals to have some specific range of values.

    Parameters
    ----------
    x: array_like, shape (Nt,)
        the signal of one element.
    y: array_like, shape (Nt,)
        the signal of one element.
    bins: int, array_like, tuple
        binning information.
    timelag: int
        the time lag considered between the first time serie with the second.

    Returns
    -------
    probs: array_like, shape (n_bins, n_bins)
        the probability of being in each possible combinations of regimes.
    bins_edges: array_like, shape (n_bins+1,)
        the edges of the bins.

    TODO
    ----
    When there is actually discretized

    """

    ## 1. Preparing discretizor
    values1 = np.unique(x)
    values2 = np.unique(y)
    if type(bins) in [list, tuple, np.array, np.ndarray]:
        # When the bins are empty
        if not np.any(bins):
            n_bins = [np.unique(x).shape[0], np.unique(y).shape[0]]
            discretizor = []
            aux = threshold_binning_builder(x, n_bins[0])
            discretizor.append(aux)
            aux = threshold_binning_builder(y, n_bins[1])
            discretizor.append(aux)
        elif type(bins[0]) == type(bins[1]) == int:
            n_bins = bins
            discretizor = []
            aux = threshold_binning_builder(x, n_bins[0])
            discretizor.append(aux)
            aux = threshold_binning_builder(y, n_bins[1])
            discretizor.append(aux)
        elif type(bins[0]) in [np.array, np.ndarray]:
            discretizor = []
            aux = threshold_binning_builder(x, bins[0])
            discretizor.append(aux)
            aux = threshold_binning_builder(y, bins[1])
            discretizor.append(aux)
    elif type(bins) == int:
        if bins == 0:
            n_bins = [np.unique(x).shape[0], np.unique(y).shape[0]]
        else:
            n_bins = [bins, bins]
        discretizor = []
        aux = threshold_binning_builder(x, n_bins[0])
        discretizor.append(aux)
        aux = threshold_binning_builder(y, n_bins[1])
        discretizor.append(aux)
    elif bins is None:
        n_bins = [np.unique(x).shape[0], np.unique(y).shape[0]]
        discretizor = []
        aux = threshold_binning_builder(x, n_bins[0])
        discretizor.append(aux)
        aux = threshold_binning_builder(y, n_bins[1])
        discretizor.append(aux)

    ## 2. Discretization
    # Discretization
    x = discretize_with_thresholds(x, discretizor[0])
    y = discretize_with_thresholds(y, discretizor[1])
    # Needed vars
    values1 = np.unique(x)
    values2 = np.unique(y)
    values = [values1, values2]

    ## 3. Preparing lag
    if timelag > 0:
        x, y = x[timelag:], y[:y.shape[0]-timelag]
    elif timelag < 0:
        x, y = x[:x.shape[0]-timelag], y[timelag:]

    ## 4. Computing probs
    probs = np.zeros((values1.shape[0], values2.shape[0]))
    for i in range(values1):
        for j in range(values2):
            val1 = values1[i]
            val2 = values2[j]
            aux = np.logical_and(x == val1, y == val2)
            probs[i, j] = np.sum(aux)

    ## 5. Normalization
    if normalize:
        probs /= np.sum(probs)

    return probs, discretizor, values


def prob_x(X, n_bins=0, individually=True, normalize=True):
    """Study the probability of the signal to be in some range of values.

    Parameters
    ----------
    X: array_like, shape (N, Melements)
        the time series.
    n_bins: int
        the number of bins selected.

    Returns
    -------
    probs: array_like, shape (n_bins, M) or shape (n_bins,)
        the density of each bins for all the signals globally or individually.
    discretizor: array_like, shape (N, Melements, n_bins+1)
        the discretizor matrix.
    values: array_like, shape (M,)
        the code of the values of each regime in the dynamics.

    TODO
    ----
    ...
    """

    ## 0. Needed variables
    X = X.reshape(-1) if individually else X
    X = X.reshape((X.shape[0], 1)) if len(X.shape) == 1 else X

    ## 1. Ensure discretization
    if np.all(n_bins == 0) or not np.any(n_bins):
        values = np.unique(X)
        n_bins = values.shape[0]
        discretizor = None
    else:
        discretizor = threshold_binning_builder(X, n_bins)
        X = discretize_with_thresholds(X, discretizor)
        values = np.unique(X)

    ## 2. Compute probabilities
    probs = np.zeros((values.shape[0], X.shape[1]))
    for i in range(values):
        val = values[i]
        probs[i, :] = np.sum(probs == val, axis=0)

    ## 3. Normalization
    if normalize:
        probs = np.divide(probs.astype(float), np.sum(probs, axis=0))

    return probs, discretizor, values
