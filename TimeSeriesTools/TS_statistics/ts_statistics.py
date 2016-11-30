
"""
This module contains the functions which computes interesting statistics. They
can be applied to continious time-series for obtaining important statistics for
the study of causality or prediction.

TODO
----
Pass to the individuals a list of values.
values_info or samevals for all.
"""

from itertools import product
import numpy as np


from TimeSeriesTools.Transformation.value_discretization import \
    threshold_binning_builder, discretize_with_thresholds


def build_ngram_from_arrays(post, pres, L):
    """
    """
    return Y


def prob_ngram_x(X, L, bins=None, samevals=True, normalize=True):
    """Function to compute the joints probability of a process and
    the states of the times before.

    Parameters
    ----------
    X: array_like, shape (N, M)
        the dynamics.
    L: int
        previous times to watch.
    bins:
        information of how to discretize.
    samevals: boolean, list or array
        all the signals of the system have the same states available.
    normalize: boolean
        to normalize the counting in order to get probabilities.

    Returns
    -------
    probs: array_like
        the joint probabilities. Probabilities of each combinations of states.
    discretizor: array_like
        the discretizor array.
    values: array_like
        the values of the probs considered.

    TODO
    ----
    Correct values.
    """

    ## 0. Formating inputs and preparing needed variables
    n = X.shape[1]

    ## 1. Discretization
    discretizor = threshold_binning_builder(X, bins)
    X = discretize_with_thresholds(X, discretizor)
    # needed variables
    if type(samevals) == bool:
        if samevals:
            samevals = [np.unique(X) for i in range(n)]
        else:
            samevals = [np.unique(X[:, i]) for i in range(n)]
    else:
        if type(samevals) == np.ndarray:
            samevals = [samevals for i in range(n)]
        elif type(samevals) == list:
            pass

    ## 2. Compute probability
    probs = [[] for i in range(n)]
    for i in range(n):
        probs[i], _ = prob_ngram_ind(X[:, i], X[:, i], L, False,
                                     samevals[i], normalize)
    # Format output
    probs = np.array(probs)
    values = samevals

    return probs, discretizor, values


def prob_ngram_xy(X, L, bins=None, auto=True, samevals=True, normalize=True):
    """Function to compute the joints probability of a process and
    the states of the times before between each one of the pairs of variables
    possible.

    Parameters
    ----------
    X: array_like, shape (N, M)
        the dynamics.
    L: int
        previous times to watch.
    bins:
        information of how to discretize.
    auto: boolean
        if we compute the auto influence.
    samevals: boolean or list or array
        all the signals of the system have the same states available.
    normalize: boolean
        to normalize the counting in order to get probabilities.

    Returns
    -------
    probs: array_like
        the joint probabilities. Probabilities of each combinations of states.
    discretizor: array_like
        the discretizor array.
    values: array_like
        the values of the probs considered.

    TODO
    ----
    Correct values.
    """

    ## 0. Formating inputs and preparing needed variables
    n = X.shape[1]
    pairs = product(range(n), range(n))

    ## 1. Discretization
    discretizor = threshold_binning_builder(X, bins)
    X = discretize_with_thresholds(X, discretizor)
    # needed variables
    if type(samevals) == bool:
        if samevals:
            samevals = [np.unique(X) for i in range(n)]
        else:
            samevals = [np.unique(X[:, i]) for i in range(n)]
    else:
        if type(samevals) == np.ndarray:
            samevals = [samevals for i in range(n)]
        elif type(samevals) == list:
            pass

    ## 2. Compute probability
    probs = [[[] for j in range(n)] for i in range(n)]
    for p in pairs:
        p0, p1 = p[0], p[1]
        probs[p0][p1], _ = prob_ngram_ind(X[:, p0], X[:, p1], L, auto,
                                          [samevals[p0], samevals[p1]],
                                          normalize)
    # Format output
    probs = np.array(probs)
    values = samevals

    return probs, discretizor, values


def prob_ngram_ind(x, y, L, auto=True, samevals=True, normalize=True):
    """Function to compute the joints probability of a process and
    the states of the times before. We assume they are discretized.

    Parameters
    ----------
    X: array_like, shape (N, M)
        the dynamics.
    L: int
        previous times to watch.
    auto: boolean
        if we compute the auto influence.
    samevals: boolean or array or list
        all the signals of the system have the same states available. If array
        we pass the actual values.
    normalize: boolean
        to normalize the counting in order to get probabilities.

    Returns
    -------
    probs: array_like
        the joint probabilities. Probabilities of each combinations of states.
    values: array_like
        the values of the probs considered.

    """
    ## 0. Compute needed variables
    assert x.shape[0] == y.shape[0]
    nt = x.shape[0]
    if type(samevals) == bool:
        if samevals:
            xvalues = np.unique(np.hstack([x, y]))
            yvalues = xvalues
            values = xvalues
        else:
            xvalues = np.unique(x)
            yvalues = np.unique(y)
            values = [xvalues, yvalues]
    else:
        if type(samevals) == np.ndarray:
            xvalues = samevals
            yvalues = samevals
            values = [xvalues, yvalues]
        elif type(samevals) == list:
            values = samevals
            xvalues = np.array(values[0])
            yvalues = np.array(values[1])

    ## 1. Formatting arrays as index
    aux = np.ones(x.shape)*np.inf
    for i in range(xvalues.shape[0]):
        aux[x == xvalues[i]] = i
    x = aux[:]
    aux = np.ones(y.shape)*np.inf
    for i in range(yvalues.shape[0]):
        aux[y == yvalues[i]] = i
    y = aux[:]

    ## 2. Building lags matrices
    xt = x[:nt-L].reshape((nt-L, 1))
    if auto:
        # Computing dependant times
        xv = np.vstack([x[l:nt-L+l] for l in range(1, L+1)]).T
        yv = np.vstack([y[l:nt-L+l] for l in range(1, L+1)]).T
        # Aggregating
        Xv = np.hstack([xt, xv, yv])
    else:
        # Compute dependant
        yv = np.vstack([y[l:nt-L+l] for l in range(1, L+1)]).T
        # Aggregating
        Xv = np.hstack([xt, yv])

    ## 3. Counting statistics
    dim = [xvalues.shape[0]]
    if auto:
        dim = dim + L*[xvalues.shape[0]]
    dim = dim + L*[yvalues.shape[0]]
    dim = tuple(dim)

    probs = np.zeros(dim)
    for i in range(Xv.shape[0]):
        probs[tuple(Xv[i, :])] = probs[tuple(Xv[i, :])] + 1

    if normalize:
        probs /= np.sum(probs)

    return probs, values


def prob_xy(X, bins=0, maxl=0, samevals=True):
    """Wrapper for prob_xy_ind. It computes the probability for all a matrix of
    dynamics.

    Parameters
    ----------
    X: array_like, shape (N, Melements)
        the signals of the system.
    bins: int
        number of bins to study the statistics.
    maxl: int
        the max lag times to compute.
    samevals: boolean or array or list
        all the signals of the system have the same states available. If array
        we pass the actual values.

    Returns
    -------
    probs: array_like, shape (Nelements, Nelements, maxl, n_bins, n_bins)
        the probability for each pair of elements, timelag and range of values.
    discretizor: array_like
        the discretizor array.
    values: array_like
        the values of the probs considered.

    """

    ## 0. Formating inputs and preparing needed variables
    n = X.shape[1]
    pairs = product(range(n), range(n))

    ## 1. Discretization
    discretizor = threshold_binning_builder(X, bins)
    X = discretize_with_thresholds(X, discretizor)
    # needed variables
    if type(samevals) == bool:
        if samevals:
            samevals = [np.unique(X) for i in range(n)]
        else:
            samevals = [np.unique(X[:, i]) for i in range(n)]
    else:
        if type(samevals) == np.ndarray:
            samevals = [samevals for i in range(n)]
        elif type(samevals) == list:
            pass

    ## 2. Compute probability
    probs = [[[[] for k in range(maxl+1)] for j in range(n)] for i in range(n)]
    for p in pairs:
        for tlag in range(maxl+1):
            p0, p1 = p[0], p[1]
#            probs[p0, p1, tlag], _, _ = prob_xy_ind(X[:, p0], X[:, p1],
#                                                    samevals, 0, tlag)
            probs[p0][p1][tlag], _, _ = prob_xy_ind(X[:, p0], X[:, p1],
                                                    samevals, 0, tlag)
    # Format output
    probs = np.array(probs)
    values = samevals

    return probs, discretizor, values


def prob_xy_ind(x, y, samevals=True, bins=0, timelag=0, normalize=True):
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
    # Descriptions of situations
    situation1 = bins is None or bins == 0
    situation2 = type(bins) in [list, tuple, np.ndarray]
    situation2 = situation2 and np.array(bins).shape[0] == 2
    # Building discretizor
    if situation1:
        bins = [None, None]
    elif situation2:
        bins = bins
    else:
        bins = [bins, bins]
    discretizor = []
    discretizor.append(threshold_binning_builder(x, bins[0]))
    discretizor.append(threshold_binning_builder(y, bins[1]))

    ## 2. Discretization
    # Discretization
    x = discretize_with_thresholds(x, discretizor[0])
    y = discretize_with_thresholds(y, discretizor[1])
    # Needed vars (from discretized arrays)
    if type(samevals) == bool:
        if samevals:
            xvalues = np.unique(np.hstack([x, y]))
            yvalues = xvalues
            values = xvalues
        else:
            xvalues = np.unique(x)
            yvalues = np.unique(y)
            values = [xvalues, yvalues]
    else:
        if type(samevals) == np.ndarray:
            xvalues = samevals
            yvalues = samevals
            values = [xvalues, yvalues]
        elif type(samevals) == list:
            values = samevals
            xvalues = values[0]
            yvalues = values[1]

    ## 3. Preparing lag
    if timelag > 0:
        x, y = x[timelag:], y[:y.shape[0]-timelag]
    elif timelag < 0:
        x, y = x[:x.shape[0]+timelag], y[-timelag:]

    ## 4. Computing probs
    X = np.vstack([x, y]).T
    probs = compute_joint_probs(X, values, normalize)

    return probs, discretizor, values


def prob_x(X, n_bins=0, individually=True, normalize=True):
    """Study the probability of the signal to be in some range of values.

    Parameters
    ----------
    X: array_like, shape (N, Melements)
        the time series.
    n_bins: int
        the number of bins selected.
    individually: boolean
        compute the probabilities for each variable or the global
        probabilities.
    normalize: boolean
        return the counts normalized as probabilities or just the counts.

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
    X = X.reshape(-1) if not individually else X
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
    for i in range(values.shape[0]):
        val = values[i]
        probs[i, :] = np.sum(X == val, axis=0)

    ## 3. Normalization
    if normalize:
        probs = np.divide(probs.astype(float), np.sum(probs, axis=0))

    return probs, discretizor, values


def compute_joint_probs(Y_t, values, normalize=True):
    """Function used to compute joint probability from a group of stochastic
    processes represented by Y_t, with different possible states given by
    values. We assume discretization.

    Parameters
    ----------
    Y_t: array_like, shape (n_t, n_vars)
        different discrete stochastic processes.
    values: list or numpy.ndarray
        the values of each stochastic variable can take.
    normalize: boolean
        if we return a normalized array.
    Returns
    -------
    probs: array_like
        the joint probability of being in each possible state of the product
        space.

    """

    ## 0. Format variables and build needed
    Y_t = Y_t if len(Y_t.shape) == 2 else Y_t.reshape((Y_t.shape[0], 1))
    n_vars = Y_t.shape[1]
    n_t = Y_t.shape[0]
    # Build values
    if values == []:
        for i in range(n_vars):
            values.append(np.unique(Y_t[:, i]))
    elif type(values) == np.ndarray:
        values = [values for i in range(n_vars)]

    ## 1. Transform to indexes
    aux = np.ones(Y_t.shape)*np.inf
    for i in range(n_vars):
        for j in range(values[i].shape[0]):
            aux[Y_t[:, i] == j, i] = j
    Y_t = aux[:, :]

    ## 2. Building probs matrix
    dim = tuple([vals.shape[0] for vals in values])
    probs = np.zeros(dim)
    for i in range(n_t):
        indices = tuple(Y_t[i, :])
        probs[indices] = probs[indices] + 1

    ## 3. Normalizing
    if normalize:
        probs /= np.sum(probs)

    return probs
