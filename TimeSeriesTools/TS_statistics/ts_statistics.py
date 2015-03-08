
"""
This module contains the functions which computes interesting statistics. They
can be applied to continious time-series for obtaining important statistics for
the study of causality or prediction.
"""

from itertools import product
import numpy as np


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
    probs: array_like, shape(Nelements, Nelements, maxl, n_bins, m_bins)
        the probability for each pair of elements, timelag and range of values.

    """

    # Formating inputs and preparing needed variables
    values = np.unique(X)
    if bins == 0:
        bins = values.shape[0]
    bins_edges = np.histogram2d(values, values, bins=bins)[1:]
    n_bins = bins_edges[0].shape[0]-1
    m_bins = bins_edges[1].shape[0]-1
    n = X.shape[1]
    nt = X.shape[0]
    pairs = product(range(n), range(n))

    # Compute probability
    probs = np.zeros((n, n, maxl+1, n_bins, m_bins))
    for p in pairs:
        for tlag in range(maxl+1):
            probs[p[0], p[1], tlag, :, :], _ = prob_xy_ind(X[tlag:, p[0]],
                                                           X[:nt-tlag, p[1]],
                                                           bins_edges,
                                                           tlag)

    return probs, bins_edges


def prob_xy_ind(x, y, bins=0, timelag=0):
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
    probs: array_like, shape (n_bins, M) or shape (n_bins,)
        the density of each bins for all the signals globally or individually.
    bins_edges: array_like, shape (n_bins+1,)
        the edges of the bins.

    """

    # Preparing bins_edges
    values1 = np.unique(x)
    values2 = np.unique(y)
    if type(bins) in [list, tuple, np.array, np.ndarray]:
        if not np.any(bins):
            n_bins = np.max(np.unique(x).shape[0], np.unique(y).shape[0])
            bins_edges = np.histogram2d(values1, values2,
                                        n_bins, normed=True)[1:]
        elif type(bins) in [np.array, np.ndarray]:
            bins_edges = (bins, bins)
        elif type(bins[0]) in [np.array, np.ndarray]:
            bins_edges = bins
        elif type(bins[0]) == type(bins[1]) == int:
            bins_edges = np.histogram2d(values1, values2, n_bins,
                                        normed=True)[1:]
    elif type(bins) == int:
        if bins == 0:
            n_bins = max(np.unique(x).shape[0], np.unique(y).shape[0])
        else:
            n_bins = bins
        bins_edges = np.histogram2d(values1, values2, n_bins,
                                    normed=True)[1:]

    # Preparing lag
    if timelag > 0:
        x, y = x[timelag:], y[:y.shape[0]-timelag]
    elif timelag < 0:
        x, y = x[:x.shape[0]-timelag], y[timelag:]

    # Computing probs
    probs = np.histogram2d(x, y, bins_edges, normed=True)[0]

    return probs, bins_edges


def prob_x(X, n_bins=0, individually=True):
    """Study the probability of the signal to be in some range of values.

    Parameters
    ----------
    X: array_like, shape (N, Melements)
        the time series.
    n_bins: int
        the number of bins selected.

    Returns
    -------
    density_matrix: array_like, shape (n_bins, M) or shape (n_bins,)
        the density of each bins for all the signals globally or individually.
    bins_edges: array_like, shape (n_bins+1,)
        the edges of the bins.

    TODO
    ----
    Danger density, normalize yourself

    """
    values = np.unique(X)
    X = X.reshape((X.shape[0], 1)) if len(X.shape) == 1 else X
    if np.all(n_bins == 0) or not np.any(n_bins):
        n_bins = values.shape[0]
    bins_edges = np.histogram(values, bins=n_bins)[1]
    if individually:
        density_matrix = np.vstack([np.histogram(X[:, i], bins=bins_edges,
                                   density=True)[0]
                                    for i in range(X.shape[1])]).T
    else:
        density_matrix = np.histogram(X.reshape(-1), bins=bins_edges,
                                      density=True)[0]
    return density_matrix, bins_edges
