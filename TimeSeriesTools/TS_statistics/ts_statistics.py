
"""
This module contains the functions which computes interesting statistics. They
can be applied to continious time-series for obtaining important statistics for
the study of causality or prediction.
"""

from itertools import combinations
import numpy as np


def prob_xy_ind(x, y, bins=0, timelag=0):
    #if type(bins) == np.array or type(bins)==list or
    #type(bins)==tuple or np.ndarray:
    if type(bins) in [np.array, list, tuple, np.ndarray]:
        if np.any(bins):
            n_bins = max(np.unique(x).shape[0], np.unique(y).shape[0])
            bins_edges = np.histogram2d(range(n_bins), range(n_bins),
                                        n_bins)[1:]
        else:
            bins_edges = bins
    elif type(bins) == int:
        if bins == 0:
            n_bins = max(np.unique(x).shape[0], np.unique(y).shape[0])
        else:
            n_bins = bins
        bins_edges = np.histogram2d(range(n_bins), range(n_bins), n_bins)[1:]

    if timelag > 0:
        x, y = x[timelag:], y[:y.shape[0]-timelag]
    elif timelag < 0:
        x, y = x[:x.shape[0]-timelag], y[timelag:]

    return np.histogram2d(x, y, bins_edges)[0]


def prob_x(X, n_bins=0):
    values = np.unique(X)
    if not n_bins:
        n_bins = values.shape[0]
    bins_edges = np.histogram(values, bins=n_bins)[1]
    density_matrix = np.vstack([np.histogram(X[:, i], bins=bins_edges,
                               density=True)[0] for i in range(X.shape[1])])
    return density_matrix, values


def prob_xy(X, n_bins=0):
    values = np.unique(X)
    if not n_bins:
        n_bins = values.shape[0]
    bins_edges = np.histogram(values, bins=n_bins)[1]
    pairs = combinations(range(X.shape[1]), 2)
    probs = [prob_xy_ind(X[:, pair[0]], X[:, pair[1]],
                         bins_edges) for pair in pairs]
    pairs = [pair for pair in pairs]
    return probs, pairs
