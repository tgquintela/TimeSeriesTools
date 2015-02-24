
"""Module in which is implemented the dynamic time warping algorithm.
"""

import numpy as np


def dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1)):
    """ Computes the DTW of two sequences.

    Parameters
    ----------
    x : array_like
        time serie.
    y : array_like
        time serie.
    dist: function
        distance function used to as the cost in dtw.
        Default value the L1 norm.

    Returns
    -------
    cum_dist : float
        the total distance between the two time-series.
    D : array_like
        the cumulative cost matrix.
    trackeback: array_like
        the optimal wrap path.

    """

    # Create array with 2-d
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # Initialize cost matrix
    r, c = x.shape[0], y.shape[0]
    D = np.zeros((r + 1, c + 1))
    D[0, 1:] = np.inf
    D[1:, 0] = np.inf

    # Compute dynamic optimization of the cost
    for i in range(r):
        for j in range(c):
            D[i+1, j+1] = dist(x[i], y[j])
    for i in range(r):
        for j in range(c):
            D[i+1, j+1] += min(D[i, j], D[i, j+1], D[i+1, j])

    # Exclude init point
    D = D[1:, 1:]
    cum_dist = D[-1, -1] / sum(D.shape) # Why division?
    trackeback = _trackeback(D)

    return cum_dist, D, trackeback


def _trackeback(D):

    i, j = np.array(D.shape) - 1
    p, q = [i], [j]

    while (i > 0 and j > 0):
        # Compute the indexes of the minimum cost
        tb = np.argmin((D[i-1, j-1], D[i-1, j], D[i, j-1]))
        if (tb == 0):
            i = i - 1
            j = j - 1
        elif (tb == 1):
            i = i - 1
        elif (tb == 2):
            j = j - 1

        p.insert(0, i)
        q.insert(0, j)

    p.insert(0, 0)
    q.insert(0, 0)
    return (np.array(p), np.array(q))



