
"""
Fractal dimensions measures
---------------------------
Measures based on computing a fractal dimension of the timeseries.

"""

import numpy as np


def pfd(X):
    """Compute Petrosian Fractal Dimension of a time series.

    Parameters
    ----------
    X : array_like, shape(N,)
        a 1-D real time series.

    Returns
    -------
    Pfd : float
        Petrosian Fractal dimension of the time series.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.random.randn(4096)
    >>> pfd(a)

    References
    ----------
    .. [1] A. Petrosian, "Kolmogorov complexity of finite sequences and
       recognition of different preictal EEG patterns," in Proceedings of 8th
       IEEE Symposium on Computer-Based Medical Systems, 1995.

    """

    # Compute First order difference
    D = np.diff(X)
    # number of sign changes in derivative of the signal
    N_delta = 0
    for i in xrange(1, D.shape[0]):
        if D[i]*D[i-1] < 0:
            N_delta += 1
    n = X.shape[0]
    Pfd = np.log10(n)/(np.log10(n)+np.log10(n/n+0.4*N_delta))
    return Pfd


def hfd(X, kMax=0):
    """Compute Higuchi's Fractal Dimension of a time series X.

    Parameters
    ----------
    X : array_like, shape(N,)
        a 1-D real time series.

    Returns
    -------
    Pfd : float
        Petrosian Fractal dimension of the time series.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.random.randn(4096)
    >>> hfd(a)

    References
    ----------
    .. [1] Higuchi, T.: Approach to an irregular time series on the basics of
       the fractal theory. Physica D: Nonlinear Phenomena 31(2), 277-283 (1988)
    .. [2] Wang, Q., Sourina, O., Nguyen, M.K.: Fractal dimension based
       algorithm for neurofeedback games. In: Proc. CGI 2010, Singapore,
       p. SP25 (2010) p. 5

    See also
    --------
    hurst_higuchi

    """

    # Set to default value kMax based on ref [2]
    if kMax == 0:
        kMax = int(2 ** (np.log2(X.shape[0]) - 4))

    L = np.zeros((kMax, 1))
    x = np.zeros((kMax, 2))
    N = X.shape[0]
    # Loop for all the sizes of kMax
    for k in xrange(1, kMax):
        Lk = np.zeros(k)
        for m in xrange(0, k):
            Lmk = 0
            for i in xrange(1, int(np.floor((N-m)/k))):
                Lmk += abs(X[m+i*k] - X[m+i*k-k])
            Lmk = Lmk*(N - 1)/np.floor((N - m)/float(k))/k
            Lk[m] = Lmk
        L[k] = np.log(np.mean(Lk))
        x[k, :] = np.array([np.log(float(1) / k), 1])

    (p, r1, r2, s) = np.linalg.lstsq(x, L)
    return p[0][0]
