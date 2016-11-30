
"""
Information theory measures
---------------------------
Collection of measures which uses the Information Theory.

"""

import numpy as np
import scipy

from ..utils.sliding_utils import sliding_embeded_transf
from ..utils.fit_utils import general_multiscale_fit


###############################################################################
#################### ENTROPY ##################################################
def entropy(ts, base=None):
    """Entropy measure of a given time-serie. That function is only appliable
    for discrete valued time-serie.

    Parameters
    ----------
    ts: np.ndarray
        the values of the measures of the time series in some time sample.

    Returns
    -------
    ent: float
        the value of the entropy of the possible values of the time-serie.

    References
    ----------
    ..[1] http://orange.biolab.si/blog/2012/06/15/joint-entropy-in-python/

    """

    # Format matrix in order to have a column-format dynamics
    if len(ts.shape) < 2:
        ts = np.atleast_2d(ts).T

    # Initialize variables
    rows, cols = ts.shape
    entropies = np.zeros(shape=(cols, 1))

    # Calculation of the entropy for each one of the columns
    for i in range(cols):
        X = ts[:, i].squeeze()
        probs = [np.mean(X == c) for c in set(X)]
        entropies[i] = scipy.stats.entropy(probs, base=base)
    entropies = float(entropies.ravel()[0])

    return entropies


#def shan_entropy(c):
#    c_normalized = c/float(np.sum(c))
#    c_normalized = c_normalized[np.nonzero(c_normalized)]
#    H = -sum(c_normalized* np.log(c_normalized))
#    return H
#
## TODEPRECATE
#def entropy(X1):
###http://orange.biolab.si/blog/2012/06/15/joint-entropy-in-python/
#    if len(X1.shape)<2:
#        X1 = np.matrix(X1).T
#    [rows, cols] = X1.shape
#    entropies = np.zeros(shape=(cols,1))
#    for i in range(cols):
#        X = X1[:,i]
#        probs = [np.mean(X == c) for c in set(X)]
#        entropies[i] = np.sum(-p * np.log2(p) for p in probs)
#    #print entropies
#    return entropies
#
## FASTER Possible alternative
#def entropy2(X1, base = None):
###http://orange.biolab.si/blog/2012/06/15/joint-entropy-in-python/
#    if len(X1.shape)<2:
#        X1 = np.matrix(X1).T
#    [rows, cols] = X1.shape
#    entropies = np.zeros(shape=(cols,1))
#    for i in range(cols):
#        X = X1[:,i]
#        probs = np.histogram(X, bins = len(set(X)) ,density=True)[0]
#        entropies[i] = scipy.stats.entropy(probs, base=base)
#    return entropies


###############################################################################
########################## Svd Entropy ########################################
def svd_entropy(X, tau=1, D=1):
    """Compute SVD Entropy from time series.

    Notes
    -------------
    """

    # Substitute to a function.
    Y = sliding_embeded_transf(X, tau, D)
    # Singular value descomposition
    W = np.linalg.svd(Y, compute_uv=0)
    # Normalize singular values
    W /= np.sum(W)
    # Compute entropy of svd
    H_svd = - np.sum(W*np.log(W))
    # Format output
    H_svd = float(H_svd)
    return H_svd


###############################################################################
########################## Spectral Entropy ###################################
def spectral_entropy(X, bins=50):
    """Compute spectral entropy of a time series. Spectral entropy is the
    entropy associated to the entropy in the distribution of the power of a
    time series between its frequency spectrum space.

    Parameters
    ----------
    X : array_like, shape(N,)
        a 1-D real time series.
    bins : int
        number of bins in which we want to discretize the frequency spectrum
        space in order to compute the entropy.

    Returns
    -------
    H_sp : float
        Spectral entropy

    TODO:
    ----
    Fs and its influence in the entropy. And part of the entropy dividing.
    Dependent on the number of bins!!!!!!!!!!!!!!!!
    """
    # Power spectral
    ps = np.abs(np.fft.fft(X))**2
    # binning:
    psd, freq = np.histogram(ps, bins, normed=True)
    # Compute entropy (?)
    H_sp = - np.sum(psd * np.log2(psd+1e-16))/np.log2(psd.shape[0])
    H_sp = float(H_sp)
    return H_sp


###############################################################################
########################## Fisher information #################################
def fisher_info(X, tau=1, D=1):
    """ Compute Fisher information of a time series.

    Parameters
    ----------
    X : array_like, shape(N,)
        a 1-D real time series.
    tau : integer
        the lag or delay when building a embedding sequence. tau will be used
        to build embedding matrix and compute singular values.
    D : integer
        the embedding dimension to build an embedding matrix from a given
        series. DE will be used to build embedding matrix and compute
        singular values if W or M is not provided.

    Returns
    -------
    FI : integer
        Fisher information

    """

    # Substitute to a function.
    Y = sliding_embeded_transf(X, tau, D)
    # Singular value descomposition
    W = np.linalg.svd(Y, compute_uv=0)
#    W /= np.sum(W)
    # Compute Fisher information
#    FI = np.sum((W[1:] - W[:W.shape[0]-1])**2)/W[:W.shape[0]-1]
#    FI = FI[0]
    FI = -1.*np.sum(W*np.log(W))
#    print FI, type(FI)
#    FI = float(FI[0])
    return FI


###############################################################################
########################## Fisher information #################################
def dfa(X):
    """Compute Detrended Fluctuation Analysis from a time series X. There is
    an adaptation function of the one provided in pyEGG.

    The first step to compute DFA is to integrate the signal. Let original
    series be X= [x(1), x(2), ..., x(N)].
    The integrated signal Y = [y(1), y(2), ..., y(N)] is obtained as follows
    y(k) = \sum_{i=1}^{k}{x(i)-Ave} where Ave is the mean of X.

    The second step is to partition/slice/segment the integrated sequence Y
    into boxes. At least two boxes are needed for computing DFA. Box sizes are
    specified by the L argument of this function. By default, it is from 1/5 of
    signal length to one (x-5)-th of the signal length, where x is the nearest
    power of 2 from the length of the signal, i.e., 1/16, 1/32, 1/64, 1/128,...
    In each box, a linear least square fitting is employed on data in the box.
    Denote the series on fitted line as Yn. Its k-th elements, yn(k),
    corresponds to y(k).

    For fitting in each box, there is a residue, the sum of squares of all
    offsets, difference between actual points and points on fitted line.
    F(n) denotes the square root of average total residue in all boxes when box
    length is n, thus
    Total_Residue = \sum_{k=1}^{N}{(y(k)-yn(k))}
    F(n) = \sqrt(Total_Residue/N)
    The computing to F(n) is carried out for every box length n. Therefore, a
    relationship between n and F(n) can be obtained. In general, F(n) increases
    when n increases.

    Finally, the relationship between F(n) and n is analyzed. A least square
    fitting is performed between log(F(n)) and log(n). The slope of the fitting
    line is the DFA value, denoted as Alpha. To white noise, Alpha should be
    0.5. Higher level of signal complexity is related to higher Alpha.

    Parameters
    ----------
    X: array_like, shape(N,)
        a time series
    L: 1-D Python list of integers
        A list of box size, integers in ascending order

    Returns
    -------
    Alpha : integer
        the result of DFA analysis, thus the slope of fitting line of log(F(n))
        vs. log(n).

    Examples
    --------
    >>> import numpy as np
    >>> a = np.random.randn(4096)
    >>> dfa(a)
    0.490035110345

    Reference
    ---------
    .. [1] Peng C-K, Havlin S, Stanley HE, Goldberger AL. Quantification of
       scaling exponents and crossover phenomena in nonstationary heartbeat
       time series. _Chaos_ 1995;5:82-87
    .. [2] http://www.physionet.org/tutorials/fmnc/node5.html

    Notes
    -----
    This value depends on the box sizes very much. When the input is a white
    noise, this value should be 0.5. But, some choices on box sizes can lead to
    the value lower or higher than 0.5, e.g. 0.38 or 0.58.

    Based on many test, I set the box sizes from 1/5 of signal length to one
    (x-5)-th of the signal length, where x is the nearest power of 2 from the
    length of the signal, i.e., 1/16, 1/32, 1/64, 1/128, ...
    """

    ## 1. Compute values
    # Size X
    N_t = len(X)
    # Compute mean
    Ave = np.mean(X)
    # Integrate of the signal
    Y = np.cumsum(X) - Ave
    # Compute an array of box size (integers) in ascending order reducing box
    # size dependant as it is explained in the Notes
    ### TODO: Revise and do it like the hurst parameter measures!!!!
    L = np.sort(np.floor(N_t*(1./(2.**np.arange(2, int(np.log2(N_t)))))))

    # F(n) of different given box length n
    F = np.zeros(len(L))
    for i in xrange(len(L)):
        # for each box length L[i]
        n = int(L[i])
        # for each box
        for j in xrange(0, N_t, n):
            if j+n < N_t:
                c = range(j, j+n)
                # coordinates of time in the box
                c = np.vstack([c, np.ones(n)]).T
                # the value of data in the box
                y = Y[j:j+n]
                # add residue in this box
                F[i] += np.linalg.lstsq(c, y)[1]
        F[i] /= ((N_t/n)*n)
    F = np.sqrt(F)

    # Computation of alpha (generalize the fit)
    Alpha = general_multiscale_fit(F, L)
#    Alpha = np.linalg.lstsq(np.vstack([np.log(L),
#                            np.ones(L.shape[0])]).T,
#                            np.log(F))[0][0]
    Alpha = float(Alpha)
    return Alpha
