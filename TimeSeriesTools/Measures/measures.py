
"""
This module is composed by a group of functions that computes some measures
in the time-series individually.
"""

import numpy as np
import scipy

import utils


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
#################### ENTROPY ##################################################
def entropy(X1, base=None):
    """Entropy measure of a given time-serie.

    References
    ----------
    ..[1] http://orange.biolab.si/blog/2012/06/15/joint-entropy-in-python/

    """

    # Format matrix in order to have a column-format dynamics
    if len(X1.shape) < 2:
        X1 = np.matrix(X1).T

    # Initialize variables
    [rows, cols] = X1.shape
    entropies = np.zeros(shape=(cols, 1))

    # Calculation of the entropy for each one of the columns
    for i in range(cols):
        X = X1[:, i]
        probs = [np.mean(X == c) for c in set(X)]
        entropies[i] = scipy.stats.entropy(probs, base=base)

    return entropies[:, 0]


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
#################### HURST ####################################################
def hurst(X):
    """Compute the Hurst exponent of X. If the output H = 0.5, the behavior of
    the time-series is similar to random walk. If H<0.5, the time-series cover
    less "distance" than a random walk, viceversa.

    Parameters
    ----------
    X : array_like, shape(N,)
        a 1-D real time series.

    Returns
    -------
    H : float
        Hurst exponent

    Examples
    --------
    >>> import numpy as np
    >>> a = np.random.randn(4096)
    >>> hurst(a)
    0.5057444

    References
    ----------
    .. [1] Hurst, H.E. (1951). Trans. Am. Soc. Civ. Eng. 116: 770.
    .. [2] Hurst, H.E.; Black, R.P.; Simaika, Y.M. (1965). Long-term storage:
       an experimental study. London: Constable.
    .. [3] Mandelbrot, Benoit B., The (Mis)Behavior of Markets, A Fractal View
       of Risk, Ruin and Reward (Basic Books, 2004), pp. 186-195

    Code
    ----
    [.] Code in Matlab
        http://www.mathworks.com/matlabcentral/fileexchange/
        19148-hurst-parameter-estimate
    [.] Code in Python
        http://www.drtomstarke.com/index.php/calculation-of-the-hurst-exponent-
        to-test-for-trend-and-mean-reversion
    [.] Code in R
        https://r-forge.r-project.org/scm/viewvc.php/pkg/PerformanceAnalytics/R
        /HurstIndex.R?view=markup&root=returnanalytics

    Applications
    ------------
    Finance, neuro, ...

    Improvements
    ------------
    - Fit the powerlaw in a likehood way.
    ..[-] Clauset, Cosma Rohilla Shalizi, M. E. J. Newman (2009).
      "Power-law distributions in empirical data". SIAM Review 51: 661-703
    - ...

    """
    # Initialization
    N = X.shape[0]
    T = np.linspace(1, N, N)
    CS_X = np.cumsum(X)
    Ave_T = CS_X/T

    # Initialazing vectors
    S_T = np.zeros((N))
    R_T = np.zeros((N))

    # Compute for each point in the serie the accumulate value of
    # std and range in order to get this values with different sample size
    for i in xrange(N):
        # Std vector calculation
        S_T[i] = np.std(X[:i+1])
        # Vector of range between trend in i and in the rest of the TS
        X_T = CS_X - T * Ave_T[i]
        R_T[i] = np.max(X_T[:i + 1]) - np.min(X_T[:i + 1])
    # Logaritmic ratio of max difference with and std
    R_S = R_T / S_T
    R_S = np.log(R_S)

    # Compute the log TS
    n = np.log(T).reshape(N, 1)

    # Fitting the linear regression by least squares
    H = np.linalg.lstsq(n[1:], R_S[1:])[0]
    return H[0]

###############################################################################


###############################################################################
#################### pfd ####################################################
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

###############################################################################


###############################################################################
###################### hfd ####################################################
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

    """

    # Set to default value kMax based on ref [2]
    if kMax == 0:
        kMax = 2 ** (np.log2(X.shape[0]) - 4)

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


###############################################################################
########################## Svd Entropy ########################################
def svd_entropy(X, tau=1, D=1):
    """Compute SVD Entropy from time series.

    Notes
    -------------
    """

    # Substitute to a function.
    Y = utils.sliding_embeded_transf(X, tau, D)

    # Singular value descomposition
    W = np.linalg.svd(Y, compute_uv=0)

    # Normalize singular values
    W /= np.sum(W)

    # Compute entropy of svd
    H_svd = - np.sum(W * np.log(W))

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
    return H_sp


###############################################################################
########################## Fisher information #################################
def fisher_info(X, tau, D):
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
    Y = utils.sliding_embeded_transf(X, tau, D)
    # Singular value descomposition
    W = np.linalg.svd(Y, compute_uv=0)
    # Compute Fisher information
    FI = np.sum((W[1:] - W[:W.shape[0]-1])**2)/W[:W.shape[0]-1]
    FI = FI[0]
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
    Ave: integer, optional
        The average value of the time series
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
    N_X = X.shape[0]
    # Compute mean
    Ave = np.mean(X)
    # Integrate of the signal
    Y = np.cumsum(X)
    Y -= Ave
    # Compute an array of box size (integers) in ascending order reducing box
    # size dependant as it is explained in the Notes
    L = np.floor(X.shape[0]*1/(2**np.array(range(4, int(np.log2(N_X))-4))))
    if np.all(L != 0):
        raise Exception("Too big box for the given time series.")

    # F(n) of different given box length n
    F = np.zeros(L.shape[0])
    for i in xrange(L.shape[0]):
        # for each box length L[i]
        n = int(L[i])
        # for each box
        for j in xrange(0, N_X, n):
            if j+n < N_X:
                c = range(j, j+n)
                # coordinates of time in the box
                c = np.vstack([c, np.ones(n)]).T
                # the value of data in the box
                y = Y[j:j+n]
                # add residue in this box
                F[i] += np.linalg.lstsq(c, y)[1]
        F[i] /= ((N_X/n)*n)
    F = np.sqrt(F)

    # Computation of alpha
    Alpha = np.linalg.lstsq(np.vstack([np.log(L),
                            np.ones(L.shape[0])]).T,
                            np.log(F))[0][0]
    return Alpha
