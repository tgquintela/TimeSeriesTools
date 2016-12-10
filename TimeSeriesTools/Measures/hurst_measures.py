
"""
Hurst measure
-------------


"""

import numpy as np
from ..utils.fit_utils import general_multiscale_fit


############################ Hurst general swicher ############################
###############################################################################
def hurst(X, scales=None, method='RS', fit_method='loglogLSQ'):
    """Compute the Hurst parameter of X. If the output H = 0.5, the behavior of
    the time-series is similar to random walk. If H<0.5, the time-series cover
    less "distance" than a random walk, viceversa.

    Parameters
    ----------
    X : array_like, shape(N,)
        a 1-D real time series.
    method: str, optional {'aggvar'}
        the method used to compute the Hurst parameter.
            * 'aggvar': aggregated variances to estimate the parameter.
            * 'RS': rescalated range analysis to estimate the parameter.
            * 'RS_alternative': alternative method of RS-like estimation.
            * 'higuchi' estimate the hurst parameter of a given sequence with
                higuchi's method.
            * 'per': 
            * 'peng': 

%               'aggvar': use aggvar function to estimate.
%               'RS': use RS function to estimate.
%               'per': use per function to estimate.

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
    [1] .. Code in Matlab
        http://www.mathworks.com/matlabcentral/fileexchange/
        19148-hurst-parameter-estimate
    [2] .. Code in Python
        http://www.drtomstarke.com/index.php/calculation-of-the-hurst-exponent-
        to-test-for-trend-and-mean-reversion
    [3] .. Code in R
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
    if method == 'RS':
        H = hurst_rs(X, scales, fit_method)
    elif method == 'RS_alternative':
        H = hurst_rs_alternative(X, scales, fit_method)
    elif method == 'aggvar':
        H = hurst_aggvar(X, scales, fit_method)
    elif method == 'per':
        H = hurst_per(X, scales, fit_method)
    elif method == 'peng':
        H = hurst_peng(X, scales, fit_method)
    elif method == 'higuchi':
        H = hurst_higuchi(X, scales, fit_method)
    return H


############################ Hurst general swicher ############################
###############################################################################
def hurst_rs(X, T=None, fit_method='loglogLSQ'):
    """Measures using Rescaled range analysis (RS).

    Parameters
    ----------
    X: np.ndarray
        the time-series.
    T: np.ndarray
        the size of the ranges we are going to use to compute R_S value.
    fit_method: str
        the fit method name.

    Returns
    -------
    H: float
        the Hurst parameter.

    """
    ## Create possible scales to inspect
    M = create_RS_scales_sequence(X, T)
    ## Compute values of length for that scales
    L, M = hurst_rs_values(X, M)
    ## Fit of the function
    measure = general_multiscale_fit(L, M, fit_method)
    ## Computation of the Hurst parameter
    H = null_hurst_measure(measure)
    return H


def hurst_rs_alternative(X, T=None, fit_method='loglogLSQ'):
    """Measures using Rescaled range analysis (RS).

    Parameters
    ----------
    X: np.ndarray
        the time-series.
    T: np.ndarray
        the size of the ranges we are going to use to compute R_S value.
    fit_method: str
        the fit method name.

    Returns
    -------
    H: float
        the Hurst parameter.

    """
    ## Create possible scales to inspect
    M = create_RS_scales_sequence(X, T)
    ## Compute values of length for that scales
    L, M = hurst_alternative_rs_values(X, M)
    ## Fit of the function
    measure = general_multiscale_fit(L, M, fit_method)
    ## Computation of the Hurst parameter
    H = null_hurst_measure(measure)
    return H


def hurst_aggvar(X, M=None, fit_method='loglogLSQ'):
    """

    Parameters
    ----------
    X: np.ndarray
        the time-series.
    T: np.ndarray
        the size of the ranges we are going to use to compute R_S value.
    fit_method: str
        the fit method name.

    Returns
    -------
    H: float
        the Hurst parameter.

    """
    ## Create possible scales to inspect
    M = create_aggvar_scales_sequence(X, M)
    ## Compute values of length for that scales
    L, M = hurst_aggvar_values(X, M)
    ## Fit of the function
    measure = general_multiscale_fit(L, M, fit_method)
    ## Computation of the Hurst parameter
    H = aggvar_hurst_measure(measure)
    return H


def hurst_peng(X, M=None, fit_method='loglogLSQ'):
    """

    Parameters
    ----------
    X: np.ndarray
        the time-series.
    T: np.ndarray
        the size of the ranges we are going to use to compute R_S value.
    fit_method: str
        the fit method name.

    Returns
    -------
    H: float
        the Hurst parameter.

    """
    ## Create possible scales to inspect
    M = create_peng_scales_sequence(X, M)
    ## Compute values of length for that scales
    L, M = hurst_peng_values(X, M)
    ## Fit of the function
    measure = general_multiscale_fit(L, M, fit_method)
    ## Computation of the Hurst parameter
    H = peng_hurst_measure(measure)
    return H


def hurst_per(X, M=None, fit_method='loglogLSQ'):
    """

    Parameters
    ----------
    X: np.ndarray
        the time-series.
    T: np.ndarray
        the size of the ranges we are going to use to compute R_S value.
    fit_method: str
        the fit method name.

    Returns
    -------
    H: float
        the Hurst parameter.

    """
    ## Create possible scales to inspect
    M = create_scales_periodogram_sequence(X, M)
    ## Compute values of length for that scales
    L, M = hurst_per_values(X, M)
    ## Fit of the function
    assert(len(L) == len(M))
#    measure = general_multiscale_fit(L, M, fit_method)
    measure = 0.5
    ## Computation of the Hurst parameter
    H = per_hurst_measure(measure)
    return H


def hurst_higuchi(X, M=None, fit_method='loglogLSQ'):
    """Hurst computation with higuchi-like computation of the fractal dimension
    of the timeseries (D).
    The method scans fluctuations of the signal by investigating the defined
    length of the curve in a given interval with different lengths.

    Parameters
    ----------
    X: np.ndarray
        the time-series.
    T: np.ndarray
        the size of the ranges we are going to use to compute R_S value.

    Returns
    -------
    H: float
        the Hurst parameter.

    References
    ----------
    [1].. Higuchi, T. (1988). Approach to an Irregular Time Series on the Basis
    of the Fractal Theory. Physica D , 31(2):277-283

    """
    ## Create possible scales to inspect
    M = create_scales_higuchi_sequence(X, M)
    ## Compute values of length for that scales
    L, M = hurst_higuchi_values(X, M)
    ## Fit of the function
    measure = general_multiscale_fit(L, M, fit_method)
    ## Computation of the Hurst parameter
    H = higuchi_hurst_measure(measure)
    return H


################### Hurst parameter with different methods ####################
###############################################################################
def hurst_rs_values(X, T=None):
    """Measures using Rescaled range analysis (RS).

    Parameters
    ----------
    X: np.ndarray
        the time-series.
    T: np.ndarray
        the size of the ranges we are going to use to compute R_S value.

    Returns
    -------
    R_S: np.ndarray
        the computed values using RS.
    T: np.ndarray
        the size of the ranges we are going to use to compute R_S value.

    """
    # Parsing inputs
    assert(len(X.shape) == 1)
    if T is None:
        T = create_RS_scales_sequence(X)

    # Initialazing vectors
    S_T = np.zeros(len(T))
    R_T = np.zeros(len(T))

    for i in xrange(len(T)):
        Z_T = np.cumsum(X[:T[i]] - X[:T[i]].mean())
        # Std vector calculation
        S_T[i] = np.std(X[:T[i]])
        # Vector of range between trend in i and in the rest of the TS
        R_T[i] = np.max(Z_T[:T[i]]) - np.min(Z_T[:T[i]])
    # Logaritmic ratio of max difference with and std
    R_S = np.divide(R_T, S_T)

    return R_S, T


def hurst_alternative_rs_values(X, T=None):
    """Measures using Rescaled range analysis (RS).

    Parameters
    ----------
    X: np.ndarray
        the time-series.
    T: np.ndarray
        the size of the ranges we are going to use to compute R_S value.

    Returns
    -------
    R_S: np.ndarray
        the computed values using RS.
    T: np.ndarray
        the size of the ranges we are going to use to compute R_S value.

    !!! WARNING

    """
    assert(len(X.shape) == 1)
    if type(T) != np.ndarray:
        T = create_RS_scales_sequence(X)
    if len(T) != len(X):
        T = create_RS_scales_sequence(X)

    # Initialization
    n_t = len(X)
    CS_X = np.cumsum(X)
    Ave_T = CS_X/T

    # Initialazing vectors
    S_T = np.zeros(n_t)
    R_T = np.zeros(n_t)

    # Compute for each point in the serie the accumulate value of
    # std and range in order to get this values with different sample size
    for i in xrange(1, n_t):
        # Std vector calculation
        S_T[i] = np.std(X[:i+1])
        # Vector of range between trend in i and in the rest of the TS
        X_T = CS_X - T * Ave_T[i]
        R_T[i] = np.max(X_T[:i+1]) - np.min(X_T[:i+1])
    # Logaritmic ratio of max difference with and std
    R_S = np.divide(R_T[1:], S_T[1:])
    T = T[1:]
    return R_S, T


def hurst_aggvar_values(X, M):
    """

    Parameters
    ----------
    X: np.ndarray
        the time-series.
    M: np.ndarray
        the size of the ranges we are going to use to compute aggregated
        variance.

    Returns
    -------
    V: np.ndarray
        the aggregated variance.
    M: np.ndarray
        the size of the ranges we are going to use to compute aggregated
        variance.

    References
    ----------
    .. [1] J. Beran. Statistics For Long-Memory Processes. Chapman and Hall,
    1994; pag. 92

    """
    if M is None:
        M = create_aggvar_scales_sequence(X)
    n_t = len(X)
    n = len(M)
    V = np.zeros(n)

    for i in range(n):
        m = M[i]
        k = int(np.floor(n_t/M[i]))
        ## Computation of the sample variance of the sample means
        V[i] = np.var(np.mean(X[:m*k].reshape((m, k)), 0))
    return V, M


def hurst_peng_values(X, M):
    """
        * 'peng' estimate the hurst parameter of a given sequence with
        residuals of regression method.

    """
    # Calculate aggregate level
    n_t = len(X)
    FBM = np.cumsum(X)
    n = len(M)

    # Calculate residuals under different aggregate level.
    Goble_residuals = np.zeros(n)
    for i in range(n):
        m = M[i]
        k = int(np.floor(np.float(n_t)/m))
        matrix_FBM = FBM[:m*k].reshape((m, k))

        # Compute local residuals
        Local_residual = np.zeros(k)
        for j in range(k):
            y = matrix_FBM[:, j]
            vv = np.stack([np.arange(1, m+1), np.ones(m)]).T
            x, resid, rank, s = np.linalg.lstsq(vv, y)
            Local_residual[j] = resid
        Goble_residuals[i] = np.mean(Local_residual)
    return Goble_residuals, M


def hurst_per_values(X, M):
    """

        * 'per' estimate the hurst parameter of a given sequence with
        periodogram method.

    """
    n_t = len(X)
    Xk = np.fft.fft(X)
    P_origin = np.abs(Xk)**2/(2*np.pi*n_t)
    P = P_origin[1:int(np.floor(n_t/2.))]

    M = create_scales_periodogram_sequence(X, M)

    # Use the lowest 20% part of periodogram to estimate the similarity.
    x = M[:int(np.floor(len(P)/5.))]
    y = P[:int(np.floor(len(P)/5.))]

    return y, x


def hurst_higuchi_values(X, M):
    """Hurst computation with higuchi-like computation of the fractal dimension
    of the timeseries (D).
    The method scans fluctuations of the signal by investigating the defined
    length of the curve in a given interval with different lengths.

    Parameters
    ----------
    X: np.ndarray
        the time-series.
    T: np.ndarray
        the size of the ranges we are going to use to compute R_S value.

    Returns
    -------
    curve_length: np.ndarray
        the lenghts for each related lag.
    M: np.ndarray
        the different lags (and inderectly scales) we want to use to compute
        lenghts of the signal for each one.

    References
    ----------
    [1].. Higuchi, T. (1988). Approach to an Irregular Time Series on the Basis
    of the Fractal Theory. Physica D , 31(2):277-283

    Example
    -------
    >>> X = (np.random.random(10000) - 0.5).cumsum()
    >>> L, M = hurst_higuchi_values(X)

    """
    ## Preparation of inputs
    n_t = len(X)
    # Creation of the sequences
    M = create_scales_higuchi_sequence(X, M)
    n = len(M)

    curve_length = np.zeros(n)
    for i in range(n):
        # Lag definition
        m = M[i]
        # Size of the timeseries with that lag step
        k = int(np.floor((n_t-m)/float(m)))
        # Defintion of all possible time series measures
        temp_length = np.zeros((m, k))
        for j1 in range(m):
            for j2 in range(k):
                temp_length[j1, j2] = np.abs(X[j1+j2*m]-X[j1+(j2-1)*m])
        C = float(n_t-1)/m**3
        curve_length[i] = C*np.sum(np.mean(temp_length, axis=1))
    return curve_length, M


######################### Create scales to inspect ts #########################
###############################################################################
def create_RS_scales_sequence(X, sequence='complete'):
    if type(sequence) == np.ndarray:
        T = sequence
    elif type(str):
        if sequence == 'power':
            T = np.array([2**x for x in range(1, int(np.log2(len(X)-1))+1)])
        elif sequence == 'complete':
            T = np.linspace(2, len(X), len(X))
    return T


def create_aggvar_scales_sequence(X, sequence=10):
    if type(sequence) == np.ndarray:
        return sequence
    n_t = len(X)
    ## This is the whole range of possible values but it is better to reduce it
#    M = np.floor(np.logspace(0, np.log10(np.floor(n_t/2)), 20))
    # Reducing the range in high M values
    M = np.floor(np.logspace(0, np.log10(np.floor(n_t/sequence)), 40))
    M = np.unique(M[M > 1])
    return M


def create_peng_scales_sequence(X, sequence=None):
    if type(sequence) == np.ndarray:
        return sequence
    n_t = len(X)
    mlarge = np.floor(n_t/5)
    msmall = max([10, np.log10(n_t)**2])
    M = np.floor(np.logspace(np.log10(msmall), np.log10(mlarge), 50))
    M = np.unique(M).astype(int)
    return M


def create_scales_higuchi_sequence(X, M=None):
    if type(M) == np.ndarray:
        return M
    n_t = len(X)
    mlarge = int(np.floor(n_t/5.))
    M = np.floor(np.logspace(0, np.log10(mlarge), 50)).astype(int)
    M = np.unique(M[M > 1])
    return M


def create_scales_periodogram_sequence(X, M=None):
    if type(M) == np.ndarray:
        return M
    n_t = len(X)
    M = (np.pi/n_t)*np.arange(1, int(np.floor(.5*n_t)))
    return M


###################### Compute hurst from fit parameter #######################
###############################################################################
def null_hurst_measure(measure):
    """Hurst computation parameter from some slope fit.

    Parameters
    ----------
    measure: float
        the slope of the fit using some method.

    Returns
    -------
    H: float
        the Hurst parameter.

    """
    # Compute measure
    return float(measure)


def higuchi_hurst_measure(measure):
    """Hurst computation parameter from higuchi slope fit.

    Parameters
    ----------
    measure: float
        the slope of the fit using higuchi method. It is related with the
        fractal dimension (D) of the timeseries.

    Returns
    -------
    H: float
        the Hurst parameter.

    """
    D = -measure
    H = float(2 - D)
    return H


def aggvar_hurst_measure(measure):
    """Hurst computation parameter from aggvar slope fit.

    Parameters
    ----------
    measure: float
        the slope of the fit using aggvar method.

    Returns
    -------
    H: float
        the Hurst parameter.

    """
    H = float((measure+2.)/2.)
    return H


def per_hurst_measure(measure):
    """Hurst computation parameter from periodogram slope fit.

    Parameters
    ----------
    measure: float
        the slope of the fit using per method.

    Returns
    -------
    H: float
        the Hurst parameter.

    """
    # Compute measure
    H = float((1-measure)/2.)
    return H


def peng_hurst_measure(measure):
    """Hurst computation parameter from peng slope fit.

    Parameters
    ----------
    measure: float
        the slope of the fit using peng method.

    Returns
    -------
    H: float
        the Hurst parameter.

    """
    # Compute measure
    H = float(measure/2.)
    return H
