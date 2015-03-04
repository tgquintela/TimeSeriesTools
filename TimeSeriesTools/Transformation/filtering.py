
"""
This module contains all the functions needed for smoothing or filtering a
time-serie.
"""

import numpy as np
from scipy import signal


########################## Wrapper to all functions ###########################
###############################################################################
def general_filtering(Y, method, kwargs):
    """Wrapper function to contain all the possible smoothing functions in
    order to be easy and quick usable for other parts of this package.
    """

#    if method == 'order_filter':
#        Ys = signal.order_filter(Y, **kwargs)
#    elif method == 'medfilt':
#        Ys = signal.medfilt(Y, **kwargs)
#    elif method == 'wiener':
#        Ys = signal.wiener(Y, **kwargs)
#    elif method == 'lfilter':
#        Ys = signal.lfilter(Y, **kwargs)
#    elif method == 'filtfilt':
#        Ys = signal.filtfilt(Y, **kwargs)
    if method == 'savgol_filter':
        Ys = signal.savgol_filter(Y, **kwargs)
    elif method == 'savitzky_golay':
        Ys = savitzky_golay_matrix(Y, **kwargs)
    elif method == 'weighted_MA':
        Ys = smooth_weighted_MA_matrix(Y, **kwargs)
    elif method == 'fft_passband':
        Ys = fft_passband_filter(Y, **kwargs)
    elif method == 'reweighting':
        Ys = general_reweighting(Y, **kwargs)
    ## DISCRETE TS
    elif method == 'collapse':
        Ys = collapser(Y, **kwargs)
    elif method == 'substitution':
        Ys = substitution(Y, **kwargs)

    return Ys


################################## functions ##################################
###############################################################################
def savitzky_golay_matrix(Y, window_size, order):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    This function acts as a wrapper of the savitzky_golay function.

    Parameters
    ----------
    Y : array_like, shape (N,M)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv : int
        the order of the derivative to compute (default=0 means only smoothing)

    Returns
    -------
    Ys : ndarray, shape (N,M)
        the smoothed signal (or it's n-th derivative).

    Examples
    --------
    import numpy as np
    t = np.linspace(-4, 4, 500)
    n = 10
    X = [np.exp(-t**2) + np.random.normal(0, 0.05, t.shape) for i in range(n)]
    X = np.vstack(X).T
    Xsg = savitzky_golay_matrix(X, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, X[:,0], label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, Xsg[:,0], 'r', label='Filtered signal')
    plt.legend()
    plt.show()

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688

    See also
    --------
    savitzky_golay

    More information
    ----------------
    TODO: vectorization. np.vectorize?
    """
    Ys = np.zeros(Y.shape)
    for i in range(Y.shape[1]):
        Ys[:, i] = savitzky_golay(Y[:, i], window_size, order)
    return Ys


def smooth_weighted_MA_matrix(Y, window_len=11, window='hanning', *args):
    '''Smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    This is the wrapper to accept a multidimensional array.

    Parameters
    ----------
    Y : numpy.array, shape (N,M)
        the input signal
    window_len : int
        the dimension of the smoothing window; should be an odd integer
    window : str
        the type of window {'flat','hanning','hamming','bartlett','blackman'}
        flat window will produce a moving average smoothing.

    Returns
    -------
    Ys : array_like
        the smoothed signal

    Examples
    --------
    import numpy as np
    t = np.linspace(-4, 4, 500)
    n = 10
    X = [np.exp(-t**2) + np.random.normal(0, 0.05, t.shape) for i in range(n)]
    X = np.vstack(X).T
    Xsg = smooth_weighted_MA_matrix(X, window_len=31)
    import matplotlib.pyplot as plt
    plt.plot(t, X[:,0], label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, Xsg[:,0], 'r', label='Filtered signal')
    plt.legend()
    plt.show()

    See also
    --------
    smooth_weighted_MA_matrix

    More information
    ----------------
    TODO: vectorization. np.vectorize?
    '''
    Ys = np.zeros(Y.shape)
    for i in range(Y.shape[1]):
        Ys[:, i] = smooth_weighted_MA(Y[:, i], window_len, window, *args)
    return Ys


def fft_passband_filter(y, f_low=0, f_high=1, axis=0):
    '''Pass band filter using fft for real 1D signal.

    Parameters
    ----------
    y : array_like shape (N,M)
        the values of the time history of the signal.
    f_low : int
        low pass niquist frequency (1 = samplin_rate/2)
    f_high : int
        high  cut niquist frequency (1 = samplin_rate/2)
    axis : int
        axis along the which each individual signal is represented.

    Returns
    -------
    ys : ndarray, shape (N,M)
        the smoothed signal (or it's n-th derivative).

    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp(-t**2) + np.random.normal(0, 0.05, t.shape)
    ysg = fft_passband_filter(y, f_low=0, f_high=0.05)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()

    See also
    --------
    smooth_weigthed_MA_matrix, savitzky_golay_matrix, np.fft.fft, np.fft.ifft

    References
    ----------
    .. [1] Nasir. Ahmed, Discrete-Time Signals and Systems
       (Reston, Virginia: Reston Publication Company, 1983), pp. 243-258.
    .. [2] Raphael C. Gonzalez and Richard E. Woods. Digital Image Processing
       (Boston: Addison-Wesley, 1992), pp. 201-213, 244.
    .. [3] Belle A. Shenoi, Introduction to digital signal processing and
       filter design. John Wiley and Sons(2006) p.120. ISBN 978-0-471-46482-2

    '''

    # Length of the transformed signal
    n = y.shape[axis]
    N = int(2**(np.ceil(np.log(n)/np.log(2))))

    # Signal to filter expressed in the frequency domain.
    SIG = np.fft.fft(y, n=N, axis=axis)

    # Transform the cuts in units of array elements.
    n_low = int(np.floor((N-1)*f_low/2)+1)
    fract_low = 1-((N-1)*f_low/2-np.floor((N-1)*f_low/2))
    n_high = int(np.floor((N-1)*f_high/2)+1)
    fract_high = 1-((N-1)*f_high/2-np.floor((N-1)*f_high/2))

    # Creation of the slide
    s = [slice(None) for i in range(y.ndim)]

    # High-pass filter
    if f_low > 0:
        # Defining the signal regarding the cuts
        s[axis] = 0
        SIG[s] = 0
        s[axis] = slice(1, n_low)
        SIG[s] = 0
        s[axis] = n_low
        SIG[s] *= fract_low
        s[axis] = -n_low
        SIG[s] *= fract_low
        if n_low != 1:
            s[axis] = slice(-n_low+1, None)
            SIG[s] = 0

    # Low-pass filter
    if f_high < 1:
        # Defining the signal regarding the cuts
        s[axis] = n_high
        SIG[s] *= fract_high
        s[axis] = slice(n_high+1, -n_high)
        SIG[s] = 0
        s[axis] = -n_high
        SIG[s] *= fract_high

    s[axis] = slice(0, n)
    # Inverse transformation in order to recover the signal smoothed.
    ys = np.real(np.fft.ifft(SIG, axis=axis)[s])
    return ys


################## 1-array smoothing
#############################################
def smooth_weighted_MA(x, window_len=11, window='hanning', *args):
    '''Smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    This function acts over continious-valued signals.

    Parameters
    ----------
    x : numpy.array, shape (N,)
        the input signal
    window_len : int
        the dimension of the smoothing window; should be an odd integer
    window : str
        the type of window {'flat','hanning','hamming','bartlett','blackman'}
        flat window will produce a moving average smoothing.

    Returns
    -------
    y : array_like
        the smoothed signal

    Examples
    --------
    import numpy as np
    t=np.linspace(-2,2,500)
    x=np.sin(t)+np.random.randn(len(t))*0.1
    y=smooth_weighted_MA(x,27)
    import matplotlib.pyplot as plt
    plt.plot(t, x, label='Noisy signal')
    plt.plot(t, np.sin(t), 'k', lw=1.5, label='Original signal')
    plt.plot(t, y, 'r', label='Filtered signal')
    plt.legend()
    plt.show()

    See also
    --------
    savitzky_golay, np.hamming, np.hanning, np.bartlett, np.blackman,
    scipy.signal.get_window

    Code
    ----
    http://wiki.scipy.org/Cookbook/SignalSmooth

    More information
    ----------------
    TODO: the window parameter could be the window itself if an array instead
          of a string
    NOTE: length(output) != length(input), to correct this:
    '''

    ## 0. Check inputs
    type0 = ['flat']
    type1 = ['hamming', 'hanning', 'bartlett', 'blackman']
    type2 = ['triang', 'flattop', 'parzen', 'bohman',
             'blackmanharris', 'nuttall', 'barthann']
    type3 = ['kaiser', 'gaussian', 'slepian', 'chebwin']
    type4 = ['general_gaussian']
    type5 = ['alpha_trim_window']
    type6 = ['median_window', 'snn_1d']

    if x.ndim != 1:
        raise ValueError("Smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in type0 + type1 + type2 + type3 + type4 + type5 + type6:
        raise ValueError("Window is on of the possible values.")
    if window in type3 and len(args) < 0:
        raise ValueError("Window selected needs an extra parameter.")

    ## 1. Creation of the window
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    if window in type0:  # moving average
        w = np.ones(window_len, 'd')
    elif window in type1:
        w = eval('np.'+window+'(window_len)')
    elif window in type2:
        inputs = "'"+window+"'"
        w = eval('signal.get_window('+inputs+',window_len)')
    elif window in type3:
        inputs = "('"+window+"',"+str(args[0])+")"
        w = eval('signal.get_window('+inputs+',window_len')
    elif window in type4:
        inputs = "('"+window+"',"+str(args[0])+','+str(args[1])+")"
        w = eval('signal.get_window('+inputs+',window_len')
    elif window in type5:
        w = eval(window+'(s,args[0])')
    elif window in type6:
        w = eval(window+'(s)')

    ## 2. Convolution
    y = np.convolve(w/w.sum(), s, mode='valid')

    ## 3. Format output: Same shape as input
    if window_len % 2:
        y = y[(window_len/2):-(window_len/2)]
    else:
        y = y[(window_len/2 - 1):-(window_len/2)]

    return y


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv : int
        the order of the derivative to compute
        (default = 0 means only smoothing)

    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).

    See also
    --------
    smooth_weigthed_MA

    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688

    Code
    ----
    http://nbviewer.ipython.org/github/pv/
    SciPy-CookBook/blob/master/ipython/SavitzkyGolay.ipynb

    """
    from math import factorial

    ## 0. Control of input and setting needed variables
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:  # , msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size - 1) // 2

    ## 1. Precompute coefficients
    b = np.mat([[k**i for i in order_range]
               for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)

    ## 2. Pad the signal at the extremes with values taken from the signal
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))

    ## 3. Convolve
    ys = np.convolve(m[::-1], y, mode='valid')
    return ys


############################# auxiliar functions ##############################
###############################################################################
from scipy.stats import tmean, scoreatpercentile


def trim_mean(arr, proportion):
    '''
    '''
    #TODO: windowing (window len) and avoid error try:
    # except: np.sort(p)[window_len/2]
    percent = proportion*100.
    lower_lim = scoreatpercentile(arr, percent/2)
    upper_lim = scoreatpercentile(arr, 100-percent/2)
    tm = tmean(arr, limits=(lower_lim, upper_lim), inclusive=(False, False))
    return tm


def alpha_trim_window(window, alpha):
    '''This function built a window in which weight each time in order to do
    the moving average for each window. It prepares the window with the
    weights in order to perform the trimmed average of the window.
    When the alpha is too big we have the median filter.

    Parameters
    ----------
    window : array_like, shape (N,)
        the values of the signal in a given window.
    alhpa : float, in the interval [0,1]
        proportion of values to be trimmed.

    Returns
    -------
    ys : array_like, shape (N)
        a {0,1}-array with the weights for the mean.

    See also
    --------
    np.hamming, np.hanning, np.bartlett, np.blackman

    Notes
    -----
    The trimmed average consist in exclude from the average the alpha
    proportion of extreme values. Trimmed mean is a non-linear smoothing
    filter, which recall the disadvantage of weighting all the points with the
    same value, and it do it truncating or 'trimming' before averaging. This
    filter it is not edge-preserving.

    Examples
    --------
    >>> w = np.random.normal(0, 0.05, 10)
    >>> alpha_trim_window(w, alpha=0.1)
    array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0])

    References
    ----------
    .. [1] Hall, M. Smooth operator: smoothing seismic horizons and attributes.
       The Leading Edge 26 (1), January 2007, p16-20. doi:10.1190/1.2431821
    .. [2] http://subsurfwiki.org/wiki/Smoothing_filter

    '''
    # calculate upper and lower limits
    percent = alpha * 100.
    lower_limit = scoreatpercentile(window, percent/2)
    upper_limit = scoreatpercentile(window, 100-percent/2)
    # Extract logical vector
    w = np.logical_and(window >= lower_limit, window <= upper_limit)
    w = window.astype(int)
    if np.sum(w) == 0:
        w[window.argsort()[window.shape[0]/2]] = 1
    window = w
    return window


def median_window(window):
    w = np.zeros(window.shape)
    w[window.argsort()[window.shape[0]/2]] = 1
    return w


def snn_1d(window):
    window_len = window.shape[0]
    w = np.zeros(window_len)
    center = window_len/2

    w[center] = 1
    w[window_len - center - 1] = 1

    value_center = np.mean((w*window)[w*window != 0])

    for i in range(window_len/2):
        res_a = abs(window[i] - value_center)
        res_o = abs(window[window_len-1-i] - value_center)
        if res_a <= res_o:
            w[i] = 1
        elif res_o <= res_a:
            w[window_len-1-i] = 1
    return w


########################### discrete functions ################################
###############################################################################
def collapser(regimes, reference, collapse_info):
    """General functions which performs filtering/smoothing in discrete
    time-series functions collapsing values to a concrete time points.

    Parameters
    ----------
    regimes: array_like, shape (N, M)
        signals in which are represented some regimes. This regimes are usually
        asigned to integer values.
    reference: int or float
        the value considered reference regime.
    collapse_info: dict, str, function and list
        the information of how collapse each possible regime of each time-serie
        of each element of the system.

    Returns
    -------
    event_ts: array_like, shape (N, M)
        the signals with the same shape as regimes but now with the collapse
        elements that are not the reference regime.

    """

    ## 0. Creation of needed variables
    values = np.unique(regimes)
    M = regimes.shape[1]

    # A collapsing for each value
    if type(collapse_info) == dict:
        assert np.all([c in values for c in collapse_info.keys()])
        # dictionary creation
        aux = {}
        for c in collapse_info.keys():
            aux[c] = lambda x: general_collapser_func(x, collapse_info[c])
        # list of dicts creation
        collapse = [aux for i in range(M)]
    # A precoded transformation for all elements and values
    elif type(collapse_info) == str:
        aux_0 = lambda x: general_collapser_func(x,  method=collapse_info)
        # dictionary creation
        aux = {}
        for v in values:
            aux[v] = aux_0
        # list of dicts creation
        collapse = [aux for i in range(M)]
    # A personal transformation for all elements and values
    elif type(collapse_info).__name__ == 'function':
        aux_0 = collapse_info
        # dictionary creation
        aux = {}
        for v in values:
            aux[v] = aux_0
        # list of dicts creation
        collapse = [aux for i in range(M)]
    # A transformation for each element and possibly each value
    elif type(collapse_info) == list:
        # list of dicts creation
        aux = []
        for coll in collapse_info:
            if type(coll) == dict:
                assert np.all([c in values for c in coll.keys()])
                # dictionary creation
                aux_d = {}
                for c in coll.keys():
                    aux_d[c] = lambda x: general_collapser_func(x, coll[c])
                # list of dicts appending
                aux.append(aux_d)
            elif type(coll) == str:
                aux_d = {}
                for v in values:
                    aux_d[v] = lambda x: general_collapser_func(x, coll)
                aux.append(aux_d)
            elif type(coll).__name__ == 'function':
                aux_d = {}
                for v in values:
                    aux_d[v] = coll
                aux.append(aux_d)
        collapse = aux

    ## 1. Collapsing process
    event_ts = reference*np.ones(regimes.shape)
    for i in range(M):
        for val in values:
            # Compute vector of changes in regime.
            APbool = (regimes[:, i] == val).astype(int)
            # Collapsing to
            APindices = collapse[i][val](APbool)
            # Inputation the result
            event_ts[APindices, i] = val

    return event_ts


def general_collapser_func(APbool, method):
    """Specific function which performs the collapsing. It is called by the
    collapser. It is used over boolean masks.

    Parameters
    ----------
    APbool: array_like boolean, shape(N,)
        boolean mask over the regime we are interested in collapse.
    method: str or function
        method to collapse regarding the available information.

    Returns
    -------
    APindices: array_like
        the integer number of the indices of this 1d array that the values are
        collapsed to.

    """

    ## 1. Preparing for collapsing
    # Obtaining ups and downs
    diffe = np.diff(APbool, axis=0)
    ups = np.where(diffe == 1)[0] + 1
    downs = np.where(diffe == -1)[0] + 1
    # Correcting the borders
    if diffe[np.where(diffe)[0][-1]] == 1:
        downs = np.hstack([downs, np.array([APbool.shape[0]])])
    if diffe[np.where(diffe)[0][0]] == -1:
        ups = np.hstack([np.array([0]), ups])
    # Ranges in which there are changes
    ranges = np.vstack([ups, downs]).T

    ## 2. Collpase process
    # Preparing for collapsing
    if type(method).__name__ == 'function':
        f = method
        method = 'personal'
    # Select the function and apply
    if method == 'center':
        APindices = ranges.mean(axis=1).round().astype(int)
    elif method == 'initial':
        APindices = ranges.min(axis=1).round().astype(int)
    elif method == 'final':
        APindices = ranges.max(axis=1).round().astype(int)
    elif method == 'personal':
        APindices = f(ranges)

    return APindices


def substitution(X, subs={}):
    """This function is used to substitute values of the signals for others.

    Parameters
    ----------
    X: array_like
        signals of the system.
    subs: dict
        the values of the time series that you want to substitute as keys and
        the values for which you want to substitute as values of the dict.

    Returns
    -------
    X: array_like
        the initial with substituted values.

    """

    # Substitution
    for val in subs.keys():
        indexs = X.where(X == val)
        X[indexs] = subs[val]

    return X


def general_reweighting(Y, method, kwargs):
    """The general reweighting methods to change the values of the time series
    depending of the global value of the system. It could be used in spiking
    systems in which we want to weight more the activity of spiking alone than
    the ones which spike alltogether.

    Parameters
    ----------
    Y: array_like, shape (N, M)
        the signals of the system.
    method: str, optional or function
        the method selected. If it is a str, we choose a precoded method. If it
        is a function we apply this non-precoded in-module function.
    kwargs: dict
        variables needed for the choosen method.

    """

    if method == 'power_sutera':
        Ys = power_sutera_reweighing(Y)

    elif type(method).__name__ == 'function':
        Ys = method(Y, **kwargs)

    return Ys


def power_sutera_reweighing(Y, f_pow=lambda x: 1):
    """Re-weights the time series giving more value to the values of the time
    serie when there are a low global activity.

    References
    ---------
    .. [1] Antonio Sutera et al. Simple connectome inference from partial
    correlation statistics in calcium imaging

    """

    ## 0. Prepare variables needed
    m = Y.shape[1]
    global_y = np.sum(Y, axis=1)

    ## 1. Transformation
    Yt = np.zeros(Y.shape)
    for j in range(m):
        Yt[:, j] = np.power((Y[:, j] + 1.),
                            np.power((1.+np.divide(1., global_y)),
                                     f_pow(global_y)))
    # Correct global 0
    Yt[global_y == 0, :] = 1.

    return Yt
