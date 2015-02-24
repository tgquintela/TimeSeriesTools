#
import numpy as np
import pandas as pd
from scipy import signal
#import statistics
from sklearn.mixture import GMM

import pyCausality.utils as Utils

#from scipy.interpolate import UnivariateSpline


############################# Time Series ##############################
########################################################################


# if there are low temporal resolution it is difficult to interpret spikes
# in other way than the hight of the point due to the noise, and the low
# information of the temporal neigthbourhood of each sample.

# TODO: Wrappers for the different functionalities.
# smoothing
# Transformation (now only ups-downs)
# Thresholds selection (now only manual)
# Regime detection (application)

### index
# Smoothing              40
# TS Transformation     460
# Thresholds selection  550
# Regime detection      560
# Align spk waveform
# Project spk waveform


## TODO list:
# MTEO peak detection (OpenElectrolophy)
#


########################################################################
############################# Smoothing ##############################
########################################################################
########
################################
###### inputs:
## matrix
## parameters
###### outputs:
## matrix
def filtering(Y, method, **kwargs):
    """
    """

    if method == 'order_filter':
        Ys = signal.order_filter(Y, **kwargs)
    elif method == 'medfilt':
        Ys = signal.medfilt(Y, **kwargs)
    elif method == 'wiener':
        Ys = signal.wiener(Y, **kwargs)
    elif method == 'lfilter':
        Ys = signal.lfilter(Y, **kwargs)
    elif method == 'filtfilt':
        Ys = signal.filtfilt(Y, **kwargs)
    elif method == 'savgol_filter':
        Ys = signal.savgol_filter(Y, **kwargs)

    return Ys


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

    for i in range(Y.shape[1]):
        Y[:, i] = savitzky_golay(Y[:, i], window_size, order)
    return Y


def smooth_weighted_MA_matrix(Y, window_len=11, window='hanning'):
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

    for i in range(Y.shape[1]):
        Y[:, i] = smooth_weighted_MA(Y[:, i], window_len, window)
    return Y


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
        w = eval('Utils.'+window+'(s,args[0])')
    elif window in type6:
        w = eval('Utils.'+window+'(s)')

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


########################################################################
######################### TS Transformation ########################
########################################################################
########
################################
###### inputs:
## array
## parameters
###### outputs:
## array
## TODO: for all the array
## TODO: Other transformation (wavelets or sthg)
def ups_downs_temporal_discretization(y, collapse_to='initial'):
    '''Temporal discretization of the Time-series only considering the ups and
    downs changes. The criteria in which are grouped the sequence is a
    consequtive share first derivative. The value of this points will be the
    whole variation along this sequence.
    The collapsed points are selected in accordance with the method selected
    with the parameter collapse_to.

    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    collapse_to : str
        the discret value to which is collapse_to a given sequence.
        there are 3 possible cases: 'initial', 'center' and final
        (initial/center/final part of the sequence)

    Returns
    -------
    positions: ndarray, shape(N)
        temporal positions which survived to the discretization
    descriptors: ndarray, shape(N,M)
        descriptors of the sequences collapsed. M = 2 descriptors,
        the total value of the change in the sequence and the shape.

    '''

    ## 1. Preparing values
    positions = []
    descriptors = []

    ## 2. Discretize up and down.
    for sign in [-1, 1]:
        ts = sign*y

        # Compute differentials
        diffe = (np.diff(ts) > 0).astype(int)
        diffe2 = np.diff(diffe)

        # Compute extremes
        ups = np.where(diffe2 == 1)[0]+1
        downs = np.where(diffe2 == -1)[0]+2

        # Corrections
        if ups[0] >= downs[0]:
            ups = np.hstack([[0], ups])
        if ups.shape[0] != downs.shape[0]:
            downs = np.hstack([downs, [ts.shape[0]-1]])

        # Compute descriptors
        if collapse_to == 'initial':
            pos = ups
        elif collapse_to == 'center':
            pos = np.mean([ups, downs], axis=0)
        elif collapse_to == 'final':
            pos = downs

        aug = ts[downs-1]-ts[ups]
        shap = sign*aug/(downs-ups)

        # Store descriptors
        positions.append(pos)
        descriptors.append([aug, shap])

    # Stack the information
    positions = np.hstack(positions)
    descriptors = np.hstack(descriptors).T

    # Order
    order = np.argsort(positions)
    positions = positions[order]
    descriptors = descriptors[order]

    return positions, descriptors


########################################################################
####################### Threshold selection #########################
########################################################################
########
################################
###### inputs:
## matrix
## parameters
###### outputs:
## matrix
## TODO:


########################################################################
####################### Regime detection #########################
########################################################################
########
################################
###### inputs:
## matrix
## parameters
###### outputs:
## matrix
## TODO:
def global_regime_detection(activation, threshold, units='gap',
                            collapse_to='initial', reference='min',
                            individual=True):
    '''Easy threshold-based method for regime detection.
    It is able to detect simple peaks which are remarkable visible.
    It uses the given threshold and selects the time of the peak in time using
    the reference.

    Parameters
    ----------
    activation: pd.pandas
        describe the measure of the voltage and the time of the measures.
    threshold: float or list of floats, str
        which is marks the relative threshold respect to the reference.
        * If float, only one threshold which this value.
        * If list of floats, there are list of thresholds.
        * If str, method to apply unsupervised model to discretize the space.
    reference: str in ['mean','min','mode']
        method to obtain the reference value.
    units: str
        what is the units in which is measured the thresholds.
    individual: bool
        if we make a treatment neuron-by-neuron or we select the top (of the
        peak) globally, and then the absolut threshold value.

    Returns
    -------
    dis_ts: pd.DataFrame
        Discretized time-series.

    Notes
    -----
    Pouzat et al., 2002 (Threshold based in multiple of std)
    Quian Quiroga et al. (2004) (Corrected threshold based in std)
    http://www.scholarpedia.org/article/Spike_sorting
    '''
    ## 1. Transformation of the inputs
    times = np.array(activation.index)
    neuronnames = list(activation.columns)
    voltage = activation.as_matrix()

    ## 2. Compute reference values
    if individual:
        axisn = 0
    else:
        axisn = None
    refs = compute_ref_value(voltage, reference, axisn)

    ## 3. Compute thresholds
    thresholds = compute_abs_thresholds(voltage, refs, threshold, axisn, units)

    ## 4. Discretization by regime
    dis = []
    for neuron in range(voltage.shape[1]):
        disi = code_from_thresholds_ind(voltage[:, neuron], neuron,
                                        thresholds[neuron], refs[neuron],
                                        True, collapse_to)
        dis.append(disi)
    dis_ts = np.vstack(dis)

    ## 5. Sorting by time
    sort_indices = np.argsort(dis_ts[:, 0])
    dis_ts = dis_ts[sort_indices, :]

    ## 6. Storing in a df and replacing the correct times and neuronnames
    dis_ts = pd.DataFrame(dis_ts, columns=['times', 'neuron', 'regime'])
    dis_ts['neuron'].replace(dict(zip(range(len(neuronnames)), neuronnames)))
    dis_ts['times'] = times[np.array(dis_ts['times']).astype(int)]

    return dis_ts


def compute_abs_thresholds(voltage, refs, threshold, axisn=None, units='gap'):
    '''Compute thresholds using the method specified in the inputs.
    '''

    # Select the method specified and compute the thresholds.
    if units == 'gap':
        max_value = np.amax(voltage, axis=axisn)
        threshold_value = (max_value - refs) * threshold
        threshold_value = refs + threshold_value
    elif units == 'std':
        stdu = np.std(voltage, axis=axisn)
        threshold_value = refs + threshold * stdu
    elif units == 'qqstd':
        qqstdu = np.median(voltage/0.6745, axis=axisn)
        threshold_value = refs + threshold * qqstdu

    return threshold_value


def code_from_thresholds_ind(ts, neuron, thresholds, reference, collapse=True,
                             collapse_to='initial'):
    '''Compute the discretized measure of the signal. It returns a 3-column
    array with the times, neuron and regime.

    Parameters
    ----------
    ts: array_like shape (N,)
        time-serie in an array representation.
    neuron: int
        index of the neuron we are representing.
    thresholds: int, float, list, array_like or object.
        Thresholds in order to discretize the signal.
        * If list or array_like, discretize ts using this points.
        * If object, discretize with its function transform.
    reference: int
        value of reference.
    units: str
        what is the units in which is measured the thresholds.
    individual: bool
        if we make a treatment neuron-by-neuron or we select the top (of the
        peak) globally, and then the absolut threshold value.

    Returns
    -------
    dis_ts: array_like shape (Nd,3)
        the discretized signal in a sparse representation.

    '''
    ## 1. Compute regimes.
    if (type(thresholds) == np.float64 or type(thresholds) == float or
       type(thresholds) == int):
        thresholds = [thresholds]
    if type(thresholds) == list or type(thresholds) == np.array:
        min_ind = np.argmin(abs(np.array(thresholds)-reference))
        coding = range(-min_ind, len(thresholds)-min_ind+1)
        regimes = Utils.discretize_with_thresholds(ts, thresholds, coding)
    else:
        regimes = thresholds.transform(ts)
        ##TODO: change regimes in order to respect reference

    values = np.unique(regimes)
    if values.shape[0] == 1:
        raise Exception("Only discretized in one value.")
    elif values.shape[0] == 2:
        values = values[1:]

    ## 2. Temporal collapsing of events
    if collapse:
        spikes = []
        for val in values:
            # Compute vector of changes in regime.
            APbool = (regimes == val).astype(int)
            absdiff = np.abs(np.diff(APbool, axis=0))
            # Compute limits of the regime in time
            ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
            if collapse_to == 'center':
                APindices = ranges.mean(axis=1).round().astype(int)
            elif collapse_to == 'initial':
                APindices = ranges.min(axis=1).round().astype(int)
            elif collapse_to == 'final':
                APindices = ranges.max(axis=1).round().astype(int)

            neuron_index = neuron * np.ones(APindices.shape).astype(int)
            regime_index = val * np.ones(APindices.shape).astype(int)
            neu_contrib = np.vstack([APindices, neuron_index, regime_index])
            spikes.append(neu_contrib)
        dis_ts = np.hstack(spikes).T
    else:
        APindices = np.arange(0, ts.shape[0])
        neuron_index = neuron * np.ones(ts.shape).astype(int)
        regime_index = regimes
        neu_contrib = np.vstack([APindices, neuron_index, regime_index])
        dis_ts = neu_contrib.T

    ## 3. Sort by times
    sort_indices = np.argsort(dis_ts[:, 0])
    dis_ts = dis_ts[sort_indices, :]

    return dis_ts


def compute_ref_value(voltage, reference='min', axisn=None):
    '''Computes the reference value according to the reference kind.
    It is computed for each neuron or globally according to the value of axisn.

    Parameters
    ----------
    voltage: array_like shape (N,M)
        describe the measure of the voltage and the time of the measures of the
        M neurons along N measurements.
    reference: str in ['mean','min','mode']
        method to obtain the reference value.
    axisn: int or None
        axis along there is the time.
        * If None, compute the statistic globally.

    Returns
    -------
    ref_value: int or array_like

    '''

    # Compute the reference value
    if reference == 'min':
        ref_value = np.amin(voltage, axis=axisn)
    elif reference == 'mean':
        ref_value = np.mean(voltage, axis=axisn)
    elif reference == 'mode':
        # TODO
        #reference_value = statistics.mode()
        pass

    return ref_value


########################################################
############# Compute_thresholds axiliary ##############
########################################################
## Global threshold finder
def gmm_spike_detection_univariate(ts, codingmethod='centered'):
    ''' Selection of the possible threshold for the spike detection.
        The possible INPUTS are:
            * ts: the univariate time-serie which represents the activation of
                    one element. It is discretized.
            * codingmethod: how to code the possible states:
                    {'centered','incremental'}
    '''
    # TODO:
    # evaluate the 2 and 3 gmm and choose.
    # TODO: return what coding
    # weights in the model?
    # Create a efficient bining histogram:
    # http://toyoizumilab.brain.riken.jp/hideaki/res/histogram.html
    # kde estimation:
    # http://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/

    ## 1. Preparation for the GMM
    obs = ts[np.nonzero(ts)[0]]
    #obs = ts #debug

    ## 2. Getting thresholds
    thresholds = gmm_spike_identification(obs)

    ## 3. Getting codes:
    if codingmethod == 'incremental':
        coding = range(len(thresholds))
    elif codingmethod == 'centered':
        min_ind = np.argmin(abs(thresholds))
        coding = range(-min_ind, len(thresholds)-min_ind)

    ## 4. Codification
    thresholds = np.array(thresholds)
    coding = np.array(coding)
    ts2 = coding_ts_from_trhesholds(ts, thresholds, coding)

    return ts2


def statistical_threshold_identification(obs, *args):
    '''Statistically global thresholds found in the space in which is
    represented the observations of the time-serie.
    It is only for one time-serie at a time.

    Parameters
    ----------
    obs: array_like (N,M)
        set of observations and values of the time serie.
        The time serie is represented in N temporal points and M features.

    Returns
    -------
    threshols: list or object
        The information for recognize the differents regimes along the time.
        * If list, the possible thresholds in which split the time-series.
        * If object, the class of the unsupervised method used.

    TODO
    ----
    The use of the algorithm of unsupervised learning and clustering.
    '''

    # 2. Aplication of an algorithm of unsupervised learning.
    return thresholds


########################################################################
####################### Align spk waveforms ########################
########################################################################
########
################################
###### inputs:
## pd.DataFrame, pd.DataFrame
## parameters
###### outputs:
## matrix
def align_spk_waveforms(dynamics, spk_info, window_info):
    '''Align spikes for all the dynamics of spikes.

    Parameters
    ----------
    dynamics : pd.DataFrame
        dynamics of activity in each measure.
    spk_info : pd.DataFrame
        dynamics described by its dynamics.

    Return
    ------
    spks_waveforms : array_like shape (Nt,Ns)
        the matrix with all the points during the spks.

    TODO
    ----
    Include neuron information.
    '''

    ## 1. Filter spikes in the extremes
    spk_info = spk_info[spk_info['times'] > - window_info[0]]
    boolfilter2 = spk_info['times'] < (dynamics.shape[0] - window_info[1])
    spk_info = spk_info[boolfilter2]

    ## 2. Define ranges, neurons and times
    spk_info = spk_info[['times', 'neuron']].as_matrix()
    ranges = np.zeros((spk_info.shape[0], 2))
    ranges[:, 0] = (spk_info[:, 0] + window_info[0])
    ranges[:, 1] = (spk_info[:, 0] + window_info[1])
    ranges = ranges.astype(int)
    times = np.arange(window_info[0], window_info[1])
    neurons = spk_info[:, 1].astype(int)
    dynamics = dynamics.as_matrix()

    ## 3. Get matrix of activities
    spks_waveforms = np.zeros((ranges.shape[0], times.shape[0]))
    for i in range(ranges.shape[0]):
        spks_waveforms[i, :] = dynamics[ranges[i, 0]:ranges[i, 1], neurons[i]]
    return spks_waveforms, times, neurons


########################################################################
####################### Feature extraction ########################
########################################################################
########
################################
###### inputs:
## matrix
## parameters
###### outputs:
## matrix
def feature_extraction(X, method_feature='', method_compression='', **kwargs):
    """This is a wrapper of functions that can perform as a feature extraction
    from time series.

    TODO
    ----
    list of method_feature. Concatenate features.
    support for more features.
    Function to create features.
    """
    import sklearn
    import pywt

    if type(method_feature) == str:
        methods_feature = [method_feature]
    elif type(method_feature) == list:
        methods_feature = method_feature

    ## Transformation to feature space
    for method_feature in methods_feature:

        if method_feature == '':
            pass
        elif method_feature == 'diff':
            X = np.diff(X)
        elif method_feature == 'diff_agg':
            X = np.concatenate([X, np.diff(X)], axis=1)
        elif method_feature in ['wavelets', 'dwavelets']:
            # Preparations
            possible = ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey',
                        'daubechies', 'bspline', 'symlets', 'coiflets',
                        'biorthogonal', 'rbiorthogonal', 'dMeyer']
            wv = kwargs['method'] if 'method' in kwargs.keys() else 'haar'
            wv = wv if wv in possible else 'haar'

            ##################################################################
            # Discrete Wavelet ransformations
            if wv == 'haar':
                X = np.array([np.concatenate(pywt.dwt(X[i, :], 'haar'))
                             for i in range(X.shape[0])])
            elif wv in ['db', 'daubechies']:
                par = kwargs['parameter1'] if 'parameter1' in kwargs else 1
                par = par if par in range(1, 21) else 1

                funct = 'db'+str(par)
                funct = funct if funct in pywt.wavelist('db') else 'db1'

                X = np.array([np.concatenate(pywt.dwt(X[i, :], funct))
                             for i in range(X.shape[0])])

            elif wv in ['sym', 'symlets']:
                par = kwargs['parameter1'] if 'parameter1' in kwargs else 2
                par = par if par in range(2, 21) else 2

                funct = 'sym'+str(par)
                funct = funct if funct in pywt.wavelist('sym') else 'sym2'

                X = np.array([np.concatenate(pywt.dwt(X[i, :], funct))
                             for i in range(X.shape[0])])

            elif wv in ['coif', 'coiflets']:
                par = kwargs['parameter1'] if 'parameter1' in kwargs else 1
                par = par if par in range(1, 6) else 1

                funct = 'coif'+str(par)
                funct = funct if funct in pywt.wavelist('coif') else 'coif1'

                X = np.array([np.concatenate(pywt.dwt(X[i, :], funct))
                             for i in range(X.shape[0])])

            elif wv in ['bior', 'biorthogonal']:
                par = kwargs['parameter1'] if 'parameter1' in kwargs else 1
                par2 = kwargs['parameter2'] if 'parameter2' in kwargs else 1
                par, par2 = str(par), str(par2)
                pars = [par, par2]

                possible = [pywt.wavelist('bior')[i][4:].split('.')
                            for i in range(len(pywt.wavelist('bior')))]

                pars = pars if pars in possible else ['1', '1']

                funct = 'bior'+'.'.join(pars)
                funct = funct if funct in pywt.wavelist('bior') else 'bior1.1'

            elif wv in ['rbior', 'rbiorthogonal']:
                par = kwargs['parameter1'] if 'parameter1' in kwargs else 1
                par2 = kwargs['parameter2'] if 'parameter2' in kwargs else 1
                par, par2 = str(par), str(par2)
                pars = [par, par2]

                possible = [pywt.wavelist('rbio')[i][4:].split('.')
                            for i in range(len(pywt.wavelist('rbio')))]

                pars = pars if pars in possible else ['1', '1']

                funct = 'rbio'+'.'.join(pars)
                funct = funct if funct in pywt.wavelist('rbio') else 'rbio1.1'

            elif wv in ['dmey', 'dMeyer']:
                X = np.array([np.concatenate(pywt.dwt(X[i, :], 'dmey'))
                             for i in range(X.shape[0])])

            elif method_feature == 'bspline':
                # TODO: include in the discrete wavelets
                # from mlpy import
                #default = 103
                pass
            ##################################################################
        elif method_feature == 'peaktovalley':
            features = []
            for i in range(X.shape[0]):
                feats = ups_downs_temporal_discretization(X[i, :])[1]
                idx = np.argmax(feats[:, 0])
                features.append(feats[idx, :])
            X = np.vstack(features)

    ## Feature compression
    if method_compression == '':
        Xtrans = X
    elif method_compression == 'pca':
        pca = sklearn.descomposition.PCA()
        Xtrans = pca.fit_transform(X)
    elif method_compression == 'ica':
        ica = sklearn.descomposition.FastICA()
        Xtrans = ica.fit_transform(X)
    return Xtrans


########################################################################
########################## Sorting peaks ###########################
########################################################################
########
################################
###### inputs:
## matrix
## parameters
###### outputs:
## array (with labels)
#
def post_filtering(Xtrans, method, **kwargs):
    import sklearn

    if method == 'manually_1d':
        threshold = kwargs['thresholds']
        ys = (Xtrans[:, 0] > threshold).astype(int)
    elif method == 'spectral clustering':
        cluster = sklearn.cluster.SpectralClustering()
        ys = cluster.fit_predict(Xtrans)
    elif method == 'gmm':
        cluster = sklearn.mixture.GMM()
        ys = cluster.fit_predict(Xtrans)
    elif method == 'kmeans':
        cluster = sklearn.cluster.KMeans()
        ys = cluster.fit_predict(Xtrans)
    elif method == 'minibatchkmeans':
        cluster = sklearn.cluster.MiniBatchKMeans()
        ys = cluster.fit_predict(Xtrans)
    elif method == 'meanshift':
        cluster = sklearn.cluster.MeanShift()
        ys = cluster.fit_predict(Xtrans)

    return ys


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
######## GLOBAL-Based peak detection ##########
def peak_detection1(activation, threshold, units='gap', collapse_to='initial',
                    reference='min', individual=True):
    '''Easy threshold-based method for peak detection.
    It is able to detect simple peaks which are remarkable visible.
    It uses the given threshold and selects the time of the peak in time using
    the reference.

    Parameters
    ----------
    activation: pd.pandas
        describe the measure of the voltage and the time of the measures.
    threshold: float or list of floats, str
        which is marks the relative threshold respect to the reference.
        * If float, only one threshold which this value.
        * If list of floats, list of threshols.
        * If str, method to apply.
    reference: str in ['mean','min','mode']
        method to obtain the reference value.
    units: str
        what is the units in which is measured the thresholds.
    individual: bool
        if we make a treatment neuron-by-neuron or we select the top (of the
        peak) globally, and then the absolut threshold value.

    Returns
    -------
    ts: pd.DataFrame
        the new dynamics.

    Notes
    -----
    Pouzat et al., 2002 (Threshold based in multiple of std)
    Quian Quiroga et al. (2004) (Corrected threshold based in std)
    http://www.scholarpedia.org/article/Spike_sorting
    '''

    ## TODO: accept np.array
    ### TODO: return correct dataframe
    ## allow np.array and tuple inputs
    ##

    ## 1. Transformation of the inputs
    times = np.array(activation.index)
    #neuronnames = list(voltage.columns)
    voltage = activation.as_matrix()

    ## 2. Creation of the threshold value  ---- > To place in other function
    #(it could be general or specific for each neuron)
    # For each neuron or collectively
    # inputs:
    ## individual, reference
    if individual:
        axisn = 0
    else:
        axisn = None

    # Compute the reference value
    if reference == 'min':
        reference_value = np.amin(voltage, axis=axisn)
    elif reference == 'mean':
        reference_value = np.mean(voltage, axis=axisn)
    elif reference == 'mode':
        # TODO
        #reference_value = statistics.mode()
        pass

    # Compute threshold value  ---> To place in other function
    if units == 'gap':
        max_value = np.amax(voltage, axis=axisn)
        threshold_value = (max_value - reference_value) * threshold
        threshold_value = threshold + reference_value
    elif units == 'std':
        stdu = np.std(activation, axis=axisn)
        threshold_value = reference_value + threshold * stdu
    elif units == 'qqstd':
        qqstdu = np.median(activation/0.6745, axis=axisn)
        threshold_value = reference_value + threshold * qqstdu

    #####################################################################
    ### TODO: get a function of thresholds and reference values and return
    # columns like: time, neuron, regime
    ## threshold_value, reference_value, collapse_to
    ### TODO: Transformation (in other features)

    ## 3. Selecting the peaks
    # Search for the indices of the AP
    APbool = (voltage > threshold_value).astype(int)
    absdiff = np.abs(np.diff(APbool, axis=0))

    spikes = []
    # Now each column has to be treated individually
    # (they could have different number of spikes)
    for i in range(voltage.shape[1]):
        ranges = np.where(absdiff[i] == 1)[0].reshape(-1, 2)
        if collapse_to == 'center':
            APindices = ranges.mean(axis=1).round().astype(int)
        elif collapse_to == 'initial':
            APindices = ranges.min(axis=1).round().astype(int)
        elif collapse_to == 'final':
            APindices = ranges.max(axis=1).round().astype(int)
        # TODO: Others weighted methods

        neuroncontribution = [[times[APindices[ind]], i] for ind in len(APindices)]
        spikes = spikes + neuroncontribution
    ########################################################################

    ## 4. Sorting and giving the correct format
    ########################################
    #TODO:
    # Sort the times
    #APTimes = np.sort(APTimes)
    ########################################

    # Return object dynamics
    ts = pd.DataFrame(spikes, columns=['times', 'neuron'])
    return ts


##### DEPRECATED
def coding_ts_from_trhesholds(ts, thresholds, coding=[]):
    ''' This function tramify a time-serie in the way we select.
        Inputs:
            * ts: 1d np.array which represents the time-serie.
            * thresholds: the thresholds between the codes.
            * coding: codes in the same order as the thresholds.
        Output:
            * ts: a 1d np.array with the same length as ts with elements of
                    coding.

    DEPRECATED
    '''
    # TODO: Globally in pandas?

    ## 0. Control of inputs
    if not coding:
        coding = np.array(range(len(thresholds)+1))

    if (type(ts) != np.ndarrray and type(thresholds) == np.ndarray and
       type(coding) == np.ndarray):
        message = 'No correct type of the inputs. They have to be numpy.arrays'
        raise Exception(message)
    if len(coding) != len(thresholds) + 1:
        message = 'No correct length of the inputs.'
        raise Exception(message)

    ## 1. Preparing boundaries
    th = np.array([min(ts)] + list(thresholds) + [max(ts)+0.1])
    th = [[th[i], th[i+1]] for i in range(len(th)-1)]

    ## 2. Transformation
    for i in range(len(th)):
        ts[np.bitwise_and(ts >= th[i][0], ts <= th[i][1])] = coding[i]

    ## 3. Check if the transformation is done correctly
    res_codes = np.unique(ts)
    if not np.all(res_codes == coding):
        message = "Error in the tramification of the Time-series."
        raise Exception(message)

    return ts


# to be included in other functions
def gmm_spike_detection(activation, codingmethod='centered',
                        caculmethod='neuwise'):
    ''' The
        Selection of the possible threshold for the spike detection.
        Inputs:
            * activation: the instantiation of the class dynamics in the
            discrete mode.
            * codingmethod: how to code the possible states:
            ['centered','incremental']
            * calculmethod: how to process ts, collectively or individually.
            ['populwise','neuwise']
        Outputs:
            * : Dynamics object with the codification of the
    '''

    ## 0. Control of the inputs

    ##### TODO:
    ## 1. Extraction of the matrix
    # TODO: matrix_activations in the class Dynamics
    obs = activation.dynamics['strength'].as_matrix()

    ## 2. Thresholds identification
    thresholds = gmm_spike_detection_univariate(obs, 'centered')

    ## 3. Thresholds application
    activation.dynamics['regime_type'] = coding_ts_from_trhesholds()  # TODO


# to be included in other functions
def gmm_spike_identification(obs):
    ''' This function aims to identify the thresholds by using a GMM.
        Inputs:
            * obs: np.array with all the discretized variations.
        Outputs:
            * thresholds: np.array with the thresholds.
    '''

    ## 1. Preparation for the GMM
    #obs = obs[nonzero(ts)[0]]

    ## 2. Creation of the GMM (evaluate the 2 and 3 gmm and choose)
    g = GMM(n_components=2, n_init=20)
    g.fit(obs)
    logprobs, components = g.decode(obs)

    ## 3. Getting thresholds
    comp_class = np.unique(components)
    means = dict(zip([e for e in comp_class], [np.mean(obs[components == e]) for e in comp_class]))
    sorted_classes = sorted(dic.keys(), key=lambda x: dic[x], reverse=True)
    thresholds = [np.mean([min(sorted_classes[i]), max(sorted_classes[i+1])]) for i in range(len(sorted_classes)-1)]
    thresholds = np.array(thresholds)
    return thresholds


##### included in spike_detection1
def variance_spike_identification(obs, ration=4):
    '''This is a dummy method for identification spikes. Make the assumption
    that there are only two regimes (spike and relax).
    Parameters
    ----------
    obs: np.array
        vector of the possible incrementations.
    ration: int
        value related with the signal/noise ratio.
    '''

    stdr = np.std(obs)
    thresholds = np.array([stdr*ration])
    return thresholds


##### included in spike_detection1
def Zscore_spike_identification(obs, likelihood_thr=4):
    '''This is a dummy method for identification spikes. Make the assumption
    that there are only two regimes (spike and relax).
    Parameters
    ----------
    obs: np.array
        vector of the possible incrementations.
    likelihood_thr: int
        value related with the signal/noise ratio.
    '''
    from scipy.stats import norm
    m, st = np.mean(obs), np.std(obs)
    x = np.linspace(np.min(obs), np.max(obs), 200)
    thresholds = x[np.argmin(np.abs(norm.logpdf(x, m, st)+likelihood_thr))]
    return thresholds
