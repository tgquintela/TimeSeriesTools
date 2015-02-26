
"""
This module groups all the functions which carry out the process of value
discretization of a time series.
"""

import numpy as np


def value_discretization(Y, discret_info, method, kwargs):
    """Function which acts as a switcher and wraps all the functions related
    with value discretization.

    Parameters
    ----------
    Y: array_like
        the signals to be discretized in values.
    discret_info: dict, np.array or pd.DataFrame
        needed information in order to discretize by value.
    method: str
        method used in order to perform a value discretization.
    kwargs: dict
        extra parameters to use in each selected method.

    Returns
    -------
    Yt: array_like
        discretized signals in values.

    """

    Yt = np.array(Y)

    if method == 'pattern_matching':
        Yt = general_pattern_matching(Y, discret_info, **kwargs)
    elif method == 'threshold_based':
        Yt = discretize_with_thresholds(Y, discret_info, **kwargs)
    elif method == 'statistical_threshold':
        Yt = statistical_discretizor(Y, discret_info, **kwargs)

    return Yt


###############################################################################
############################### threshold_based ###############################
###############################################################################
def discretize_with_thresholds(array, thres, values=[]):
    '''This function uses the given thresholds in order to discretize the array
    in different possible values, given in the variables with this name.

    Parameters
    ----------
    array: array_like
        the values of the signal
    thres: float, list of floats or np.ndarray
        the threshold values to discretize the array.
    values: list
        the values assigned for each discretized part of the array.

    Returns
    -------
    aux: array
        a array with discretized values

    '''

    ## 1. Preparing thresholds and values
    nd_input = len(array.shape)

    # From an input of a float
    if type(thres) == float:
        nd_thres = 1
        if nd_input == 1:
            thres = thres*np.ones((array.shape[0], 1))
        elif nd_input == 2:
            thres = thres*np.ones((array.shape[0], array.shape[1], 1))
    # From an input of a list
    elif type(thres) == list:
        nd_thres = len(thres)
        aux = np.ones((array.shape[0], array.shape[1], nd_thres))
        if nd_input == 1:
            for i in range(nd_thres):
                aux[:, i] = thres[i]*aux[:, i]
        elif nd_input == 2:
            for i in range(nd_thres):
                aux[:, :, i] = thres[i]*aux[:, :, i]
        thres = aux
    # From an input of a array (4 possibilities)
    elif type(thres) == np.ndarray:
        # each threshold for each different elements (elements)
        if len(thres.shape) == 1 and thres.shape[0] == array.shape[1]:
            nd_thres = 1
            aux = np.ones((array.shape[0], array.shape[1], nd_thres))
            for i in range(array.shape[1]):
                aux[:, i, 0] = thres[i]*aux[:, i, 0]
            thres = aux
        # each threshold for each different times (times)
        elif len(thres.shape) == 1 and thres.shape[0] == array.shape[0]:
            nd_thres = 1
            aux = np.ones((array.shape[0], array.shape[1], nd_thres))
            for i in range(array.shape[0]):
                aux[i, :, 0] = thres[i]*aux[i, :, 0]
            thres = aux
        # some threshold for each different elements (elemnts-thres)
        elif len(thres.shape) == 2 and thres.shape[0] == array.shape[1]:
            nd_thres = thres.shape[1]
            aux = np.ones((array.shape[0], array.shape[1], nd_thres))
            for i in range(array.shape[1]):
                aux[:, i, :] = thres[i, :]*aux[:, i, :]
            thres = aux
        # some threshods for each time shared by all elements (times-thres)
        elif len(thres.shape) == 2 and thres.shape[0] == array.shape[0]:
            nd_thres = thres.shape[1]
            aux = np.ones((array.shape[0], array.shape[1], nd_thres))
            for i in range(array.shape[0]):
                aux[i, :, :] = thres[i, :]*aux[i, :, :]
            thres = aux
        # one threshold for each time and element (times-elements)
        elif len(thres.shape) == 2 and thres.shape[:2] == array.shape[:2]:
            nd_thres = 1
            thres = thres.reshape((thres.shape[0], thres.shape[1], nd_thres))
        # some thresholds for each time and element
        elif len(thres.shape) == 3:
            nd_thres = thres.shape[2]

    # Setting values
    if values == []:
        values = range(nd_thres+1)
    elif type(thres) == list:
        assert nd_thres == len(values)-1

    # Creation of the limit min and max thresholds
    mini = np.ones((array.shape[0], array.shape[1], 1))*np.min(array)
    maxi = np.ones((array.shape[0], array.shape[1], 1))*np.max(array)
    # Concatenation
    thres = np.concatenate([mini, thres, maxi], axis=2)
    array = array.reshape((array.shape[0], array.shape[1]))

    ## 2. Fill the new vector discretized signal to the given values
    aux = np.zeros(array.shape)
    for i in range(len(values)):
        indices = np.logical_and(array >= thres[:, :, i],
                                 array <= thres[:, :, i+1])
        indices = np.nonzero(indices)
        aux[indices] = values[i]

    return aux


###############################################################################
############################### pattern_matching ##############################
###############################################################################
def general_pattern_matching(signals, patterns, values=[], tols=[]):
    """Molde para functiones de pattern matching o general.
    """
    pass


def pattern_matching_detection(activation, patterns, method, **kwargs):
    """General function to perform pattern matching based detection.

    Parameters
    ----------
    activation: array_like
        description of the activity of the elements of the system.
    patterns: array_like
        patterns we want to match in order to detect a wanted regime.
    method: str, optional
        the method used to perform pattern matching.
    kwargs: dict
        variables needed to call the method selected.

    Returns
    -------
    spks: pd.DataFrame
        spikes detected.

    """

    possible = ['dtw']
    method = method if method in possible else 'dtw'
    if method == 'dtw':
        spks = dtw_based_detection(activation, patterns, **kwargs)

    return spks


def dtw_based_detection(activation, patterns):
    """This function is based on dynamic time warping and uses examples to
    determine the parameters for dtw and the actual pattern shape in order to
    detect the parts of the time series which has this kind of shape.
    """

    # Infer parameter from patterns

    # Transformation

    # Dynamic time warping detection

    return times_spks


###############################################################################
############################ statistical_threshold ############################
###############################################################################


############################# auxiliary functions #############################
###############################################################################
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
