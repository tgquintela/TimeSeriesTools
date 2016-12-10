
"""
This module groups all the functions which carry out the process of value
discretization of a time series.
"""

import numpy as np


def value_discretization(Y, discret_info, method, kwargs={}):
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
## TODO: Create an object which save the information and generalize the task
## for other systems and arrays. fit transform
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
    aux: array, shape (array.shape)
        a array with discretized values

    '''

    ## Formatting and building variables needed
    inshape = array.shape
    thres = threshold_formatter(array, thres)
    if thres is None:
        return array
    # Number of threshold values
    nd_thres = thres.shape[2]
    # Setting values
    if values == []:
        values = range(nd_thres-1)
        assert nd_thres == len(values)+1
    ## 2. Fill the new vector discretized signal to the given values
    aux = np.zeros(array.shape)
    for i in range(len(values)):
        indices = np.logical_and(array >= thres[:, :, i],
                                 array <= thres[:, :, i+1])
        indices = np.nonzero(indices)[0]
        if len(indices):
            aux[indices] = values[i]

    aux = aux.reshape(inshape)
    return aux


def threshold_formatter(array, thres, extend_limits=True):
    """Function which transforms the thresholds given to a array format to
    apply easy the discretization.

    Parameters
    ----------
    array: array_like, shape ()
        the array we want to discretize.
    thres:
        the threshold information.

    Returns
    -------
    thres: array_like, shape (array.shape[0], n_elem, nd_thres)
        the matrix of thresholds to easy threshold the array.

    """

    ## 1. Preparing thresholds and values
    nd_input = len(array.shape)
    n_elem = 1 if nd_input == 1 else array.shape[1]

    # From an input of a float
    if thres is None:
        return None
    elif type(thres) == float:
        nd_thres = 1
        thres = thres*np.ones((array.shape[0], n_elem, nd_thres))

    # From an input of a list or tuple:
    # {list of floats, list of arrays or tuple of arrys}
    elif type(thres) in [list, tuple]:
        tp = [type(e) for e in thres]
        assert np.all([t == tp[0] for t in tp])
        tp = tp[0]
        # List of floats
        if tp == float:
            nd_thres = len(thres)
            aux = np.ones((array.shape[0], n_elem, nd_thres))
            for i in range(nd_thres):
                aux[:, :, i] = aux[:, :, i] * thres[i]
            thres = aux
        # List or tuple of arrays
        elif tp in [np.ndarray, np.array]:
            nd_thres1 = len(thres)
            nd_thres2 = [e.shape[0] for e in thres]
            assert np.all([e == nd_thres2[0] for e in nd_thres2])
            nd_thres2 = nd_thres2[0]
            # nd_thres could be n_elem or nd_thres
            if n_elem == nd_thres1:
                nd_thres = thres[0].shape[0]
                aux = np.ones((array.shape[0], n_elem, nd_thres))
                for i in range(n_elem):
                    for j in range(nd_thres):
                        aux[:, i, j] = aux[:, i, j] * thres[i][j]
                thres = aux
            elif n_elem == nd_thres2:
                nd_thres = thres[0].shape[0]
                aux = np.ones((array.shape[0], n_elem, nd_thres))
                for i in range(nd_thres):
                    for j in range(n_elem):
                        aux[:, j, i] = aux[:, j, i] * thres[i][j]
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

    ## 2. Creation of the limit min and max thresholds
    if extend_limits:
        mini = -np.ones((array.shape[0], n_elem, 1))*np.inf
        maxi = np.ones((array.shape[0], n_elem, 1))*np.inf
        # Concatenation
        thres = np.concatenate([mini, thres, maxi], axis=2)

    return thres


def threshold_binning_builder(array, n_bins):
    """Function to create the thresholds from binning arrays.

    Parameters
    ----------
    array: array_like
        the array to be discretized.
    n_bins: int or array
        the information about binning.

    Returns
    -------
    thres: array_like, shape (N, n_elem, nd_thres)
        the thresholding matrix.

    """

    ## 0. Create needed variables and format inputs
    # Number of dimensions of array
    ndim = len(array.shape)
    # Number of elements
    n_elem = 1 if ndim == 1 else array.shape[1]
    # Possible values
    values = np.unique(array)
    # Individually
    individually = True
    # Format array
    array = array if ndim == 2 else array.reshape((array.shape[0], 1))
    # Description of the possible situations
    situation1 = n_bins == 0 or n_bins is None or not np.any(n_bins)
    situation2 = type(n_bins) == int
    situation3 = type(n_bins) in [list, tuple, np.ndarray]
    if situation3:
        situation3check1 = np.all(np.array(n_bins) == type(n_bins[0]))
        situation3check2 = np.array(n_bins).shape[0] == n_elem
        situation3a = type(n_bins[0]) == int
        situation3b = type(n_bins[0]) == np.ndarray
    # Prepare bins
    if situation1:
        return None
    elif situation2:
        pass
    elif situation3:
        assert situation3check1
        assert situation3check2
        if situation3a:
            individually = True
        elif situation3b:
            individually = True
        else:
            raise Exception("Not correct n_bins input.")

    ## 1. Binning thresholds
    # bins edges
    if not individually:
        _, bins_edges = np.histogram(array.reshape(-1), n_bins)
        thres = bins_edges
    else:
        thres = []
        for i in range(n_elem):
            _, bins_edges = np.histogram(array[:, i], n_bins)
            thres.append(bins_edges)
        thres = tuple(thres)

    ## 2. Format to matrix thresholding
    thres = threshold_formatter(array, thres, False)

    return thres


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
