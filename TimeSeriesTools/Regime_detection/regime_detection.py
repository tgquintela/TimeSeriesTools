
"""
Collection of functions which receive time series and discretize them into a
series of discrete values. They could be discretized temporally and by value.
This module wrappes the module of TimeSeries and specially the one of
transformation in order to do the required specific tasks in order to perform
regime detection.

TODO
----
1. Support for pattern matching
2. Support for statistical thresholding
"""

import numpy as np
import pandas as pd

import transformations


###############################################################################
############################## Global detection ###############################
###############################################################################
def general_regime_detection(signals, itrans_info, discret_info, val_disc_info,
                             ptrans_info, norm_info):
    """
    This function acts as a pipeline of the whole process of regime detection.
    We input a general activity time series and the information of each step of
    the process and returns a matrix of dense regimes states.

    Parameters
    ----------
    signals: array_like, shape(N, M)
        the signals of the system.
    itrans_info: dict, array_like, pd.DataFrame or function
        information of the initial transformation.
    discret_info: dict, array_like, pd.DataFrame or function
        information of how to obtain the discretization element.
    val_disc_info: dict or function
        information of how to discretize the time-series.
    ptrans_info: dict
        information of the final transformation.
    norm_info: dict
        information of the Normalization transformation in which we format the
        output properly.

    Returns
    -------
    X: array_like
        it returns the dense representation of the regimes dynamics.

    """

    ## 1. Initial transformation
    if type(itrans_info) == dict:
        X = general_transformation(signals, **itrans_info)
    elif type(itrans_info) == pd.DataFrame:
        X = itrans_info.as_matrix()
    elif type(itrans_info) == np.ndarray:
        X = itrans_info
    elif type(itrans_info).__name__ == 'function':
        X = itrans_info(signals)

    ## 2. Preparing discretizor
    if type(discret_info) == dict:
        discretizor = discretizor_builder(X, **discret_info)
    elif type(discret_info) == pd.DataFrame:
        discretizor = discret_info.as_matrix()
    elif type(discret_info) == np.ndarray:
        discretizor = discret_info
    elif type(discret_info).__name__ == 'function':
        # TO CHECK:
        # Probably you need to pass a function not their transf (GMM inside)
        discretizor = discret_info(X)

    ## 3. Value discretization
    # Creation of the refs and call the function
    # Use pattern matching (it is possible that there are different patterns)
    if type(val_disc_info).__name__ == 'function':
        X = val_disc_info(X, discretizor)
    elif type(val_disc_info) == dict:
        X = value_discretization(X, discretizor, **val_disc_info)
    else:
        raise TypeError('Not correct type of val_disc_info.')

    ## 4. Post transformation (generally a median filter)
    X = general_transformation(X, **ptrans_info)

    ## 5. Normalization (format to output, collapsing and this stuff)
    X = general_transformation(X, **norm_info)

    return X


###############################################################################
###############################################################################
###############################################################################
def discretizor_builder(signals, method, discret_info):
    """Main function to built the discretizor element.

    Parameters
    ----------
    signals: array_like, shape (N, M)
        the signals of the system.
    method: str
        method used for discretize the signals.
    discret_info: dict
        information to build the discretizor element.

    Returns
    -------
    discretizor: object, array_like
        the information to discretize our signals.

    """

    if method == 'pattern_matching':
        # Read file or pass matrix of examples of waveforms
        pass
    elif method == 'threshold_based':
        discretizor = ref_based_thres_builder(signals, **discret_info)
    elif method == 'statistical_threshold':
        # pass the object to instantiate or something
        pass

    return discretizor


###############################################################################
######################## Threshold based discretization #######################
###############################################################################
def ref_based_thres_builder(signals, refs_info, units_info, thresholds_info):
    """Main function for compute the thresholds

    Parameters
    ----------
    signals: array_like, shape (N, M)
        the signals of the system.
    refs_info: array_like, dict or function
        information needed to compute references.
    units_info: array_like, dict or function
        information needed to compute unit values.
    thresholds_info: array_like, dict or function
        information needed to compute thresholds from references and units.

    Returns
    -------
    thresholds: array_like, shape (N, M)
        the thresholds for each time and element.

    """
    # Needed variables and tools
    def equality_elements_list(a, b):
        return a == b or a[-1::-1] == b

    ## 1. Reference creation
    if type(refs_info) == np.ndarray:
        refs = refs_info
    elif type(refs_info) == dict:
        if equality_elements_list(units_info.keys(), ['method', 'axisn']):
            refs = compute_ref_value(signals, **refs_info)
        else:
            refs = general_transformation(signals, **refs_info)
    elif type(refs_info).__name__ == 'function':
        refs = refs_info(signals)

    ## 2. Units creation
    if type(units_info) == np.ndarray:
        units = units_info
    elif type(units_info) == dict:
        if equality_elements_list(units_info.keys(), ['method', 'axisn']):
            units = compute_units(signals, **units_info)
        else:
            units = general_transformation(signals, **units_info)
    elif type(units_info).__name__ == 'function':
        units = units_info(signals)

    ## 3. Threshold creation
    if type(thresholds_info) == np.ndarray:
        thresholds = thresholds_info
    elif type(thresholds_info) == dict:
        thresholds = compute_threshold(refs, units, **thresholds_info)
    elif type(thresholds_info).__name__ == 'function':
        thresholds = thresholds_info(refs, units, **thresholds_info)

    return thresholds


def compute_ref_value(signals, method='min', axisn=None):
    """Computes the reference value according to the reference kind.
    It is computed for each neuron or globally according to the value of axisn.

    Parameters
    ----------
    signals: array_like, shape (N,M)
        describe the measure of the signals and the time of the measures of the
        M neurons along N measurements.
    method: str optional, dict or function
        method to obtain the reference value.
    axisn: int or None
        axis along there is the time.
        * If None, compute the statistic globally.

    Returns
    -------
    ref_value: array_like, shape (N,M)
        reference value to start comparing whith the actual value of the ts in
        order to determine if it is a peak. It represents the normal state or
        behaviour of the system for systems with two states (active or idle).

    """

    # Setting the inputs
    if type(method) == dict:
        kwargs = method
        method = 'precoded_transf'
    elif type(method).__name__ == 'function':
        f = method
        method = 'personal'

    # Compute the reference value
    if method == 'null':
        ref_value = 0
        axisn = None
    elif method == 'min':
        ref_value = np.amin(signals, axis=axisn)
    elif method == 'mean':
        ref_value = np.mean(signals, axis=axisn)
    elif method == 'mode':
        # TODO
        #reference_value = statistics.mode()
        pass
    # Personal method applying a given transformation (TO CHANGE)
    elif method == 'precoded_transf':
        ref_value = general_transformation(signals, **kwargs)
    elif method == 'personal':
        ref_value = f(signals, axisn)

    # Format output
    if axisn is None:
        ref_value = ref_value*np.ones(signals.shape)
    elif axisn == 0:
        aux = np.zeros(signals.shape)
        for i in range(signals.shape[0]):
            aux[i, :] = ref_value
        ref_value = aux
    elif axisn == 1:
        aux = np.zeros(signals.shape)
        for i in range(signals.shape[1]):
            aux[:, i] = ref_value
        ref_value = aux

    return ref_value


def compute_units(signals, method, axisn=None):
    """Compute units using the method specified in the inputs.

    Parameters
    ----------
    signals: array_like, shape(Ntimes, Nelements)
        the signals of the elements of the system or some part of it.
    method: str, function or dict
        method used to compute unit values
    aixn: int or None
        the axis over which the time extends in the ts. In the case we want to
        study as a homogenous stationary system we can use None.

    Returns
    -------
    units: array_like, shape(Ntimes, Nelements)
        how to measure the threshold respect the reference

    References
    ----------
    .. [1] Pouzat et al., 2002 (Threshold based in multiple of std)
    .. [2] Quian Quiroga et al. (2004) (Corrected threshold based in std)

    """

    # Setting the inputs
    if type(method) == dict:
        kwargs = method
        method = 'precoded_transf'
    elif type(method).__name__ == 'function':
        f = method
        method = 'personal'

    # Compute the reference value
    if method == 'unitary':
        units = 1
        axisn = None
    elif method == 'gap':
        units = np.max(signals, axis=axisn)-np.min(signals, axis=axisn)
    elif method == 'std':
        units = np.std(signals, axis=axisn)
    elif method == 'qqstd':
        units = np.median(signals/0.6745, axis=axisn)

    # Personal method applying a given transformation (TO CHANGE)
    elif method == 'precoded_transf':
        units = general_transformation(signals, **kwargs)
    elif method == 'personal':
        units = f(signals, axisn)

    # Format output
    if axisn is None:
        units = units*np.ones(signals.shape)
    elif axisn == 0:
        aux = np.zeros(signals.shape)
        for i in range(signals.shape[0]):
            aux[i, :] = units
        units = aux
    elif axisn == 1:
        aux = np.zeros(signals.shape)
        for i in range(signals.shape[1]):
            aux[:, i] = units
        units = aux

    return units


def compute_threshold(refs, units, threshold_value):
    """Compute the thresholds from reference values, unit values and the
    threshold value.

    Parameters
    ----------
    refs: array_like, shape(N, M)
        the reference values for all the times and elements.
    units: array_like, shape(N, M)
        the units values for all the the times and elements.
    threshold_value: float
        the number of units that the threshold is over the reference value.

    Returns
    -------
    thresholds: array_like, shape (N, M)
        the tresholds for each value and element.
    """

    thresholds = refs + threshold_value*units

    return thresholds
