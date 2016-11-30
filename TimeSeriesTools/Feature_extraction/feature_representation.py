
"""
This module group all the funcions related with the feature representation
of time-series system.
"""

import numpy as np
import pywt
from ..Transformation.temporal_discretization import\
    ups_downs_temporal_discretization_matrix, ups_downs_temporal_discretization
from ..Transformation.aux_transformation import collapse3waveform_matrix


def feature_representation(X, methods_feature=[], args=[]):
    """This is a wrapper of functions that can perform as a feature extraction
    from time series.

    Parameters
    ----------
    X: array_like, shape (N, M)
        N times measures for M temporal series.
    methods_feature: list str or str, optional
        methods used to represent the time series.
    args: list of dicts
        the needed arguments for each method selected.

    Returns
    -------
    Xt: array_like, shape (Nt, M)
        transformed times series in which we have Nt feaures for each series,
        number which depends on the method, keeping the original M elements
        given.

    TODO
    ----

    """

    # Uniformation of the inputs
    if type(methods_feature) == str:
        methods_feature = [methods_feature]
        args = [args] if type(args) == dict else [{}]
    elif type(methods_feature) == list:
        methods_feature = methods_feature
    possible_methods = ['', 'diff', 'diff_agg', 'wavelets', 'dwavelets',
                        'collapse3waveform', 'peaktovalley']
    assert(all([m in possible_methods for m in methods_feature]))

    ## Transformation to feature space
    Xt = []
    i = 0
    for method_feature in methods_feature:
        if method_feature == '':
            aux = null_feat_repres(X)
            Xt.append(aux)
        elif method_feature == 'diff':
            aux = diff_feat_repres(X)
            Xt.append(aux)
        elif method_feature == 'diff_agg':
            aux = diff_agg_feat_repres(X)
            Xt.append(aux)
        elif method_feature == 'peaktovalley':
            aux = peak2valley_feat_repres(X)
            Xt.append(aux)
        elif method_feature in ['wavelets', 'dwavelets']:
            aux = dwavelet_feat_repres(X, **args[i])
            Xt.append(aux)
        elif method_feature == 'collapse3waveform':
            Xt.append(collapse3waveform_matrix(X, **args[i]))
        i += 1

    # Format output
    Xt = np.vstack(Xt)

    return Xt


###############################################################################
################################### Methods ###################################
###############################################################################
def null_feat_repres(X):
    """The null feature representation represents the time series with its same
    values without any Transformation.

    Parameters
    ----------
    X: array_like, shape (N, M)
        represents the signals of the system

    Returns
    -------
    X: array_like, shape (N, M)
        the transformated signals of the system.

    """
    return X


def diff_feat_repres(X):
    """The diff feature representation represents the time series with the
    first derivative of the time series.

    Parameters
    ----------
    X: array_like, shape (N, M)
        represents the signals of the system.

    Returns
    -------
    Xt: array_like, shape (N-1, M)
        the transformated signals of the system.

    """
    Xt = np.diff(X, axis=0)
    return Xt


def diff_agg_feat_repres(X):
    """The diff_agg feature representation represents the time series with the
    first derivative of the time series added to the actual timeseries.

    Parameters
    ----------
    X: array_like, shape (2*N-1, M)
        represents the signals of the system.

    Returns
    -------
    Xt: array_like
        the transformated signals of the system.

    """
    Xt = diff_feat_repres(X)
    Xt = np.vstack([X, Xt])
    return Xt


def peak2valley_feat_repres(X):
    """The peak2valley feature representation represents the time series with
    the maximum consecutive incremental change in the time series.

    Parameters
    ----------
    X: array_like, shape (N, M)
        represents the signals of the system.

    Returns
    -------
    Xt: array_like, shape (1, M)
        the transformated signals of the system.

    """
    features = []
    for i in range(X.shape[1]):
        feats = ups_downs_temporal_discretization(X[:, i])[1]
        idx = np.argmax(feats[:, 0])
        features.append(feats[idx, :])
    Xt = np.vstack(features).T
    return Xt


#def collapse3waveform_matrix(waveform, window_info, axisn=0):
#    """Funcions to collapse a waveform in 3 stages, pre-spike, spike and
#    post-spike in order to be able to identify the spikes easily.
#
#    Parameters
#    ---------
#    waveform: array_like, shape(Nf, M)
#        representation of the waveform or part of the time series we want to
#        transform and extracting representative features of the agrregated
#        state of pre-event, event and post-event situation.
#    window_info: int or array_like of two elements
#        the information of the window. If the number is an integer we will
#        consider as a symmetrical window. If there is a list, array or tuple,
#        it is the information of the event time or the borders of the window
#        respect the position of the event.
#    axisn: int, optional
#        there is the option of
#
#    Returns
#    -------
#    collapsed: array_like, shape (3, M)
#        collapsed representation of the waveform passed.
#    """
#
#    # Formatting inputs
#    if type(window_info) == int:
#        idx = window_info
#    else:
#        idx = -window_info[0]
#
#    # How many time series we have.
#    m = waveform.shape[axisn]
#    collapsed = np.zeros((3, m))
#    axisn = (axisn - 1) % 2
#
#    collapsed[0, :] = np.sum(waveform[:idx, :], axisn).reshape(-1)
#    collapsed[1, :] = waveform[idx, :].reshape(-1)
#    collapsed[2, :] = np.sum(waveform[idx+1:, :], axisn).reshape(-1)
#    return collapsed


def dwavelet_feat_repres(X, method='haar', kwargs={}):
    """The peak2valley feature representation represents the time series with
    the maximum consecutive incremental change in the time series.

    Parameters
    ----------
    X: array_like, shape (N, M)
        represents the signals of the system.
    method: str, optional
        the discrete wavelet to use in the feature representation.
    kwargs: dict
        specific variables for each method.

    Returns
    -------
    Xt: array_like, shape (Nt, M)
        the transformated signals of the system. The number of features given
        depends on the method used.

    """

    # Preparations
    possible = ['haar', 'db', 'sym', 'coif', 'bior', 'rbior', 'dmey',
                'daubechies', 'bspline', 'symlets', 'coiflets',
                'biorthogonal', 'rbiorthogonal', 'dMeyer']

    method = method if method in possible else 'haar'
    ##################################################################
    # Discrete Wavelet ransformations
    if method == 'haar':
        Xt = haar_wavelet_transf(X)
    elif method in ['db', 'daubechies']:
        Xt = daubechies_wavelet_transf(X, **kwargs)
    elif method in ['sym', 'symlets']:
        Xt = symlets_wavelets_transf(X, **kwargs)
    elif method in ['coif', 'coiflets']:
        Xt = coiflets_wavelets_transf(X)
    elif method in ['bior', 'biorthogonal']:
        Xt = biorthogonal_wavelets_transf(X, **kwargs)
    elif method in ['rbior', 'rbiorthogonal']:
        Xt = rbiorthogonal_wavelets_transf(X, **kwargs)
    elif method in ['dmey', 'dMeyer']:
        Xt = dMeyer_wavelets_transf(X)
    elif method == 'bspline':
        # TODO: include in the discrete wavelets
        # from mlpy import
        #default = 103
        pass
    ##################################################################
    return Xt


###############################################################################
################################## wavelests ##################################
################################## functions ##################################
########################### (No support for the db) ###########################
###############################################################################
def haar_wavelet_transf(X):
    Xt = np.array([np.concatenate(pywt.dwt(X[:, i], 'haar'))
                   for i in range(X.shape[1])])
    return Xt


def daubechies_wavelet_transf(X, parameter1=1):
    parameter1 = parameter1 if parameter1 in range(1, 21) else 1
    funct = 'db'+str(parameter1)
    funct = funct if funct in pywt.wavelist('db') else 'db1'
    Xt = np.array([np.concatenate(pywt.dwt(X[:, i], funct))
                   for i in range(X.shape[1])])
    return Xt


def symlets_wavelets_transf(X, parameter1=2):
    parameter1 = parameter1 if parameter1 in range(2, 21) else 2
    funct = 'sym'+str(parameter1)
    funct = funct if funct in pywt.wavelist('sym') else 'sym2'
    Xt = np.array([np.concatenate(pywt.dwt(X[:, i], funct))
                  for i in range(X.shape[1])])
    return Xt


def coiflets_wavelets_transf(X, parameter1=1):
    parameter1 = parameter1 if parameter1 in range(1, 6) else 1
    funct = 'coif'+str(parameter1)
    funct = funct if funct in pywt.wavelist('coif') else 'coif1'
    Xt = np.array([np.concatenate(pywt.dwt(X[:, i], funct))
                   for i in range(X.shape[1])])
    return Xt


def biorthogonal_wavelets_transf(X, parameter1=1, parameter2=1):
    parameter1, parameter2 = str(parameter1), str(parameter2)
    pars = [parameter1, parameter2]
    possible = [pywt.wavelist('bior')[i][4:].split('.')
                for i in range(len(pywt.wavelist('bior')))]
    pars = pars if pars in possible else ['1', '1']
    funct = 'bior'+'.'.join(pars)
    funct = funct if funct in pywt.wavelist('bior') else 'bior1.1'
    Xt = np.array([np.concatenate(pywt.dwt(X[:, i], funct))
                   for i in range(X.shape[1])])
    return Xt


def rbiorthogonal_wavelets_transf(X, parameter1=1, parameter2=1):
    parameter1, parameter2 = str(parameter1), str(parameter2)
    pars = [parameter1, parameter2]
    possible = [pywt.wavelist('rbio')[i][4:].split('.')
                for i in range(len(pywt.wavelist('rbio')))]
    pars = pars if pars in possible else ['1', '1']
    funct = 'rbio'+'.'.join(pars)
    funct = funct if funct in pywt.wavelist('rbio') else 'rbio1.1'
    Xt = np.array([np.concatenate(pywt.dwt(X[:, i], funct))
                   for i in range(X.shape[1])])
    return Xt


def dMeyer_wavelets_transf(X):
    Xt = np.array([np.concatenate(pywt.dwt(X[:, i], 'dmey'))
                   for i in range(X.shape[1])])
    return Xt
