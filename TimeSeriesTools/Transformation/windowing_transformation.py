
"""
This module contains all the possible transformations performing and operation
over a window. The result could be an array_like multidimensional if the
transformation is not R1 to R1.
"""

import numpy as np
#import pandas as pd


def windowing_transformation(X, method, kwargs):
    """Switcher function to wrap the windowing transformations possibles and
    allow to select by string some specific functions.

    Parameters
    ----------
    X: array_like, shape (N,M) or pandas.DataFrame
        signals of the system.
    method: str, optional or function
        method to use windowing transformation.
    kwargs: dict
        variables needed

    Returns
    -------
    Yt: array_like, shape(N, M, nt) or pandas.DataFrame
        transformed signals. The transformation could be R^1 to R^nt

    """

    if type(method) != str:
        f = method
        method = 'personal'

    ## Switch
    if method == 'windowing_transformation_f':
        Yt = windowing_transformation_f(X, **kwargs)
    elif method == 'personal':
        Yt = f(X, **kwargs)
    elif method not in ['personal', 'windowing_transformation_f']:
        pass

    return Yt


def windowing_transformation_f(Y, window_info, step, borders=True,
                               transf=lambda x: x.mean(0)):
    """Transformation by windowing a time-series and applying to each
    symmetrical window a transformation given.
    There are the assumption that the time is represented in axis 0. It is
    accepted functions which operate over a 2d matrix returning the same
    number of elements dimmension 0 as there are in the input matrix.

    Parameters
    ----------
    Y : array_like, shape(N,)
        signal of all the elements of the system
    window_info : int or list
        number of the size of the window or the information of the previous and
        the next border of the window
    step : int
        the step of the window along the time-series.
    borders: boolean, float
        the possibility to create artificially borders with zero values in
        order to keep size of the matrix.
    transf: handle function
        the transformation to be applied to the time-series.

    Returns
    -------
    Yt : arry_like
        transformed signals. It could have a extra dimension if we want to
        represent some parameters or features per point.

    """

    ## 0. Initialization
    # Dimensions
    N = Y.shape[0]
    m = Y.shape[1] if len(Y.shape) != 1 else 1
    # Reshaping in order to keep dimensions
    reshaping = lambda X: X.reshape(X.shape[0], 1)
    if m == 1:
        Y = reshaping(Y)
    # Window setting
    if type(window_info) == int:
        w_init = window_info/2
        w_end = -w_init
    elif type(window_info) in [list, tuple]:
        w_init = window_info[0]
        w_end = window_info[1]
    elif window_info is None:
        w_init = -Y.shape[0]/2+(Y.shape[0] % 2)
        w_end = -w_init-((Y.shape[0]+1) % 2)
    ind = np.arange(N)
    wind = lambda p: ind[p+w_init:p+w_end+1]
    # Steps setting
    points = np.arange(-w_init, N-w_end, step)
    maxl = points.shape[0]
    # Test and quantify multidimensional output
    try:
        b_0 = transf(Y[:-w_init+w_end+1, 0])
        nt = b_0.shape[1]
    except:
        nt = 1
    #borders
    if type(borders) == bool:
        if borders:
            Yt = np.zeros((N, m, nt))
        else:
            Yt = np.zeros((maxl, m, nt))
    else:
        Yt = np.ones((N, m, nt))*borders

    # 1. Transformation
#    i = -w_init
    i = 0
    for p in points:
        ind_p = wind(p)
        aux = transf(Y[ind_p, :])
        if len(aux.shape) == 1:
            aux = reshaping(aux)
        Yt[i, :, :] = aux
        i += 1

    # 2. Formatting output
    if window_info is None:
        if type(borders) == bool:
            if borders:
                Yt = np.ones((N, m, nt))*aux
            else:
                Yt = np.ones((maxl, m, nt))*aux

    return Yt


#############################
###### TO DEPRECATE #########
#def windowing_discrete_transformation1(Y, times, window_info, step,
#                                       trans=lambda x: x.mean()):
#    """Transformation by windowing a time-series and applying to each window
#    a transformation given.
#    """
#
#    # Required transformations
#    if type(window_info) == int:
#        window_info = [-window_info/2, window_info/2]
#
#    # Variables needed
#    init = - window_info[0]
#    final = Y.shape[0] - window_info[1]
#    t_iter = times[init:final]
#    n_neu = Y.shape[1]
#
#    # Determine the number of descriptor per time.
#    try:
#        b_0 = np.logical_and(times > 1, times < window_info[0]+window_info[1])
#        nt = collapse3waveform(Y[b_0, 0], init).shape[0]
#    except:
#        nt = 1
#
#    # Build a descriptor matrix
#    Yt = np.zeros((Y.shape[0], Y.shape[1], nt))
#    for i in t_iter:
#        bool_i = np.logical_and(times >= i+window_info[0],
#                                times <= i+window_info[1])
#        Yt[i, :, :] = collapse3waveform_matrix(Y[bool_i, :], init, 1)
#
#    return Yt
