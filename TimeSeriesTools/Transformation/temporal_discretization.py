
"""
This module groups all the functions which carry out the process of temporal
discretization of a time series.
"""

import numpy as np


def temporal_discretization(Y, method='ups_downs', kwargs={}):
    """This function acts as a switcher for temporal discretizations and wraps
    all the functions which carry out discretization over time of time-series.

    Parameters
    ----------
    Y : array_like, shape (N, M)
        the signal of each element of the system. M time-series.
    method: str
        method used for performing temporal discretization.
    kwargs: dict
        required arguments for the method choosen.

    Returns
    -------
    Yt : array_like, shape (N, M)
        discretized activation matrix.

    TODO
    ----
    More than one descriptor.
    """

    if method == 'ups_downs':
        Yt = ups_downs_temporal_discretization_matrix(Y, **kwargs)
    elif method not in ['ups_downs']:
        pass

    return Yt


########################### transformation functions ##########################
###############################################################################
def ups_downs_temporal_discretization_matrix(Y, collapse_to='initial',
                                             feats=['augment']):
    '''Temporal discretization of the Time-series only considering the ups and
    downs changes. The criteria in which are grouped the sequence is a
    consequtive share first derivative. The value of this points will be the
    whole variation along this sequence.
    The collapsed points are selected in accordance with the method selected
    with the parameter collapse_to.

    Parameters
    ----------
    y : array_like, shape (N, M)
        the values of the time history of the signal.
    kwargs: dict
        the needed variables to use this method.

    Returns
    -------
    Yt: array_like, shape (N, M)
        transformed signals in dense representation.

    '''

    # Initialization
    Yt = np.zeros([Y.shape[0], Y.shape[1], len(feats)])
    # Loop for each element of the system
    for i in range(Y.shape[1]):
        pos, desc = ups_downs_temporal_discretization(Y[:, i], collapse_to,
                                                      feats)
        Yt[pos, i, :] = np.array(desc)

    return Yt


def ups_downs_temporal_discretization(y, collapse_to='initial',
                                      feats=['augment']):
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

        # Compute descriptors (GENERALIZE)
        aug = sign*(ts[downs-1]-ts[ups])
        shap = sign*aug/(downs-ups)

        # Store descriptors
        positions.append(pos)
        descriptors.append([aug])  # , shap])

    # Stack the information
    positions = np.hstack(positions)
    descriptors = np.hstack(descriptors).T

    # Order
    order = np.argsort(positions)
    positions = positions[order]
    descriptors = descriptors[order]

    return positions, descriptors
