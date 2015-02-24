
"""
Module which groups all the auxiliary functions for transforming each point of
a time series regarding their vicinity to a descriptor or descriptors.
The resultant of the application of these functions to a time series is a
2d array in which the first dim represents the descriptors and the other
the different elements of the system.
"""

import numpy as np


############################# auxiliary functions #############################
###############################################################################
def aggregating_ups_downs(Yt, n, desc=''):
    ## TODO: 2 ways. Measure or number

    neur = np.array(range(Yt.shape[1]))
    Ytt = np.zeros(Yt.shape)
    n1 = (n-1)*2+1
    for neu in neur:
        idx = np.where(Yt[:, neu] != 0)[0]
        for i in range(idx.shape[0]-n1+1):
            y = Yt[idx[i]:idx[i]+n1, neu]
            if desc == 'normal':
                Ytt[idx[i]] = np.sum(y)
            elif desc == 'shape':
                Ytt[idx[i]] = np.sum(y)/y.shape[0]
    return Ytt


def collapse3waveform(waveform_diff, event_t=None):
    """Funcions to collapse a waveform in 3 stages, pre-spike, spike and
    post-spike in order to be able to identify the spikes easily.
    The input it is supposed that is ups_downs discretized.
    """

    collapsed = np.zeros(3)
    if event_t is None:
        event_t = np.argmax(waveform_diff)

    collapsed[0] = np.sum(waveform_diff[:event_t])
    collapsed[1] = waveform_diff[event_t]
    collapsed[2] = np.sum(waveform_diff[event_t+1:])
    return collapsed


def collapse3waveform_matrix(waveform, event_t, axisn=0):
    """Funcions to collapse a waveform in 3 stages, pre-spike, spike and
    post-spike in order to be able to identify the spikes easily.

    Parameters
    ---------
    waveform: array_like, shape(Nf, M)
        representation of the waveform or part of the time series we want to
        transform and extracting representative features of the agrregated
        state of pre-event, event and post-event situation.
    event_t: int
        the index of the time in which the interested event occurs.
    axisn: int, optional
        the axis in which is represented the time.

    Returns
    -------
    collapsed: array_like, shape (3, M)
        collapsed representation of the waveform passed.
    """

    # How many waveforms we have
    axisn2 = (axisn - 1) % 2
    m = waveform.shape[axisn2]
    # Initialization
    collapsed = np.zeros((3, m))

    # Computation of the features
    collapsed[0, :] = np.sum(waveform[:event_t, :], axisn).reshape(-1)
    collapsed[1, :] = waveform[event_t, :].reshape(-1)
    collapsed[2, :] = np.sum(waveform[event_t+1:, :], axisn).reshape(-1)
    collapsed = collapsed.T
    return collapsed
