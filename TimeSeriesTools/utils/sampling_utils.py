
"""
Sampling times
--------------
Collection of functions which sample times in an interval with different random
properties.

"""

import numpy as np


def create_times_randompoints(t_lim, t_inter=1.):
    """Create random temporal points with a regular intertemporal time.

    Parameters
    ----------
    t_lim: float
        the limit time.
    t_inter: float
        the average intertemporal time events

    Returns
    -------
    times_sp: np.ndarray
        the random sample time events.

    """
    times_sp = []
    t_last = 0
    while True:
        t = np.random.random()*t_inter
        t_last = t_last+t
        if t_last >= t_lim:
            break
        times_sp.append(t_last)
    times_sp = np.array(times_sp)
    return times_sp


def create_times_randomregimes(t_lim, t_inters):
    """Create different random point regimes with same time window but
    different properties.

    Parameters
    ----------
    t_lim: float
        the limit time.
    t_inters: list
        the average intertemporal time events.

    Returns
    -------
    ts_regimes: np.ndarray
        the random sample time events.

    """
    n_regimes = len(t_inters)
    ts_regimes = []
    for i in range(n_regimes):
        ts_regimes.append(create_times_randompoints(t_lim, t_inters[i]))
    return ts_regimes


def create_times_randombursts(ts0, next_levels):
    """Create different random point regimes with same time window but
    different properties.

    Parameters
    ----------
    ts0: np.ndarray
        the initial random sample times.
    next_levels: list
        the levels information for each considered regimes.

    Returns
    -------
    ts1: np.ndarray
        the random sample times.

    """
    if len(next_levels) > 1:
        ts1 = create_times_randombursts(ts0, [next_levels[0]])
        return create_times_randombursts(ts1, next_levels[1:])
    else:
        assert(len(next_levels) == 1)
        ts1 = []
        for i in range(len(ts0)):
            ts1.append(create_times_randompoints(*next_levels[0])+ts0[i])
        ts1 = np.sort(np.concatenate(ts1))
        return ts1
