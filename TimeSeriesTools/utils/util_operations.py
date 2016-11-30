
"""
Utils operations
----------------
Collection of util operations for timeseries.

"""

import numpy as np
from scipy import signal, interpolate


def join_regimes(times, magnitudes):
    """Join different time series which represents events time series of
    different regimes and join altogether creating random values for the
    actual time series and scaling them using magnitudes
    information.

    Parameters
    ----------
    times: list of np.ndarray
        the list of the event time timeseries considered.
    magnitudes: list
        the scale of the random values of these event series.

    Returns
    -------
    times: np.ndarray
        the times samples for which we have values measured.
    values: np.ndarray
        the values of each time sample for the joined time-series.

    """
    assert(len(times) == len(magnitudes))
    values = [np.random.random(len(times[i]))*magnitudes[i]
              for i in range(len(times))]
    times = np.concatenate(times)
    values = np.concatenate(values)
    idxs = np.argsort(times)
    times = times[idxs]
    values = values[idxs]
    return times, values


def format_as_regular_ts(times, values, intervals):
    """Format timeseries as regular timeseries.

    Parameters
    ----------
    times: np.ndarray, shape (n,)
        the times sample we have values measured.
    values: np.ndarray, shape (n,)
        the values of the measured time samples.
    intervals: tuple (init, endit, step)
        the information needed to define the regular times sample.

    Returns
    -------
    x: np.ndarray, shape (n,)
        the regular times samples for which we want the values measured.
    v: np.ndarray, shape (n,)
        the measured values for the regular times sample.

    """
    x = np.arange(*intervals)
    v = interpolate.griddata(np.atleast_2d(times).T, values,
                             np.atleast_2d(x).T, 'linear')
    v = v.squeeze()
    return x, v


def apply_gaussianconvolved_ts(gridvalues, n_wind, stds):
    """Apply gaussian convolution to the given regular time series.

    Parameters
    ----------
    gridvalues: np.ndarray, shape (n,)
        the values for the regular sample times.
    n_wind: int
        the size of the window in which we want to apply the convolution.
    stds: float
        the value of the standard deviation we want to have the gaussian
        filter we want to apply.

    Returns
    -------
    convvalues: np.ndarray, shape (n,)
        the convolved resultant time series.

    """
    wind = signal.gaussian(n_wind, stds)
    convvalues = signal.convolve(gridvalues, wind, 'same')
    return convvalues
