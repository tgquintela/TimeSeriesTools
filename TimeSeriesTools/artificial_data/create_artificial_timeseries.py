
"""
Create artificial timeseries
----------------------------


"""

import numpy as np
try:
    from pyfftw.interfaces.numpy_fft import irfft
except ImportError:
    from numpy.fft import irfft

from ..utils.sampling_utils import create_times_randombursts
from ..utils import join_regimes


########################## Regular timespaced series ##########################
###############################################################################
def create_white_noise_regular_ts(n_t, std=1.0):
    """Create random time series with spikes and bursts in every level we want.

    Parameters
    ----------
    n_t: int
        the number of time steps of the timeseries.
    std: float
        the standard deviation of the white noise.

    Returns
    -------
    times: np.ndarray
        the times sampled.
    values: np.ndarray
        the values of the signal in each times sampled.

    """
    values = np.random.randn(n_t)*std
    times = np.arange(n_t)
    return times, values


def create_brownian_noise_regular_ts(n_t, std=1.0):
    """Create random time series with brownian noise.

    Parameters
    ----------
    n_t: int
        the number of time steps of the timeseries.
    std: float
        the standard deviation of the white noise.

    Returns
    -------
    times: np.ndarray
        the times sampled.
    values: np.ndarray
        the values of the signal in each times sampled.

    """
    times, values = create_white_noise_regular_ts(n_t, std)
    values = values.cumsum()
    return times, values


def create_blue_noise_regular_ts(n_t, par=1.0):
    """Create random time series with blue noise.

    Parameters
    ----------
    n_t: int
        the number of time steps of the timeseries.
    par: float
        the parameter which determines the scale of the variations.

    Returns
    -------
    times: np.ndarray
        the times sampled.
    values: np.ndarray
        the values of the signal in each times sampled.

    """
    uneven = n_t % 2
    X = np.random.randn(n_t//2+1+uneven) + 1j*np.random.randn(n_t//2+1+uneven)
    S = np.sqrt(np.arange(len(X)))
    y = (irfft(X*S)).real
    if uneven:
        y = y[:-1]
    values = y*np.sqrt(par/(np.abs(y)**2.0).mean())
    times = np.arange(n_t)
    assert(len(values) == len(times))
    return times, values


def create_brown_noise_regular_ts(n_t, par=1.0):
    """Create random time series with brown noise.

    Parameters
    ----------
    n_t: int
        the number of time steps of the timeseries.
    par: float
        the parameter which determines the scale of the variations.

    Returns
    -------
    times: np.ndarray
        the times sampled.
    values: np.ndarray
        the values of the signal in each times sampled.

    """
    uneven = n_t % 2
    X = np.random.randn(n_t//2+1+uneven) + 1j*np.random.randn(n_t//2+1+uneven)
    S = np.arange(len(X))+1
    y = (irfft(X/S)).real
    if uneven:
        y = y[:-1]
    values = y*np.sqrt(par/(np.abs(y)**2.0).mean())
    times = np.arange(n_t)
    assert(len(values) == len(times))
    return times, values


def create_violet_noise_regular_ts(n_t, par=1.0):
    """Create random time series with violet noise.

    Parameters
    ----------
    n_t: int
        the number of time steps of the timeseries.
    par: float
        the parameter which determines the scale of the variations.

    Returns
    -------
    times: np.ndarray
        the times sampled.
    values: np.ndarray
        the values of the signal in each times sampled.

    """
    uneven = n_t % 2
    X = np.random.randn(n_t//2+1+uneven) + 1j*np.random.randn(n_t//2+1+uneven)
    S = np.arange(len(X))
    y = (irfft(X*S)).real
    if uneven:
        y = y[:-1]
    values = y*np.sqrt(par/(np.abs(y)**2.0).mean())
    times = np.arange(n_t)
    assert(len(values) == len(times))
    return times, values


######################### Irregular timespaced series #########################
###############################################################################
def create_random_ts(levels, magnitudes, t0=None):
    """Create random time series with spikes and bursts in every level we want.

    Parameters
    ----------
    levels: list of list of tuples
        the level information for each regime.
    magnitudes: list
        the magnitude of each regime.

    Returns
    -------
    times: np.ndarray
        the times sampled.
    values: np.ndarray
        the values of the signal in each times sampled.

    """
    ## 0. Prepare inputs
    t0 = np.array([0.]) if t0 is None else t0
    assert(len(levels) == len(magnitudes))
    assert(all([type(e) == list for e in levels]))
    assert(all([all([type(e) == tuple for e in l]) for l in levels]))

    times = []
    for i in range(len(levels)):
        times.append(create_times_randombursts(t0, levels[i]))
    times, values = join_regimes(times, magnitudes)
    return times, values


def create_random_raster(n_times, n_elements, regimes_info):
    """Create random rasters with different times, elements and regimes with
    uniform probability of regimes and non-uniform probability of regimes.

    Parameters
    ----------
    n_times: int
        the number of times we want to use.
    n_elements: int
        the number of elements we want to create.
    regimes_info: list of list or list of int
        the regimes information to create each raster regime.

    Returns
    -------
    raster: np.ndarray, shape (n_times, n_elements, len(regimes_info))
        the raster created.

    TODO
    ----
    Random quantities in the probability sample.

    """
    assert(type(regimes_info) == list)
    raster = []
    for i in range(len(regimes_info)):
        ## Prepare inputs
        if type(regimes_info[i]) == list:
            bool_uniform, bool_direct = True, False
            regime_list = regimes_info[i]
        elif type(regimes_info[i]) == tuple:
            bool_uniform, bool_direct = False, False
            assert(type(regimes_info[i][1]) in [np.ndarray, list])
            probs_regimes = regimes_info[i][1]/np.sum(regimes_info[i][1])
            if type(regimes_info[i][0]) == list:
                regime_list = regimes_info[i][0]
            else:
                regime_list = range(regimes_info[i][0])
            assert(len(probs_regimes) == len(regime_list))
        else:
            bool_uniform, bool_direct = True, True
        ## Compute raster_i
        if not bool_uniform:
            n_regimes = [int(probs_regimes[i]*(n_times*n_elements))
                         for i in range(len(regime_list))]
            n_regimes[-1] = (n_times*n_elements) - np.sum(n_regimes[:-1])
            assert(np.sum(n_regimes) == (n_times*n_elements))
            raster_i = np.zeros(n_times*n_elements).astype(int)
            permuts = np.random.permutation(n_times*n_elements)
            init = 0
            for j in range(len(n_regimes)):
                endit = init + n_regimes[j]
                raster_i[permuts[init:endit]] = regime_list[j]
                init = endit
        else:
            if bool_direct:
                raster_i = np.random.randint(0, regimes_info[i],
                                             n_times*n_elements)
            else:
                aux_i = np.random.randint(0, len(regimes_info[i]),
                                          n_times*n_elements)
                raster_i = np.zeros(n_times*n_elements).astype(int)
                for j in range(len(regimes_info[i])):
                    raster_i[aux_i == j] = regimes_info[i][j]
        raster_i = raster_i.reshape((n_times, n_elements)).astype(int)
        raster.append(raster_i)
    raster = np.stack(raster, axis=2)
    return raster
