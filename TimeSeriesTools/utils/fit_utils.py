
"""
Fit_utils
---------
Utils to fit measures.

"""

import numpy as np


################################### Fit utils #################################
###############################################################################
def general_multiscale_fit(measures, scales, method='loglogLSQ'):
    """General fitting functions multi-scales methods.

    Parameters
    ----------
    measures: np.ndarray
        the measures associated with the method for the given scales.
    scales: np.ndarray
        the scales we measured.
    method: str, optional
        the methods availables for doing the fit.
            * 'logLSQ': logarithmic least squares fit.

    Returns
    -------
    beta: float
        value of the slope of the fit, related with the final measure
        we want to obtain.

    """
    assert(method == 'loglogLSQ')
    measure = fit_loglogleastsquares(measures, scales)
    return measure


def fit_loglogleastsquares(measures, scales):
    """Fit RS log least squares method.

    Parameters
    ----------
    measures: np.ndarray
        the measures associated with the method for the given scales.
    scales: np.ndarray
        the scales we measured.

    Returns
    -------
    beta: float
        value of fits, related with the Hurst parameter.

    """
    scales = np.vstack([np.log(scales), np.ones(len(scales))]).T
    beta = np.linalg.lstsq(np.log(scales), np.log(measures))[0][0]
    return beta
