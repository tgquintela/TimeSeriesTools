
"""
Pattern matching detection.
"""

import numpy as np
import pandas as pd


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
