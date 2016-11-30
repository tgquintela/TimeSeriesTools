
"""
Test burst detection
--------------------
Collection of tests for the burst detection functions.

"""

import numpy as np
from scipy import signal, interpolate

from ..Burst_detection.bursts_detection import dummy_burst_detection,\
    general_burst_detection, kleinberg_burst_detection, kleinberg
from ..artificial_data.create_artificial_timeseries import create_random_ts,\
    create_random_raster, create_brownian_noise_regular_ts


def test():
    ## Artificial data
    levels_rand, magnitudes_rand = [[(200, .1)]], [1.]
    levels = [[(200, .05)], [(200, 20.), (5, 1.)], [(200, 50.0)],
              [(200, 70.), (5, .2)]]
    magnitudes = [.1, -10., -5., 15.]
    ts_rand, vals_rand = create_random_ts(levels_rand, magnitudes_rand)
    ts, vals = create_random_ts(levels, magnitudes)

    dummy_burst_detection(np.atleast_2d(vals[:200]).T, 10)
#    kleinberg_burst_detection
    kleinberg(np.atleast_2d(vals[:2000]).T)
    general_burst_detection(np.atleast_2d(vals[:200]).T, 'dummy',
                            {'t_max': 10})
