
"""
test_artificial_data
--------------------

"""

from ..artificial_data.create_artificial_timeseries import create_random_ts,\
    create_random_raster, create_white_noise_regular_ts,\
    create_brownian_noise_regular_ts, create_blue_noise_regular_ts,\
    create_brown_noise_regular_ts, create_violet_noise_regular_ts


def test():
    levels = [[(1000, .01)], [(1000, 20.), (5, 1.)], [(1000, 50.0)],
              [(1000, 15.), (10, .2)]]
    magnitudes = [.1, -10., -5., 15.]
    ts, vals = create_random_ts(levels, magnitudes)
    raster = create_random_raster(1000, 10, [2, [0, 4, 5]])
    raster = create_random_raster(1000, 10, [(2, [.5, .5]),
                                             ([0, 3, 4], [6., 4., 5.])])

    ####### Create random timeseries
    ################################
    create_white_noise_regular_ts(1000)
    create_brownian_noise_regular_ts(1000)
    create_blue_noise_regular_ts(1000)
    create_brown_noise_regular_ts(1000)
    create_violet_noise_regular_ts(1000)
    create_white_noise_regular_ts(999)
    create_brownian_noise_regular_ts(999)
    create_blue_noise_regular_ts(999)
    create_brown_noise_regular_ts(999)
    create_violet_noise_regular_ts(999)
