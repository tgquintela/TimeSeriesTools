
"""
test_utils
----------

"""

import numpy as np

from ..utils.sampling_utils import create_times_randompoints,\
    create_times_randomregimes, create_times_randombursts
from ..utils.util_operations import join_regimes, format_as_regular_ts,\
    apply_gaussianconvolved_ts
from ..utils.sliding_utils import sliding_embeded_transf
from ..utils.fit_utils import general_multiscale_fit,\
    fit_loglogleastsquares


def test():
    create_times_randompoints(100)
    create_times_randomregimes(100, [.1, .2])
    ts0 = create_times_randombursts(np.array([0.]), [(100, 1.), (5, .01)])
    ts1 = create_times_randombursts(np.array([0.]), [(100, 1.)])

    #### Operations
    # Parameters
    tss = [ts0, ts1]
    regimes = [.4, 2.]
    intervals = (0, 100, .2)

    times, values = join_regimes(tss, regimes)
    gridtimes, gridvalues = format_as_regular_ts(times, values, intervals)
    apply_gaussianconvolved_ts(gridvalues, 5, .2)

    ### Sliding utils
    #################
    D, step = 2, 1
    tau = 3

    sliding_embeded_transf(values, tau, D,  step)
    try:
        tau = .4
        boolean = False
        sliding_embeded_transf(values, tau, D,  step)
        boolean = True
    except:
        if boolean:
            raise Exception()
    try:
        tau = len(values)
        boolean = False
        sliding_embeded_transf(values, tau, D,  step)
        boolean = True
    except:
        if boolean:
            raise Exception()

    ### Fit utils
    ##############
    x = np.random.random(1000).cumsum()
    y = x + np.random.randn(1000)*0.01
    x = x + x.min()+0.001
    y = y + y.min()+0.001
    general_multiscale_fit(x, y, 'loglogLSQ')
    fit_loglogleastsquares(x, y)
