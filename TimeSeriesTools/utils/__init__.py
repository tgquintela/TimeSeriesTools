
"""
utils
=====
Collection of basic utils for the treatment of time series.

"""

## Sampling utils
from sampling_utils import create_times_randompoints,\
    create_times_randomregimes, create_times_randombursts
## Util operations
from util_operations import join_regimes, format_as_regular_ts,\
    apply_gaussianconvolved_ts
