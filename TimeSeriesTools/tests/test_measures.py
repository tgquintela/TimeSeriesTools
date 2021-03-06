
"""
test_measures
-------------
Collections of measures.

"""

import numpy as np
from ..artificial_data.create_artificial_timeseries import create_random_ts,\
    create_random_raster, create_brownian_noise_regular_ts
from ..Measures.hurst_measures import hurst
from ..Measures.hurst_measures import create_RS_scales_sequence,\
    create_aggvar_scales_sequence, create_peng_scales_sequence,\
    create_scales_higuchi_sequence, create_scales_periodogram_sequence
from ..Measures.fractal_dimension_measures import pfd, hfd
from ..Measures.information_theory_measures import entropy, svd_entropy,\
    spectral_entropy, fisher_info, dfa
from ..Measures.measures import hjorth


def test():
    levels_rand, magnitudes_rand = [[(200, .1)]], [1.]
    levels = [[(200, .01)], [(200, 20.), (5, 1.)], [(200, 50.0)],
              [(200, 15.), (10, .2)]]
    magnitudes = [.1, -10., -5., 15.]
    ts_rand, vals_rand = create_random_ts(levels_rand, magnitudes_rand)
    ts, vals = create_random_ts(levels, magnitudes)
    n_times, n_elements, n_regs = 200, 1, 10
    regimes_info = [(n_regs, np.random.random(n_regs))]
    raster = create_random_raster(len(ts), n_elements, regimes_info).squeeze()

    times_rd, rand_ts = create_brownian_noise_regular_ts(n_times)
    normal_rand = np.random.randn(n_times)
    uniform_rand = np.random.random(n_times)

    #### Test measures
    #################################################################

    ## Entropy measure
    mea = entropy(raster)
    assert(type(mea) == float)

    #### Hurst measure
    ##################
    ## Some failure. Not correct values obtained
    ##################
    T = create_RS_scales_sequence(rand_ts, sequence='power')
    T = create_RS_scales_sequence(rand_ts, sequence='complete')
    T = create_RS_scales_sequence(rand_ts, sequence=T)
    H = hurst(rand_ts, T, method='RS')
    H = hurst(normal_rand, T, method='RS')
    H = hurst(uniform_rand, T, method='RS')

    H = hurst(rand_ts, scales='power', method='RS_alternative')
    H = hurst(normal_rand, scales='power', method='RS_alternative')
    H = hurst(uniform_rand, scales='power', method='RS_alternative')

    T = create_aggvar_scales_sequence(rand_ts)
    T = create_aggvar_scales_sequence(rand_ts, T)
    H = hurst(rand_ts, T, method='aggvar')
    H = hurst(normal_rand, T, method='aggvar')
    H = hurst(uniform_rand, T, method='aggvar')

    T = create_peng_scales_sequence(rand_ts)
    T = create_peng_scales_sequence(rand_ts, T)
    H = hurst(rand_ts, T, method='peng')
    H = hurst(normal_rand, T, method='peng')
    H = hurst(uniform_rand, T, method='peng')

    T = create_scales_higuchi_sequence(rand_ts)
    T = create_scales_higuchi_sequence(rand_ts, T)
    H = hurst(rand_ts, T, method='higuchi')
    H = hurst(normal_rand, T, method='higuchi')
    H = hurst(uniform_rand, T, method='higuchi')

    T = create_scales_periodogram_sequence(rand_ts)
    T = create_scales_periodogram_sequence(rand_ts, T)
    ## Problems with the fit
    H = hurst(rand_ts, T, method='per')
    H = hurst(normal_rand, T, method='per')
    H = hurst(uniform_rand, T, method='per')

    #### Fractal dimenstion measure
    ###############################
    ## Use fit utilities and generalize fit
    ##################
    pfd(rand_ts)
    pfd(normal_rand)
    pfd(uniform_rand)
    hfd(rand_ts)
    hfd(normal_rand)
    hfd(uniform_rand)

    #### Information theory measure
    ###############################
    ##
    ##################
    entropy(rand_ts)
    entropy(normal_rand)
    entropy(uniform_rand)
    svd_entropy(rand_ts)
    svd_entropy(normal_rand)
    svd_entropy(uniform_rand)
    spectral_entropy(rand_ts)
    spectral_entropy(normal_rand)
    spectral_entropy(uniform_rand)
    fisher_info(rand_ts)
    fisher_info(normal_rand)
    fisher_info(uniform_rand)
    dfa(rand_ts)
    dfa(normal_rand)
    dfa(uniform_rand)

    #### Fractal dimenstion measure
    ###############################
    ## Use fit utilities and generalize fit
    ##################
    hjorth(rand_ts)
    hjorth(normal_rand)
    hjorth(uniform_rand)
