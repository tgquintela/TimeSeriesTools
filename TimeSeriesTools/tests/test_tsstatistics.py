
"""

"""

import numpy as np
from ..TS_statistics.utils import build_ngram_arrays, create_ngram,\
    uniform_input_samevals, uniform_input_lags
from ..TS_statistics.probabilitytools import compute_conditional_probs,\
    compute_marginal_probs
from ..TS_statistics.ts_statistics import prob_ngram_x, prob_ngram_xy,\
    prob_ngram_ind, prob_xy, prob_xy_ind, prob_x, compute_joint_probs
from ..TS_statistics.regime_statistics import prob_regimes_x, temporal_counts,\
    temporal_average_counts, count_repeated, temporal_densities, parse_spks,\
    isis_computation, isi_distribution, general_count, counts_normalization,\
    temporal_si, prob_spk_xy, count_into_bursts


def test():
    #####
    X = np.random.randn(1000).cumsum()
    X_disc = np.random.randint(0, 20, 1000)
    X_mdisc = np.random.randint(0, 20, (1000, 5))

    ################
    #Test functions
    ##################

    ## Utils
    #########
    uniform_input_lags([5], X)
    lags = uniform_input_lags(np.array([5]), X)
    uniform_input_samevals(True, X_disc)
    uniform_input_samevals(False, np.atleast_2d(X_disc).T)
    uniform_input_samevals(range(20), X_disc)
    uniform_input_samevals(np.arange(20), X_disc)

    ngram = create_ngram(X_disc, lags, samevals=True)
    lags = lags = uniform_input_lags(np.array([5]), X_disc)
    ngram = create_ngram(X_disc, range(5), samevals=True)

    pres, post = [0], [0]
    L = 1
    build_ngram_arrays(np.atleast_2d(X_disc).T, post, pres, L)
    L = 2
    build_ngram_arrays(np.atleast_2d(X_disc).T, post, pres, L)

    ## TO IMPLEMENT
#    X_mdisc = np.random.randint(0, 20, (1000, 5))
#    pres, post = [0, 1, 2], [[1, 2, 3], [2, 3, 4], [0, 1, 4]]
#    build_ngram_arrays(X_mdisc, post, pres, L)
#
    ## probabilitytools
    ###################
    probs = np.random.random((10, 7, 5))
    p_x = compute_marginal_probs(probs, [0])
    assert(p_x.shape == (7, 5))
    p_x = compute_marginal_probs(probs, [1])
    assert(p_x.shape == (10, 5))
    p_x = compute_marginal_probs(probs, [1, 2])
    assert(p_x.shape == (10,))

    p_x_y = compute_conditional_probs(probs, [0])
    assert(p_x_y.shape == probs.shape)
    p_x_y = compute_conditional_probs(probs, [1])
    assert(p_x_y.shape == probs.shape)
    p_x_y = compute_conditional_probs(probs, [1, 2])
    assert(p_x_y.shape == probs.shape)

    ## probabilitytools
    ###################
    L = 3
    X_m = np.random.randint(0, 5, (1000, 4))
    x, y = np.random.randint(0, 5, 1000), np.random.randint(0, 5, 1000)

    prob_ngram_ind(x, y, 1, auto=False, samevals=False, normalize=True)
    prob_ngram_ind(x, y, 1, auto=True, samevals=[range(5)]*2, normalize=True)
    prob_ngram_ind(x, y, 2, auto=False, samevals=False, normalize=True)
    prob_ngram_ind(x, y, 2, auto=True, samevals=[np.arange(5)]*2,
                   normalize=True)
    prob_ngram_ind(x, y, L, auto=False, samevals=np.arange(5), normalize=True)
    prob_ngram_ind(x, y, L, auto=True, samevals=True, normalize=True)

    prob_ngram_xy(X_m, 1, bins=None, auto=True, samevals=True, normalize=True)
    prob_ngram_xy(X_m, 1, bins=None, auto=True, samevals=False, normalize=True)
    prob_ngram_xy(X_m, 1, bins=None, auto=True, samevals=np.arange(5),
                  normalize=True)
    prob_ngram_xy(X_m, 1, bins=None, auto=True, samevals=[range(5)]*4,
                  normalize=True)

    prob_ngram_x(X_m, 1, bins=None, samevals=True, normalize=True)
    prob_ngram_x(X_m, 1, bins=None, samevals=False, normalize=True)
    prob_ngram_x(X_m, 1, bins=None, samevals=np.arange(5), normalize=True)
    prob_ngram_x(X_m, 1, bins=None, samevals=[np.arange(5)]*4, normalize=True)

#    compute_joint_probs(X_m, values=[], normalize=True)
#    compute_joint_probs(X_m, values=np.arange(5), normalize=True)
#    compute_joint_probs(X_m, values=np.arange(5), normalize=True)
#
#    prob_xy_ind(x, y, samevals=True, bins=0, timelag=0, normalize=True)
#    prob_xy_ind(x, y, samevals=False, bins=0, timelag=1, normalize=True)
#    prob_xy_ind(x, y, samevals=True, bins=0, timelag=0, normalize=True)
#    prob_xy_ind(x, y, samevals=False, bins=0, timelag=2, normalize=True)
#    prob_xy_ind(x, y, samevals=False, bins=0, timelag=-2, normalize=True)
#    prob_xy_ind(x, y, samevals=False, bins=0, timelag=-1, normalize=True)
#    prob_xy_ind(x, y, samevals=False, bins=[None, None], timelag=-1,
#                normalize=True)
#    prob_xy_ind(x, y, samevals=False, bins=[0, 0], timelag=-1, normalize=True)
#    prob_xy_ind(x, y, samevals=np.arange(4), bins=2, timelag=-1,
#                normalize=True)
#
#    prob_xy(X_m, bins=0, maxl=1, samevals=True)
#    prob_xy(X_m, bins=0, maxl=1, samevals=False)
#    prob_xy(X_m, bins=0, maxl=1, samevals=np.arange(5))
#    prob_xy(X_m, bins=0, maxl=1, samevals=[np.arange(5)]*4)
#
#    prob_x(X_m, n_bins=0, individually=True, normalize=True)
#    prob_x(X_m, n_bins=2, individually=False, normalize=True)
#
#    #### Regime statistics
#    #######################
#    raster = np.random.randint(0, 10, (1000, 4))
#    times_event = np.random.randn(1000).cumsum()
#    element_event = np.random.randint(0, 5, 1000)
#    regimes_event = np.random.randint(0, 2, 1000)
#    spks = times_event, element_event, regimes_event
#
#    parse_spks(raster)
#    parse_spks(spks)
#
#    prob_regimes_x(spks)
#    prob_regimes_x(spks, 10)
#    prob_regimes_x(spks, normalized=True)
#    prob_regimes_x(spks, 10, normalized=True)
#
#    temporal_counts(spks, normalized=False, collapse_reg=True)
#    temporal_counts(spks, normalized=False, collapse_reg=False)
#    temporal_counts(spks, normalized=True, collapse_reg=True)
#    temporal_counts(spks, normalized=True, collapse_reg=False)
#
#    temporal_average_counts(spks, window=0, collapse_reg=True)
#    temporal_average_counts(spks, window=20, collapse_reg=True)
#    temporal_average_counts(spks, window=0, collapse_reg=False)
#    temporal_average_counts(spks, window=20, collapse_reg=False)
#
#    count_repeated(raster[:, 0])
#
#    temporal_densities(spks, w_limit=0, collapse_reg=True)
#    temporal_densities(spks, w_limit=20, collapse_reg=True)
#    temporal_densities(spks, w_limit=0, collapse_reg=False)
#    temporal_densities(spks, w_limit=20, collapse_reg=False)
#
#    ## Interevent statistics
#    ##
#    isis_computation(spks, logscale=False)
#    isis_computation(spks, logscale=True)
#
#    isi_distribution(spks, 10, globally=False, normalized=True,
#                     logscale=True)
#    isi_distribution(spks, 10, globally=True, normalized=True,
#                     logscale=False)
#    isi_distribution(spks, 10, globally=False, normalized=False,
#                     logscale=False)
#    isi_distribution(spks, 10, globally=True, normalized=False,
#                     logscale=True)
#
#    temporal_si(spks)
#
#    ## Probs utils
#    ##
#    c_xy, max_l = np.random.randint(0, 100, (10, 10, 6)), 5
#    general_count(c_xy)
#    general_count(c_xy, max_l)
#
#    counts_normalization(np.random.randint(0, 100, 1000), 1000)
#    counts_normalization(c_xy, 1000)
#
#    ## Counting probs in the events information information
#    ##
#    prob_spk_xy(spks, max_l=8, normalized=False)
#    prob_spk_xy(spks, max_l=8, normalized=True)
#
#    ## Statistics bursts
#    bursts = [np.arange(1, 10), np.arange(50, 60)]
#    count_into_bursts(spks, bursts, elements=None)
#    count_into_bursts(spks, bursts, elements=range(5))
