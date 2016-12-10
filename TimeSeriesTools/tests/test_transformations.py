
"""
test_transformations
--------------------

"""

import numpy as np
from ..artificial_data.create_artificial_timeseries import create_random_ts,\
    create_random_raster, create_brownian_noise_regular_ts
from ..utils.util_operations import format_as_regular_ts
from ..Transformation.filtering import general_filtering,\
    general_collapser_func, trim_mean
from ..Transformation.aux_transformation import aggregating_ups_downs,\
    collapse3waveform, collapse3waveform_matrix
from ..Transformation.temporal_discretization import temporal_discretization,\
    ups_downs_temporal_discretization_matrix, ups_downs_temporal_discretization
from ..Transformation.windowing_transformation import \
    windowing_transformation, windowing_transformation_f
from ..Transformation.transformations import general_transformation
from ..Transformation.value_discretization import value_discretization,\
    general_pattern_matching, discretize_with_thresholds,\
    statistical_discretizor


def test():
    ## Artificial data
    levels_rand, magnitudes_rand = [[(200, .1)]], [1.]
    levels = [[(200, .05)], [(200, 20.), (5, 1.)], [(200, 50.0)],
              [(200, 70.), (5, .2)]]
    magnitudes = [.1, -10., -5., 15.]
    ts_rand, vals_rand = create_random_ts(levels_rand, magnitudes_rand)
    ts, vals = create_random_ts(levels, magnitudes)
    n_times, n_elements, n_regs = 200, 1, 5
    regimes_info = [(n_regs, np.random.random(n_regs))]
    raster = create_random_raster(len(ts), n_elements, regimes_info).squeeze()
    intervals = (0, int(ts[-1]), 2*ts[-1]/len(ts))
    ts_reg, vals_reg = format_as_regular_ts(ts, vals, intervals)

    times_rd, rand_ts = create_brownian_noise_regular_ts(n_times)
    normal_rand = np.random.randn(n_times)
    uniform_rand = np.random.random(n_times)

    ##### Filtering
    ###############
    # Extend the options
    #################

    ## savitzky_golay
    #################
    ys = general_filtering(np.atleast_2d(rand_ts).T, method='savitzky_golay',
                           parameters={'window_size': 9, 'order': 2})
    try:
        boolean = False
        ys = general_filtering(np.atleast_2d(rand_ts).T,
                               method='savitzky_golay',
                               parameters={'window_size': 8, 'order': 2})
        boolean = True
        raise Exception("It should halt.")
    except:
        if boolean:
            raise Exception("It should halt.")
    try:
        boolean = False
        ys = general_filtering(np.atleast_2d(rand_ts).T,
                               method='savitzky_golay',
                               parameters={'window_size': .8, 'order': 2})
        boolean = True
        raise Exception("It should halt.")
    except:
        if boolean:
            raise Exception("It should halt.")
    try:
        boolean = False
        ys = general_filtering(np.atleast_2d(rand_ts).T,
                               method='savitzky_golay',
                               parameters={'window_size': .8, 'order': 2})
        boolean = True
        raise Exception("It should halt.")
    except:
        if boolean:
            raise Exception("It should halt.")
    try:
        boolean = False
        ys = general_filtering(np.atleast_2d(rand_ts).T,
                               method='savitzky_golay',
                               parameters={'window_size': 5, 'order': 4})
        boolean = True
        raise Exception("It should halt.")
    except:
        if boolean:
            raise Exception("It should halt.")

    ## weighted_MA
    ###############
    ys = general_filtering(np.atleast_2d(rand_ts).T, method='weighted_MA',
                           parameters={'window_len': 2, 'window': 'hanning'})
    ys = general_filtering(np.atleast_2d(rand_ts).T, method='weighted_MA',
                           parameters={'window_len': 11, 'window': 'hanning'})

    type0 = ['flat']
    type1 = ['hamming', 'hanning', 'bartlett', 'blackman']
    type2 = ['triang', 'flattop', 'parzen', 'bohman',
             'blackmanharris', 'nuttall', 'barthann']
    type3 = ['kaiser', 'gaussian', 'slepian', 'chebwin']
    type4 = ['general_gaussian']
    type5 = ['alpha_trim_window']
    type6 = ['median_window', 'snn_1d']
    windows_pos = type0+type1+type2
    for w in windows_pos:
        ys = general_filtering(np.atleast_2d(rand_ts).T, method='weighted_MA',
                               parameters={'window_len': 11,
                                           'window': w})
    i, args = 0, [4.0, 4.0, 0.001, 1.0]
    for w in type3:
        ys = general_filtering(np.atleast_2d(rand_ts).T, method='weighted_MA',
                               parameters={'window_len': 11,
                                           'window': w, 'args': [args[i]]})
        i += 1
    for w in type4:
        ys = general_filtering(np.atleast_2d(rand_ts).T, method='weighted_MA',
                               parameters={'window_len': 11,
                                           'window': w, 'args': [0.4, 2.]})
        ys = general_filtering(np.atleast_2d(rand_ts).T[:-1],
                               method='weighted_MA',
                               parameters={'window_len': 11,
                                           'window': w, 'args': [0.4, 2.]})
    for w in type5:
        ys = general_filtering(np.atleast_2d(rand_ts).T, method='weighted_MA',
                               parameters={'window_len': 11,
                                           'window': w, 'args': [0.5]})
        ys = general_filtering(np.atleast_2d(rand_ts).T[:-1],
                               method='weighted_MA',
                               parameters={'window_len': 11,
                                           'window': w, 'args': [0.5]})

    for w in type6:
        ys = general_filtering(np.atleast_2d(rand_ts).T, method='weighted_MA',
                               parameters={'window_len': 11,
                                           'window': w})

    try:
        boolean = False
        ys = general_filtering(np.atleast_2d(rand_ts).T, method='weighted_MA',
                               parameters={'window_len': 11,
                                           'window': ''})
        boolean = True
        raise Exception("It should halt.")
    except:
        if boolean:
            raise Exception("It should halt.")
    try:
        boolean = False
        ys = general_filtering(np.atleast_2d(rand_ts).T, method='weighted_MA',
                               parameters={'window_len': 1000,
                                           'window': 'hanning'})
        boolean = True
        raise Exception("It should halt.")
    except:
        if boolean:
            raise Exception("It should halt.")
    try:
        boolean = False
        ys = general_filtering(np.atleast_2d(rand_ts).T, method='weighted_MA',
                               parameters={'window_len': 3,
                                           'window': 'kaiser'})
        boolean = True
        raise Exception("It should halt.")
    except:
        if boolean:
            raise Exception("It should halt.")

    ## fft_passband_filter
    ######################
#    fft_passband_filter(y, f_low=0, f_high=1, axis=0)
    ys = general_filtering(np.atleast_2d(rand_ts).T, method='fft_passband',
                           parameters={'f_low': 0, 'f_high': 1, 'axis': 0})
    ys = general_filtering(np.atleast_2d(rand_ts).T, method='fft_passband',
                           parameters={'f_low': 0.2, 'f_high': .05, 'axis': 0})

    ## reweighting
    ##############
    def null_reweighting(X):
        return X
    ys = general_filtering(np.atleast_2d(rand_ts).T, method='reweighting',
                           parameters={'method': 'power_sutera'})
    ys = general_filtering(np.atleast_2d(rand_ts).T, method='reweighting',
                           parameters={'method': null_reweighting})

    ## collapse
    ##############
    def f_method(ranges):
        return ranges.max(axis=1).round().astype(int)
    methods_col = ['center', 'initial', 'final', f_method]
    APbool = np.random.randint(0, 2, 1000)

    for m in methods_col:
        general_collapser_func(APbool, m)

    collapse_info0 = {0: methods_col[0], 1: methods_col[1], 2: methods_col[2],
                      3: methods_col[3], 4: methods_col[3]}
    collapse_info1 = 'center'
    collapse_info2 = lambda x: general_collapser_func(x, 'center')
    collapses_infos = [collapse_info0, collapse_info1, ['center'],
                       collapse_info2, [collapse_info2], [collapse_info0]]
    for col in collapses_infos:
        ys = general_filtering(np.atleast_2d(raster).T, method='collapse',
                               parameters={'reference': 0.,
                                           'collapse_info': col})

    ## substitution
    ###############
    u_raster = np.unique(raster)
    new_u = u_raster[np.random.permutation(len(u_raster))]
    subs = dict(zip(u_raster, new_u))
    ys = general_filtering(np.atleast_2d(raster).T, method='substitution',
                           parameters={'subs': subs})

    ## Auxiliar functions filtering
    ###############################
    trim_mean(rand_ts, .7)

    ## Auxiliar functions filtering
    ###############################
    cut_reg = vals_reg
    cut_reg[vals_reg < 0.5] = 0
    aggregating_ups_downs(cut_reg, n=5, desc='shape')
    aggregating_ups_downs(vals_reg, n=10, desc='normal')
    collapse3waveform(np.random.randn(100))
    collapse3waveform_matrix(np.random.randn(100, 4), 50)

    ## Temporal discretization transformation
    #########################################
    temporal_discretization(np.atleast_2d(rand_ts).T, 'ups_downs',
                            {'collapse_to': 'initial'})
    temporal_discretization(np.atleast_2d(rand_ts).T, 'ups_downs',
                            {'collapse_to': 'center'})
    temporal_discretization(np.atleast_2d(rand_ts).T, 'ups_downs',
                            {'collapse_to': 'final'})
#    ups_downs_temporal_discretization_matrix
#    ups_downs_temporal_discretization

    ## Windowing transformation
    ############################
    kwargs0 = {'window_info': None, 'step': 1, 'borders': False}
    kwargs1 = {'window_info': 10, 'step': 2, 'borders': True}
    kwargs2 = {'window_info': (0, 10), 'step': 3, 'borders': True}
    windowing_transformation(rand_ts,
                             'windowing_transformation_f', kwargs0)
    windowing_transformation(np.atleast_2d(rand_ts).T,
                             'windowing_transformation_f', kwargs1)
    windowing_transformation(np.atleast_2d(rand_ts).T,
                             windowing_transformation_f, kwargs2)
    windowing_transformation(np.atleast_2d(rand_ts).T,
                             'windowing_transformation_f', kwargs0)
    windowing_transformation_f(np.atleast_2d(rand_ts).T, **kwargs0)
    windowing_transformation_f(np.atleast_2d(rand_ts).T, **kwargs1)
    windowing_transformation_f(np.atleast_2d(rand_ts).T, **kwargs2)

    ## General transformation
    ############################
    args_wind = {'method': 'windowing_transformation_f', 'kwargs': kwargs0}
    general_transformation(np.atleast_2d(rand_ts).T,
                           'windowing_transformation', args_wind)
    general_transformation(np.atleast_2d(rand_ts).T,
                           ['windowing_transformation'], [args_wind])

    args_filt = {'method': 'weighted_MA', 'parameters': {'window_len': 11,
                                                         'window': 'hanning'}}
    general_transformation(np.atleast_2d(rand_ts).T, 'filtering', args_filt)

    args_disc = {'method': 'ups_downs', 'kwargs': {'collapse_to': 'final'}}
    general_transformation(np.atleast_2d(rand_ts).T,
                           'temporal_discretization', args_disc)
