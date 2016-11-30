
"""
Test for regime detection
-------------------------
Collection of tests for the regime detection module functions.

"""

import numpy as np
import pandas as pd
from ..Regime_detection.aux_functions import sum_conservative_detectors
from ..Regime_detection.regime_detection import compute_threshold,\
    compute_units, compute_ref_value, ref_based_thres_builder,\
    discretizor_builder, general_regime_detection
from ..Transformation.transformations import general_transformation


def test():
    def generate_spks(n):
        times_event = np.random.randn(n).cumsum()
        element_event = np.random.randint(0, 5, n)
        regimes_event = np.random.randint(0, 2, n)
        return times_event, element_event, regimes_event

    def generate_detections(n):
        times_event, element_event, regimes_event = generate_spks(n)
        values_event = np.random.random(n)
        df = pd.concat([pd.DataFrame(times_event), pd.DataFrame(element_event),
                        pd.DataFrame(regimes_event),
                        pd.DataFrame(values_event)], axis=1)
        df.columns = ['times', 'neuron', 'regime', 'value']
        return df

    n = 1000
    detections = [generate_detections(n) for i in range(10)]

    ##### Auxiliar functions
    ########################
    sum_conservative_detectors(detections, n, 'or', True)
    sum_conservative_detectors(detections, n, 'or', False)

    refs, units = np.random.random((1000, 4)),  np.random.random((1000, 4))
    threshold_value = 0.5
    thrs = compute_threshold(refs, units, threshold_value)

    ### Signals
    signals = np.random.random((1000, 4))

    pars = {'method': 'windowing_transformation_f',
            'kwargs': {'window_info': None, 'step': 1, 'borders': False}}
    par_gen = {'method': 'windowing_transformation', 'args': pars}

    methods = ['unitary', 'gap', 'std', 'qqstd', par_gen]
    for method in methods:
        units = compute_units(signals, method, axisn=0)
#        compute_units(signals.T, method, axisn=1)
#        compute_units(signals, method, axisn=None)

    methods = ['null', 'min', 'mean', par_gen]
    for method in methods:
        refs = compute_ref_value(signals, method, axisn=0)
#        compute_ref_value(signals.T, method, axisn=1)
#        compute_ref_value(signals, method, axisn=None)

    thrs = compute_threshold(refs, units, threshold_value)

    def f_trans(x):
        return general_transformation(x, **par_gen)

    def f_trhes(x, y):
        return compute_threshold(x, y, .5)

    refs_infos = [{'method': 'null', 'axisn': 0}, par_gen, f_trans, refs]
    units_infos = [{'method': 'unitary', 'axisn': 0}, par_gen, f_trans, units]
    thres_infos = [{'threshold_value': 0.5}, f_trhes, f_trhes, thrs]

    for i in range(4):
        abs_refs = ref_based_thres_builder(signals, refs_infos[i],
                                           units_infos[i], thres_infos[i])

    discret_info = {'refs_info': refs_infos[i], 'units_info': units_infos[i],
                    'thresholds_info': thres_infos[i]}
    discretizor_builder(signals, 'threshold_based', discret_info)

    discdict_info = {'method': 'threshold_based', 'kwargs': {}}
    discdict_creation_info = {'method': 'threshold_based',
                              'discret_info': discret_info}
    nulldict_trans = {'method': '', 'args': {}}

    def disc_creation_f(x):
        return x-.01

    def null_trans_f(x, y):
        return x

    def null_f(x):
        return x

    general_regime_detection(signals, signals, signals-.01, discdict_info,
                             nulldict_trans, nulldict_trans)
    general_regime_detection(signals, signals, discdict_creation_info,
                             discdict_info, nulldict_trans, nulldict_trans)
    general_regime_detection(signals, nulldict_trans, disc_creation_f,
                             null_trans_f, nulldict_trans, nulldict_trans)
    general_regime_detection(signals, pd.DataFrame(signals), disc_creation_f,
                             null_trans_f, nulldict_trans, nulldict_trans)
    general_regime_detection(signals, null_f, pd.DataFrame(signals-.01),
                             null_trans_f, nulldict_trans, nulldict_trans)
    general_regime_detection(signals, pd.DataFrame(signals), disc_creation_f,
                             null_trans_f, nulldict_trans, nulldict_trans)
