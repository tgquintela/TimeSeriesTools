
"""
Test_feature_extraction
-----------------------

"""

import numpy as np
from ..Feature_extraction.feature_transformation import feature_transformation
from ..Feature_extraction.feature_extraction import feature_extraction
from ..Feature_extraction.feature_representation import haar_wavelet_transf,\
    daubechies_wavelet_transf, symlets_wavelets_transf,\
    coiflets_wavelets_transf, biorthogonal_wavelets_transf,\
    rbiorthogonal_wavelets_transf, dMeyer_wavelets_transf,\
    dwavelet_feat_repres, feature_representation


def test():
    ## Create artificial data
    Xt = np.random.random((10000, 10))
    Xt = np.random.randn(10000, 10).cumsum(0)

    ## Feature extraction
    #####################
    feature_transformation(Xt, method_transformation='')
    feature_transformation(Xt, method_transformation='pca')
    feature_transformation(Xt, method_transformation='ica')

    ## Feature representation
    #########################
    haar_wavelet_transf(Xt)
    daubechies_wavelet_transf(Xt)
    symlets_wavelets_transf(Xt)
    coiflets_wavelets_transf(Xt)
    biorthogonal_wavelets_transf(Xt)
    rbiorthogonal_wavelets_transf(Xt)
    dMeyer_wavelets_transf(Xt)

    methods = ['haar', 'db', 'sym', 'coif', 'bior', 'rbior', 'dmey',
               'daubechies', 'symlets', 'coiflets',
               'biorthogonal', 'rbiorthogonal', 'dMeyer']
    for m in methods:
        dwavelet_feat_repres(Xt, m)

    methods = ['', 'diff', 'diff_agg', 'peaktovalley', 'wavelets', 'dwavelets']
    for m in methods:
        feature_representation(Xt, m)
    feature_representation(Xt, 'collapse3waveform', {'event_t': 10})
    feature_representation(Xt, ['collapse3waveform'], [{'event_t': 10}])

    ## Feature transformation
    #########################
    transf_pars = {'method_transformation': ''}
    repres_pars = {'methods_feature': 'diff'}
    feature_extraction(Xt, repres_pars, transf_pars)
