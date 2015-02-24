
"""
Module to group the feature extraction process.
"""

from feature_representation import feature_representation
from feature_transformation import feature_transformation


def feature_extraction(X, repres_pars, transf_pars):
    """This is a wrapper of functions that can perform as a feature extraction
    from time series.
    """

    # Representation in the selected feature space
    Xt = feature_representation(X, **repres_pars)
    # Transformation of the feature space
    Xt = feature_transformation(Xt, **transf_pars)
    return Xt
