
"""
Module which groups all the functions which transforms the feature space
in order to get a representation of the timeseries with wanted properties or
clarity denoising the possible initial result.

TODO
----
Return also the function class.
"""

from sklearn.decomposition import PCA, FastICA


def feature_transformation(Xt, method_transformation='', kwargs={}):
    """This function acts as a wrapper to all functions of transformation of
    the feature representation of the time-series.

    """

    # Format inputs
    Xt = Xt.T

    ## Feature compression
    if method_transformation == '':
        Xdec = Xt
    elif method_transformation == 'pca':
        pca = PCA()
        Xdec = pca.fit_transform(Xt)
    elif method_transformation == 'ica':
        ica = FastICA()
        Xdec = ica.fit_transform(Xt)

    # Format outputs ??
    Xdec = Xdec.T

    return Xdec
