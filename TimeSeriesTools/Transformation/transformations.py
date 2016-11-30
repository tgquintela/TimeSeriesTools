
"""This module contains typical transformation for time-series.
"""

from temporal_discretization import temporal_discretization
from value_discretization import value_discretization
from filtering import general_filtering
from windowing_transformation import windowing_transformation


########################## Wrapper to all functions ###########################
###############################################################################
def general_transformation(X, method, args):
    """General wrapper to transformation of a time series.

    Parameters
    ----------
    X: array_like, shape (N, M)
        signals of the system.
    method: str, optional
        method used to do the transformation.
    kwargs: dict
        variables needed for the choosen method.

    Returns
    -------
    Xt: array_like
        transformed signals.

    """

    if type(method) == list:
        methods = method[:]
        assert(type(args) == list)
    elif type(method) == str:
        methods = [method]
        assert(type(args) == dict)
        args = [args]

    # Preparing the transformation
    Xt = X[:]

    for i in range(len(methods)):
        method = methods[i]
        if method == 'windowing_transformation':
            Xt = windowing_transformation(Xt, **args[i])
        elif method == 'filtering':
            Xt = general_filtering(Xt, **args[i])
        elif method == 'temporal_discretization':
            Xt = temporal_discretization(Xt, **args[i])
        elif method == 'value_discretization':
            value_discretization(Xt, **args[i])
        elif method not in ['windowing_transformation', 'filtering',
                            'temporal_discretization', 'value_discretization']:
            pass

    return Xt
