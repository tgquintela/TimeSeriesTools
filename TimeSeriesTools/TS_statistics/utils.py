
import numpy as np


def build_ngram_arrays(X, post, pres, L):
    """This matrix build another one with all the given arrays and compute
    the lags of the pres variables.

    Parameters
    ----------
    X: array_like
        discretize dynamics.
    post: list
        the indices of the variables that are posterior.
    pres: list
        the indices of the variables that are previous and we want to compute
        the lag arrays.
    L: int
        the max lag to compute. It is computed from lag 1 to lag L.

    Returns
    -------
    Y: array_like
        the given arrays preparated to compute the joint probability.
    """

    npost = len(post)
    npres = len(pres)
    nvars = npost+npres*(L+1)
    nt = X.shape[0]

    Y = np.zeros((nt-L, nvars))
    for i in range(npost):
        Y[:, i] = X[:nt-L, post[i]]
    for i in range(npres):
        for l in range(1, L+1):
            Y[:, npost+i*npres+l-1] = X[l:nt-L+l, pres[i]]

    return Y


def create_ngram(X, lags, samevals=True):
    """Auxiliary function which serves to compute a lag matrices of the given
    time-series.
    """

    ## Formatting inputs
    lags = uniform_input_lags(lags, X)
    samevals = uniform_input_samevals(samevals, X)
    nt = X.shape[0]
    m = X.shape[1] if len(X.shape) == 2 else 1
    L = np.max([np.max(e) for e in lags])
    X = X if len(X.shape) == 2 else X.reshape((X.shape[0], 1))

    ## Computing matrix of lags
    Y = []
    for i in range(m):
        xv = np.vstack([X[l:nt-L+l, i] for l in lags[i]])
        Y.append(xv)

    Y = np.vstack(Y).T

    return Y


def uniform_input_samevals(samevals, X):
    """Auxiliary function to format samevals.
    """
    n = X.shape[1] if len(X.shape) == 2 else 1
    # needed variables (TODO: specific function)
    if type(samevals) == bool:
        if samevals:
            samevals = [np.unique(X) for i in range(n)]
        else:
            samevals = [np.unique(X[:, i]) for i in range(n)]
    else:
        if type(samevals) == np.ndarray:
            samevals = [samevals for i in range(n)]
        elif type(samevals) == list:
            pass

    return samevals


def uniform_input_lags(lags, X):
    """Auxiliary function of format inputs of lags.
    """

    n = X.shape[1] if len(X.shape) == 2 else 1

    if type(lags) == list:
        flag = np.all([type(e) == list for e in lags])
        if not flag:
            lags = [lags]
        else:
            pass
    elif type(lags) == np.ndarray:
        lags = [lags for i in range(n)]
    return lags
