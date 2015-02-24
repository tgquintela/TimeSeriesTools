
import numpy as np


def sliding_embeded_transf(X, tau, D,  step=1, f=lambda x: x):
    """Build a set of embedding sequences from given time series X with lag Tau
    and embedding dimension D. Let X = [x(1), x(2), ... , x(N)], then for each
    i such that 1 < i <  N - (D - 1) * Tau, we build an embedding sequence,
    Y(i) = [x(i), x(i + Tau), ... , x(i + (D - 1) * Tau)]. All embedding
    sequence are placed in a matrix Y.

    Parameters
    ----------
    X : array_like, shape(N,)
        a time series
    tau : int
        the lag or delay when building embedding sequence
    D : integer
        the embedding dimension
    step: int
        the step for which we compute the sequence.
    f: function
        transformation function to be applied to each element of the sequence.

    Returns
    -------
    Y : 2-D list
        embedding matrix built

    """

    N = X.shape[0]

    # Check inputs
    if D * tau > N:
        message = "Cannot build such a matrix, because D * tau > N"
        raise Exception(message)
    if tau < 1:
        message = "Tau has to be at least 1"
        raise Exception(message)

    Y = np.zeros((N - (D - 1) * tau, D))
    for i in xrange(0, N - (D - 1) * tau, step):
        for j in xrange(0, D):
            Y[i][j] = f(X[i + j * tau])
    return Y
