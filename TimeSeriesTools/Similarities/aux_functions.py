
"""
"""

import numpy as np
from scipy.stats import entropy


def KL_divergence(p, q):
    """Compute KL divergence of two vectors, K(p || q).

    Parameters
    ----------
    p: array_like
        a probability distribution. np.sum(p) == 1
    q: array_like
        a probability distribution. np.sum(q) == 1

    Returns
    -------
    div: float
        KL_divergence value.

    """

    div = np.sum(p[x] * np.log((p[x]) / (q[x]))
                 for x in range(len(p)) if p[x] != 0.0 or p[x] != 0)

    return div


def Jensen_Shannon_divergence(list_p, list_w=[]):
    """Compute the Jensen-Shannon divergence generically.

    Parameters
    ----------
    list_p: list, array_like
        the list of probability distributions.
    list_w: list
        the list of weights for each probability distribution.

    Returns
    -------
    div: float
        JS divergence value.

    """

    # Check and format inputs
    assert(len(list_p) > 1)
    list_w = [1/2., 1/2.] if list_w == [] else list_w
    w = np.array(list_w)[np.newaxis]
    probs = np.array(list_p) if type(list_p) == list else list_p

    # Compute measure
    div = entropy(np.sum(np.multiply(w.T, probs))) -\
        np.sum(np.multiply(w, entropy(probs.T)))
    return div


def average_prob(pk, qk=[], val=[]):
    """It is used to compute the average value of the distribution and the
    disimilarity to the mean distribution. It could be used to compute how
    spontaneous is an element to generate bursts.

    pk: array_like, shape (N,) or shape (N, M)
        distributions to operate with.
    qk: array_like, shape (N,)
        representative distribution or mean distribution.
    val: list or array_like
        values of each bin of the distribution.

    """

    # Initial variables
    m = len(pk.shape)
    n = pk.shape[0] if m == 1 else pk.shape[1]
    val = range(n) if val == [] else val
    qk = 1./n*np.ones(n) if qk == [] else qk

    # Position value
    if m == 1:
        val_avg = np.mean(np.multiply(pk-qk, val))
    else:
        val_avg = np.mean(np.multiply(pk-qk, val), axis=1)

    # Relative entropy over the average
    if m == 1:
        rel_dis = entropy(pk, qk)
    else:
        rel_dis = np.array([entropy(pk[i], qk) for i in range(pk.shape[0])])

    return val_avg, rel_dis
