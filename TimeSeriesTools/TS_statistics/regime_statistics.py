
"""
This module contains the functions which computes interesting statistics for
the study of the causality between variables. They can be applied to discrete
time-series for obtaining important statistics.
"""

import numpy as np


def parse_spks(spks):
    if type(spks) == tuple:
        times_event, elements_event, regimes_event = spks
    elif type(spks) == np.ndarray:
        times_event = spks[:, 0]
        elements_event = spks[:, 1]
        regimes_event = spks[:, 2]
    return times_event, elements_event, regimes_event


def prob_regimes_x(spks, max_t=None, normalized=False):
    """Estimation of the probability of each element to be in an specific
    state or regime. It is done from the sparse representation of the dynamics.
    There are two main reasons for explaining the probability to be in some
    specific regime:
    - Intrinsec probability of being in the given regime.
    - The influence of its connections to be in this regime.

    Parameters
    ----------
    spks: array_like, shape (N, variables)
        description of the regimes detected.
    max_t: int
        number of observations obtained along the experiment.
    normalized: bool
        return probabilities of spike in time or the number of times in a
        selected regimes.

    Returns
    -------
    probs: array_like, shape(Nneur, Nreg)
        the number of times in a given regime or the probability of spike in
        time.
    elements: array_like
        the code for each element.
    regimes: array_like, shape (Nreg)
        the regime codes.
    """

    # Initialization
    times_event, elements_event, regimes_event = parse_spks(spks)
    elements = np.unique(elements_event)
    regimes = np.unique(regimes_event)
    if max_t is None:
        max_t = np.max(times_event)

    probs = np.zeros((elements.shape[0], regimes.shape[0]))

    for i in range(elements.shape[0]):
        for j in range(regimes.shape[0]):
            indices = np.logical_and(elements_event == elements[i],
                                     regimes_event == regimes[j])
            num = indices.nonzero()[0].shape[0]
            if normalized:
                probs[i, j] = num/float(max_t)
            else:
                probs[i, j] = num

    return probs, elements, regimes


def temporal_counts(spks, normalized=False, collapse_reg=True):
    """This function count the quantity of element actives in each time.

    Parameters
    ----------
    spks: tuple of array_like, shape (N,variables)
        description of the spikes detected.
    normalized: bool
        return probabilities of spike in each active time for for any element.
    collapse_reg: boolean
        collapse regimes or treat each one independently.

    Returns
    -------
    probs: array_like, shape (Ntimes,) or shape (Ntimes, Nregimes)
        the number of spikes or the probability of spike in each time.
    utimes: array_like, shape (Ntimes,)
        times in which the system has at least one element active.
    regimes: None or array_like
        the regime codes.

    """

    # Temporal counts for any regime
    times_event, elements_event, regimes_event = parse_spks(spks)
    if collapse_reg:
        # "Active" times
        utimes = np.unique(times_event)
        # Regimes
        regimes = None
        # Counts
        counts = np.zeros(utimes.shape)
        for i in range(utimes.shape[0]):
            counts[i] = np.sum(times_event == utimes[i])

    # Temporal count for each regime
    else:
        # "Active" times
        utimes = np.unique(times_event)
        # Regimes
        regimes = np.unique(regimes_event)
        # Counts
        counts = np.zeros((utimes.shape[0], regimes.shape[0]))
        for i in range(utimes.shape[0]):
            for j in range(regimes.shape[0]):
                indices = np.logical_and(times_event == utimes[i],
                                         regimes_event == regimes[j])
                counts[i, j] = np.sum(indices)

    # Normalization
    if normalized:
        # Number of elemens
        n = np.unique(elements_event).shape[0]
        # Times available
        max_t = np.max(times_event)
        # Probability in times and elements
        probs = counts/float(max_t)/float(n)
    else:
        probs = counts

    return probs, utimes, regimes


def temporal_average_counts(spks, window=0, collapse_reg=True):
    """Temporal average calculates the density in time that the spikes occur in
    the time there is an spike using moving average with a given window.

    Parameters
    ----------
    spks: array_like, shape (N,variables)
        description of the spikes detected.
    window: int
        window size. Distance from the center.
    collapse_reg: boolean
        collapse regimes or treat each one independently.

    Returns
    -------
    counts: array_like, shape (Ntimes,) or shape (Ntimes, Nregimes)
        the number of spikes in each time.
    utimes: array_like, shape (Ntimes,)
        times in which the system has at least one element active.
    regimes: None or array_like
        the regime codes.

    """
    times_event, _, regimes_event = parse_spks(spks)
    utimes = np.unique(times_event)
    regimes = np.unique(regimes_event) if not collapse_reg else None
    if collapse_reg:
        counts = np.zeros(utimes.shape)
        for i in range(utimes.shape[0]):
            # Acumulative logical
            logical = times_event == utimes[i]
            for t in range(window):
                aux = np.logical_or(times_event == utimes[i]+t,
                                    times_event == utimes[i]-t)
                logical = np.logical_or(aux, logical)
            counts[i] = np.sum(logical)
    else:
        counts = np.zeros((utimes.shape[0], regimes.shape[0]))
        for i in range(utimes.shape[0]):
            for j in range(regimes.shape[0]):
                # Acumulative logical
                logical = np.logical_and(times_event == utimes[i],
                                         regimes_event == regimes[j])
                for t in range(window):
                    aux = np.logical_or(times_event == utimes[i]+t,
                                        times_event == utimes[i]-t)
                    aux = np.logical_and(aux, regimes_event == regimes[j])
                    logical = np.logical_or(aux, logical)
                counts[i, j] = np.sum(logical)

    return counts, utimes, regimes


from collections import Counter


def count_repeated(array):
    """Auxiliary functions for complementing numpy which performs a counting of
    how many elements are repeated in the array.
    """
    c_repeated = np.sum(np.array(Counter(array).values()) > 1)
    return c_repeated


def temporal_densities(spks, w_limit=10, collapse_reg=True):
    """Computing the temporal average scrolling through possible window sizes.

    Parameters
    ----------
    spks: array_like, shape (N,variables)
        description of the spikes detected.
    w_limit: int
        window size. Distance from the center.
    collapse_reg: boolean
        collapse regimes or treat each one independently.

    Returns
    -------
    counts: array_like, shape (Ntimes,) or shape (Ntimes, Nregimes)
        the number of spikes in each time.
    utimes: array_like, shape (Ntimes,)
        times in which the system has at least one element active.
    repeated: array_like, shape (w_limit,) or shape (w_limit, Nregimes)
        the number of repeated elements active in the window for the same
        regime if collapse_reg=True.
    regimes: None or array_like, shape (N,)
        the regime codes.

    """
    times_event, elements_event, regimes_event = parse_spks(spks)
    utimes = np.unique(times_event)

    if collapse_reg:
        repeated = np.zeros(w_limit+1)
        regimes = None
        counts = np.zeros((utimes.shape[0], w_limit+1))
        for window in range(w_limit):
            for i in range(utimes.shape[0]):
                logical = times_event == utimes[i]
                for t in range(window):
                    aux = np.logical_or(times_event == utimes[i]+t,
                                        times_event == utimes[i]-t)
                    logical = np.logical_or(aux, logical)
                # Counts and repeated
                counts[i, window] = np.sum(logical)
                repeated[window] += count_repeated(elements_event[logical])
    else:
        utimes = np.unique(times_event)
        regimes = np.unique(regimes_event)
        repeated = np.zeros((w_limit+1, regimes.shape[0]))
        counts = np.zeros((utimes.shape[0], w_limit+1, regimes.shape[0]))
        for window in range(w_limit):
            for i in range(utimes.shape[0]):
                for j in range(regimes.shape[0]):
                    logical = np.logical_and(times_event == utimes[i],
                                             regimes_event == regimes[j])
                    for t in range(window):
                        aux = np.logical_or(times_event == utimes[i]+t,
                                            times_event == utimes[i]-t)
                        aux = np.logical_and(aux, regimes_event == regimes[j])
                        logical = np.logical_or(aux, logical)
                    # Counts and repeated
                    counts[i, window, j] = np.sum(logical)
                    repeated[window, j] +=\
                        count_repeated(elements_event[logical])

    return counts, utimes, repeated, regimes


###############################################################################
###############################################################################
def isi_distribution(spks, n_bins, globally=False, normalized=True,
                     logscale=False):
    """Computation of the ISI distribution.

    Parameters
    ----------
    spks: array_like, shape(N,variables)
        the spikes descriptions
    n_bins: int
        number of bins in which we want to describe the isi distribution.
    globally: bool
        if we want to compute the isi distribution globally or for each
        element individually.
    normalized: bool
        normalize the results if True
    logscale: bool
        return the histogram in x logscale.

    Returns
    -------
    isi: array_like, shape (M, Nelements)
        frequencies or probabilities of the ISI in the given intervals
    bin_edges: array_like, shape (M+1,)
        edges of the intervals

    """

    # Computation of isis
    isis, elements = isis_computation(spks, logscale)
    # Concatenate in an array
    isis_concat = np.hstack(isis)

#    # Global computing of ISI distribution
    hist, bin_edges = np.histogram(isis_concat, bins=n_bins)

    # Compute the ISI distribution
    if globally:
        if normalized:
            isi = hist/float(np.sum(hist))
        else:
            isi = hist
    else:
        isi = np.zeros((elements.shape[0], n_bins))
        for i in range(elements.shape[0]):
            aux = np.histogram(isis[i], bins=bin_edges)[0]
            if normalized:
                isi[i, :] = aux/float(np.sum(aux))
            else:
                isi[i, :] = aux

    return isi, bin_edges


def isis_computation(spks, logscale=False):
    """Computation of isis.

    Parameters
    ----------
    spks: array_like, shape (N,variables)
        description of the spikes detected.
    logscale: bool
        return the inter spikes intervals in logscale.

    Returns
    -------
    isis: list of arrays, shape (M,)
        isis for each element.
    """

    # Initialization
    times_event, elements_event, _ = parse_spks(spks)
    isis = []
    elements = np.unique(elements_event)
    # Loop for computing the ISIs
    for i in range(elements.shape[0]):
        aux = np.where(elements_event == elements[i])[0]
        aux_t = times_event[aux]
        if logscale:
            isis.append(np.log(np.diff(aux_t)))
        else:
            isis.append(np.diff(aux_t))
        assert(np.all(np.isfinite(isis[i])))
    return isis, elements


###############################################################################
###############################################################################
def temporal_si(spks):
    """Computation of the intervals between times in which any spike happens.

    Parameters
    ----------
    spks: array_like, shape (N,variables)
        description of the spikes detected.

    Returns
    -------
    spk_intervals: array_like, shape(Nt-1,)
        the intevals between the spikes times.

    """
    times_event, _, _ = parse_spks(spks)
    # "Active" times
    utimes = np.unique(times_event)

    # Calculate spike intervals
    spk_intervals = np.diff(utimes)

    return spk_intervals


def count_into_bursts(spks, bursts, elements=None):
    """Counting into the detected bursts the probability of appearance of a
    spike of each given element.

    Parameters
    ----------
    spks: array_like, shape (N,variables)
        description of the spikes detected.
    bursts: list of arrays
        the active times in each burst.
    elements: None, or list
        The elements considered to count into the burst.
        - if None, it consider the global statistics.
        - if void list it will be considered all the elemnts.

    Returns
    -------
    counts: array_like, shape(n)
        number of spikes in the time position into the bursts.

    """

    ## 0. Set needed variables
    times_event, elements_event, regimes_event = parse_spks(spks)
    # Number of bursts
    n_bursts = len(bursts)
    # Set bursts correctly
    bursts = [np.array(bursts[i]) for i in range(len(bursts))]
    # Compute maximum length of a burst considered
    difs = [bursts[i][-1]-bursts[i][0]+1 for i in range(n_bursts)]
    n = np.max(difs)
    # Number of elements in the system
    if elements is None:
        pass
    elif elements == []:
        elements = np.unique(elements_event)
        m = len(elements)
    else:
        aux = np.zeros(len(times_event)).astype(bool)
        for e in elements:
            aux = np.logical_or(aux, elements_event == e)
        times_event = times_event[aux]
        elements_event = elements_event[aux]
        regimes_event = regimes_event[aux]
        elements = np.array(elements)
        m = len(elements)

    ## 1. Compute counts
    if elements is None:
        counts = np.zeros(n)
        for i in range(n_bursts):
            for j in range(len(bursts[i])):
                l = int(bursts[i][-1]-bursts[i][0])
                aux = times_event == bursts[i][j]
                counts[l] += np.sum(aux)
    else:
        counts = np.zeros((n, m))
        for k in range(m):
            for i in range(n_bursts):
                for j in range(len(bursts)):
                    l = int(bursts[i][-1]-bursts[i][0])
                    aux = np.logical_and(times_event == bursts[i][j],
                                         elements_event == k)
                    counts[l, k] += np.sum(aux)

    return counts


def general_count(c_xy, max_l=0):
    """Global count about the relative temporal position of spiking for each
    element.

    Parameters
    ----------
    c_xy: array_like, shape(Nneur, Nneur, maxl+1)
        the number of the coincident spikes given a lag time.
    max_l: int
        maximum size of a bursts.

    Returns
    -------
    counts: array_like, shape(max_l)
        number of spikes in a temporal position of a bursts.

    """
    max_l = c_xy.shape[2]-1 if max_l == 0 else max_l
    counts = np.zeros(max_l+1)
    for i in range(max_l+1):
        if i == 0:
            m = np.triu(c_xy[:, :, 0])
            d = np.eye(c_xy.shape[0])*np.diag(c_xy[:, :, 0])
            counts[i] = np.sum(m-d)
        counts[i] = np.sum(c_xy[:, :, i])
    return counts


def prob_spk_xy(spks, max_l=8, normalized=False):
    """Compute of the number of coincidences in with lag of the spikes between
    the elements of the system.

    Parameters
    ----------
    spks: array-like, shape(N,variables)
        the spikes descriptions.
    max_l: int
        maximum number to inspect lag.

    Returns
    -------
    probs: array_like, shape(Nneur, Nneur, maxl+1)
        the number of the coincident spikes given a lag time.
    elements: array_like
        the code for each element.
    regimes: array_like
        the regimes in the spks.

    TODO
    ----
    Extract normalization
    Update normalization

    """
    # Initialization
    times_event, elements_event, regimes_event = parse_spks(spks)

    elements = np.unique(elements_event)
    regimes = np.unique(regimes_event)
    n = len(elements)
    m = len(regimes)
    probs = np.zeros((n, n, m, max_l+1))

    # Loop for each timelag possible
    for timelag in range(max_l+1):
        # Loop for each element pair
        for i in range(n):
            for j in range(i, n):
                for k in range(m):
                    times1 = times_event[np.logical_and(elements_event == i,
                                                        regimes_event == k)]
                    times2 = times_event[np.logical_and(elements_event == j,
                                                        regimes_event == k)]
                    if i == j:
                        inttimes = np.intersect1d(times1, times2+timelag)
                        probs[i, j, k, timelag] = inttimes.shape[0]
                    else:
                        inttimes = np.intersect1d(times1+timelag, times2)
                        probs[i, j, k, timelag] = inttimes.shape[0]
                        inttimes = np.intersect1d(times1, times2+timelag)
                        probs[j, i, k, timelag] = inttimes.shape[0]
    # Normalization (extern function, and compute intially normalization?)
    ### TODO: can be normalized in many ways
    if normalized:
        for r in range(len(regimes)):
            diag = np.diag(probs[:, :, r, 0])
            for z in range(max_l+1):
                d = np.eye(n)*np.diag(probs[:, :, r, z])
                aux_u = np.divide(np.triu(probs[:, :, r, z]).T, diag).T
                aux_l = np.divide(np.tril(probs[:, :, r, z])-d, diag)
                probs[:, :, r, z] = aux_u + aux_l

    return probs, elements, regimes


def counts_normalization(counts, max_t):
    """This functions acts over the counts matrices and vectors in order to
    normalize their elements into a probabilities.

    Parameters
    ----------
    counts: array_like, shape(n,n,nlags)
        the counts of the coincident spikes considering lag times.

    Returns
    -------
    probs: array_like, shape(Nneur, Nneur, maxl+1)
        the number of the coincident spikes given a lag time.

    TODO
    ----
    Update
    """
    s = counts.shape
    if len(s) == 1:
        probs = counts/float(max_t)
    elif len(s) == 3:
        diag = np.diag(counts[:, :, 0])
        n = counts.shape[0]
        max_l = counts.shape[2]
        probs = np.zeros(counts.shape)
        for z in range(max_l):
            d = np.eye(n)*np.diag(counts[:, :, z])
            aux_u = np.divide(np.triu(counts[:, :, z]).T, diag).T
            aux_l = np.divide(np.tril(counts[:, :, z])-d, diag)
            probs[:, :, z] = aux_u + aux_l

    return probs
