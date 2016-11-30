
"""This module contains the group of functions which carry out the pocess of
separate a sequence of spikes in bursts.
They share as input the sequence of spikes detected by some algorithm of spike
detection and returns a list of arrays which contains the times in which spikes
exists.
"""

import numpy as np
import math


def general_burst_detection(spks, method='', kwargs={}):
    """Burst detection is the problem of group the spikes into bursts
    considering temporal information and temporal density. Usually is done with
    spikes produced for the same element, but in this package we consider the
    general situation of many elements.

    Parameters
    ----------
    spks: pandas.DataFrame
        spikes format with columns: 'times', 'neuron', 'regime', 'descriptor'
    method: str, optional
        which method use. Depending of the selected method we have to input
        the required variables.
    kwargs: dict
        parameters needed for the called functions selected by the method
        parameter.

    Returns
    -------
    bursts: list of lists
        active times grouped by bursts.

    """
    possible_methods = ['dummy']
    method = method if method in possible_methods else ''
    if method == 'dummy':
        bursts = dummy_burst_detection(spks, **kwargs)

    return bursts


def dummy_burst_detection(spks, t_max):
    """This algorithm works under the supposition that all the bursts are
    separated at least by a gap with lenght which is a upper bound of the
    bursts length.

    Parameters
    ----------
    spks: array-like, shape (Ns, variables)
        the spikes descriptions.
    t_max: int
        lower-bound of gaps beween bursts or upper bound of bursts length.
        It is expressed in index units.

    Returns
    -------
    bursts: list of arrays
        the active times in each burst.

    """

    utimes = np.unique(spks[:, 0])
    gaps = np.diff(utimes)

    edges = np.where(gaps >= t_max)[0] + 1
    edges = np.hstack([[0], edges, [utimes.shape[0]]])

    bursts = []
    for i in range(edges.shape[0]-1):
        bursts.append(utimes[edges[i]:edges[i+1]])

    return bursts


############################## Under supervision ##############################
###############################################################################
def kleinberg_burst_detection(spks, s=2, gamma=1):
    import pybursts
    utimes = list(spks[:, 0])
    bursts = pybursts.kleinberg(utimes, s, gamma)
    return bursts


############################ Under implementation #############################
###############################################################################
def kleinberg(spks, s=2, gamma=1):
    # Control of inputs
    if s <= 1:
        raise ValueError("s must be greater than 1!")
    if gamma <= 0:
        raise ValueError("gamma must be positive!")

    utimes = np.unique(spks[:, 0])
    utimes = np.sort(utimes)

    # Return bursts for only 1 time
    if utimes.size == 1:
        bursts = [utimes[0]]
        return bursts

    # Computation of gaps
    gaps = np.diff(utimes)

    # Computation of needed magnitudes
    T = np.sum(gaps)
    n = np.size(gaps)
    g_hat = T / n

    k = int(math.ceil(float(1+math.log(T, s) + math.log(1/np.amin(gaps), s))))
    gamma_log_n = gamma * math.log(n)

    alpha_function = np.vectorize(lambda x: s ** x / g_hat)
    alpha = alpha_function(np.arange(k))

    # Definition complementary functions
    def tau(i, j):
        if i >= j:
            return 0
        else:
            return (j - i) * gamma_log_n

    def f(j, x):
        return alpha[j] * math.exp(-alpha[j] * x)

    # Intialization of C (?)
    C = np.repeat(float("inf"), k)
    C[0] = 0

    q = np.empty((k, 0))
    for t in range(n):
        C_prime = np.repeat(float("inf"), k)
        q_prime = np.empty((k, t+1))
        q_prime.fill(np.nan)

        for j in range(k):
            cost_function = np.vectorize(lambda x: C[x] + tau(x, j))
            cost = cost_function(np.arange(0, k))
            el = np.argmin(cost)
            if f(j, gaps[t]) > 0:
                C_prime[j] = cost[el] - math.log(f(j, gaps[t]))
            if t > 0:
                q_prime[j, :t] = q[el, :]
            q_prime[j, t] = j + 1
        C = C_prime
        q = q_prime
    j = np.argmin(C)
    q = q[j, :]
    prev_q = 0

    N = 0
    for t in range(n):
        if q[t] > prev_q:
            N = N + q[t] - prev_q
        prev_q = q[t]

    bursts = np.array([np.repeat(np.nan, N), np.repeat(utimes[0], N),
                      np.repeat(utimes[0], N)], ndmin=2,
                      dtype=object).transpose()

    burst_counter = -1
    prev_q = 0
    stack = np.repeat(np.nan, N)
    stack_counter = -1

    for t in range(n):
        if q[t] > prev_q:
            num_levels_opened = q[t] - prev_q
            for i in range(int(num_levels_opened)):
                burst_counter += 1
                bursts[burst_counter, 0] = prev_q + i
                bursts[burst_counter, 1] = utimes[t]
                stack_counter += 1
                stack[stack_counter] = burst_counter
        elif q[t] < prev_q:
            num_levels_closed = prev_q - q[t]
            for i in range(int(num_levels_closed)):
                bursts[stack[stack_counter], 2] = utimes[t]
                stack_counter -= 1
        prev_q = q[t]

    while stack_counter >= 0:
        bursts[stack[stack_counter], 2] = utimes[n]
        stack_counter -= 1

    return bursts
