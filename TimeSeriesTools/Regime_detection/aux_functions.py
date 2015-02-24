
"""
This module is used for grouping auxiliary functions in order to perform regime
detection.
"""

import numpy as np
import pandas as pd


def sum_conservative_detectors(dist_tss, maxt, union='or', use_desc=True,
                               collapsing=lambda x: int(x.mean())):
    """
    """

    # Variables needed
    n = len(dist_tss)
    Ms = [e.as_matrix() for e in dist_tss]
    M = np.vstack(Ms)
    times = np.unique(M[:, 0])
    neur = np.unique(M[:, 1])

    # Creation of spks
    spks = []
    for t in times:
        # Detect the times around a detection.
        bool_t = [np.logical_and(m[:, 0] > t-maxt, m[:, 0] <= t+maxt)
                  for m in Ms]
        for neu in neur:
            # Compute boolean vector for t and neu
            bool_neu = [np.logical_and(bool_t[i], Ms[i][:, 1] == neu)
                        for i in range(n)]
            # Check union type
            ns = np.array([np.sum(bool_neu[i]) for i in range(n)])
            if union == 'and' and not np.all(ns > 0):
                continue
            # Collapse time
            if use_desc:
                unique_desc = [Ms[i][bool_neu[i], 3] for i in range(n)]
                unique_desc = np.unique(np.hstack(unique_desc))
                for uniq in unique_desc:
                    bool_desc = [np.logical_and(Ms[i][:, 3] == uniq,
                                                bool_neu[i])
                                 for i in range(n)]
                    ts = np.hstack([Ms[i][bool_desc[i], 0] for i in range(n)])
                    spks.append([collapsing(ts), neu, 1, uniq])
            else:
                ts = np.hstack([Ms[i][bool_neu[i], 0] for i in range(n)])
                spks.append([collapsing(ts), neu])
        ## As pandas dataframe structure
        if use_desc:
            columns = ['times', 'neuron', 'regime', 'value']
        else:
            columns = ['times', 'neuron']
        spks = pd.DataFrame(spks, columns=columns)

        return spks
