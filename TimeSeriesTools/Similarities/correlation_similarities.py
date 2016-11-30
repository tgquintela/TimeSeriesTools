
"""
This module groups orrelation based measures of similarity between different
time series.
"""

import numpy as np


################################ CORRELATION ##################################
# See more in
# http://scikit-learn.org/stable/modules/covariance.html
# dynamics.corr('pearson')
# np.corrcoef(X[tlag:, i], X[:X.shape[0]-tlag, j])[1,0]


def lagged_PearsonCorrelation(X, timelag=0):
    """Computes the Pearson Correlation pairwise in between the lagged
    time-series of all the elements of the system.
    """

    # initialization
    n = X.shape[1]
    m = 1 if type(timelag) in [int, float] else len(timelag)
    if m == 1:
        timelag = [timelag]
    M = np.zeros((n, n, m))

    # loop over the required timelags
    for lag in range(m):
        tlag = timelag[lag]
        # loop over the possible combinations of elements
        for i in range(n):
            for j in range(n):
                aux = np.corrcoef(X[tlag:, i], X[:X.shape[0]-tlag, j])
                # assignation
                M[i, j, lag] = aux[1, 0]

    return M


#def myxcorr_1to1(x1, y1, timeLag=0):
#    # Not mine
#    # Not used
#    # TODO: Test this
#    crossCorr = []
#    x = ((np.array(x1))[np.newaxis])
#    y = ((np.array(y1))[np.newaxis])
#    n = x.shape[1]
#    for i in range(-timeLag-1, 0, 1):
#        j = -i
#        suma = np.dot(y[:, j-1:n], x[:, 0:n-j+1])
#        crossCorr.append(suma)
#    for i in range(0, timeLag, 1):
#        suma = np.dot(x[:, i:n], y[:, 0:n-i])
#        crossCorr.append(suma)
#    crossCorr_array = np.asarray(crossCorr)
#    x_sum = np.dot(x, x)
#    final_result = crossCorr_array/math.sqrt(x_sum)
#    y_sum = np.dot(y, y)
#    final_result_1 = final_result/math.sqrt(y_sum)
#    return final_result_1
