
"""
Module which groups magnitude based similarities. The distances are computed by
using the value of the time series.

The main method implemented is a wrap to the R package.
"""

import numpy as np

try:
    from rpy2.robjects.packages import importr
    from rpy2.robjects.numpy2ri import numpy2ri
    from rpy2.robjects.numpy2ri import ri2numpy
except:
    pass


def general_dtw(ts1, ts2, method, kwargs={}):
    """The implementation of the dynamic time warping measure between two ts.

    Parameters
    ----------
    ts1: array_like
        pattern to be queried in the next array.
    ts2: array_like
        array to be searched in order to find the pattern given in ts1.
    method: str, optional
        method to apply selected.
    kwargs: dict
        the variables needed for the method selected.

    Returns
    -------
    distance: float
        distance computed with dtw.
    traceback_: array_like, shape (2, Nsteps)
        the trajectory of the minimal warping.
    alignment: tuple
        information of the process in R mode and python mode.

    """

    if method == 'rpy2':
        distance, traceback_, alignment = dtw_specific_rpy2(ts1, ts2, **kwargs)

    return distance, traceback_, alignment


def dtw_specific_rpy2(ts1, ts2, dist_method="Euclidean", step_pattern=None,
                      open_beging=False, open_end=False, window_type="none",
                      window_size=30, distance_only=False):
    """This function acts as a wrapper for the package dtw and its functions
    related with the computation of the dynamic time warping with its different
    options.

    ts1: array_like
        pattern to be queried in the next array.
    ts2: array_like
        array to be searched in order to find the pattern given in ts1.
    dist_method: str, function (?)
        the distance definition used.
    step_pattern: DTW object
        a stepPattern object describing the local warping steps allowed with
        their cost.
    open_beging: boolean
        perform open-ended alignment in the starting point.
    open_end: boolean
        perform open-ended alignment in the ending point.
    window_type: str, optional or function
        the window type information. It is precoded using str in {"none",
        "itakura", "sakoechiba", "slantedband"}.
    window_size: int
        the distance size of the bands in the algorithms with window_type which
        needed.
    distance_only: boolean
        compute only the distance in order to save computational time.

    Returns
    -------
    distance: float
        distance computed with dtw.
    traceback_: array_like, shape (2, Nsteps)
        the trajectory of the minimal warping.
    alignment: tuple
        information of the process in R mode and python mode.

    Reference
    ---------
    .. [1] Toni Giorgino (2009). Computing and Visualizing Dynamic Time Warping
    Alignments in R: The dtw Package. Journal of Statistical Software, 31(7),
    1-24. URL www.jstatsoft.org/v31/i07/.
    .. [2] Paolo Tormene, Toni Giorgino, Silvana Quaglini, Mario Stefanelli
    (2008). Matching Incomplete Time Series with Dynamic Time Warping: An
    Algorithm and an Application to Post-Stroke Rehabilitation. Artificial
    Intelligence in Medicine, 45(1), 11-34. doi:10.1016/j.artmed.2008.11.007


    TODO
    ----
    Problems slantedband and sakoechiba with window_size
    """

    DTW = importr('dtw')
    ## Set step pattern
    if step_pattern is None:
        step_pattern = DTW.symmetric2

    ## Transform to ri
    query = numpy2ri(ts1)
    template = numpy2ri(ts2)

    ## Call the algorithm
    alignment = DTW.dtw(query, template, keep=True,
                        step_pattern=step_pattern, dist_method=dist_method,
                        open_beging=open_beging, open_end=open_end,
                        window_size=window_size, window_type=window_type,
                        distance_only=distance_only)

    ## Output variables
    dtw_info = from_dtw2dict(alignment)
    distance = dtw_info['distance']

    ## Preparing traceback
    if not distance_only:
        traceback_ = np.vstack([dtw_info['index1'], dtw_info['index2']])
    else:
        traceback_ = None

    return distance, traceback_, (alignment, dtw_info)


def from_dtw2dict(alignment):
    """Auxiliar function which transform useful information of the dtw function
    applied in R using rpy2 to python formats.
    """

    dtw_keys = list(alignment.names)
    bool_traceback = 'index1' in dtw_keys and 'index2' in dtw_keys
    bool_traceback = bool_traceback and 'stepsTaken' in dtw_keys

    ## Creating a dict to save all the information in python format
    dtw_dict = {}
    # Transformation into a dict
    dtw_dict['stepPattern'] = ri2numpy(alignment.rx('stepPattern'))
    dtw_dict['N'] = alignment.rx('N')[0]
    dtw_dict['M'] = alignment.rx('M')[0]
    dtw_dict['call'] = alignment.rx('call')
    dtw_dict['openEnd'] = alignment.rx('openEnd')[0]
    dtw_dict['openBegin'] = alignment.rx('openBegin')[0]
    dtw_dict['windowFunction'] = alignment.rx('windowFunction')
    dtw_dict['jmin'] = alignment.rx('jmin')[0]
    dtw_dict['distance'] = alignment.rx('distance')[0]
    dtw_dict['normalizedDistance'] = alignment.rx('normalizedDistance')[0]
    if bool_traceback:
        aux = np.array(ri2numpy(alignment.rx('index1')).astype(int))
        dtw_dict['index1'] = aux
        aux = np.array(ri2numpy(alignment.rx('index2')).astype(int))
        dtw_dict['index2'] = aux
        dtw_dict['stepsTaken'] = ri2numpy(alignment.rx('stepsTaken'))
    elif 'localCostMatrix' in dtw_keys:
        aux = np.array(ri2numpy(alignment.rx('localCostMatrix')))
        dtw_dict['localCostMatrix'] = aux
    elif 'reference' in dtw_keys and 'query' in dtw_keys:
        dtw_dict['reference'] = alignment.rx('reference')
        dtw_dict['query'] = alignment.rx('query')

    return dtw_dict


###############################################################################
############################### Move to plotting ##############################
###############################################################################
def plot_dtw(alignment, typeplot, filename=None):
    """Wrapper to the plot functions of the R package dtw.
    """
    grdevices = importr('grDevices')
    DTW = importr('dtw')
    # Create filename
    if filename is None:
        filename = typeplot+".png"
    grdevices.png(file=filename, width=512, height=512)
    # plotting code here
    if typeplot == 'alignment':
        DTW.dtwPlotAlignment(alignment)
    elif typeplot == 'density':
        DTW.dtwPlotDensity(alignment)
    elif typeplot == 'twoway':
        DTW.dtwPlotTwoWay(alignment)
    elif typeplot == 'threeway':
        DTW.dtwPlotThreeWay(alignment)
    elif typeplot == 'window_plot':
        DTW.dtwWindow_plot(alignment)
    grdevices.dev_off()
