


# Spike-based detection
# Fusion spikes
# Aligning spike detected
# Statistic-based detection
## Feature extraction from spikes detected
## Method of clustering


###### Spike detection
# Choose possible method for spike detection in list
# Default values?
# 
# visualization


def visualization_interval_ts(tss, w_init=0, w_end=None):

    times = np.array(tss.index)
    activity_n = np.array(tss.as_matrix())

    if w_end == None:
        w_end = times.shape[0]

    fig = plot_signals(times[w_init:w_end], activity_n[w_init:w_end, :])
    return fig


def visualization_threshold_ts(tss, **kwargs):

    # Create reference function
    # Create threshold function

    reference_ts = 
    units_ts = 
    thr_ts = 

    times = np.array(tss.index)
    activity_n = np.array(tss.as_matrix())

    fig = plot_signals_thresholding(times[w_init:w_end],
                                    activity_n[w_init:w_end, :],
                                    reference_ts, thr_ts)

    return fig


###################################################################
### TO MOVE:
###################################################################
###################################################################
def plot_reference_building(times, signal, reference_ts):
    """Function to plot the process of thresholding a signal.

    Parameters
    ----------
    times: array_like, shape (N,)
        the times array.
    signal: array_like, shape (N,)
        the signal array.
    reference_ts: array_like, shape (N,)
        the reference array.

    Returns
    -------
    fig : matplotlib figure
        plot described.

    """

    fig = plt.figure()

    #Plot different waveforms
    m_line = plt.plot(times, signal, label='Signal')

    # Plot references
    ref_line = plt.plot(times, reference_ts, 'b--', label='Reference')

    labs = ['Signal', 'Reference']
    lines = [m_line, ref_line]

    plt.xlabel('Time')
    plt.ylabel('Signal')

    plt.legend(lines, labs, loc='upper left')

    return fig


def plot_signal_thresholding(times, signal, reference_ts, thr_ts):
    """
    Function to plot the process of thresholding a signal.

    Parameters
    ----------
    times: array_like, shape (N,)
        the times array.
    signal: array_like, shape (N,)
        the signal array.
    reference_ts: array_like, shape (N,)
        the reference array.
    thr_ts: array_like, shape (N,)
        the threshold array.

    Returns
    -------
    fig : matplotlib figure
        plot described.

    TODO
    ----
    Support for more than 1 threshold.
    """

    fig = plt.figure()

    #Plot different waveforms
    m_line = plt.plot(times, signal, label='Signal')

    # Plot references and thresholds
    ref_line = plt.plot(times, reference_ts, 'b--', label='Reference')
    thr_line = plt.plot(times, thr_ts, 'r--', label='Threshold')

    labs = ['Signal', 'Reference', 'Threshold']
    lines = [m_line, ref_line, thr_line]

    plt.xlabel('Time')
    plt.ylabel('Signal')

    plt.legend(lines, labs, loc='upper left')

    return fig


def plot_signals_matching(times, signal, patterns, matchings):
    """Plot signals for pattern matching task.

    Parameters
    ----------
    times: array_like, shape (N,)
        the times array.
    signal: array_like, shape (N,)
        the signal array.
    patterns: array_like, shape (N,Mpatterns)
        the collection of patterns to be matched in order to detect a concrete
        regime.
    matchings: array_like, list
        times in which they are matchings.

    Returns
    -------
    fig : matplotlib figure
        plot described.

    """

    fig = plt.figure()

    #Plot different waveforms
    m_line = plt.plot(times, signal, label='Signal')

    representante = np.mean(voltage, axis=0)
    uncertainty = np.std(voltage, axis=0)

    t_patterns = np.array(patterns.index)
    # Plot 
    matches = [m for m in matchings if m in times]
    for m in matches:
        t_pat = matchings+t_patterns
        bool_filter = np.logical_and(t_pat<times[0], t_pat>times[-1])

        t_pat = t_pat[bool_filter]
        uncertainty_pat = uncertainty[bool_filter]
        representante_pat = representante[bool_filter]

        plt.plot(t_pat, representante, label='Mean spike')
        plt.fill_between(t_pat, representante_pat-uncertainty_pat,
                         representante_pat+uncertainty_pat,
                         facecolor='blue', alpha=0.3)

    labs = ['Signal']
    lines = [m_line]

    plt.xlabel('Time')
    plt.ylabel('Signal')

    plt.legend(lines, labs, loc='upper left')

    return fig


###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################


def regime_detection_ui(db, ):
    """
    """

    q1 = ""
    name1 = ""
    options1 = ['relative_regime_detection', 'global_regime_detection', 'pattern_matching']

    # Subsample (n_ts_subsample)
    complete = False
    # Ask for a possible value 
    while not complete:
        # Choose method for detection

        method_dtc = selection_options(q1, name1, options1)
        # Ask for a name.

        # Default values?
        default_bool = # print values

        if not default:
            # Ask for values
            automatic_questioner()
            # Apply
            # Show results
            # Confirm
            pass
        else:
            # Apply
            # Show results
            # Confirm
            pass

    # dictionary of output
    return regime_detections


def aggregation_detections():

    # Select parameters for a correct aggregation
    # dictionary to list
    # Call a function 

    return regime_dt


def aligning_detections():

    # default
    # ask default
    # 
    # only has to select init, end

    return ex_detections


def sorting_detections():

    # select method
    # ask default? / ask parameters
    # apply extraction

    feature_extraction()

    # select method
    # ask default? / ask parameters
    # apply clustering
    # show clustering
    # ask correctness

    clustering()

    return detections

