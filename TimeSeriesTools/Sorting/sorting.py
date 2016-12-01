
import numpy as np


# Wrapper TODO:
# align spikes + transformation + post-filtering
# possibility to ask for threshold
#from pyCausality.Plotting.sorting_plots import *
#from regime_detection import *
#from feature_extraction import *


############################# Auxiliary functions #############################
###############################################################################
def set_threshold_detection(activation, n_ts=10):
    """Function to ask for the threshold in order to perform regime detection.

    Parameters
    ----------
    activation : pd.DataFrame
        the dynamics of the system or subsystem.
    n_ts : int
        number of time series to be represented in the visualization.

    Returns
    -------
    threshold : float
        the selection threshold for the dynamics.

    TODO
    ----
    Plot more than one time serie
    Select transformation and other things of regime_detection
    """

    # Subsampling
    if n_ts < activation.shape[1]:
        idx = np.random.permutation(activation.shape[1])[:n_ts]
    else:
        idx = range(activation.shape[1])

    # Extract usable variables from input
    times = np.array(activation.index)
    activity_n = activation.as_matrix()[:, idx]

    ## 1. Window visualization selection
    # Questions
    question00 = "Set the initial frame of the window. (int)\n"
    question01 = "Set the final frame of the window. (int)\n"
    question02 = "Is it correct your option? (Y/N)\n"

    # Loop in order to set window visualization
    complete = False
    while not complete:
        # Plot time serie
        fig = plot_signals(times, activity_n)
        print type(fig)
        fig.show()
        # Ask for window to show
        w_init = raw_input(question00)
        w_end = raw_input(question01)
        w_init = int(w_init)
        w_end = int(w_end)
        # Show result
        fig = plot_signals(times[w_init:w_end], activity_n[w_init:w_end])
        fig.show()
        # Ask for correctness
        correctness = raw_input(question02)
        complete = True if correctness == 'Y' else False

    ## 2. Transformation selection
    # Questions
    unit = ['gap', 'std', 'qqstd']
    collapse = ['initial', ]
    ref = ['min']
    question_def = "Do you want to keep default values? (Y/N)\n"
    question1 = "Is it correct your selection? (Y/N)\n"
    q_units = "Units for measuring threshold?\n%s\nunits=" % str(unit)
    q_collap = "Collapse to option?\n%s\ncollapse_to=" % str(collapse)
    q_ref = " \n%s\n" % str(ref)
    q_individual = "Detection each ts individually? (T/F)"
    q_transformation = """Select the desired transformation method from:
    (1) 'ups_downs'

    [introduce the number]
    """
    # Loop in order to set window visualization
    complete = False
    while not complete:
        kwargs = {}
        default = raw_input(question_def)
        if default == 'Y':
            pass
        else:
            pass
        # Apply transformation

        # Completeness
        completeness = raw_input(question1)
        complete = True if completeness == 'Y' else False

    ## 3. Threshold selection
    # Questions
    question0 = "What is your desired threshold? (Relative value)\n"
    question1 = "Is it correct your selection? (Y/N)\n"

    # Loop of questions
    complete = False
    while not complete:
        # Plot time serie
        fig = plot_signals(times, activity_n)
        fig.show()
        # Ask for threshold
        threshold = raw_input(question0)
        threshold = float(threshold)
        # Plot results (TODO) 2 plots 1 with threshold and other with results

        # Check if correct
        completeness = raw_input(question1)
        complete = True if completeness == 'Y' else False

    return threshold


def set_align_waveforms(activation, dis_ts):
    """Selection of the window information.

    Parameters
    ----------
    activation : pd.DataFrame
        activity information of the system.
    dis_ts : pd.DataFrame
        discretize information of the time-series after regime detection.

    Returns
    -------
    window_info : list or tuple
        it is a two element array like which describe how many frames are the
        borders of the window from the center.
        * 0: negative integer distance from the center in the left border.
        * 1: positive integer distance from the center in the right border.

    """

    # Initialization
    window_info = [-50, 50]

    complete0 = False
    complete1 = False

    question0 = "Choose the correct window left border? (Integer number)\n"
    question1 = "Choose the correct window right border? (Integer number)\n"
    question2 = "It is your final guess for this question? (Y/N)\n"

    # Subsampling
    nmax = 10000
    if nmax < dis_ts.shape[0]:
        idx = np.random.permutation(dis_ts.shape[0])
        dis_ts_ss = dis_ts.irow(idx[:nmax])
    else:
        dis_ts_ss = dis_ts

    # Align with the given window
    spks_waveforms, times, neurons = align_spk_waveforms(activation,
                                                         dis_ts_ss,
                                                         window_info)
    # Create plot
    fig = plot_waveform_typo(times, spks_waveforms)
    fig.show()

    # Decision of the window
    while True:
        # Set intial border window
        if not complete0:
            begin = raw_input(question0)
            window_info[0] = int(begin)
        elif not complete1:
            end = raw_input(question1)
            window_info[1] = int(end)
        else:
            break

        # Align with the given window
        spks_waveforms, times, neurons = align_spk_waveforms(activation,
                                                             dis_ts_ss,
                                                             window_info)
        # Create plot
        fig = plot_waveform_typo(times, spks_waveforms)
        fig.show()

        # Check the correctness
        correctness = raw_input(question2)
        if correctness == 'Y':
            if not complete0:
                complete0 = True
            else:
                break
        else:
            pass

    return window_info


def filter_borders(activation, dis_ts, window_info):
    """This function is an auxiliary function to filter the spikes which fall
    in the borders and the window will cover more time that the one available
    by the discretized dynamics.

    Parameters
    ----------
    activation : pd.DataFrame
        activity information of the system.
    dis_ts : pd.DataFrame
        discretize information of the time-series after regime detection.
    window_info : list or tuple
        it is a two element array like which describe how many frames are the
        borders of the window from the center.
        * 0: negative integer distance from the center in the left border.
        * 1: positive integer distance from the center in the right border.

    Returns
    -------
    dis_ts_co : pd.DataFrame
        spikes in the border removed.

    """

    # Filter borders
    times_dis = np.array(dis_ts['times'])
    idx_fil = np.logical_and(times_dis <= activation.shape[0] - window_info[1],
                             times_dis >= - window_info[0])
    dis_ts_co = dis_ts[:][idx_fil]

    return dis_ts_co


def set_feature_extraction(spks_waveforms, nmax=10000):
    """Set the method of transformation (in which feature space represent the
    spikes waveforms) and the method of statistical compression.
    This function provides and interface to interact and decide these 2
    parameters in order to carry out the feature extraction from the waveforms.

    Parameters
    ----------
    spks_waveforms : array_like, shape(N,M)
        the N spikes represented by M measures.
    nmax : integer
        number of samples to use in order to save computational cost in the
        parameters setting phase.

    Returns
    -------
    method_feature : str, optional
        the method to use in order to extract features.
    method_compression:
        the method to compress the information of the spikes getting rid of the
        statistical noise and minimizing the representation variables.

    TODO
    ----
    include nmax
    """
    poss_q1 = ['', 'diff', 'diff_agg', 'wavelets', 'peaktovalley']
    poss_q1a = ['haar', 'daubechies', 'bspline', 'symlets', 'coiflets',
                'biorthogonal', 'rbiorthogonal', 'dMeyer']
    poss_q2 = ['pca', 'ica']

    question0 = "Use default values? ["'"diff"'", "'"pca"'"] (Y/N)\n"
    question1 = "Introduce the a feature space from:\n%s\n" % str(poss_q1)
    question1a = "Introduce a wavelet form from:\n%s\n" % str(poss_q1a)
    question2 = "Introduce the a compression method from:\n%s\n" % str(poss_q2)

    # Subsampling
    if nmax < spks_waveforms.shape[0]:
        idx = np.random.permutation(spks_waveforms.shape[0])
        spks_waveforms_ss = spks_waveforms[idx[:nmax], :]
    else:
        spks_waveforms_ss = spks_waveforms

    # Selecting parameters
    complete = False
    while not complete:
        kwargs = {}
        default_op = raw_input(question0)
        if default_op == 'N':
            method_feature = raw_input(question1)
            if method_feature == 'wavelets':
                kwargs['method'] = raw_input(question1a)
            method_compression = raw_input(question2)
        else:
            method_feature = 'diff'
            method_compression = 'pca'

        Xtrans = feature_extraction(spks_waveforms_ss, method_feature,
                                    method_compression, **kwargs)
        # Plot features of Xtrans (import plotting)
        fig = plot_spks_2d_featurespc(Xtrans)
        fig.show()
        question3 = """It is correct? (Y/N)\n"""
        correctness = raw_input(question3)
        if correctness == 'Y':
            complete = True

    return method_feature, method_compression, kwargs


def set_manual_postfiltering(Xtrans, spks_waveforms, window_info):
    """This function acts as an interface of interaction with the user in order
    to select the correct threshold and direction of the inequality in the
    filter process of the undesired spike detectections.

    Parameters
    ----------
    Xtrans : array_like, shape(N,M)

    Returns
    -------
    thr : float
        the value of the threshold to be applied with the given method in the
        given feature space.
    inequality : str, optional
        the value of the inequality:
        * '1': <=
        * '2': >

    TODO
    ----
    Other detection methods
    """

    # Questions
    question1 = "Introduce the selected threshold for variable 1 (x axis)\n"
    question1a = "Introduce (1/2)\n'1' Smaller or equal\n'2' Bigger\n"
    question2 = "It is your best result? Y/N (If yes, end the detection)\n"

    # Subsampling
    nmax = 10000
    if nmax < Xtrans.shape[0]:
        idx = np.random.permutation(Xtrans.shape[0])
        Xtrans_ss = Xtrans[idx[:nmax], :]
    else:
        Xtrans_ss = Xtrans

    # Set variables
    times = np.arange(window_info[0], window_info[1])

    # Loop for setting the correct parameters
    complete = False
    while not complete:
        # Plot features of Xtrans (import plotting)
        fig = plot_spks_2d_featurespc(Xtrans_ss)
        fig.show()
        # Ask for threshold
        thr = raw_input(question1)
        thr = float(thr)
        inequality = raw_input(question1a)
        # Apply threshold with inequality
        kwargs = {'threshold': thr, 'inequality': inequality}
        ys = post_filtering(Xtrans_ss, 'manually_1d', **kwargs)
        # Show results (import plotting)
        fig = plot_waveforms_sorting(times, spks_waveforms[ys == 1, :],
                                     spks_waveforms[ys == 0, :])
        fig.show()
        # Ask correctness
        correctness = raw_input(question2)
        complete = True if correctness == 'Y' else False

    return thr, inequality


################### Wrapper function to sorting detection #####################
###############################################################################
def sorting_detection(activation, threshold=None, window_info=None, thr=None):
    """It is possible to use sorting techniques and conservatice peak detection
    algorithms convined in order to improve the tasks of peak detection.

    The first step is to detect in a conservative way (low threshold) possible
    peak candidates.
    Secondly, we take this candidates and using domain knowledge about the
    correct time scale of the peaks, we are going to align the candidates in
    order to unify its temporal domain and could use the own time serie as
    features or extract other ones.
    Finally, it is shown in 2D the candidate spikes and we try to select a
    threshold in the first feature in order to classify the ones which actually
    are peaks and the others.

    TODO
    ----
    Selection of the window_info along the running.
    Select the 100 biggest up deviations and plot with different sizes.
    Selection of threshold along the time.
    """

    ## 1. Conservative detection
    # selection of the threshold
    if threshold is None:
        threshold = set_threshold_detection(activation)
    # regime detection
    dis_ts = global_regime_detection(activation, threshold)

    ## 2. Aligning
    # selection of the aligning
    if window_info is None:
        window_info = set_align_waveforms(activation, dis_ts)
    # filter dis_ts borders
    dis_ts = filter_borders(activation, dis_ts, window_info)
    # align waveforms
    spks_waveforms, times, neurons = align_spk_waveforms(activation, dis_ts,
                                                         window_info)

    ## 3. Feature extraction
    # selection of the parameters
    m_feature, m_compression, kwargs = set_feature_extraction(spks_waveforms)

    # transformation
    Xtrans = feature_extraction(spks_waveforms, m_feature,
                                m_compression)

    ## 4. Select threshold and perform post-filtering
    # selection of the parameters
    thr, inequality = set_manual_postfiltering(Xtrans, spks_waveforms,
                                               window_info)
    # transformation
    kwargs = {'threshold': thr, 'inequality': inequality}
    ys = post_filtering(Xtrans, 'manually_1d', **kwargs)

    return dis_ts, ys


############################## Useful functions ###############################
###############################################################################
def align_spk_waveforms(dynamics, spk_info, window_info):
    """Align spikes for all the dynamics of spikes.

    Parameters
    ----------
    dynamics : pd.DataFrame
        dynamics of activity in each measure.
    spk_info : pd.DataFrame
        dynamics described by its dynamics.

    Return
    ------
    spks_waveforms : array_like shape (Nt,Ns)
        the matrix with all the points during the spks.

    TODO
    ----
    Include neuron information.
    """

    ## 1. Filter spikes in the extremes
    spk_info = spk_info[spk_info['times'] > - window_info[0]]
    boolfilter2 = spk_info['times'] < (dynamics.shape[0] - window_info[1])
    spk_info = spk_info[boolfilter2]

    ## 2. Define ranges, neurons and times
    spk_info = spk_info[['times', 'neuron']].as_matrix()
    ranges = np.zeros((spk_info.shape[0], 2))
    ranges[:, 0] = (spk_info[:, 0] + window_info[0])
    ranges[:, 1] = (spk_info[:, 0] + window_info[1])
    ranges = ranges.astype(int)
    times = np.arange(window_info[0], window_info[1])
    neurons = spk_info[:, 1].astype(int)
    dynamics = dynamics.as_matrix()

    ## 3. Get matrix of activities
    spks_waveforms = np.zeros((ranges.shape[0], times.shape[0]))
    for i in range(ranges.shape[0]):
        spks_waveforms[i, :] = dynamics[ranges[i, 0]:ranges[i, 1], neurons[i]]
    return spks_waveforms, times, neurons


def post_filtering(Xtrans, method, **kwargs):
    import sklearn

    if method == 'manually_1d':
        threshold = kwargs['threshold']
        inequality = kwargs['inequality']
        if inequality == '1':
            ys = np.array(Xtrans[:, 0]) <= threshold
        elif inequality == '2':
            ys = np.array(Xtrans[:, 0]) > threshold
        ys = np.array(ys).astype(int)

    elif method == '2d-zscore':
        zscore = kwargs['zscore']
        cluster = sklearn.mixture.GMM(n_components=1)
        cluter.fit(Xtrans[:, :2])
        ys = cluster.predict_proba(Xtrans[:, :2])
        ys = ys >= zscore
        ys = np.array(ys).astype(int)

    elif method == 'spectral clustering':
        cluster = sklearn.cluster.SpectralClustering()
        ys = cluster.fit_predict(Xtrans)

    elif method == 'gmm':
        cluster = sklearn.mixture.GMM()
        ys = cluster.fit_predict(Xtrans)

    elif method == 'kmeans':
        cluster = sklearn.cluster.KMeans()
        ys = cluster.fit_predict(Xtrans)

    elif method == 'minibatchkmeans':
        cluster = sklearn.cluster.MiniBatchKMeans()
        ys = cluster.fit_predict(Xtrans)

    elif method == 'meanshift':
        cluster = sklearn.cluster.MeanShift()
        ys = cluster.fit_predict(Xtrans)

    return ys
