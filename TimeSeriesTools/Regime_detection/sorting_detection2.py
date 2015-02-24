
"""Module that group all the functions related directly with detection of peaks
using sorting techniques.
"""


from pyCausality.Plotting.sorting_plots import *
from pyCausality.TimeSeries.Sorting.sorting import *

from regime_detection import *
from feature_extraction import *
from tui import *


##################### Wrapper class to sorting detection ######################
###############################################################################
class Sorting_detection():
    '''Sorting detection represents the operation of transform the time series
    of the dynamics of the system to a regime based dynamics representation by
    using statistical information of the whole set of spike detections by
    simpler algorithms of regime detection.
    '''

    def __init__(self):
        self.thr_detection = None
        self.align_wv = None
        self.feat_extraction = None
        self.postfilter = None

    def set_threshold_detection(self, activation):
        aux = set_threshold_detection(activation)
        self.thr_detection = {}
        self.thr_detection['threshold'] = aux

    def set_align_waveforms(self, activation, dis_ts):
        self.align_wv = {}
        self.align_wv['window_info'] = set_align_waveforms(activation, dis_ts)

    def set_feature_extraction(self, spks_waveforms):
        aux = set_feature_extraction(spks_waveforms)
        self.feat_extraction = {}
        self.feat_extraction['method_feature'] = aux[0]
        self.feat_extraction['method_compression'] = aux[1]
        self.feat_extraction['kwargs'] = aux[2]

    def set_manual_postfiltering(self, Xtrans, spks_waveforms):
        w_info = self.align_wv['window_info']
        thr, inequality = set_manual_postfiltering(Xtrans, spks_waveforms,
                                                   w_info)
        self.postfilter = {'threshold': thr, 'inequality': inequality}

    def sort_detection(self, activation):
        ## 1. Conservative detection
        # selection of the threshold
        if self.thr_detection is None:
            self.set_threshold_detection(activation)
        # regime detection
        dis_ts = global_regime_detection(activation, **self.thr_detection)

        ## 2. Aligning
        # selection of the aligning
        if self.align_wv is None:
            self.set_align_waveforms(activation, dis_ts)
        # filter dis_ts borders
        dis_ts = filter_borders(activation, dis_ts, **self.align_wv)
        # align waveforms
        spks_waveforms, times, neurons = align_spk_waveforms(activation,
                                                             dis_ts,
                                                             **self.align_wv)
        ## 3. Feature extraction
        # selection of the parameters
        if self.feat_extraction is None:
            self.set_feature_extraction(spks_waveforms)
        # transformation
        Xtrans = feature_extraction(spks_waveforms, **self.feat_extraction)

        ## 4. Select threshold and perform post-filtering
        # selection of the parameters
        if self.postfilter is None:
            self.set_manual_postfiltering(Xtrans, spks_waveforms)
        # transformation
        ys = post_filtering(Xtrans, 'manually_1d', **self.postfilter)

        return dis_ts, ys


############################# Auxiliary functions #############################
###############################################################################
def set_threshold_detection(activation, n_ts=10):
    '''Function to ask for the threshold in order to perform regime detection.

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
    '''

    # Extract usable variables from input
    times = np.array(activation.index)
    # Subsampling
    activity_n = subsampling_matrix(activation.as_matrix(), 1, samples=n_ts)

#    if n_ts < activation.shape[1]:
#        idx = np.random.permutation(activation.shape[1])[:n_ts]
#    else:
#        idx = range(activation.shape[1])
#    activity_n = activation.as_matrix()[:, idx]

    q_init_frame = "Set the initial frame of the window. (int)"
    q_final_frame = "Set the final frame of the window. (int)"
    q_default_values = "Do you want to keep default values?"

    ## 1. Window visualization selection
    # Loop in order to set window visualization
    complete = False
    while not complete:
        # Plot time serie
        fig = plot_signals(times, activity_n)
        fig.show()
        # Ask for window to show
        w_init = simple_input(q_init_frame, int)
        w_end = simple_input(q_final_frame, int)
        # Show result
        fig = plot_signals(times[w_init:w_end], activity_n[w_init:w_end, :])
        fig.show()
        # Ask for completeness
        complete = confirmation_question()

    ## 2. Transformation selection  (TO COMPLETE)
    # Loop in order to set window visualization
    complete = False
    while not complete:
        kwargs = {}
        default_bool = confirmation_question(q_default_values)

        ####### TODO
        if not default_bool:
            # Read parameters
            method = 'ups_downs'
        else:
            # Ask parameters
            method = 'ups_downs'

        # Apply transformation
        activity_nt = temporal_discretization(activity_n, method, sparse=False)
        # Visualize transformation
        fig = plot_signals(times[w_init:w_end], activity_nt[w_init:w_end, :])
        fig.show()
        # Ask for completeness
        complete = confirmation_question()

    ## 3. Threshold selection
    # Init default values

    # Loop of questions
    complete = False
    while not complete:
        # Plot time serie
        fig = plot_signals_thr_ind(times, activity_n, default_thres)
        fig.show()
        # Ask for threshold values
        units_spkdetect = selection_options(q_units_spkdetect, 'units',
                                            op_units_spkdetect)
        collap_spkdetect = selection_options(q_collap_spkdetect, 'collapse_to',
                                             op_collapse_spkdetect)
        ref_spkdetect = selection_options(q_ref_spkdetect, 'reference',
                                          op_ref_spkdetect)
        threshold = simple_input(q_thresh_spkdetect, float)

        # Apply threshold

        # Create extras
        extras = {'reference': ref_spkdetect,
                  'units_reference': units_spkdetect,
                  'threshold': threshold}
        # Plot results (TOTEST) 2 plots 1 with threshold and other with results
        fig = plot_confirmation_thr_ind(times, signals, extras, spks)
        fig.show()
        # Ask for completeness
        complete = confirmation_question()

    return threshold


def set_align_waveforms(activation, dis_ts):
    '''Selection of the window information.

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

    '''

    # Initialization (TO DEFAULT)
    window_info = [-50, 50]

    # Subsampling
    dis_ts_ss = subsampling_spks(dis_ts)

    # Align with the given window
    spks_waveforms, times, neurons = align_spk_waveforms(activation,
                                                         dis_ts_ss,
                                                         window_info)
    # Create plot
    fig = plot_waveform_typo(times, spks_waveforms)
    fig.show()

    # Loop booleans
    complete0 = False
    complete1 = False
    # Decision of the window
    while True:
        # Set intial border window
        if not complete0:
            window_info[0] = simple_input(q_w_info_left, int)
        elif not complete1:
            window_info[1] = simple_input(q_w_info_right, int)
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
        correctness = confirmation_question()
        if correctness:
            if not complete0:
                complete0 = True
            else:
                break

    return window_info


def filter_borders(activation, dis_ts, window_info):
    '''This function is an auxiliary function to filter the spikes which fall
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

    '''

    # Filter borders
    times_dis = np.array(dis_ts['times'])
    idx_fil = np.logical_and(times_dis <= activation.shape[0] - window_info[1],
                             times_dis >= - window_info[0])
    dis_ts_co = dis_ts[:][idx_fil]

    return dis_ts_co


def set_feature_extraction(spks_waveforms, nmax=10000):
    '''Set the method of transformation (in which feature space represent the
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
    '''

    # Subsampling
    if nmax < spks_waveforms.shape[0]:
        idx = np.random.permutation(spks_waveforms.shape[0])
        spks_waveforms_ss = spks_waveforms[idx[:nmax], :]
    else:
        spks_waveforms_ss = spks_waveforms

    # Selecting parameters
    complete = False
    while not complete:
        default_bool = confirmation_question(q_default_values)
        if default_bool:
            #Reset or assign the default values
            method_feature = 'diff'
            method_compression = 'pca'
            break

        method_feature = selection_options(q_fext_featspace, 'feature space',
                                           op_trans_featspace)

        kwargs_feat =  # function to built kwargs

        if method_feature == 'wavelets':
            kwargs['method'] = raw_input(question1a)
            method_compression = raw_input(question2)

        Xtrans = feature_extraction(spks_waveforms_ss, method_feature,
                                    method_compression, **kwargs)
        # Plot features of Xtrans (import plotting)
        fig = plot_spks_2d_featurespc(Xtrans)
        fig.show()
        # Ask for completeness
        complete = confirmation_question()

    return method_feature, method_compression, kwargs


def set_manual_postfiltering(Xtrans, spks_waveforms, window_info):
    '''This function acts as an interface of interaction with the user in order
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
    '''

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
    '''It is possible to use sorting techniques and conservatice peak detection
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
    '''

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
