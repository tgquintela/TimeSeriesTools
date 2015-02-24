

"""
Module that group all the functions related directly with detection of peaks
using sorting techniques.
"""


##################### Wrapper class to regime detection #######################
###############################################################################
class Regime_detection():
    '''General regime detection represents the operation of transform the
    time series which describes the dynamics of the system to a regime based
    dynamics representation usually domain knowledge of the system and
    statistical information of the whole set of spike detections by simpler
    algorithms of regime detection.
    '''

    def __init__(self, interaction='default', method=''):

        # Inputs correction
        possible = ['default', 'read_file', 'tui', 'gui']
        interaction = interaction if interaction in possible else 'default'

        # Instantiation of the method
        self.method = None
        if interaction == 'default':
            # TODO: Transform methods to paths
            self.method = Automatic_detection(method)
        elif interaction == 'read_file':
            self.method = Automatic_detection(method)
        elif interaction == 'tui':
            self.method = Sorting_detection_tui()
        elif interaction == 'gui':
            self.method = Sorting_detection_gui()

    def detect(self, activation):
        dis_ts, parameters = self.method.detect(activation)
        return dis_ts, parameters


#################### Wrapper classes to regime detection ######################
###############################################################################
class Automatic_detection():
    '''Class oriented to perform detection task in time series automatically,
    following the instructions given by a file or other data structure.

    TODO
    ----
    Support for more than one process concatenated.
    '''

    def __init__(self, instructions):
        self.pars = None
        if type(instructions) == str:
            # TODO: Read file and transform to special dict structure
            pass
        elif type(instructions) == dict:
            self.pars = instructions

    def set_parameters(self, activation):
        # TO program setting_parameters
        self.pars = setting_parameters(activation, self.pars)

    def detect(self, activation):
        if self.pars is None or self.pars == {}:
            self.set_parameters(activation)
        regimes = general_regime_detection(activation, **self.pars)
        regimes = dense2sparse(regimes)
        return regimes, parameters


#############
##############
#### pipeline
###########
#### Possible tasks:
### Visualization ts
### Transformation
### Regime detection:   general_regime_detection
### 


def setting_parameters(activation, pars):


    ## Prepare needed variables
    # Prepare dictionary
    if pars is None:
        pars = {}

    # Load db (TODO)

    # Subsample of X
    # Extract usable variables from input
    times = np.array(activation.index)
    # Subsampling
    activity_n = subsampling_matrix(activation.as_matrix(), 1, samples=n_ts)

    # Set plots limits
    # w_init, w_final

    # Set initial transformation
    pars = set_transf_signals(times, activity_n, pars)

    # Set thresholds





def set_filter_signals(times, activity_n, pars=None):
    """
    TODO
    ----
    Link to plots.
    db loader
    """
    ## Prepare needed variables
    # Prepare dictionary
    if pars is None:
        pars = {}
    # Load db

    ## 1. Filteringdict
    # Ask filter dict
    filterdict = {}
    filterdict = {'method': 'filtering'}
    filterdict['args'] = []

    ## 2. Question
    while not complete:
        # preplot (TODO)
        fig = plot_signals(times, activity_n)
        fig.show()
        # Asking
        aux = automatic_questioner('general_transformation', db, filterdict)
        # Application
        Xaux = general_transformation(activity_n, **aux)
        # postplot (TODO)
        fig = plot_signals(times, Xaux)
        fig.show()        
        # Ask for correctness
        complete = confirmation_question()

    filterdict = aux
    return filterdict


def set_transf_signals(times, activity_n, pars=None):
    """
    TODO
    ----
    Link to plots.
    db loader
    """
    ## Prepare needed variables
    # Prepare dictionary
    if pars is None:
        pars = {}
    # Load db (TODO)

    ## 1. Transformation
    # Define plot to see filters
    # Ask filter dict
    transfdict = pars
    while not complete:
        # preplot (TODO)
        fig = plot_signals(times, activity_n)
        fig.show()
        # Asking
        aux = automatic_questioner('general_transformation', db, transfdict)
        # Application
        Xaux = general_transformation(activity_n, **aux)
        # postplot (TODO)
        fig = plot_signals(times, Xaux)
        fig.show()        
        # Ask for correctness
        complete = confirmation_question()

    transfdict = aux
    return transfdict


def set_refs_signals(times, activity_n, pars=None):
    """
    TODO
    ----
    Link to plots.
    db loader
    """
    ## Prepare needed variables
    # Prepare dictionary
    if pars is None:
        pars = {}
    # Load db (TODO)

    ## 1. Transformation
    # Define plot to see filters
    # Ask filter dict
    transfdict = pars
    while not complete:
        # preplot (TODO)
        fig = plot_signals(times, activity_n)
        fig.show()
        # Asking
        aux = automatic_questioner('general_transformation', db, transfdict)
        # Application
        Xaux = general_transformation(activity_n, **aux)
        # postplot (TODO)
        fig = plot_signals(times, Xaux)
        fig.show()        
        # Ask for correctness
        complete = confirmation_question()

    transfdict = aux
    return transfdict


def set_units_signals(times, activity_n, pars=None):
    """
    TODO
    ----
    Link to plots.
    db loader
    """
    ## Prepare needed variables
    # Prepare dictionary
    if pars is None:
        pars = {}
    # Load db (TODO)

    ## 1. Transformation
    # Define plot to see filters
    # Ask filter dict
    unitsdict = pars
    while not complete:
        # preplot (TODO)

        fig.show()
        # Asking (TODO)
        aux = automatic_questioner('', db, unitsdict)
        # Application
        Xaux = general_transformation(activity_n, **aux)
        # postplot (TODO)

        fig.show()        
        # Ask for correctness
        complete = confirmation_question()

    unitsdict = aux
    return unitsdict


def set_thres_signals(times, activity_n, pars=None):
    """
    TODO
    ----
    Link to plots.
    db loader
    """
    ## Prepare needed variables
    # Prepare dictionary
    if pars is None:
        pars = {}
    # Load db (TODO)

    ## 1. Transformation
    # Define plot to see filters
    # Ask filter dict
    thresdict = pars
    while not complete:
        # preplot (TODO)

        fig.show()
        # Asking
        aux = automatic_questioner('', db, thresdict)
        # Application
        Xaux = general_transformation(activity_n, **aux)
        # postplot (TODO)

        fig.show()        
        # Ask for correctness
        complete = confirmation_question()

    thresdict = aux
    return thresdict




class Residual():
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
